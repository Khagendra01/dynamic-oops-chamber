"""Batched execution of *different* frozen-DSL programs, one per batch row.

Candidate-evaluation kernel for GPU OOPS prefix search: each row runs its own
program on its own scene, and instruction-trace events plus the weighted work
proxy are reproduced exactly so a row's EpisodeResult matches
`oops_training.replay` field for field.

Two properties matter for speed and for torch.compile stability:
  * every safety query of a control step is stacked into ONE
    `TorchChamberBatch.motion_safe` call (kernel launches, not FLOPs, decide);
  * the per-step Python structure is batch-invariant (fixed program width L and
    every rule kind evaluated unconditionally) so a single compiled graph
    serves every batch instead of recompiling for ~10 minutes each time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from oops_language import Program
from torch_chamber import TorchChamberBatch

# Canonical lexical order: CHOOSE_AXIS tie-breaks by ascending action name.
DIRECTIONS = ("X_NEG", "X_POS", "Y_NEG", "Y_POS", "Z_NEG", "Z_POS")
RULE_KINDS = ("GOAL", "CHOOSE_AXIS", "WAIT_GAP", "CONTINUE", "LAST") + DIRECTIONS
GOAL, CHOOSE_AXIS, WAIT_GAP, CONTINUE, LAST = range(5)
KIND_ID = {name: i for i, name in enumerate(RULE_KINDS)}
PAD_RULE = len(RULE_KINDS)

_DIR_VECTORS = {
    "X_POS": (1.0, 0.0, 0.0), "X_NEG": (-1.0, 0.0, 0.0),
    "Y_POS": (0.0, 1.0, 0.0), "Y_NEG": (0.0, -1.0, 0.0),
    "Z_POS": (0.0, 0.0, 1.0), "Z_NEG": (0.0, 0.0, -1.0),
}

# Safety queries per kind. CHOOSE_AXIS probes each axis now plus the goal
# follow-up one step later; WAIT_GAP probes direct, standing, and 1..4 ahead.
_QUERY_COUNT = (1, 12, 6, 1, 1, 1, 1, 1, 1, 1, 1)
QUERY_OFFSET: list[int] = []
_cursor = 0
for _count in _QUERY_COUNT:
    QUERY_OFFSET.append(_cursor)
    _cursor += _count
N_QUERIES = _cursor


@dataclass(frozen=True)
class CompiledProgram:
    """CALL-expanded rule list with per-position VM overhead accounting.

    A `CALL` costs one VM instruction plus one trace entry (also one work
    unit); a frame that runs off its end costs one more VM instruction
    (`END_NO_MATCH`) with no trace entry. `pre_*` is owed before rule i is
    attempted; `post_*` when no rule fires at all.
    """

    rules: tuple[str, ...]
    pre_ev: tuple[int, ...]
    pre_wk: tuple[int, ...]
    pre_in: tuple[int, ...]
    post_ev: int
    post_wk: int
    post_in: int

    @property
    def length(self) -> int:
        return len(self.rules)


def compile_program(program: Program, frozen: dict[str, Program],
                    *, max_call_depth: int = 6) -> CompiledProgram:
    """Flatten CALLs exactly as Interpreter executes them, counting overhead."""
    flat: list[str] = []
    pre_ev: list[int] = []
    pre_wk: list[int] = []
    pre_in: list[int] = []
    pend_ev = pend_wk = pend_in = 0
    post_ev = post_wk = post_in = 0

    def walk(prog: Program, stack: tuple[str, ...]) -> None:
        nonlocal pend_ev, pend_wk, pend_in, post_ev, post_wk, post_in
        if len(stack) >= max_call_depth or prog.id in stack:
            # "CALL DEPTH LIMIT" / "RECURSIVE CALL BLOCKED": one trace entry,
            # no bytecode instruction inside the blocked frame.
            pend_ev += 1
            pend_wk += 1
            return
        for rule in prog.rules:
            if rule.startswith("CALL:"):
                pend_in += 1                      # the CALL instruction itself
                pend_ev += 1                      # "CALL id" / "MISSING CALL id"
                pend_wk += 1
                sub = frozen.get(rule.split(":", 1)[1])
                if sub is not None:
                    walk(sub, (*stack, prog.id))
            else:
                pre_ev.append(pend_ev)
                pre_wk.append(pend_wk)
                pre_in.append(pend_in)
                pend_ev = pend_wk = pend_in = 0
                flat.append(rule)
        if stack:
            pend_in += 1                          # sub-frame END_NO_MATCH
        else:
            post_in += 1                          # top-frame END_NO_MATCH

    walk(program, ())
    return CompiledProgram(
        tuple(flat), tuple(pre_ev), tuple(pre_wk), tuple(pre_in),
        post_ev + pend_ev, post_wk + pend_wk, post_in + pend_in)


def _aim_costs(work: int) -> tuple[int, int, int, int, int, int]:
    """(fail events, fail work, fail instr, ok events, ok work, ok instr)."""
    return (2, work + 1, 3, 4, work + 3, 5)


_ROW_CACHE: dict[tuple, tuple[bytes, bytes, bytes, bytes]] = {}
_CONST_CACHE: dict[tuple, tuple] = {}


def _packed_row(prog: CompiledProgram, width: int):
    """Padded [width] int64 rows as bytes, memoised per program."""
    key = (prog, width)
    cached = _ROW_CACHE.get(key)
    if cached is None:
        rules = np.full(width, PAD_RULE, dtype=np.int64)
        pre_ev = np.zeros(width, dtype=np.int64)
        pre_wk = np.zeros(width, dtype=np.int64)
        pre_in = np.zeros(width, dtype=np.int64)
        for i, name in enumerate(prog.rules):
            rules[i] = KIND_ID[name]
        n = prog.length
        pre_ev[:n] = prog.pre_ev
        pre_wk[:n] = prog.pre_wk
        pre_in[:n] = prog.pre_in
        cached = (rules.tobytes(), pre_ev.tobytes(), pre_wk.tobytes(),
                  pre_in.tobytes())
        _ROW_CACHE[key] = cached
    return cached


def _column(blobs: list[bytes], rows: int, width: int) -> np.ndarray:
    return np.frombuffer(b"".join(blobs), dtype=np.int64).reshape(rows, width).copy()


def _constants(device, dtype):
    """Per-kind cost table and direction vectors, built once per device."""
    key = (str(device), dtype)
    cached = _CONST_CACHE.get(key)
    if cached is None:
        cost = torch.zeros((len(RULE_KINDS), 6), dtype=torch.int64, device=device)
        for name, kind in KIND_ID.items():
            if kind == WAIT_GAP:
                row = (1, 4, 2, 3, 6, 3)
            else:
                work = 8 if kind == CHOOSE_AXIS else 4 if kind == GOAL else 1
                row = _aim_costs(work)
            cost[kind] = torch.tensor(row, dtype=torch.int64, device=device)
        dir_table = torch.tensor(
            [_DIR_VECTORS.get(name, (0.0, 0.0, 0.0)) for name in RULE_KINDS],
            dtype=dtype, device=device)
        axis_ids = torch.tensor([KIND_ID[n] for n in DIRECTIONS],
                                dtype=torch.int64, device=device)
        cached = (cost, dir_table, axis_ids)
        _CONST_CACHE[key] = cached
    return cached


class TorchJobPolicy:
    """One control step for J rows, each running its own compiled program."""

    def __init__(self, env: TorchChamberBatch, programs: list[CompiledProgram],
                 *, program_width: int, stride: float = .12,
                 instruction_limit: int = 128):
        if len(programs) != env.B:
            raise ValueError("One compiled program is required per batch row")
        self.env = env
        self.stride = float(stride)
        self.instruction_limit = int(instruction_limit)
        self.J = env.B
        self.L = int(program_width)
        device, dtype = env.device, env.dtype
        if any(p.length > self.L for p in programs):
            raise ValueError("Flattened program exceeds program_width")

        packed = [_packed_row(p, self.L) for p in programs]
        rows = len(packed)
        self.rules = torch.from_numpy(_column([r[0] for r in packed], rows,
                                              self.L)).to(device)
        self.pre_ev = torch.from_numpy(_column([r[1] for r in packed], rows,
                                               self.L)).to(device)
        self.pre_wk = torch.from_numpy(_column([r[2] for r in packed], rows,
                                               self.L)).to(device)
        self.pre_in = torch.from_numpy(_column([r[3] for r in packed], rows,
                                               self.L)).to(device)
        self.length = torch.from_numpy(
            np.array([p.length for p in programs], dtype=np.int64)).to(device)
        self.post_ev = torch.from_numpy(
            np.array([p.post_ev for p in programs], dtype=np.int64)).to(device)
        self.post_wk = torch.from_numpy(
            np.array([p.post_wk for p in programs], dtype=np.int64)).to(device)
        self.post_in = torch.from_numpy(
            np.array([p.post_in for p in programs], dtype=np.int64)).to(device)
        self.cost, self.dir_table, self.axis_ids = _constants(device, dtype)

        self.last_direction = torch.zeros((self.J, 3), dtype=dtype, device=device)
        self.explicit_waits = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.fallback_waits = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.continued = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.directional = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.trace_events = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.work_units = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.interventions = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.emissions = torch.zeros(self.J, dtype=torch.int64, device=device)
        self.max_instructions = torch.zeros(self.J, dtype=torch.int64, device=device)
        self._row = torch.arange(self.J, device=device)

    # ------------------------------------------------------------------ helpers
    def _velocity(self, direction: torch.Tensor, *, cap: bool = False) -> torch.Tensor:
        length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        dist = min(self.stride, self.env.max_speed * self.env.dt)
        if cap:
            dist = torch.minimum(length, torch.full_like(length, dist))
        return direction / length.clamp_min(1e-12) * dist / self.env.dt

    def _build_queries(self, goal_dir: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Stack all N_QUERIES safety probes of one control step."""
        env = self.env
        J, dtype, device = env.B, env.dtype, env.device
        pos, last = env.position, self.last_direction
        t = env.step_index.to(dtype) * env.dt
        zero = torch.zeros_like(pos)
        direct = self._velocity(goal_dir, cap=True)
        axis_v = self._velocity(self.dir_table[self.axis_ids].expand(6, -1))
        start: list[torch.Tensor] = [None] * N_QUERIES
        end: list[torch.Tensor] = [None] * N_QUERIES
        t0: list[torch.Tensor] = [None] * N_QUERIES

        def put(base: int, s: torch.Tensor, e: torch.Tensor,
                tt: torch.Tensor) -> None:
            """s/e are [J,N,3]; tt is [J,N] or [J] (broadcast over N)."""
            n = s.shape[1]
            if tt.dim() == 1:
                tt = tt.unsqueeze(1).expand(J, n)
            for k in range(n):
                start[base + k] = s[:, k]
                end[base + k] = e[:, k]
                t0[base + k] = tt[:, k]

        def one(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return pos.unsqueeze(1), (pos + v * env.dt).unsqueeze(1), t.unsqueeze(1)

        s, e, tt = one(direct)
        put(QUERY_OFFSET[GOAL], s, e, tt)

        cb = QUERY_OFFSET[CHOOSE_AXIS]
        after = pos.unsqueeze(1) + axis_v.unsqueeze(0) * env.dt    # [J,6,3]
        put(cb, pos.unsqueeze(1).expand(J, 6, 3), after,
            t.unsqueeze(1).expand(J, 6))
        goal_v = self._velocity(env.goal.unsqueeze(1) - after, cap=True)
        put(cb + 6, after, (after + goal_v * env.dt),
            (t + env.dt).unsqueeze(1).expand(J, 6))

        gb = QUERY_OFFSET[WAIT_GAP]
        s, e, tt = one(direct)
        put(gb, s, e, tt)
        put(gb + 1, pos.unsqueeze(1), pos.unsqueeze(1), t.unsqueeze(1))
        ahead = torch.arange(1, 5, device=device, dtype=dtype)
        put(gb + 2, pos.unsqueeze(1).expand(J, 4, 3),
            (pos + direct * env.dt).unsqueeze(1).expand(J, 4, 3),
            t.unsqueeze(1) + ahead * env.dt)

        for kind in (CONTINUE, LAST):
            s, e, tt = one(self._velocity(last))
            put(QUERY_OFFSET[kind], s, e, tt)
        for name in DIRECTIONS:
            v = self._velocity(self.dir_table[KIND_ID[name]].expand(J, -1))
            s, e, tt = one(v)
            put(QUERY_OFFSET[KIND_ID[name]], s, e, tt)

        s = torch.stack(start, dim=1)                      # [J,Q,3]
        e = torch.stack(end, dim=1)
        a = torch.stack(t0, dim=1)                         # [J,Q]
        return (s.reshape(J * N_QUERIES, 3), e.reshape(J * N_QUERIES, 3),
                a.reshape(J * N_QUERIES))

    # ------------------------------------------------------- per-kind proposals
    def _kind_table(self, safe: torch.Tensor, goal_dir: torch.Tensor):
        """fire[J,11] and proposal[J,11,3] for the current observation."""
        env = self.env
        J, dtype, device = env.B, env.dtype, env.device
        row = self._row
        probe = safe.view(J, N_QUERIES)

        fire = torch.zeros(J, len(RULE_KINDS), dtype=torch.bool, device=device)
        prop = torch.zeros(J, len(RULE_KINDS), 3, dtype=dtype, device=device)

        goal_v = self._velocity(goal_dir, cap=True)
        fire[:, GOAL] = probe[:, QUERY_OFFSET[GOAL]]
        prop[:, GOAL] = goal_v

        dist0 = torch.linalg.vector_norm(goal_dir, dim=-1)
        mem_len = torch.linalg.vector_norm(self.last_direction,
                                           dim=-1).clamp_min(1e-12)
        axis_v = self._velocity(self.dir_table[self.axis_ids].expand(6, -1))
        scores, opts = [], []
        for a in range(6):
            v = axis_v[a].expand(J, -1)
            ok = probe[:, QUERY_OFFSET[CHOOSE_AXIS] + a]
            after = env.position + v * env.dt
            goal_follow = self._velocity(env.goal - after, cap=True)
            next_ok = probe[:, QUERY_OFFSET[CHOOSE_AXIS] + 6 + a]
            align = (dist0 - torch.linalg.vector_norm(env.goal - after,
                                                      dim=-1)) / max(self.stride, 1e-12)
            momentum = .18 * (v * self.last_direction).sum(dim=-1) / (
                torch.linalg.vector_norm(v, dim=-1) * mem_len)
            score = align + next_ok.to(dtype) * .65 + momentum
            scores.append(torch.where(ok, score, torch.full_like(score, -float("inf"))))
            opts.append(v)
        stacked = torch.stack(scores, dim=-1)
        # Numerical ties are common in symmetric chambers: take the first
        # maximum in lexical order, matching Interpreter's sorted() key.
        indices = torch.argmax(stacked, dim=-1)
        chosen = torch.stack(opts, dim=1)[row, indices]
        possible = torch.isfinite(stacked).any(dim=-1)
        fire[:, CHOOSE_AXIS] = possible
        prop[:, CHOOSE_AXIS] = torch.where(possible[:, None], chosen,
                                           torch.zeros_like(chosen))

        go = QUERY_OFFSET[WAIT_GAP]
        blocked = ~probe[:, go]
        standing = probe[:, go + 1]
        moving = ((env.vel0.abs() > 1e-9).any(dim=-1) & env.valid).any(dim=-1)
        opens = torch.zeros(J, dtype=torch.bool, device=device)
        for ahead in range(4):
            opens = opens | probe[:, go + 2 + ahead]
        fire[:, WAIT_GAP] = blocked & standing & moving & opens

        for kind in (CONTINUE, LAST):
            v = self._velocity(self.last_direction)
            prior = torch.linalg.vector_norm(self.last_direction, dim=-1) > 1e-12
            if kind == CONTINUE:
                nxt = torch.linalg.vector_norm(goal_dir - v * env.dt, dim=-1)
                prior = prior & (nxt <= dist0 + .02)
            fire[:, kind] = prior & probe[:, QUERY_OFFSET[kind]]
            prop[:, kind] = v

        for name in DIRECTIONS:
            kind = KIND_ID[name]
            v = self._velocity(self.dir_table[kind].expand(J, -1))
            fire[:, kind] = probe[:, QUERY_OFFSET[kind]]
            prop[:, kind] = v
        return fire, prop

    # ----------------------------------------------------------------- one step
    @torch.no_grad()
    def step(self):
        """One control step: run the program, then advance the scene."""
        env = self.env
        J, L, device = self.J, self.L, env.device
        active = ~env.done
        fired = torch.zeros(J, dtype=torch.bool, device=device)
        velocity = torch.zeros((J, 3), dtype=env.dtype, device=device)
        goal_dir = env.goal - env.position

        start, end, t0 = self._build_queries(goal_dir)
        job = self._row.repeat_interleave(N_QUERIES)
        safe = env.motion_safe(start, end, t0, job_index=job)
        fire_all, prop_all = self._kind_table(safe, goal_dir)

        events = torch.zeros(J, dtype=torch.int64, device=device)
        work = torch.zeros(J, dtype=torch.int64, device=device)
        instr = torch.zeros(J, dtype=torch.int64, device=device)
        row = self._row

        for i in range(L):
            attempt = active & (i < self.length) & ~fired
            # Padding slots are masked by `attempt`; clamp only to keep gathers in range.
            col = self.rules[:, i].clamp(max=PAD_RULE - 1)
            peel, pwk, pin = self.pre_ev[:, i], self.pre_wk[:, i], self.pre_in[:, i]
            zero_i = torch.zeros_like(events)
            events = events + torch.where(attempt, peel, zero_i)
            work = work + torch.where(attempt, pwk, zero_i)
            instr = instr + torch.where(attempt, pin, zero_i)
            would_fire = fire_all[row, col]
            fire = attempt & would_fire
            miss = attempt & ~would_fire
            costs = self.cost[col]                        # [J,6]
            events = events + torch.where(fire, costs[:, 3],
                                          torch.where(miss, costs[:, 0], zero_i))
            work = work + torch.where(fire, costs[:, 4],
                                      torch.where(miss, costs[:, 1], zero_i))
            instr = instr + torch.where(fire, costs[:, 5],
                                        torch.where(miss, costs[:, 2], zero_i))
            proposal = prop_all[row, col]
            velocity = torch.where(fire[:, None], proposal, velocity)
            aim = col != WAIT_GAP
            keep = fire & aim & (torch.linalg.vector_norm(proposal, dim=-1) > 1e-12)
            self.last_direction = torch.where(keep[:, None], proposal,
                                              self.last_direction)
            self.explicit_waits += (fire & ~aim).to(torch.int64)
            self.continued += (fire & (col == CONTINUE)).to(torch.int64)
            self.directional += (fire & (col == CHOOSE_AXIS)).to(torch.int64)
            fired = fired | fire

        fallback = active & ~fired
        events = events + torch.where(fallback, self.post_ev + 1,
                                      torch.zeros_like(events))
        work = work + torch.where(fallback, self.post_wk + 1,
                                  torch.zeros_like(work))
        instr = instr + torch.where(fallback, self.post_in,
                                    torch.zeros_like(instr))
        self.fallback_waits += fallback.to(torch.int64)
        self.trace_events += events
        self.work_units += work
        self.emissions += active.to(torch.int64)
        self.max_instructions = torch.maximum(self.max_instructions, instr)

        result = env.step(velocity, check_speed=False)
        self.interventions += result.intervention.to(torch.int64)
        return result

    @torch.no_grad()
    def run(self, *, max_steps: int, check_every: int = 8, step_fn=None):
        """Advance until every row is done or the horizon is exhausted."""
        fn = self.step if step_fn is None else (lambda: step_fn(self))
        remaining = int(max_steps)
        while remaining > 0:
            block = min(check_every, remaining)
            for _ in range(block):
                fn()
            remaining -= block
            if bool(self.env.done.all()):
                break
        if bool((self.max_instructions > self.instruction_limit).any()):
            raise RuntimeError("INSTRUCTION LIMIT hit: batched accounting diverges")
