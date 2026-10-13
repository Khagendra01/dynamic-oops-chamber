"""GPU-prefetched OOPS prefix search with bit-exact CPU reference parity.

`PrefixOOPS.train_task` is a sequential, adaptive depth-first search: a node's
child budget depends on that node's own evaluation, and the walk stops at the
first accepting prefix. Replaying it on the GPU naively would need one
synchronised kernel launch per search node.

Instead the candidate episodes of a whole stage are evaluated *ahead of the
walk*, in short-circuit waves:

  * the search tree a node can ever belong to is bounded by assuming every
    evaluation costs the minimum possible (1 unit instead of the real >= 3),
    which yields a strict superset of the executable prefixes - no evaluation
    results are needed to enumerate it;
  * wave 0 runs every prefix of that superset against the last task of the
    stage; wave w only runs prefixes that already passed tasks -1..-w, matching
    `visit()`'s `reversed(stage_tasks)` short circuit exactly, so the waves
    shrink geometrically instead of evaluating every task for every prefix;
  * `PrefixOOPS`'s own `_run_lane` then walks the real search against the warm
    cache, so every decision (found program, lane budget, visit and test counts)
    is the reference's own code and is exact by construction; keys the
    predictor missed fall back to a plain CPU replay.

All episodes of one wave go into fixed-size device batches, so the batch shape
is independent of the program and a single compiled graph serves the run.
"""
from __future__ import annotations

import time

import torch

from oops_language import Program
from oops_prefix import PrefixOOPS, Task
from oops_training import EpisodeResult, extended_curriculum
from torch_chamber import TorchChamberBatch
from torch_job_policy import CompiledProgram, TorchJobPolicy, compile_program

_COMPILED: dict[tuple, CompiledProgram] = {}


class BatchedPrefixOOPS(PrefixOOPS):
    """PrefixOOPS whose evaluations are batched on `device` before each walk."""

    def __init__(self, *, device: str = "cpu",
                 dtype: torch.dtype = torch.float64, batch_size: int = 8192,
                 program_width: int = 12, speculative_phase_limit: int = 32768,
                 merge_waves: bool = False, step_fn=None,
                 verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.program_width = int(program_width)
        self.speculative_phase_limit = int(speculative_phase_limit)
        self.merge_waves = bool(merge_waves)
        self.step_fn = step_fn
        self.verbose = verbose
        self.templates = [factory() for _name, factory, *_ in extended_curriculum()]
        self.task_index = {name: i
                           for i, (name, *_) in enumerate(extended_curriculum())}
        self.base_batch = TorchChamberBatch(self.templates,
                                            device=str(self.device),
                                            dtype=self.dtype)
        self._prefetched: set = set()
        self._stage_prefetched = False
        self._spec_limit = 0
        self._frozen_sig: tuple | None = None
        self.gpu_episodes = 0
        self.gpu_batches = 0
        self.gpu_steps = 0
        self.gpu_row_steps = 0
        self.gpu_slots = 0
        self.gpu_episode_steps = 0
        self.prefetch_seconds = 0.0
        self.walk_seconds = 0.0
        self.timers = {"env": 0.0, "programs": 0.0, "policy": 0.0,
                       "run": 0.0, "read": 0.0}

    # ----------------------------------------------------------- device runner
    def _episode_chunk(self, chunk: list) -> list[EpisodeResult]:
        n = len(chunk)
        batch = self.batch_size
        ids = [self.task_index[task.name] for _rules, task in chunk]
        ids += [0] * (batch - n)
        mark = time.perf_counter()
        env = TorchChamberBatch.from_templates(
            self.templates, torch.tensor(ids, dtype=torch.long),
            device=str(self.device), dtype=self.dtype, base=self.base_batch)
        if n < batch:
            # Padded rows must be inert: done from step zero, any program.
            env.done[n:] = True
        self.timers["env"] += time.perf_counter() - mark

        mark = time.perf_counter()
        programs = []
        for rules, _task in chunk:
            key = (tuple(rules), tuple(self.frozen))
            prog = _COMPILED.get(key)
            if prog is None:
                prog = compile_program(Program(tuple(rules)), self.frozen)
                _COMPILED[key] = prog
            programs.append(prog)
        if n < batch:
            programs = programs + [programs[0]] * (batch - n)
        self.timers["programs"] += time.perf_counter() - mark

        mark = time.perf_counter()
        policy = TorchJobPolicy(env, programs, program_width=self.program_width)
        self.timers["policy"] += time.perf_counter() - mark

        mark = time.perf_counter()
        policy.run(max_steps=int(env.max_steps_t.max()), step_fn=self.step_fn)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timers["run"] += time.perf_counter() - mark

        mark = time.perf_counter()
        done = env.done[:n].cpu().tolist()
        collision = env.collision[:n].cpu().tolist()
        distance = torch.linalg.vector_norm(
            (env.position[:n] - env.goal[:n]), dim=-1).cpu().tolist()
        tolerance = env.tolerance_t[:n].cpu().tolist()
        steps = env.step_index[:n].cpu().tolist()
        fields = {name: getattr(policy, name)[:n].cpu().tolist()
                  for name in ("interventions", "fallback_waits", "explicit_waits",
                               "continued", "directional", "trace_events",
                               "work_units")}
        self.gpu_batches += 1
        self.gpu_episodes += n
        ran = max(steps) if steps else 0
        self.gpu_steps += ran
        self.gpu_row_steps += n * ran
        self.gpu_slots += self.batch_size
        self.gpu_episode_steps += sum(steps)
        results = [EpisodeResult(
            success=bool(done[i] and not collision[i] and distance[i] <= tolerance[i]),
            collision=bool(collision[i]),
            interventions=int(fields["interventions"][i]),
            steps=int(steps[i]),
            fallback_waits=int(fields["fallback_waits"][i]),
            explicit_waits=int(fields["explicit_waits"][i]),
            continued_actions=int(fields["continued"][i]),
            directional_choices=int(fields["directional"][i]),
            trace_events=int(fields["trace_events"][i]),
            work_units=int(fields["work_units"][i]),
        ) for i in range(n)]
        self.timers["read"] += time.perf_counter() - mark
        return results

    def _episodes(self, pairs: list) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        batch = self.batch_size
        for start in range(0, len(pairs), batch):
            if self.verbose:
                print(f"    wave batch {start}/{len(pairs)} pairs", flush=True)
            results.extend(self._episode_chunk(pairs[start:start + batch]))
        return results

    # ------------------------------------------------------------- bookkeeping
    def _key(self, rules: tuple, task_name: str):
        """Cache key for a (program rules, task) episode.

        The frozen-program signature is constant for the length of a stage, so
        it is built once instead of once per pair (tens of thousands of pairs
        per stage).
        """
        sig = self._frozen_sig
        if sig is None or len(sig) != len(self.frozen):
            sig = tuple((pid, p.rules) for pid, p in self.frozen.items())
            self._frozen_sig = sig
        return (sig, rules, task_name)

    def _evaluate(self, rules: tuple, task: Task):
        key = self._key(rules, task.name)
        if key in self._cache:
            if key in self._prefetched:
                # Count it exactly as a reference cache miss - but only if the
                # walk actually asks for it.
                self._prefetched.discard(key)
                self.real_replay_calls += 1
                self.simulated_steps += self._cache[key][0].steps
            return self._cache[key]
        return super()._evaluate(rules, task)

    def train_task(self, task: Task):
        self._prefetched = set()
        self._stage_prefetched = False
        self._spec_limit = 0
        self._frozen_sig = None
        return super().train_task(task)

    # -------------------------------------------------------------- prefetching
    def _enumerate_lane(self, lane: str, budget: int, prev: Program | None):
        """Cost-is-one superset of the executable prefixes of one lane run.

        Every node consumes the smallest possible search cost, so no real node
        can lie outside this tree: real costs are larger, which only removes
        descendants.
        """
        base = prev.rules if lane == "extend" and prev is not None else ()
        if len(base) > self.max_depth:
            return []
        vocab = self._vocabulary(lane)
        nodes: list = []
        visited = 0
        limit = self.max_prefixes_per_phase
        max_rel_depth = self.max_depth - len(base)

        def walk(prefix: tuple, allocation: float, depth: int) -> None:
            nonlocal visited
            if visited >= limit or allocation < 1:
                return
            visited += 1
            if prefix:
                nodes.append((prefix, allocation))
            if depth >= max_rel_depth:
                return
            remaining = max(0.0, allocation - 1.0)
            for token, weight in vocab:
                if token in prefix:
                    continue
                child = remaining * weight
                if child < 1:
                    continue
                walk(prefix + (token,), child, depth + 1)

        walk(base, float(budget), 0)
        return nodes

    def _speculative_lane_runs(self):
        old = next(reversed(self.frozen.values())) if self.frozen else None
        runs = []
        phase = self.starting_budget
        while phase <= self._spec_limit:
            lanes = ([("extend", phase // 2), ("fresh", phase - phase // 2)]
                     if self.reuse and old else [("fresh", phase)])
            for lane, lane_budget in lanes:
                runs.append((lane, lane_budget, old,
                             self._enumerate_lane(lane, lane_budget, old)))
            phase *= 2
        return runs

    def _prefetch_stage(self, stage_tasks: list[Task]) -> None:
        runs = self._speculative_lane_runs()
        prefixes = sorted({prefix for _lane, _budget, _prev, nodes in runs
                           for prefix, _alloc in nodes if prefix})
        started = time.perf_counter()
        if self.merge_waves:
            # One batch per stage: evaluate every (prefix, task) pair up front.
            # More episodes than the short-circuit waves, but a stage's tasks no
            # longer cost one empty device batch each.
            pending = [(rules, task) for rules in prefixes
                       for task in stage_tasks
                       if self._key(rules, task.name)
                       not in self._cache]
            if pending:
                self._store(pending, self._episodes(pending))
        else:
            for wave, task in enumerate(reversed(stage_tasks)):
                pending = []
                for rules in prefixes:
                    if wave:
                        # visit() stops at the first rejecting task, so only
                        # survivors of tasks[-1..-wave] are ever asked for this
                        # one.
                        survives = True
                        for offset in range(1, wave + 1):
                            prior = stage_tasks[-offset]
                            hit = self._cache.get(
                                self._key(rules, prior.name))
                            if hit is None or not prior.accepts(hit[0]):
                                survives = False
                                break
                        if not survives:
                            continue
                    key = self._key(rules, task.name)
                    if key not in self._cache:
                        pending.append((rules, task))
                if not pending:
                    break
                self._store(pending, self._episodes(pending))
        self.prefetch_seconds += time.perf_counter() - started

    def _store(self, pairs: list, results: list) -> None:
        for index, (rules, task) in enumerate(pairs):
            key = self._key(rules, task.name)
            if key in self._cache:
                continue
            result = results[index]
            self._cache[key] = (result, max(1, result.work_units + len(rules)))
            self._prefetched.add(key)

    def _run_lane(self, lane: str, budget: int, stage_tasks: list[Task],
                  prev: Program | None) -> dict:
        if not self._stage_prefetched:
            self._spec_limit = max(self.speculative_phase_limit, budget)
            self._prefetch_stage(stage_tasks)
            self._stage_prefetched = True
        elif budget > self._spec_limit:
            # Search ran past the speculative phase ceiling; widen and top up.
            self._spec_limit = budget
            self._prefetch_stage(stage_tasks)
        started = time.perf_counter()
        out = super()._run_lane(lane, budget, stage_tasks, prev)
        self.walk_seconds += time.perf_counter() - started
        return out

    # ------------------------------------------------------------------ reports
    def stats(self) -> dict:
        return {
            "gpu_episodes": self.gpu_episodes,
            "gpu_batches": self.gpu_batches,
            "gpu_steps": self.gpu_steps,
            "gpu_row_steps": self.gpu_row_steps,
            "gpu_slots": self.gpu_slots,
            "gpu_episode_steps": self.gpu_episode_steps,
            "fill": (round(self.gpu_episodes / self.gpu_slots, 4)
                     if self.gpu_slots else 0.0),
            "straggler_overrun": (round(self.gpu_row_steps
                                        / max(1, self.gpu_episode_steps), 3)),
            "prefetch_seconds": round(self.prefetch_seconds, 4),
            "walk_seconds": round(self.walk_seconds, 4),
            "timers": {k: round(v, 4) for k, v in self.timers.items()},
            "cache_entries": len(self._cache),
        }
