"""A deliberately small, deterministic reactive program language for 3-D chambers.

This is a research *DSL and VM*, not a complete implementation of Schmidhuber's
bias-optimal OOPS. A program is a sequence of fall-through guarded rules:
AIM(candidate) -> CHECK_SAFE -> IF_UNSAFE_SKIP -> STORE_DIRECTION -> EMIT.
CALL(program_id) can reuse frozen code, and LOAD_SAVED gives persistent state.

All geometric prediction below uses current Observation only, not the oracle
future obstacle positions from the simulator. It assumes the simulator's known
constant-velocity-and-wall-reflection dynamics over one control period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from chamber import (Observation, ObstacleObservation, Vec3, Cuboid, add, sub, mul, norm, lerp,
                     _segment_hits_aabb)

ACTIONS = ("GOAL", "Y_POS", "Y_NEG", "Z_POS", "Z_NEG",
           "X_POS", "X_NEG", "LAST", "CHOOSE_AXIS", "CONTINUE", "WAIT_GAP")
DIRECTIONS = {
    "X_POS": (1.0, 0.0, 0.0), "X_NEG": (-1.0, 0.0, 0.0),
    "Y_POS": (0.0, 1.0, 0.0), "Y_NEG": (0.0, -1.0, 0.0),
    "Z_POS": (0.0, 0.0, 1.0), "Z_NEG": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class Program:
    rules: tuple[str, ...]
    id: str = "unfrozen"

    def __post_init__(self):
        if not self.rules:
            raise ValueError("A program requires at least one rule")
        if any(r not in ACTIONS and not r.startswith("CALL:") for r in self.rules):
            raise ValueError(f"Invalid rule in {self.rules!r}")

    def to_dict(self):
        return {"id": self.id, "rules": list(self.rules),
                "bytecode": [list(x) for x in compile_program(self)]}


@dataclass
class VMState:
    """One persistent register across sensor observations within an episode."""
    last_direction: Vec3 = (0.0, 0.0, 0.0)
    emissions: int = 0
    last_position: Vec3 | None = None
    last_mode: str = ""
    explicit_waits: int = 0


@dataclass(frozen=True)
class MotionConfig:
    radius: float = .035
    dt: float = .5
    max_speed: float = .7
    stride: float = .12  # one command moves at most this distance


def predicted_safe(obs: Observation, velocity: Vec3, config: MotionConfig) -> bool:
    """Swept conservative collision prediction from *observable* state only.

    This is not an arbitrary-dynamics prediction. Each cuboid is extrapolated
    using its observed instantaneous velocity and known chamber reflections.
    The safety envelope is inflated by agent radius; false positives possible.
    """
    if norm(velocity) > config.max_speed + 1e-9:
        return False
    p0 = obs.position
    p1 = add(p0, mul(velocity, config.dt))
    if any(not (config.radius-1e-9 <= p[i] <= obs.chamber_size[i]-config.radius+1e-9)
           for p in (p0, p1) for i in range(3)):
        return False
    for data in obs.obstacles:
        ob = Cuboid(data.name, data.center, data.size, data.velocity)
        times = [0.0] + ob.bounce_times(0.0, config.dt, obs.chamber_size) + [config.dt]
        inflated = tuple(data.size[i] / 2 + config.radius for i in range(3))
        for t0,t1 in zip(times,times[1:]):
            a0 = lerp(p0,p1,t0/config.dt)
            a1 = lerp(p0,p1,t1/config.dt)
            b0 = ob.center_at(t0,obs.chamber_size)
            b1 = ob.center_at(t1,obs.chamber_size)
            if _segment_hits_aabb(sub(a0,b0),sub(a1,b1),inflated):
                return False
    return True


def _shift_observation(obs: Observation, seconds: float) -> Observation:
    """Produce a short-term prediction using ONLY the current observation.

    This is a known-dynamics simulation assumption, not an oracle observation.
    No future scene object or simulator env is passed to the VM.
    """
    shifted = []
    for item in obs.obstacles:
        box = Cuboid(item.name, item.center, item.size, item.velocity)
        shifted.append(ObstacleObservation(
            item.name, box.center_at(seconds, obs.chamber_size), item.size,
            box.velocity_at(seconds, obs.chamber_size)))
    return Observation(obs.time_s + seconds, obs.step_index, obs.position,
                       obs.goal, tuple(shifted), obs.chamber_size)


def _one_step_velocity(vector: Vec3, config: MotionConfig,
                       *, cap_length: bool = False) -> Vec3:
    length = norm(vector)
    if length < 1e-12:
        return (0., 0., 0.)
    distance = min(config.stride, config.max_speed * config.dt)
    if cap_length:
        distance = min(distance, length)
    return mul(vector, distance / (length * config.dt))


def _safe_axis_score(action: str, obs: Observation, cfg: MotionConfig,
                     memory: VMState) -> tuple[float,Vec3] | None:
    """Rank an axis candidate by a cheap one-step goal lookahead.

    It is a HAND-DESIGNED primitive. Program synthesis discovers whether/when
    to call it, not the underlying scoring function. No future oracle data.
    """
    v = _one_step_velocity(DIRECTIONS[action], cfg)
    if not predicted_safe(obs, v, cfg):
        return None
    position = add(obs.position, mul(v, cfg.dt))
    after = _shift_observation(obs, cfg.dt)
    next_obs = Observation(after.time_s, after.step_index, position, obs.goal,
                           after.obstacles, after.chamber_size)
    direct = _one_step_velocity(sub(obs.goal, position), cfg, cap_length=True)
    # A direction that opens a safe follow-up route is preferable to a blind
    # detour that is expected to terminate at a local dead end.
    next_safe = predicted_safe(next_obs, direct, cfg)
    initial_distance = norm(sub(obs.goal, obs.position))
    moved_distance = norm(sub(obs.goal, position))
    alignment = (initial_distance - moved_distance) / max(cfg.stride, 1e-12)
    persistence = 0.0
    if norm(memory.last_direction) > 1e-12:
        dot = sum(v[i]*memory.last_direction[i] for i in range(3))
        persistence = .18 * dot / (norm(v)*norm(memory.last_direction))
    # Progress matters, but direct safety next step adds a substantial bonus.
    score = alignment + (.65 if next_safe else 0.0) + persistence
    return score,v


def _moving_gap_wait(obs: Observation, cfg: MotionConfig, lookahead: int = 4) -> bool:
    """Wait only if stationary is safe now and motion may reopen soon.

    Not a general dynamic-obstacle predictor: assumes observed velocity and
    known wall reflection. It is a DSL primitive tested by synthesized code.
    """
    direct = _one_step_velocity(sub(obs.goal,obs.position),cfg,cap_length=True)
    if predicted_safe(obs,direct,cfg) or not predicted_safe(obs,(0.,0.,0.),cfg):
        return False
    if not any(norm(item.velocity) > 1e-9 for item in obs.obstacles):
        return False
    for steps in range(1,lookahead+1):
        future = _shift_observation(obs,steps*cfg.dt)
        if predicted_safe(future,direct,cfg):
            return True
    return False


def candidate_velocity(action: str, obs: Observation, config: MotionConfig,
                       state: VMState) -> Vec3:
    if action == "GOAL":
        return _one_step_velocity(sub(obs.goal,obs.position),config,cap_length=True)
    if action == "LAST":
        return _one_step_velocity(state.last_direction,config)
    if action == "CONTINUE":
        if norm(state.last_direction) < 1e-12:
            return (0.,0.,0.)
        v = _one_step_velocity(state.last_direction,config)
        after = add(obs.position,mul(v,config.dt))
        # Do not persist when it creates substantial backwards motion.
        if norm(sub(obs.goal,after)) > norm(sub(obs.goal,obs.position)) + .02:
            return (0.,0.,0.)
        return v
    if action == "CHOOSE_AXIS":
        options = [(_safe_axis_score(a,obs,config,state),a)
                   for a in ("X_POS","X_NEG","Y_POS","Y_NEG","Z_POS","Z_NEG")]
        valid = [(score,a) for score,a in options if score is not None]
        if not valid:
            return (0.,0.,0.)
        # Tie-breaking by lexical action name ensures replay determinism.
        best = sorted(valid,key=lambda entry:(-entry[0][0],entry[1]))[0]
        return best[0][1]
    if action == "WAIT_GAP":
        return (0.,0.,0.)
    return _one_step_velocity(DIRECTIONS[action],config)


def compile_program(program: Program) -> tuple[tuple, ...]:
    """Rule compiler. Explicit conditionals and persistent state writes.

    CALL attempts frozen code; if it cannot propose a safe move, execution
    continues with the next rule. WAIT is a final, controlled no-op fallback.
    """
    bytecode = []
    for rule in program.rules:
        if rule.startswith("CALL:"):
            bytecode.append(("CALL", rule.split(":", 1)[1]))
        elif rule == "WAIT_GAP":
            bytecode.extend((("CHECK_WAIT_GAP",), ("IF_UNSAFE_SKIP", 2),
                             ("EMIT_WAIT",)))
        else:
            bytecode.extend((("AIM", rule), ("CHECK_SAFE",),
                             ("IF_UNSAFE_SKIP", 3), ("STORE_DIRECTION",),
                             ("EMIT",)))
    bytecode.append(("END_NO_MATCH",))
    return tuple(bytecode)


class Interpreter:
    def __init__(self, frozen: Mapping[str, Program] | None = None,
                 config: MotionConfig = MotionConfig(), instruction_limit: int = 128,
                 max_call_depth: int = 6):
        self.frozen = frozen or {}
        self.config = config
        self.instruction_limit = instruction_limit
        self.max_call_depth = max_call_depth

    def act(self, program: Program, obs: Observation, memory: VMState) -> tuple[Vec3, list[str]]:
        """Return next velocity and an interpretable instruction trace."""
        remaining = [self.instruction_limit]
        trace: list[str] = []
        output = self._run(program, obs, memory, remaining, trace, ())
        if output is None:
            output = (0.0, 0.0, 0.0)
            trace.append("FALLBACK WAIT")
        memory.emissions += 1
        memory.last_position = obs.position
        return output, trace

    def _run(self, program: Program, obs: Observation, memory: VMState,
             remaining: list[int], trace: list[str], stack: tuple[str,...]) -> Vec3 | None:
        if len(stack) >= self.max_call_depth:
            trace.append("CALL DEPTH LIMIT")
            return None
        if program.id in stack:
            trace.append("RECURSIVE CALL BLOCKED")
            return None
        bytecode = compile_program(program)
        pc = 0
        candidate: Vec3 = (0.,0.,0.)
        is_safe = False
        last_action = ''
        while pc < len(bytecode):
            if remaining[0] <= 0:
                trace.append("INSTRUCTION LIMIT")
                return None
            remaining[0] -= 1
            inst = bytecode[pc]
            op = inst[0]
            if op == "AIM":
                last_action = inst[1]
                candidate = candidate_velocity(inst[1], obs, self.config, memory)
                trace.append(f"AIM {inst[1]}")
            elif op == "CHECK_SAFE":
                # An uninitialized LAST register must not emit a zero vector
                # and mask later rules (or create a fake successful branch).
                origin = bytecode[pc-1][1] if pc > 0 and bytecode[pc-1][0] == "AIM" else ""
                undefined = origin in ("LAST","CONTINUE") and norm(memory.last_direction) < 1e-12
                undefined |= origin in ("CHOOSE_AXIS", "CONTINUE") and norm(candidate) < 1e-12
                is_safe = (not undefined and
                           predicted_safe(obs, candidate, self.config))
                trace.append("SAFE" if is_safe else "BLOCKED")
            elif op == "CHECK_WAIT_GAP":
                is_safe = _moving_gap_wait(obs,self.config)
                trace.append("GAP_WAIT_READY" if is_safe else "GAP_WAIT_NOT_NEEDED")
            elif op == "IF_UNSAFE_SKIP":
                if not is_safe:
                    pc += inst[1]
                    continue
            elif op == "STORE_DIRECTION":
                if norm(candidate) > 1e-12:
                    memory.last_direction = candidate
            elif op == "EMIT":
                memory.last_mode = last_action
                trace.append("SELECTED:" + last_action)
                trace.append("EMIT")
                return candidate
            elif op == "EMIT_WAIT":
                memory.last_mode = "WAIT_GAP"
                memory.explicit_waits += 1
                trace.append("SELECTED:WAIT_GAP")
                trace.append("EMIT_WAIT_GAP")
                return (0.,0.,0.)
            elif op == "CALL":
                subprogram = self.frozen.get(inst[1])
                if subprogram is None:
                    trace.append("MISSING CALL " + inst[1])
                else:
                    trace.append("CALL " + inst[1])
                    output = self._run(subprogram, obs, memory, remaining, trace,
                                       (*stack, program.id))
                    if output is not None:
                        return output
            elif op == "END_NO_MATCH":
                return None
            else:
                raise ValueError(f"Unknown bytecode opcode: {op}")
            pc += 1
        return None
