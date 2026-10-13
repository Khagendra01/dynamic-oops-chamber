"""Joint-space DSL for the v1.3 OOPS experiments.

A deliberately small, deterministic reactive language.  A program is a
sequence of fall-through guarded rules; each control period the VM runs the
program on the current observation and emits one bounded joint increment
``dq in R^6`` (radians).  The language is *joint-space*: instructions name
individual joints or memory/stall tests — there is **no** inverse kinematics
instruction and **no** hidden trajectory planner.  Forward kinematics is used
only inside two disclosed primitives (``CHOOSE_JOINT`` scoring and the
``CHECK_PROGRESS`` guard), each documented with its exact algorithm.

Vocabulary (rule names)
-----------------------
``JOINT_POS_i`` / ``JOINT_NEG_i``   full-speed one-period step on joint i
                                   (clipped to remaining position range);
                                   guarded by CHECK_LIMIT + CHECK_COLLISION.
``...?`` suffix                    additionally guarded by CHECK_PROGRESS
                                   (emit only if EE error would decrease).
``...!`` suffix                    additionally guarded by CHECK_STALLED
                                   (emit only after >= 2 stagnant steps —
                                   an escape move for local minima).
``HOLD``                           explicit zero increment, emitted only when
                                   holding is certified safe for the period.
``LAST``                           repeat the previous emitted increment
                                   (guarded like a direction rule).
``CALL:<id>``                      run a frozen program as a subroutine.
``CHOOSE_JOINT``                   hand-designed disclosed primitive: score
                                   all 12 single-joint candidates by one-step
                                   EE-error reduction (FK lookahead), keep
                                   only conservatively certified-safe ones,
                                   deterministic tie-break.  Synthesis
                                   discovers *whether/when* to use it, not
                                   the scoring rule.  Cost: 12 FK + up to 12
                                   certificates per invocation.

Compilation
-----------
Each direction rule compiles to::

    AIM a; CHECK_LIMIT; IF_LIMIT_SKIP L1; CHECK_COLLISION; IF_UNSAFE_SKIP L2;
    [CHECK_PROGRESS; IF_NO_PROGRESS_SKIP 3 |          # "?" form
     CHECK_STALLED; IF_NOT_STALLED_SKIP 3] ;          # "!" form
    STORE; EMIT

so a candidate is emitted only when it is inside joint limits *and*
conservatively certified collision-free (the same certify_motion the shield
executes — programs cannot bypass the shield, they can only pre-screen).

Execution cost weights (search-cost proxy, NOT latency): CHECK_COLLISION=8,
AIM CHOOSE_JOINT=20, EMIT=1, everything else=1.  Disclosed infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from experiments_v13.forward_kinematics import ee_position
from experiments_v13.robot_model import RobotSpec

_EPS = 1e-9

JOINT_COUNT = 6
_DIRECTION_TOKENS = tuple(
    [f"JOINT_POS_{i + 1}" for i in range(JOINT_COUNT)]
    + [f"JOINT_NEG_{i + 1}" for i in range(JOINT_COUNT)])
BASE_TOKENS = _DIRECTION_TOKENS + ("HOLD", "LAST", "CHOOSE_JOINT")

# Search priors (ordering only, not probabilities of any physical event).
BASE_WEIGHTS_JOINT: dict[str, float] = {
    "CHOOSE_JOINT": 0.22,
    "JOINT_POS_2": 0.075, "JOINT_NEG_2": 0.075,
    "JOINT_POS_3": 0.070, "JOINT_NEG_3": 0.070,
    "JOINT_POS_1": 0.045, "JOINT_NEG_1": 0.045,
    "JOINT_POS_5": 0.030, "JOINT_NEG_5": 0.030,
    "JOINT_POS_4": 0.025, "JOINT_NEG_4": 0.025,
    "JOINT_POS_6": 0.020, "JOINT_NEG_6": 0.020,
    "HOLD": 0.10, "LAST": 0.07,
}

COST_COLLISION_CHECK = 8
COST_CHOOSE_JOINT_AIM = 20
STALL_THRESHOLD = 2
PROGRESS_EPSILON = 1e-4


def normalize_rule(rule: str) -> str:
    """Validate a rule token; returns the canonical form."""
    if rule.startswith("CALL:"):
        target = rule[5:]
        if not target:
            raise ValueError("CALL requires a program id")
        return rule
    base = rule
    suffix = ""
    if base.endswith("?") or base.endswith("!"):
        suffix = base[-1]
        base = base[:-1]
    if base not in BASE_TOKENS:
        raise ValueError(f"unknown rule {rule!r}")
    if suffix and base not in _DIRECTION_TOKENS:
        raise ValueError(f"suffix not allowed on {base}")
    return base + suffix


@dataclass(frozen=True)
class JointProgram:
    """A frozen or candidate program: an ordered tuple of DSL rules."""

    rules: tuple[str, ...]
    id: str = "unfrozen"

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("a program requires at least one rule")
        for rule in self.rules:
            normalize_rule(rule)

    def to_dict(self) -> dict:
        return {"id": self.id, "rules": list(self.rules)}


@dataclass
class JointVMState:
    """Persistent register across observations within an episode."""

    last_delta: tuple[float, ...] = (0.0,) * JOINT_COUNT
    saved_delta: tuple[float, ...] = (0.0,) * JOINT_COUNT
    last_mode: str = ""
    emissions: int = 0
    explicit_waits: int = 0
    stall_steps: int = 0
    last_ee_error: float = math.inf


def clip_to_limits(spec: RobotSpec, q: Sequence[float],
                   dq: Sequence[float]) -> np.ndarray:
    """Clip a candidate increment so q+dq stays inside joint limits."""
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    room_hi = spec.joint_hi - q
    room_lo = spec.joint_lo - q
    return np.clip(dq, room_lo, room_hi)


def candidate_dq(action: str, obs, memory: JointVMState, spec: RobotSpec,
                 dt: float) -> np.ndarray:
    """Map one AIM action to a bounded joint increment.

    ``CHOOSE_JOINT`` is the disclosed hand-designed scoring primitive (see
    module docstring); everything else is a direct bounded mapping.
    """
    q = np.asarray(obs.q, dtype=float)
    vmax = spec.velocity_limits
    if action == "HOLD":
        return np.zeros(JOINT_COUNT)
    if action == "LAST":
        return clip_to_limits(spec, q, np.asarray(memory.last_delta))
    if action == "CHOOSE_JOINT":
        return choose_joint(spec, obs, memory, dt)
    if action.startswith("JOINT_"):
        sign_part, index = action.rsplit("_", 1)
        joint = int(index) - 1
        if not 0 <= joint < JOINT_COUNT:
            raise ValueError(f"bad joint index in {action}")
        sign = 1.0 if sign_part == "JOINT_POS" else -1.0
        dq = np.zeros(JOINT_COUNT)
        dq[joint] = sign * vmax[joint] * dt
        return clip_to_limits(spec, q, dq)
    raise ValueError(f"unknown AIM action {action}")


def choose_joint(spec: RobotSpec, obs, memory: JointVMState, dt: float,
                 certify=None) -> np.ndarray:
    """Hand-designed joint chooser (disclosed primitive).

    Scores all 12 single-joint full-speed candidates by one-step EE-error
    reduction using FK at the position-limit-clipped configuration.  When a
    ``certify`` callable is supplied (the VM passes its conservative motion
    checker), only certified-safe candidates remain eligible; unsafe-all
    yields a zero increment (hold).  Ties break by action-name order for
    deterministic replay.
    """
    q = np.asarray(obs.q, dtype=float)
    goal = np.asarray(obs.goal, dtype=float)
    base_err = float(np.linalg.norm(np.asarray(ee_position(spec, q)) - goal))
    best_score = -math.inf
    best = np.zeros(JOINT_COUNT)
    scored = []
    for action in sorted(_DIRECTION_TOKENS):
        dq = candidate_dq(action, obs, memory, spec, dt)
        q2 = np.clip(q + dq, spec.joint_lo, spec.joint_hi)
        err2 = float(np.linalg.norm(np.asarray(ee_position(spec, q2)) - goal))
        score = base_err - err2
        if certify is not None:
            status = certify(dq)
            if status != "safe":
                continue
        scored.append((score, action, dq))
    if not scored:
        return best
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[0][2]


def in_limits(spec: RobotSpec, q: Sequence[float],
              dq: Sequence[float], tol: float = 1e-9) -> bool:
    q2 = np.asarray(q, dtype=float) + np.asarray(dq, dtype=float)
    return bool(np.all(q2 >= spec.joint_lo - tol) and
                np.all(q2 <= spec.joint_hi + tol))


def reduces_error(spec: RobotSpec, obs, dq: Sequence[float],
                  eps: float = PROGRESS_EPSILON) -> bool:
    """CHECK_PROGRESS: one-step FK lookahead (hand-designed, disclosed)."""
    q = np.asarray(obs.q, dtype=float)
    goal = np.asarray(obs.goal, dtype=float)
    q2 = np.clip(q + np.asarray(dq, dtype=float), spec.joint_lo, spec.joint_hi)
    before = float(np.linalg.norm(np.asarray(ee_position(spec, q)) - goal))
    after = float(np.linalg.norm(np.asarray(ee_position(spec, q2)) - goal))
    return after < before - eps


def _rule_uses_progress(rule: str) -> bool:
    return rule.endswith("?")


def _rule_uses_stall(rule: str) -> bool:
    return rule.endswith("!")


def _base_action(rule: str) -> str:
    if rule.endswith("?") or rule.endswith("!"):
        return rule[:-1]
    return rule


def compile_program(program: JointProgram) -> tuple[tuple, ...]:
    """Compile rules to VM bytecode with explicit skip offsets."""
    bytecode: list[tuple] = []
    for rule in program.rules:
        if rule.startswith("CALL:"):
            bytecode.append(("CALL", rule.split(":", 1)[1]))
            continue
        action = _base_action(rule)
        if action == "HOLD":
            # emit an explicit wait only when holding is certified safe
            bytecode.extend((("AIM", "HOLD"), ("CHECK_COLLISION",),
                             ("IF_UNSAFE_SKIP", 1), ("EMIT_WAIT",)))
            continue
        guards_extra = 0
        if _rule_uses_progress(rule):
            guards_extra = 2
        elif _rule_uses_stall(rule):
            guards_extra = 2
        # base: AIM CHECK_LIMIT [guards] CHECK_COLLISION skip STORE EMIT
        # indices: 0 AIM, 1 CHECK_LIMIT, then optional 2 guards,
        # then CHECK_COLLISION, IF_UNSAFE_SKIP, [progress/stall pair], STORE, EMIT
        if_action = action
        bytecode.append(("AIM", if_action))
        bytecode.append(("CHECK_LIMIT",))
        if _rule_uses_progress(rule):
            # IF_LIMIT_SKIP must jump past the whole rule (to next rule)
            # rule length = 2 + 1(limit skip) + 2 collision pair + 2 progress pair + 2 store/emit = 9
            limit_skip = 7  # from pc=2: 2+7=9 = next rule
            bytecode.append(("IF_LIMIT_SKIP", limit_skip))
            bytecode.append(("CHECK_COLLISION",))
            unsafe_skip = 5  # from pc=4: 4+5=9
            bytecode.append(("IF_UNSAFE_SKIP", unsafe_skip))
            bytecode.append(("CHECK_PROGRESS",))
            progress_skip = 3  # from pc=6: 6+3=9
            bytecode.append(("IF_NO_PROGRESS_SKIP", progress_skip))
            bytecode.append(("STORE",))
            bytecode.append(("EMIT",))
        elif _rule_uses_stall(rule):
            limit_skip = 7
            bytecode.append(("IF_LIMIT_SKIP", limit_skip))
            bytecode.append(("CHECK_COLLISION",))
            unsafe_skip = 5
            bytecode.append(("IF_UNSAFE_SKIP", unsafe_skip))
            bytecode.append(("CHECK_STALLED",))
            stall_skip = 3  # skip STORE+EMIT when NOT stalled
            bytecode.append(("IF_NOT_STALLED_SKIP", stall_skip))
            bytecode.append(("STORE",))
            bytecode.append(("EMIT",))
        else:
            # rule length = 7: AIM CHECK_LIMIT IFLIM CHECK_COL IFUNSAFE STORE EMIT
            limit_skip = 5  # from pc=2: 2+5=7
            bytecode.append(("IF_LIMIT_SKIP", limit_skip))
            bytecode.append(("CHECK_COLLISION",))
            unsafe_skip = 3  # from pc=4: 4+3=7
            bytecode.append(("IF_UNSAFE_SKIP", unsafe_skip))
            bytecode.append(("STORE",))
            bytecode.append(("EMIT",))
    bytecode.append(("END_NO_MATCH",))
    return tuple(bytecode)


def instruction_cost(op: str, arg: str = "") -> int:
    if op == "CHECK_COLLISION":
        return COST_COLLISION_CHECK
    if op == "AIM" and arg == "CHOOSE_JOINT":
        return COST_CHOOSE_JOINT_AIM
    return 1
