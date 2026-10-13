"""Instruction interpreter for the v1.3 joint-space DSL.

One ``act`` call runs one program on one observation and returns a bounded
joint increment plus an interpretable instruction trace.  Guards use only:

* the served observation (q, obstacles, goal) — never simulator futures;
* the *same* conservative ``certify_motion`` the execution shield uses, so a
  program can pre-screen candidates but can never bypass the shield;
* forward kinematics inside ``CHECK_PROGRESS`` / ``CHOOSE_JOINT`` (both
  disclosed hand-designed primitives, documented in ``joint_dsl``).

Persistent memory (last increment, stall counter, last EE error) carries
across control periods within an episode via ``JointVMState``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np

from experiments_v13.collision_geometry import CollisionChecker
from experiments_v13.joint_dsl import (JOINT_COUNT, JointProgram, JointVMState,
                                       STALL_THRESHOLD, candidate_dq,
                                       compile_program, in_limits,
                                       instruction_cost, reduces_error)
from experiments_v13.robot_model import RobotSpec
from experiments_v13.robot_observations import JointObservation, observed_field
from experiments_v13.swept_collision import SAFE, certify_motion


class JointInterpreter:
    """Deterministic fall-through interpreter with frozen CALL support."""

    def __init__(self, spec: RobotSpec, frozen: Mapping[str, JointProgram] | None = None,
                 *, dt: float = 0.1, instruction_limit: int = 128,
                 max_call_depth: int = 6, certify_max_checks: int = 32):
        self.spec = spec
        self.frozen = dict(frozen or {})
        self.dt = float(dt)
        self.instruction_limit = instruction_limit
        self.max_call_depth = max_call_depth
        self.checker = CollisionChecker(spec)
        self.certify_max_checks = certify_max_checks

    # ------------------------------------------------------------ prediction
    def _certify(self, obs: JointObservation, dq: np.ndarray) -> str:
        field = observed_field(obs.chamber_size, obs.obstacles)
        verdict = certify_motion(self.spec, self.checker, obs.q, dq,
                                 obs.time_s, self.dt, field.obstacle_at,
                                 max_checks=self.certify_max_checks)
        return verdict.status

    # ----------------------------------------------------------------- act
    def act(self, program: JointProgram, obs: JointObservation,
            memory: JointVMState) -> tuple[np.ndarray, list[str]]:
        remaining = [self.instruction_limit]
        trace: list[str] = []
        output = self._run(program, obs, memory, remaining, trace, ())
        if output is None:
            output = np.zeros(JOINT_COUNT)
            trace.append("FALLBACK WAIT")
            memory.explicit_waits += 0  # fallback is not an explicit wait
        memory.emissions += 1
        # stall bookkeeping from true post-step EE error available next call
        err = obs.ee_error
        if err < memory.last_ee_error - 1e-4:
            memory.stall_steps = 0
        else:
            memory.stall_steps += 1
        memory.last_ee_error = min(memory.last_ee_error, err)
        return output, trace

    def _run(self, program: JointProgram, obs: JointObservation,
             memory: JointVMState, remaining: list[int], trace: list[str],
             stack: tuple[str, ...]) -> np.ndarray | None:
        if len(stack) >= self.max_call_depth:
            trace.append("CALL DEPTH LIMIT")
            return None
        if program.id in stack and program.id != "unfrozen":
            trace.append("RECURSIVE CALL BLOCKED")
            return None
        bytecode = compile_program(program)
        pc = 0
        candidate = np.zeros(JOINT_COUNT)
        safe = False
        last_action = ""
        while pc < len(bytecode):
            if remaining[0] <= 0:
                trace.append("INSTRUCTION LIMIT")
                return None
            remaining[0] -= 1
            op = bytecode[pc][0]
            if op == "AIM":
                last_action = bytecode[pc][1]
                certify_fn = None
                if last_action == "CHOOSE_JOINT":
                    certify_fn = (lambda dq, _obs=obs: self._certify(_obs, dq))
                candidate = candidate_dq(last_action, obs, memory, self.spec,
                                         self.dt)
                if last_action == "CHOOSE_JOINT" and certify_fn is not None:
                    # re-run chooser with certification (candidate_dq without
                    # certify is the uncertified fallback; certified path
                    # replaces it when safe candidates exist)
                    from experiments_v13.joint_dsl import choose_joint
                    certified = choose_joint(self.spec, obs, memory, self.dt,
                                             certify=certify_fn)
                    candidate = certified
                trace.append(f"AIM {last_action}")
            elif op == "CHECK_LIMIT":
                safe = in_limits(self.spec, obs.q, candidate)
                trace.append("LIMIT_OK" if safe else "LIMIT_BLOCKED")
            elif op == "CHECK_COLLISION":
                status = self._certify(obs, candidate)
                safe = status == SAFE
                trace.append("SAFE" if safe else "BLOCKED")
            elif op == "CHECK_PROGRESS":
                safe = reduces_error(self.spec, obs, candidate)
                trace.append("PROGRESS" if safe else "NO_PROGRESS")
            elif op == "CHECK_STALLED":
                safe = memory.stall_steps >= STALL_THRESHOLD
                trace.append("STALLED" if safe else "MOVING_WELL")
            elif op == "IF_LIMIT_SKIP":
                if not safe:
                    pc += bytecode[pc][1]
                    continue
            elif op == "IF_UNSAFE_SKIP":
                if not safe:
                    pc += bytecode[pc][1]
                    continue
            elif op == "IF_NO_PROGRESS_SKIP":
                if not safe:
                    pc += bytecode[pc][1]
                    continue
            elif op == "IF_NOT_STALLED_SKIP":
                if not safe:
                    pc += bytecode[pc][1]
                    continue
            elif op == "STORE":
                memory.last_delta = tuple(float(v) for v in candidate)
                memory.saved_delta = memory.last_delta
            elif op == "EMIT":
                memory.last_mode = last_action
                trace.append("SELECTED:" + last_action)
                trace.append("EMIT")
                return candidate
            elif op == "EMIT_WAIT":
                memory.last_mode = "HOLD"
                memory.explicit_waits += 1
                trace.append("SELECTED:HOLD")
                trace.append("EMIT_WAIT")
                return np.zeros(JOINT_COUNT)
            elif op == "CALL":
                child = self.frozen.get(bytecode[pc][1])
                if child is None:
                    trace.append("MISSING CALL " + bytecode[pc][1])
                else:
                    trace.append("CALL " + bytecode[pc][1])
                    value = self._run(child, obs, memory, remaining, trace,
                                      stack + (program.id,))
                    if value is not None:
                        return value
                    # child declined; continue with the next rule
            elif op == "END_NO_MATCH":
                return None
            else:
                raise ValueError(f"unknown opcode {op}")
            pc += 1
        return None

    def program_work_units(self, trace: list[str]) -> int:
        total = 0
        for event in trace:
            if event.startswith("AIM "):
                total += instruction_cost("AIM", event[4:])
            elif event in ("SAFE", "BLOCKED", "LIMIT_OK", "LIMIT_BLOCKED",
                           "PROGRESS", "NO_PROGRESS", "STALLED", "MOVING_WELL"):
                if event in ("SAFE", "BLOCKED"):
                    total += instruction_cost("CHECK_COLLISION")
            else:
                total += 1
        return max(total, 1)
