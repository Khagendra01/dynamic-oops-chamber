"""Deterministic kinematic joint controller for v1.3.

Velocity/acceleration-limited first-order tracking of commanded joint
increments.  No torques, no contact dynamics: OOPS commands bounded joint
increments and this module realizes them identically every run.

Command model
-------------
An action is a desired joint increment ``dq_cmd`` (rad).  Each command period
``dt``:

1. per-joint clip:  ``|dq_cmd_i| <= velocity_limit_i * dt``;
2. desired velocity: ``v_i = clip(dq_cmd_i / dt, +-velocity_limit_i)``;
3. acceleration limit: ``v_i <- clip(v_i, v_prev_i +- accel_limit_i * dt)``;
4. integrate: ``q_i <- clip(q_i + v_i * dt, limit_lo_i, limit_hi_i)``.

Position-limit clipping is reported so callers can count violations of the
*command* (attempted motion beyond a limit) separately from the enforced
state constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from experiments_v13.robot_model import RobotSpec

_EPS = 1e-12


@dataclass(frozen=True)
class JointState:
    """Immutable snapshot of the controlled joint state."""

    q: np.ndarray
    qd: np.ndarray

    def copy(self) -> "JointState":
        return JointState(self.q.copy(), self.qd.copy())


@dataclass(frozen=True)
class JointStep:
    """Outcome of one command period."""

    state: JointState
    executed_dq: np.ndarray
    clamped_command: bool
    limit_clipped: bool
    stalled: bool


class JointController:
    """Deterministic kinematic controller enforcing pos/vel/accel limits."""

    def __init__(self, spec: RobotSpec, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.spec = spec
        self.dt = float(dt)
        self.vmax = spec.velocity_limits
        self.amax = spec.acceleration_limits
        self.lo = spec.joint_lo
        self.hi = spec.joint_hi

    def start_state(self, q0: Sequence[float]) -> JointState:
        return JointState(self.spec.clamp_configuration(q0),
                          np.zeros(self.spec.joint_count, dtype=float))

    def step(self, state: JointState, dq_cmd: Sequence[float]) -> JointStep:
        """Advance one command period from ``state`` under ``dq_cmd``."""
        cmd = np.asarray(dq_cmd, dtype=float)
        if cmd.shape != (self.spec.joint_count,):
            raise ValueError(f"dq_cmd must have shape ({self.spec.joint_count},)")
        if not np.all(np.isfinite(cmd)):
            raise ValueError("dq_cmd contains non-finite values")

        max_step = self.vmax * self.dt
        clamped = bool(np.any(np.abs(cmd) > max_step + _EPS))
        cmd = np.clip(cmd, -max_step, max_step)

        v_des = np.clip(cmd / self.dt, -self.vmax, self.vmax)
        v_min = state.qd - self.amax * self.dt
        v_max = state.qd + self.amax * self.dt
        v = np.clip(v_des, v_min, v_max)
        v = np.clip(v, -self.vmax, self.vmax)

        q_unclipped = state.q + v * self.dt
        q_next = np.clip(q_unclipped, self.lo, self.hi)
        limit_clipped = bool(np.any(q_next != q_unclipped))
        qd = np.where(np.abs(q_next - state.q) < _EPS, 0.0, v)
        # zero velocity components fully blocked by a position limit
        blocked_low = (q_next <= self.lo + _EPS) & (qd < 0.0)
        blocked_high = (q_next >= self.hi - _EPS) & (qd > 0.0)
        qd = np.where(blocked_low | blocked_high, 0.0, qd)
        executed = q_next - state.q
        stalled = bool(np.all(np.abs(executed) < 1e-9))
        return JointStep(JointState(q_next, qd), executed, clamped,
                         limit_clipped, stalled)

    def max_increment(self) -> np.ndarray:
        """Per-joint maximum commanded increment over one period."""
        return self.vmax * self.dt

    def unit_increment(self, joint: int, sign: float) -> np.ndarray:
        """Full-speed one-period increment on a single joint (for DSL AIM)."""
        out = np.zeros(self.spec.joint_count, dtype=float)
        step = self.vmax[joint] * self.dt
        out[joint] = math.copysign(step, sign)
        return out
