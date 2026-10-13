"""Conventional kinematic baselines for v1.3.

B1 ``DLSIKPolicy`` — damped least-squares Jacobian inverse kinematics used as
a receding-horizon position controller.  Each control period it takes one
regularized Newton step toward the goal in joint space.  Joint limits are
enforced by clamping; velocity/acceleration limits by the shared joint
controller.  **No collision handling** — B1 is a reachability/kinematics
baseline, not a dynamic motion planner; its collision rate is a measured
outcome, not an oversight.

B3 ``ReactiveRolloutPolicy`` — short-horizon reactive controller: proposes a
small set of candidate joint increments (DLS step, per-joint probes, hold),
screens each with the *same causal conservative predictor* the shield uses
(policy-side, from its served observation), and executes the best one-step
candidate by predicted EE-error reduction.  It re-plans every period from
live observations and never reads future obstacle schedules.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from experiments_v13.forward_kinematics import (ee_position,
                                                position_jacobian)
from experiments_v13.robot_observations import JointObservation, observed_field
from experiments_v13.robot_model import RobotSpec
from experiments_v13.swept_collision import SAFE, certify_motion


def dls_step(spec: RobotSpec, q: Sequence[float], goal: Sequence[float],
             *, damping: float = 0.05, gain: float = 1.0,
             max_dq: float | None = None) -> np.ndarray:
    """One damped-least-squares joint increment toward ``goal``."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(ee_position(spec, q))
    err = np.asarray(goal, dtype=float) - p
    jac = position_jacobian(spec, q)
    jjt = jac @ jac.T + (damping ** 2) * np.eye(3)
    dq = gain * jac.T @ np.linalg.solve(jjt, err)
    if max_dq is not None:
        norm = float(np.linalg.norm(dq))
        if norm > max_dq:
            dq *= max_dq / norm
    return dq


class DLSIKPolicy:
    """B1: pure damped-least-squares IK, receding horizon. No collision logic."""

    name = "B1_dls_ik"

    def __init__(self, spec: RobotSpec, *, damping: float = 0.05,
                 gain: float = 1.0):
        self.spec = spec
        self.damping = damping
        self.gain = gain
        self.deadline_misses = 0
        self.decisions = 0

    def reset(self) -> None:
        self.deadline_misses = 0
        self.decisions = 0

    def act(self, obs: JointObservation) -> np.ndarray:
        self.decisions += 1
        started = time.perf_counter()
        dq = dls_step(self.spec, obs.q, obs.goal, damping=self.damping,
                      gain=self.gain)
        if time.perf_counter() - started > 0.5:
            self.deadline_misses += 1
        return dq

    def extras(self) -> dict:
        return {"deadline_misses": self.deadline_misses,
                "decisions": self.decisions}


class ReactiveRolloutPolicy:
    """B3: collision-screened one-step rollout controller (shared predictor)."""

    name = "B3_reactive_rollout"

    def __init__(self, spec: RobotSpec, *, damping: float = 0.05,
                 probe_scale: float = 0.5, deadline_s: float = 0.5,
                 certify_max_checks: int = 32):
        self.spec = spec
        self.damping = damping
        self.probe_scale = probe_scale
        self.deadline_s = deadline_s
        self.certify_max_checks = certify_max_checks
        self.deadline_misses = 0
        self.decisions = 0
        self.blocked_stops = 0
        self.replans = 0

    def reset(self) -> None:
        self.deadline_misses = 0
        self.decisions = 0
        self.blocked_stops = 0
        self.replans = 0

    def act(self, obs: JointObservation) -> np.ndarray:
        self.decisions += 1
        self.replans += 1
        started = time.perf_counter()
        vmax = np.asarray(obs.velocity_limits)
        q = np.asarray(obs.q)
        goal_err = float(np.linalg.norm(
            np.asarray(obs.ee_position) - np.asarray(obs.goal)))

        base = dls_step(self.spec, q, obs.goal, damping=self.damping)
        max_step = vmax * 0.1  # matches env dt; policy is told velocity limits
        candidates = [np.clip(base, -max_step, max_step)]
        # per-joint probes: reinforce the largest DLS components, both signs
        order = np.argsort(-np.abs(base))[:3]
        for j in order:
            for sign in (1.0, -1.0):
                dq = np.zeros_like(q)
                dq[j] = sign * max_step[j]
                candidates.append(dq)
        candidates.append(np.zeros_like(q))  # hold

        field = observed_field(obs.chamber_size, obs.obstacles)
        checker = _checker_for(self.spec)
        best_dq = np.zeros_like(q)
        best_score = -np.inf
        for dq in candidates:
            verdict = certify_motion(self.spec, checker, q, dq, obs.time_s,
                                     0.1, field.obstacle_at,
                                     max_checks=self.certify_max_checks)
            if verdict.status != SAFE:
                continue
            p2 = np.asarray(ee_position(self.spec, np.clip(
                q + dq, obs.joint_lo, obs.joint_hi)))
            score = -float(np.linalg.norm(p2 - np.asarray(obs.goal)))
            # small penalty for joint travel
            score -= 0.01 * float(np.linalg.norm(dq))
            if score > best_score:
                best_score = score
                best_dq = dq
        if best_score == -np.inf:
            self.blocked_stops += 1
        if time.perf_counter() - started > self.deadline_s:
            self.deadline_misses += 1
        return best_dq

    def extras(self) -> dict:
        return {"deadline_misses": self.deadline_misses,
                "decisions": self.decisions,
                "blocked_stops": self.blocked_stops,
                "replans": self.replans}


_CHECKER_CACHE: dict[int, object] = {}


def _checker_for(spec: RobotSpec):
    key = id(spec)
    if key not in _CHECKER_CACHE:
        from experiments_v13.collision_geometry import CollisionChecker
        _CHECKER_CACHE[key] = CollisionChecker(spec)
    return _CHECKER_CACHE[key]


def ik_solve(spec: RobotSpec, q0: Sequence[float], goal: Sequence[float],
             *, tol: float = 1e-4, max_iters: int = 400,
             damping: float = 0.05, gain: float = 1.0):
    """Batch DLS solve (for feasibility probes and diagnostics).

    Returns (q, success, iterations, path).  Pure kinematics — ignores
    collisions and dynamics; a solution is a *configuration* witness only.
    """
    q = np.clip(np.asarray(q0, dtype=float), spec.joint_lo, spec.joint_hi)
    path = [q.copy()]
    success = False
    iters = 0
    for iters in range(1, max_iters + 1):
        err = np.linalg.norm(np.asarray(ee_position(spec, q))
                             - np.asarray(goal, dtype=float))
        if err <= tol:
            success = True
            break
        dq = dls_step(spec, q, goal, damping=damping, gain=gain)
        q = np.clip(q + dq, spec.joint_lo, spec.joint_hi)
        path.append(q.copy())
    return q, success, iters, path
