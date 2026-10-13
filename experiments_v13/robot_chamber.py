"""Six-DOF robot chamber simulator for v1.3.

Public surface mirrors the v1.2.1 ``DynamicChamber`` contract so the same
evaluation harness patterns, causal-observer attachment and shield semantics
carry over:

* ``step(dq_cmd)`` realizes a bounded joint increment through the kinematic
  controller, then executes it with the **causal execution shield**:
  motion is filtered by ``certify_motion`` evaluated on a *served
  observation* only (the policy's degraded view when an observer is
  attached, otherwise the perfect current snapshot).  Ground-truth futures
  are never consulted for filtering.
* after execution, ground truth (exact DynamicBox trajectories, including
  velocity events and appearance times) is used **only to detect** whether
  the executed motion — or a forced wait while obstacles keep moving —
  collided.  A collision sets ``done``; it is a metric, never a filter.
* holding position is *not* assumed safe: a moving obstacle striking a
  stationary arm is detected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from experiments_v13.collision_geometry import CollisionChecker
from experiments_v13.forward_kinematics import ee_position, forward_pose
from experiments_v13.joint_dynamics import JointController, JointState
from experiments_v13.robot_model import RobotSpec, Vec3
from experiments_v13.robot_observations import (JointObservation, ObsBox,
                                                observed_field)
from experiments_v13.swept_collision import (SAFE, dense_motion_check,
                                             certify_motion)

_EPS = 1e-12


@dataclass(frozen=True)
class RobotStepResult:
    observation: JointObservation
    reached: bool
    collision: bool
    collision_category: str        # "", "environment", "self", "boundary"
    collision_bodies: tuple[str, str]
    intervention: bool             # shield refused the commanded motion
    limit_clipped: bool            # command hit a joint position/step limit
    done: bool
    commanded_dq: tuple[float, ...]
    executed_dq: tuple[float, ...]


class RobotChamber:
    """Bounded chamber containing one articulated arm and moving cuboids."""

    def __init__(self, spec: RobotSpec, q0: Sequence[float],
                 goal: Sequence[float], obstacles: Sequence = (),
                 *, dt: float = 0.1, max_steps: int = 60,
                 goal_tolerance: float = 0.03,
                 detection_samples: int = 12,
                 certify_max_checks: int = 64):
        self.spec = spec
        self.q0 = spec.clamp_configuration(q0)
        self.goal = (float(goal[0]), float(goal[1]), float(goal[2]))
        self.obstacles = tuple(obstacles)  # DynamicBox-compatible
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_tolerance = float(goal_tolerance)
        self.detection_samples = int(detection_samples)
        self.certify_max_checks = int(certify_max_checks)
        self.controller = JointController(spec, dt)
        self.checker = CollisionChecker(spec)
        self.shield_observer = None
        # ground-truth obstacle field: full schedule (events, appearances)
        from experiments_v13.swept_collision import MovingObstacleField
        self._gt_field = MovingObstacleField.from_dynamic(spec.chamber_size,
                                                          self.obstacles)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> JointObservation:
        self.state = self.controller.start_state(self.q0)
        self.step_index = 0
        self.collision_flag = False
        self.collision_category = ""
        self.collision_bodies = ("", "")
        start_error = self._error(self.state.q)
        self.done = bool(start_error <= self.goal_tolerance)
        self.reached_start = self.done
        if self.shield_observer is not None and hasattr(self.shield_observer, "reset"):
            self.shield_observer.reset()
        return self.observe()

    @property
    def q(self) -> np.ndarray:
        return self.state.q

    @property
    def time_s(self) -> float:
        return self.step_index * self.dt

    def _error(self, q: Sequence[float]) -> float:
        p = np.asarray(ee_position(self.spec, q))
        g = np.asarray(self.goal)
        return float(np.linalg.norm(p - g))

    # ------------------------------------------------------------ observations
    def raw_obstacle_snapshot(self) -> list[ObsBox]:
        out: list[ObsBox] = []
        t = self.time_s
        chamber = self.spec.chamber_size
        for i, ob in enumerate(self.obstacles):
            if not ob.active_at(t):
                continue
            out.append(ObsBox(
                ob.name,
                tuple(ob.center_at(t, chamber)),
                tuple(ob.size),
                tuple(ob.velocity_at(t, chamber))))
        return out

    def make_observation(self, obstacles: Sequence[ObsBox]) -> JointObservation:
        pose = forward_pose(self.spec, self.state.q)
        return JointObservation(
            time_s=round(self.time_s, 10),
            step_index=self.step_index,
            q=tuple(float(v) for v in self.state.q),
            qd=tuple(float(v) for v in self.state.qd),
            ee_position=pose.ee_position,
            goal=self.goal,
            obstacles=tuple(obstacles),
            chamber_size=self.spec.chamber_size,
            joint_lo=tuple(float(v) for v in self.spec.joint_lo),
            joint_hi=tuple(float(v) for v in self.spec.joint_hi),
            velocity_limits=tuple(float(v) for v in self.spec.velocity_limits))

    def observe(self) -> JointObservation:
        return self.make_observation(self.raw_obstacle_snapshot())

    def served_observation(self) -> JointObservation:
        if self.shield_observer is not None:
            return self.shield_observer.observe(self)
        return self.observe()

    # ------------------------------------------------------------------ step
    def step(self, dq_cmd: Sequence[float]) -> RobotStepResult:
        if self.done:
            raise RuntimeError("Episode is complete; call reset()")
        cmd = np.asarray(dq_cmd, dtype=float)
        trial = self.controller.step(self.state, cmd)

        # --- causal execution shield (served observation only) -------------
        shield_obs = self.served_observation()
        predictor = observed_field(shield_obs.chamber_size, shield_obs.obstacles)
        verdict = certify_motion(
            self.spec, self.checker, self.state.q, trial.executed_dq,
            self.time_s, self.dt, predictor.obstacle_at,
            max_checks=self.certify_max_checks)
        intervention = verdict.status != SAFE
        if intervention:
            executed = np.zeros(self.spec.joint_count)
            next_state = JointState(self.state.q.copy(),
                                    np.zeros(self.spec.joint_count))
        else:
            executed = trial.executed_dq
            next_state = trial.state

        # --- ground-truth detection (metric only) --------------------------
        collided, hit = dense_motion_check(
            self.spec, self.checker, self.state.q, executed,
            self.time_s, self.dt, self._gt_field.obstacle_at,
            samples=self.detection_samples)

        self.state = next_state
        self.step_index += 1
        error = self._error(self.state.q)
        reached = bool(error <= self.goal_tolerance + _EPS)
        if collided:
            self.collision_flag = True
            self.collision_category = hit[0] if hit else "environment"
            self.collision_bodies = (hit[1], hit[2]) if hit else ("", "")
        self.done = bool(reached and not self.collision_flag) or \
            self.collision_flag or self.step_index >= self.max_steps
        return RobotStepResult(
            observation=self.observe(), reached=reached,
            collision=bool(collided),
            collision_category=self.collision_category if collided else "",
            collision_bodies=self.collision_bodies if collided else ("", ""),
            intervention=bool(intervention),
            limit_clipped=bool(trial.clamped_command or trial.limit_clipped),
            done=self.done,
            commanded_dq=tuple(float(v) for v in cmd),
            executed_dq=tuple(float(v) for v in executed))

    # ------------------------------------------------------------- utilities
    def state_free(self, q: Sequence[float] | None = None, t: float | None = None) -> bool:
        """Ground-truth static check at one configuration (for witnesses)."""
        q = self.state.q if q is None else q
        t = self.time_s if t is None else t
        boxes, names = self._gt_field.obstacle_at(t)
        return self.checker.state_free(q, boxes)

    def straight_line_witness(self, q_target: Sequence[float], *,
                              samples: int = 24, time_invariant: bool = True) -> bool:
        """Dense-verify the straight joint-space path q0 -> q_target.

        With ``time_invariant=True`` obstacles are frozen at t=0 (static
        witness).  Returns True only when every sample configuration is
        collision-free.  This is ground truth, used for scenario feasibility
        labels — never by the policy.
        """
        q0 = self.q0
        for k in range(samples + 1):
            q = np.asarray(q0) + (k / samples) * (np.asarray(q_target) - np.asarray(q0))
            t = 0.0 if time_invariant else k / samples * self.max_steps * self.dt
            if not self.state_free(q, t):
                return False
        return True

    def ee_error_at(self, q: Sequence[float] | None = None) -> float:
        return self._error(self.state.q if q is None else q)

    def render_frames(self) -> list[dict]:
        """Recordable frames for visualization (actual simulated state)."""
        return [{
            "q": self.state.q.copy(),
            "time_s": self.time_s,
            "goal": self.goal,
            "obstacles": [(tuple(ob.center_at(self.time_s, self.spec.chamber_size)),
                           tuple(ob.size))
                          for ob in self.obstacles if ob.active_at(self.time_s)],
        }]
