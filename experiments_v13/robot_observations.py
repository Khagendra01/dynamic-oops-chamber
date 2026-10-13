"""Causal observations for the v1.3 robot chamber.

Preserves the v1.2.1 scientific contract:

* proprioception (joint positions/velocities, EE pose, goal) is always
  current;
* exteroception (obstacle centers/velocities) may be delayed/noised by an
  optional :class:`CausalObserver`;
* the simulator's true future obstacle schedule is never exposed;
* the execution shield is attached to the *same* observer the policy sees
  (``env.shield_observer``), never to ground-truth futures.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from experiments_v13.robot_model import RobotSpec, Vec3

_EPS = 1e-12


@dataclass(frozen=True)
class ObsBox:
    """Observable obstacle snapshot: axis-aligned center/size/velocity."""

    name: str
    center: Vec3
    size: Vec3
    velocity: Vec3


@dataclass(frozen=True)
class JointObservation:
    """What a policy may read at one decision epoch."""

    time_s: float
    step_index: int
    q: tuple[float, ...]
    qd: tuple[float, ...]
    ee_position: Vec3
    goal: Vec3
    obstacles: tuple[ObsBox, ...]
    chamber_size: Vec3
    joint_lo: tuple[float, ...]
    joint_hi: tuple[float, ...]
    velocity_limits: tuple[float, ...]

    @property
    def ee_error(self) -> float:
        return math.sqrt(sum((self.ee_position[i] - self.goal[i]) ** 2
                             for i in range(3)))


@dataclass(frozen=True)
class ObservationConfig:
    """Degradation settings.  Defaults are perfect causal observations."""

    perfect: bool = True
    obstacle_delay_steps: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "ObservationConfig":
        data = data or {}
        return cls(perfect=bool(data.get("perfect", True)),
                   obstacle_delay_steps=int(data.get("obstacle_delay_steps", 0)))


class CausalObserver:
    """Serves delayed exteroception + current proprioception.

    Maintains a short history of raw obstacle snapshots keyed by episode
    step.  Joint state and EE pose are always read live from the simulator —
    proprioceptive sensors are assumed current.  No future information is
    buffered or accessible.
    """

    def __init__(self, config: ObservationConfig, robot: RobotSpec):
        self.config = config
        self.robot = robot
        self._history: deque[list[ObsBox]] = deque(maxlen=64)

    def reset(self) -> None:
        self._history.clear()

    def observe(self, env) -> JointObservation:
        raw = env.raw_obstacle_snapshot()
        self._history.append(raw)
        if self.config.perfect or self.config.obstacle_delay_steps <= 0:
            obstacles = raw
        else:
            index = len(self._history) - 1 - self.config.obstacle_delay_steps
            obstacles = self._history[index] if index >= 0 else []
        return env.make_observation(obstacles)


def observed_field(chamber_size: Sequence[float],
                   obstacles: Sequence[ObsBox]):
    """Build a predictor field from an observation (constant-velocity model).

    Uses only the served snapshot: center + instantaneous velocity + known
    wall reflections.  Velocity-event schedules and appearance times from the
    simulator are *not* available here by construction.
    """
    from experiments_v13.swept_collision import MovingObstacleField
    return MovingObstacleField(
        chamber_size,
        [(b.center, b.size, b.velocity) for b in obstacles])
