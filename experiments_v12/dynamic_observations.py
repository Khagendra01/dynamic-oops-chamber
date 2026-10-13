"""Causal observation models (v1.2 level 4 and beyond).

The planner only ever receives an :class:`chamber.Observation`.  This module
produces those observations from the simulator's ground truth under a configured
degradation model:

* ``position_noise``   additive Gaussian noise on the reported agent position
* ``velocity_noise``   additive Gaussian noise on reported obstacle velocities
* ``delay_steps``      observations are ``k`` control periods old
* ``serve_every``      observations are only refreshed every ``k`` steps
* ``visibility_range`` obstacles farther than this from the agent are hidden

Causality contract (asserted by ``tests/test_observations.py``):

* the observer only ever reads simulator state at time ``t <= now``;
* a delayed/stale observation reports its own (older) ``time_s`` and
  ``step_index`` so the observation age is visible to the planner;
* noise is a pure function of ``(episode_seed, step, channel)``, so replaying a
  scenario with the same seed reproduces the same observation sequence.

Ground-truth future obstacle states are never exposed here.  Every evaluated
method receives the same ``Observation`` object produced by this observer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

from chamber import Observation, ObstacleObservation, norm, sub, vec


@dataclass(frozen=True)
class ObservationConfig:
    position_noise: float = 0.0
    velocity_noise: float = 0.0
    delay_steps: int = 0
    serve_every: int = 1
    visibility_range: float = math.inf
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "ObservationConfig":
        data = dict(data or {})
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict:
        payload = asdict(self)
        if math.isinf(self.visibility_range):
            payload["visibility_range"] = "inf"
        return payload

    @property
    def perfect(self) -> bool:
        return (self.position_noise == 0.0 and self.velocity_noise == 0.0
                and self.delay_steps == 0 and self.serve_every == 1
                and math.isinf(self.visibility_range))


class CausalObserver:
    """Serve degraded-but-causal observations of a live environment."""

    def __init__(self, config: "ObservationConfig | dict | None" = None,
                 episode_seed: int = 0):
        if not isinstance(config, ObservationConfig):
            config = ObservationConfig.from_dict(config)
        self.config = config
        self.episode_seed = int(episode_seed)
        self.last: Observation | None = None

    def _noise(self, step: int, channel: int, sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        rng = random.Random(self.episode_seed * 1_000_003 + step * 97 + channel)
        return rng.gauss(0.0, sigma)

    def _truth_at(self, env, step: int) -> Observation:
        """Ground truth at ``step``; ``step`` is never greater than the now."""
        if step == env.step_index:
            return env.observe()
        saved = env.step_index
        try:
            env.step_index = step
            return env.observe()
        finally:
            env.step_index = saved

    def observe(self, env) -> Observation:
        now = env.step_index
        cfg = self.config
        if cfg.serve_every > 1 and now % cfg.serve_every != 0 and self.last is not None:
            return self.last
        step = max(0, now - max(0, cfg.delay_steps))
        truth = self._truth_at(env, step)

        position = truth.position
        if cfg.position_noise > 0.0:
            position = vec(tuple(
                min(max(position[i] + self._noise(step, i, cfg.position_noise),
                        env.radius),
                    env.chamber_size[i] - env.radius)
                for i in range(3)))

        obstacles = []
        for item in truth.obstacles:
            center = item.center
            velocity = item.velocity
            channel = 100 + (sum(ord(c) for c in item.name) % 89)
            if cfg.velocity_noise > 0.0:
                velocity = vec(tuple(
                    velocity[i] + self._noise(step, channel + i, cfg.velocity_noise)
                    for i in range(3)))
            if cfg.visibility_range < math.inf:
                if norm(sub(center, position)) > cfg.visibility_range:
                    continue
            obstacles.append(ObstacleObservation(item.name, center, item.size,
                                                 velocity))
        served = Observation(time_s=truth.time_s, step_index=truth.step_index,
                             position=position, goal=truth.goal,
                             obstacles=tuple(obstacles),
                             chamber_size=truth.chamber_size)
        self.last = served
        return served


def observer_for(spec, env) -> CausalObserver:
    """Build the observer described by a ``ScenarioSpec`` observation block."""
    return CausalObserver(ObservationConfig.from_dict(spec.observation),
                          episode_seed=spec.seed)
