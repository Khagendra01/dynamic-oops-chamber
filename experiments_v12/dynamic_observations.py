"""Causal observation models (v1.2 level 4 and beyond).

The planner only ever receives an :class:`chamber.Observation`.  This module
produces those observations from the simulator's ground truth under a configured
degradation model:

* ``position_noise``   additive Gaussian noise on the reported agent position
* ``velocity_noise``   additive Gaussian noise on reported obstacle velocities
* ``delay_steps``      *exteroception* is ``k`` control periods old
* ``serve_every``      observations are only refreshed every ``k`` steps
* ``visibility_range`` obstacles farther than this from the agent are hidden

v1.2.1 semantics — proprioception vs exteroception
-------------------------------------------------
Delay applies to **exteroception only** (obstacle states and the observation
timestamp used for motion prediction).  The agent's own pose and goal are
**current proprioception**: a real robot always knows where it is now, but
perceives obstacles with latency.  The served observation therefore pairs a
current ``position``/``goal`` with obstacle snapshots and ``time_s`` taken
``delay_steps`` periods in the past.  (v1.2 accidentally produced this split
via a state-mutation hack while *labelling* the whole observation as delayed;
v1.2.1 makes the split explicit, removes the hack, and tests it.)

Causality contract (asserted by ``tests/test_observations.py``):

* the observer only ever reads simulator state at time ``t <= now``;
* a delayed observation reports its own (older) ``time_s``/``step_index``
  for obstacles while ``position``/``goal`` remain current;
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

    def observe(self, env) -> Observation:
        now = env.step_index
        cfg = self.config
        if cfg.serve_every > 1 and now % cfg.serve_every != 0 and self.last is not None:
            return self.last
        step = max(0, now - max(0, cfg.delay_steps))
        t_old = round(step * env.dt, 10)

        # Exteroception: obstacle states as last seen at t_old (never future).
        obstacles = []
        for ob in env.obstacles:
            if not ob.active_at(t_old):
                continue
            center = ob.center_at(t_old, env.chamber_size)
            velocity = ob.velocity_at(t_old, env.chamber_size)
            channel = 100 + (sum(ord(c) for c in ob.name) % 89)
            if cfg.velocity_noise > 0.0:
                velocity = vec(tuple(
                    velocity[i] + self._noise(step, channel + i, cfg.velocity_noise)
                    for i in range(3)))
            if cfg.visibility_range < math.inf:
                if norm(sub(center, env.position)) > cfg.visibility_range:
                    continue
            obstacles.append(ObstacleObservation(ob.name, center, ob.size,
                                                 velocity))

        # Proprioception: current pose and goal (never delayed).
        position = env.position
        if cfg.position_noise > 0.0:
            position = vec(tuple(
                min(max(position[i] + self._noise(now, i, cfg.position_noise),
                        env.radius),
                    env.chamber_size[i] - env.radius)
                for i in range(3)))

        served = Observation(time_s=t_old, step_index=step,
                             position=position, goal=env.goal,
                             obstacles=tuple(obstacles),
                             chamber_size=env.chamber_size)
        self.last = served
        return served


def observer_for(spec, env) -> CausalObserver:
    """Build the observer described by a ``ScenarioSpec`` observation block."""
    return CausalObserver(ObservationConfig.from_dict(spec.observation),
                          episode_seed=spec.seed)
