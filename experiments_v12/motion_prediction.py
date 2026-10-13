"""Single shared causal motion-prediction model (v1.2).

Both the synthesized programs and every baseline planner must use the *same*
prediction and collision rules, otherwise differences in the results could come
from the rules rather than from the planner.  This module is that single source:

* ``predict``        extrapolate an observation forward using the observed
                     velocity and the known reflecting-wall model (no oracle);
* ``segment_safe``   swept safety of an arbitrary agent segment over one control
                     period, evaluated through ``oops_language.predicted_safe``
                     so the geometry is literally the interpreter's geometry;
* ``config_from_env`` build the shared :class:`oops_language.MotionConfig`.

Nothing here reads the simulator's future obstacle states.
"""
from __future__ import annotations

from chamber import Observation, Vec3, mul, sub
from oops_language import MotionConfig, predicted_safe
from oops_language import _shift_observation as _shift


def config_from_env(env) -> MotionConfig:
    return MotionConfig(radius=env.radius, dt=env.dt, max_speed=env.max_speed)


def predict(obs: Observation, seconds: float) -> Observation:
    """Extrapolate ``obs`` ``seconds`` into the future from observed state."""
    if seconds <= 0:
        return obs
    return _shift(obs, seconds)


class Predictor:
    """Stateless constant-velocity predictor bound to one control period."""

    def __init__(self, dt: float):
        self.dt = float(dt)

    def at_time(self, obs: Observation, t_target: float) -> Observation:
        delta = t_target - obs.time_s
        return predict(obs, delta) if delta > 1e-12 else obs

    def after_steps(self, obs: Observation, steps: int) -> Observation:
        if steps <= 0:
            return obs
        return predict(obs, steps * self.dt)

    def segment_safe(self, obs: Observation, p0: Vec3, p1: Vec3,
                     config: MotionConfig) -> bool:
        """Swept safety of p0->p1 over one control period at ``obs.time_s``."""
        return segment_safe(obs, p0, p1, config)


def segment_safe(obs: Observation, p0: Vec3, p1: Vec3,
                 config: MotionConfig) -> bool:
    """Safety of an arbitrary segment using the interpreter's own rule.

    Implemented by probing ``oops_language.predicted_safe`` with a synthetic
    observation rooted at ``p0`` so there is exactly one swept-check
    implementation in the project.
    """
    velocity = mul(sub(p1, p0), 1.0 / config.dt)
    probe = Observation(time_s=obs.time_s, step_index=obs.step_index,
                        position=p0, goal=obs.goal, obstacles=obs.obstacles,
                        chamber_size=obs.chamber_size)
    return predicted_safe(probe, velocity, config)
