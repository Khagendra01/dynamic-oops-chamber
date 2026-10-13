"""Extended dynamic chamber for experiment series v1.2.

Adds, without touching ``chamber.py`` (the v1.1 correctness oracle):

* piecewise-constant velocity schedules (random direction changes,
  stop-and-start, speed changes) with exact wall reflections;
* obstacles that appear / disappear during an episode;
* configurable chamber, time step, obstacle counts, dimensions, speeds and
  episode limits.

The public surface mirrors :class:`chamber.ChamberEnv` (``reset``, ``observe``,
``step``, ``done``, ``position``, ``goal``, ``radius``, ``dt``, ``max_speed``,
``goal_tolerance``, ``max_steps``, ``is_motion_safe``) so the existing
interpreter, ``replay()`` and search code work unchanged.

Collision semantics are the same continuous swept-AABB test used by
``chamber.ChamberEnv``: the agent moves linearly over ``[t0, t1]``, each box
trajectory is split wherever its velocity changes (wall bounce, schedule knot,
appearance or disappearance) and the relative motion is tested with the exact
slab intersection from ``chamber._segment_hits_aabb``.

A pure-``Cuboid`` scene built here reproduces ``ChamberEnv`` exactly; that is
asserted by ``tests/test_dynamic_chamber.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from chamber import (Cuboid, Observation, ObstacleObservation, StepResult, Vec3,
                     _segment_hits_aabb, add, lerp, mul, norm, sub, vec)

_EPS = 1e-10


@dataclass(frozen=True)
class VelocityEvent:
    """Velocity change at absolute time ``t`` (must not precede appearance)."""
    t: float
    velocity: Vec3


class DynamicBox:
    """Axis-aligned cuboid with a piecewise-constant velocity schedule.

    Motion inside one segment is the triangular wall-bounce trajectory used by
    :class:`chamber.Cuboid`, restarted from the segment entry state, so a
    single-segment box is equivalent to a ``Cuboid`` with the same parameters.
    """

    def __init__(self, name: str, center: Vec3, size: Vec3,
                 velocity: Vec3 = (0.0, 0.0, 0.0),
                 events: Sequence[VelocityEvent] = (),
                 appear_at: float = 0.0,
                 disappear_at: float | None = None):
        if not name:
            raise ValueError("Obstacle name cannot be empty")
        if appear_at < 0:
            raise ValueError("appear_at must be >= 0")
        if disappear_at is not None and disappear_at <= appear_at + _EPS:
            raise ValueError("disappear_at must be after appear_at")
        self.name = name
        self.center = vec(center)
        self.size = vec(size)
        self.velocity = vec(velocity)
        self.appear_at = float(appear_at)
        self.disappear_at = None if disappear_at is None else float(disappear_at)
        knots = sorted((VelocityEvent(float(e.t), vec(e.velocity)) for e in events),
                       key=lambda e: e.t)
        if knots and knots[0].t < self.appear_at - _EPS:
            raise ValueError("velocity event precedes appearance")
        if self.disappear_at is not None:
            self.events = tuple(e for e in knots if e.t < self.disappear_at - _EPS)
        else:
            self.events = tuple(knots)
        self._segment_cache: dict[tuple, list[tuple]] = {}

    # ------------------------------------------------------------------ model
    def _center_in(self, seg_start: float, seg_vel: Vec3, seg_pos: Vec3,
                   t: float, chamber_size: Vec3) -> Vec3:
        out = []
        for i in range(3):
            lo = self.size[i] / 2
            hi = chamber_size[i] - self.size[i] / 2
            span = hi - lo
            if span < _EPS:
                out.append(lo)
                continue
            phase = (seg_pos[i] - lo + seg_vel[i] * (t - seg_start)) % (2 * span)
            out.append(lo + (phase if phase <= span else 2 * span - phase))
        return vec(out)

    def segments(self, chamber_size: Vec3) -> list[tuple[float, float, Vec3, Vec3]]:
        """[(t_start, t_end, center_at_t_start, velocity)] for this chamber."""
        key = tuple(chamber_size)
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached
        segments: list[tuple[float, float, Vec3, Vec3]] = []
        pos, vel, t = self.center, self.velocity, self.appear_at
        for knot in self.events:
            if knot.t <= t + _EPS:
                vel = knot.velocity
                continue
            segments.append((t, knot.t, pos, vel))
            pos = self._center_in(t, vel, pos, knot.t, chamber_size)
            t = knot.t
            vel = knot.velocity
        if self.disappear_at is None:
            segments.append((t, math.inf, pos, vel))
        elif self.disappear_at > t + _EPS:
            segments.append((t, self.disappear_at, pos, vel))
        if not segments:
            segments.append((t, math.inf, pos, vel))
        self._segment_cache[key] = segments
        return segments

    def active_at(self, t: float) -> bool:
        if t < self.appear_at - _EPS:
            return False
        if self.disappear_at is not None and t >= self.disappear_at - _EPS:
            return False
        return True

    def _segment_for(self, t: float, chamber_size: Vec3) -> tuple | None:
        if not self.active_at(t):
            return None
        for seg in self.segments(chamber_size):
            t0, t1, _pos, _vel = seg
            if t0 - _EPS <= t <= t1 + _EPS:
                return seg
        return self.segments(chamber_size)[-1]

    def center_at(self, t: float, chamber_size: Vec3) -> Vec3:
        if t < 0:
            raise ValueError("time cannot be negative")
        seg = self._segment_for(t, chamber_size)
        if seg is None:
            return self.center if t < self.appear_at else self._center_at_edge(
                t, chamber_size)
        return self._center_in(seg[0], seg[3], seg[2], t, chamber_size)

    def _center_at_edge(self, t: float, chamber_size: Vec3) -> Vec3:
        if t < self.appear_at:
            return self.center
        boundary = self.disappear_at if self.disappear_at is not None else t
        for seg in reversed(self.segments(chamber_size)):
            if seg[0] <= boundary + _EPS:
                return self._center_in(seg[0], seg[3], seg[2],
                                       min(boundary, seg[1]), chamber_size)
        return self.center

    def velocity_at(self, t: float, chamber_size: Vec3) -> Vec3:
        seg = self._segment_for(t, chamber_size)
        if seg is None:
            return (0.0, 0.0, 0.0)
        t0, _t1, pos, vel = seg
        out = []
        for i in range(3):
            span = chamber_size[i] - self.size[i]
            if span < _EPS or abs(vel[i]) < _EPS:
                out.append(0.0)
                continue
            origin = pos[i] - self.size[i] / 2
            phase = (origin + vel[i] * (t - t0)) % (2 * span)
            out.append(vel[i] if phase < span else -vel[i])
        return vec(out)

    def breakpoints(self, t0: float, t1: float, chamber_size: Vec3) -> list[float]:
        """Times strictly inside (t0, t1) where this box velocity is discontinuous."""
        marks: list[float] = []
        for seg_start, seg_end, pos, vel in self.segments(chamber_size):
            if seg_end <= t0 + _EPS or seg_start >= t1 - _EPS:
                continue
            lo, hi = max(seg_start, t0), min(seg_end, t1)
            if hi - lo <= _EPS:
                continue
            if seg_start > t0 + _EPS and seg_start < t1 - _EPS:
                marks.append(seg_start)
            for i in range(3):
                span = chamber_size[i] - self.size[i]
                if span < _EPS or abs(vel[i]) < _EPS:
                    continue
                origin = pos[i] - self.size[i] / 2
                z0 = origin + vel[i] * (lo - seg_start)
                z1 = origin + vel[i] * (hi - seg_start)
                lower = math.floor(min(z0, z1) / span) - 1
                upper = math.ceil(max(z0, z1) / span) + 1
                for k in range(lower, upper + 1):
                    candidate = seg_start + (k * span - origin) / vel[i]
                    if lo + _EPS < candidate < hi - _EPS:
                        marks.append(candidate)
        for boundary in (self.appear_at, self.disappear_at):
            if boundary is not None and t0 + _EPS < boundary < t1 - _EPS:
                marks.append(boundary)
        return sorted(set(round(x, 12) for x in marks))

    def validate(self, chamber_size: Vec3) -> None:
        for axis in range(3):
            if self.size[axis] <= 0 or self.size[axis] > chamber_size[axis]:
                raise ValueError(f"Invalid {self.name} size on axis {axis}")
            lo, hi = self.size[axis] / 2, chamber_size[axis] - self.size[axis] / 2
            if not lo - _EPS <= self.center[axis] <= hi + _EPS:
                raise ValueError(f"{self.name} starts outside chamber on axis {axis}")

    @classmethod
    def from_cuboid(cls, box: Cuboid) -> "DynamicBox":
        return cls(box.name, box.center, box.size, box.velocity)


def to_dynamic(obstacles: Sequence) -> tuple[DynamicBox, ...]:
    out = []
    for item in obstacles:
        if isinstance(item, DynamicBox):
            out.append(item)
        elif isinstance(item, Cuboid):
            out.append(DynamicBox.from_cuboid(item))
        else:
            raise TypeError(f"Unsupported obstacle type: {type(item)!r}")
    return tuple(out)


class DynamicChamber:
    """Moving-cuboid chamber with scheduled velocity changes and appearances."""

    def __init__(self, chamber_size: Vec3 = (1.0, 1.0, 1.0),
                 start: Vec3 = (0.1, 0.1, 0.1),
                 goal: Vec3 = (0.9, 0.9, 0.9),
                 obstacles: Sequence = (),
                 radius: float = 0.04,
                 dt: float = 0.5,
                 max_speed: float = 0.70,
                 goal_tolerance: float = 0.06,
                 max_steps: int = 60):
        self.chamber_size = vec(chamber_size)
        self.start = vec(start)
        self.goal = vec(goal)
        self.obstacles: tuple[DynamicBox, ...] = to_dynamic(obstacles)
        self.radius = float(radius)
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.goal_tolerance = float(goal_tolerance)
        self.max_steps = int(max_steps)
        if any(v <= 0 for v in self.chamber_size):
            raise ValueError("chamber dimensions must be positive")
        if self.radius <= 0 or self.dt <= 0 or self.max_speed <= 0:
            raise ValueError("radius, dt and max_speed must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if len({b.name for b in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle names must be unique")
        for box in self.obstacles:
            box.validate(self.chamber_size)
        if not self.is_position_inside(self.start) or not self.is_position_inside(self.goal):
            raise ValueError("start and goal must be inside the chamber")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> Observation:
        self.position = self.start
        self.step_index = 0
        self.done = norm(sub(self.start, self.goal)) <= self.goal_tolerance
        return self.observe()

    @property
    def time_s(self) -> float:
        return self.step_index * self.dt

    def is_position_inside(self, point: Vec3) -> bool:
        return all(self.radius - _EPS <= point[i] <= self.chamber_size[i] - self.radius + _EPS
                   for i in range(3))

    def is_state_free(self, position: Vec3, t: float) -> bool:
        if not self.is_position_inside(position):
            return False
        for box in self.obstacles:
            if not box.active_at(t):
                continue
            center = box.center_at(t, self.chamber_size)
            if all(abs(position[i] - center[i]) <= box.size[i] / 2 + self.radius + _EPS
                   for i in range(3)):
                return False
        return True

    def is_motion_safe(self, p0: Vec3, p1: Vec3, t0: float, t1: float) -> bool:
        """Continuous swept check against every box active inside [t0, t1]."""
        if t1 < t0 or t0 < 0:
            raise ValueError("Require 0 <= t0 <= t1")
        if not (self.is_position_inside(p0) and self.is_position_inside(p1)):
            return False
        if t1 - t0 <= _EPS:
            return self.is_state_free(p0, t0) and self.is_state_free(p1, t1)
        for box in self.obstacles:
            lo = max(t0, box.appear_at)
            hi = t1 if box.disappear_at is None else min(t1, box.disappear_at)
            if hi - lo <= _EPS:
                continue
            times = [lo] + box.breakpoints(lo, hi, self.chamber_size) + [hi]
            half = vec(tuple(box.size[i] / 2 + self.radius for i in range(3)))
            for a, b in zip(times, times[1:]):
                if b - a <= _EPS:
                    continue
                a0 = lerp(p0, p1, (a - t0) / (t1 - t0))
                a1 = lerp(p0, p1, (b - t0) / (t1 - t0))
                b0 = box.center_at(a, self.chamber_size)
                b1 = box.center_at(b, self.chamber_size)
                if _segment_hits_aabb(sub(a0, b0), sub(a1, b1), half):
                    return False
        return True

    def observe(self) -> Observation:
        boxes = []
        for ob in self.obstacles:
            if not ob.active_at(self.time_s):
                continue
            boxes.append(ObstacleObservation(
                ob.name, ob.center_at(self.time_s, self.chamber_size), ob.size,
                ob.velocity_at(self.time_s, self.chamber_size)))
        return Observation(
            time_s=round(self.time_s, 10), step_index=self.step_index,
            position=self.position, goal=self.goal,
            chamber_size=self.chamber_size, obstacles=tuple(boxes))

    def step(self, velocity: Vec3) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is complete; call reset()")
        velocity = vec(velocity)
        if norm(velocity) > self.max_speed + _EPS:
            raise ValueError(f"speed {norm(velocity):.3f} exceeds max_speed {self.max_speed}")
        desired = add(self.position, mul(velocity, self.dt))
        end_time = self.time_s + self.dt
        safe = self.is_motion_safe(self.position, desired, self.time_s, end_time)
        intervention = not safe
        collision = False
        if safe:
            self.position = desired
        else:
            collision = not self.is_motion_safe(self.position, self.position,
                                                self.time_s, end_time)
        self.step_index += 1
        reached = norm(sub(self.position, self.goal)) <= self.goal_tolerance
        self.done = reached or collision or self.step_index >= self.max_steps
        return StepResult(self.observe(), reached, collision, intervention,
                          self.done, velocity)
