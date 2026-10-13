"""Deterministic 3-D chamber with moving, axis-aligned cuboids.

No rendering, physics engine, ML framework, or third-party package required.
Units: meters and seconds. The agent is a sphere, NOT a robot arm.

All randomness lives in the explicit scene generator. Given the same scene and
control sequence, every trajectory is reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Sequence

Vec3 = tuple[float, float, float]
_EPS = 1e-10


def vec(values: Sequence[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError("Expected three coordinates")
    return (float(values[0]), float(values[1]), float(values[2]))


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def norm(a: Vec3) -> float:
    return math.sqrt(sum(v * v for v in a))


def lerp(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return add(a, mul(sub(b, a), alpha))


@dataclass(frozen=True)
class Cuboid:
    """An axis-aligned box whose center bounces elastically at chamber walls.

    center is the center at t=0. velocity is the nominal initial velocity.
    There is no randomness in movement. AABB geometry and constant size only.
    """
    name: str
    center: Vec3
    size: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)

    def validate(self, chamber_size: Vec3) -> None:
        if not self.name:
            raise ValueError("Obstacle name cannot be empty")
        for axis in range(3):
            if self.size[axis] <= 0 or self.size[axis] > chamber_size[axis]:
                raise ValueError(f"Invalid {self.name} size on axis {axis}")
            lo, hi = self.size[axis] / 2, chamber_size[axis] - self.size[axis] / 2
            if not lo - _EPS <= self.center[axis] <= hi + _EPS:
                raise ValueError(f"{self.name} starts outside chamber on axis {axis}")

    def center_at(self, t: float, chamber_size: Vec3) -> Vec3:
        if t < 0:
            raise ValueError("time cannot be negative")
        result = []
        for i in range(3):
            lo = self.size[i] / 2
            hi = chamber_size[i] - self.size[i] / 2
            span = hi - lo
            if span < _EPS:
                result.append(lo)
                continue
            phase = (self.center[i] - lo + self.velocity[i] * t) % (2 * span)
            result.append(lo + (phase if phase <= span else 2 * span - phase))
        return vec(result)

    def velocity_at(self, t: float, chamber_size: Vec3) -> Vec3:
        result = []
        for i in range(3):
            span = chamber_size[i] - self.size[i]
            if span < _EPS or abs(self.velocity[i]) < _EPS:
                result.append(0.0)
            else:
                phase = (self.center[i] - self.size[i] / 2 + self.velocity[i] * t) % (2 * span)
                result.append(self.velocity[i] if phase < span else -self.velocity[i])
        return vec(result)

    def bounce_times(self, t0: float, t1: float, chamber_size: Vec3) -> list[float]:
        """Exact times at which any component of the box's velocity reverses."""
        ts: list[float] = []
        for i in range(3):
            v = self.velocity[i]
            span = chamber_size[i] - self.size[i]
            if abs(v) < _EPS or span < _EPS:
                continue
            p0 = self.center[i] - self.size[i] / 2
            z0, z1 = p0 + v * t0, p0 + v * t1
            lower = math.floor(min(z0, z1) / span) - 1
            upper = math.ceil(max(z0, z1) / span) + 1
            for k in range(lower, upper + 1):
                candidate = (k * span - p0) / v
                if t0 + _EPS < candidate < t1 - _EPS:
                    ts.append(candidate)
        return sorted(set(round(x, 12) for x in ts))


@dataclass(frozen=True)
class ObstacleObservation:
    name: str
    center: Vec3
    size: Vec3
    velocity: Vec3


@dataclass(frozen=True)
class Observation:
    time_s: float
    step_index: int
    position: Vec3
    goal: Vec3
    obstacles: tuple[ObstacleObservation, ...]
    chamber_size: Vec3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reached_goal: bool
    collision: bool
    safety_intervention: bool
    done: bool
    attempted_velocity: Vec3

    def to_dict(self) -> dict:
        return asdict(self)


def _segment_hits_aabb(start: Vec3, end: Vec3, half_extent: Vec3) -> bool:
    """Exact slab intersection for a moving *point* relative to an AABB.

    The AABB is expanded on each axis by the spherical agent radius. This is a
    conservative approximation of a sphere-vs-box sweep (false positives near
    corners possible; undetected collisions from this approximation are not).
    """
    v = sub(end, start)
    u0, u1 = 0.0, 1.0
    for i in range(3):
        if abs(v[i]) < _EPS:
            if abs(start[i]) > half_extent[i] + _EPS:
                return False
        else:
            t_enter = (-half_extent[i] - start[i]) / v[i]
            t_exit = (half_extent[i] - start[i]) / v[i]
            u0 = max(u0, min(t_enter, t_exit))
            u1 = min(u1, max(t_enter, t_exit))
            if u0 > u1 + _EPS:
                return False
    return True


class ChamberEnv:
    """Deterministic moving-obstacle environment with swept-motion validation.

    step() rejects an unsafe commanded segment (a safety-shield intervention)
    and keeps the agent stationary, then advances time. If remaining stationary
    is ALSO unsafe due to an incoming moving obstacle, the episode terminates
    with collision=True. This is not a validated real-robot safety controller.
    """

    def __init__(
        self,
        chamber_size: Vec3 = (1.0, 1.0, 1.0),
        start: Vec3 = (0.1, 0.1, 0.1),
        goal: Vec3 = (0.9, 0.9, 0.9),
        obstacles: Sequence[Cuboid] = (),
        radius: float = 0.04,
        dt: float = 0.5,
        max_speed: float = 0.70,
        goal_tolerance: float = 0.06,
        max_steps: int = 60,
    ):
        self.chamber_size = vec(chamber_size)
        self.start = vec(start)
        self.goal = vec(goal)
        self.obstacles = tuple(obstacles)
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
        if len({ob.name for ob in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle names must be unique")
        for box in self.obstacles:
            box.validate(self.chamber_size)
        if not self.is_position_inside(self.start) or not self.is_position_inside(self.goal):
            raise ValueError("start and goal must be inside the chamber with radius margin")
        if not self.is_state_free(self.start, 0):
            raise ValueError("start is inside an inflated obstacle at t=0")
        self.reset()

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
        for ob in self.obstacles:
            center = ob.center_at(t, self.chamber_size)
            if all(abs(position[i] - center[i]) <= ob.size[i] / 2 + self.radius + _EPS
                   for i in range(3)):
                return False
        return True

    def is_motion_safe(self, p0: Vec3, p1: Vec3, t0: float, t1: float) -> bool:
        """Check continuous movement against all moving cuboids between times.

        Assumes the agent moves linearly p0->p1 at constant speed during [t0,t1].
        Splits at every obstacle-wall bounce, then uses relative swept AABBs.
        """
        if t1 < t0 or t0 < 0:
            raise ValueError("Require 0 <= t0 <= t1")
        if not (self.is_position_inside(p0) and self.is_position_inside(p1)):
            return False
        if t1 - t0 <= _EPS:
            return self.is_state_free(p0, t0) and self.is_state_free(p1, t1)
        for box in self.obstacles:
            times = [t0] + box.bounce_times(t0, t1, self.chamber_size) + [t1]
            half = vec(tuple(box.size[i] / 2 + self.radius for i in range(3)))
            for lo, hi in zip(times, times[1:]):
                a0 = lerp(p0, p1, (lo - t0) / (t1 - t0))
                a1 = lerp(p0, p1, (hi - t0) / (t1 - t0))
                b0 = box.center_at(lo, self.chamber_size)
                b1 = box.center_at(hi, self.chamber_size)
                if _segment_hits_aabb(sub(a0, b0), sub(a1, b1), half):
                    return False
        return True

    def observe(self) -> Observation:
        return Observation(
            time_s=round(self.time_s, 10), step_index=self.step_index,
            position=self.position, goal=self.goal,
            chamber_size=self.chamber_size,
            obstacles=tuple(ObstacleObservation(
                ob.name, ob.center_at(self.time_s, self.chamber_size), ob.size,
                ob.velocity_at(self.time_s, self.chamber_size)
            ) for ob in self.obstacles),
        )

    def step(self, velocity: Vec3) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is complete; call reset()")
        velocity = vec(velocity)
        if norm(velocity) > self.max_speed + _EPS:
            raise ValueError(f"speed {norm(velocity):.3f} exceeds max_speed {self.max_speed:.3f}")
        desired = add(self.position, mul(velocity, self.dt))
        end_time = self.time_s + self.dt
        safe = self.is_motion_safe(self.position, desired, self.time_s, end_time)
        intervention = not safe
        collision = False
        if safe:
            self.position = desired
        else:
            # The shield refuses the unsafe action, but a moving obstacle can
            # collide with a stopped agent too. Detect that explicitly.
            collision = not self.is_motion_safe(self.position, self.position, self.time_s, end_time)
        self.step_index += 1
        reached = norm(sub(self.position, self.goal)) <= self.goal_tolerance
        self.done = reached or collision or self.step_index >= self.max_steps
        return StepResult(self.observe(), reached, collision, intervention, self.done, velocity)


def example_scene(kind: str = "moving", seed: int = 7) -> ChamberEnv:
    """Reproducible canonical scenarios. The agent starts/ends on a 0.2-m grid."""
    common = dict(chamber_size=(1, 1, 1), start=(.1, .1, .1), goal=(.9, .9, .9),
                  radius=.035, dt=.5, max_speed=.70, goal_tolerance=.055, max_steps=40)
    if kind == "empty":
        boxes = ()
    elif kind == "static":
        boxes = (Cuboid("central", (.5, .5, .5), (.23, .36, .36)),)
    elif kind == "moving":
        boxes = (
            Cuboid("central", (.5, .5, .5), (.22, .30, .30)),
            Cuboid("moving", (.60, .26, .72), (.14, .14, .14), (-.13, .18, 0)),
        )
    elif kind == "crossing":
        common["start"], common["goal"] = (.1, .5, .5), (.9, .5, .5)
        boxes = (Cuboid("crossing", (.5, .2, .5), (.20, .18, .22), (0, .20, 0)),)
    elif kind == "random":
        rng = random.Random(seed)
        built = []
        for index in range(rng.randint(1, 5)):
            for _ in range(200):
                size = vec([rng.uniform(.10, .23) for _ in range(3)])
                center = vec([rng.uniform(size[i] / 2, 1 - size[i] / 2) for i in range(3)])
                velocity = vec([rng.uniform(-.20, .20) for _ in range(3)])
                ob = Cuboid(f"random_{index}", center, size, velocity)
                if all(not all(abs(p[i] - center[i]) <= size[i] / 2 + common["radius"] + .03
                                   for i in range(3)) for p in (common["start"], common["goal"])):
                    built.append(ob)
                    break
            else:
                raise RuntimeError("Failed to sample non-overlapping initial obstacle")
        boxes = tuple(built)
    else:
        raise ValueError(f"Unknown scenario: {kind}")
    return ChamberEnv(**common, obstacles=boxes)
