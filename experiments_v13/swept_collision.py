"""Conservative continuous-motion collision certification for joint motions.

Two independent mechanisms (kept separate on purpose):

``dense_motion_check``
    High-resolution *detection*: samples the joint interpolation and moving
    obstacles at ``samples`` sub-steps and runs exact SAT at each.  Used by
    the simulator's post-hoc ground-truth collision detector, scenario
    trajectory-witness verification and baseline edge validation.  Dense
    sampling alone is *not* claimed to be a continuous guarantee.

``certify_motion``
    *Conservative certification*: adaptive interval subdivision with rigorous
    displacement bounds.  A subinterval is certified SAFE only when the
    exact-cleared start state plus a proven displacement bound keeps every
    link clear of every obstacle, the boundary and nonadjacent links for the
    whole subinterval.  If the budget is exhausted before certification the
    result is UNKNOWN (callers must treat it as unsafe-to-execute).  UNSAFE
    is returned only when an exact sampled collision is found.

Displacement bound (per link, rigorous)
--------------------------------------
For a joint increment ``dq`` applied over the interval, any point ``p`` of a
link driven by joint ``a`` moves by at most::

    D(p) <= sum_{j<=a} |dq_j| * rho_j

with the lever chain (velocity-propagation bound, exact over the interval by
the mean-value argument with the recursive lever definition)::

    rho_1 = R_1
    rho_j = R_j + sum_{k<j} |dq_k| * rho_k

where ``R_j = max_p |p - origin_j|`` at the interval start over all tracked
points downstream of joint ``j``.  Rotations are isometries, so intermediate
configurations cannot stretch distances beyond this chain.

Obstacle points move at most ``|v| * tau`` (elastic wall bounces preserve
speed).  Relative displacement is the sum.  Boundary safety requires the
start corner margin to exceed the link bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from experiments_v13.collision_geometry import (CollisionChecker, OBB,
                                                corners_clear_of_walls,
                                                obb_sphere_gap)
from experiments_v13.forward_kinematics import forward_pose
from experiments_v13.robot_model import RobotSpec, link_corners_world

_EPS = 1e-12
ObstacleAt = Callable[[float], tuple[list[OBB], list[str]]]
# (time) -> (obstacle boxes, names)


SAFE = "safe"
UNSAFE = "unsafe"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class MotionVerdict:
    status: str                 # SAFE | UNSAFE | UNKNOWN
    reason: str
    checks: int
    hit: tuple[str, str, str] | None = None   # category, body_a, body_b


def link_lever_chain(spec: RobotSpec, q: Sequence[float]) -> np.ndarray:
    """rho_j per joint: rigorous downstream lever bound at configuration q."""
    n = spec.joint_count
    pose = forward_pose(spec, q)
    # tracked points: all link corners + tool tip
    points: list[np.ndarray] = []
    link_of_point: list[int] = []
    for i, link in enumerate(spec.links):
        frame = pose.frames[link.drive_joint + 1 if link.drive_joint >= 0 else 0]
        for corner in link_corners_world(link, frame):
            points.append(corner)
            link_of_point.append(i)
    tool = pose.ee[:3, 3]
    points.append(tool)
    link_of_point.append(len(spec.links) - 1)
    pts = np.asarray(points)

    rho = np.zeros(n, dtype=float)
    origins = np.array([pose.frames[j + 1][:3, 3] for j in range(n)])
    # R_j = max |p - origin_j| over points attached downstream of joint j
    for j in range(n):
        worst = 0.0
        for p, li in zip(pts, link_of_point):
            drive = spec.links[li].drive_joint
            if drive >= 0 and drive < j:
                continue  # point is upstream of joint j: does not rotate with it
            if drive == -1:
                continue  # static base body never moves with any joint
            worst = max(worst, float(np.linalg.norm(p - origins[j])))
        rho[j] = max(worst, 1e-9)
    return rho


def displacement_bound(spec: RobotSpec, q: Sequence[float],
                       dq: Sequence[float]) -> np.ndarray:
    """Per-link max displacement over the interval (array over links)."""
    rho = link_lever_chain(spec, q)
    absdq = np.abs(np.asarray(dq, dtype=float))
    n = spec.joint_count
    # enforce the recursive definition rho_j' = R_j + sum_{k<j} |dq_k| rho_k'
    r = rho.copy()
    for j in range(1, n):
        r[j] = rho[j] + float(np.sum(absdq[:j] * r[:j]))
    total_joint = float(np.sum(absdq * r))
    out = np.full(len(spec.links), total_joint, dtype=float)
    for i, link in enumerate(spec.links):
        a = link.drive_joint
        if a < 0:
            out[i] = 0.0
        else:
            out[i] = float(np.sum(absdq[: a + 1] * r[: a + 1]))
    return out


def dense_motion_check(spec: RobotSpec, checker: CollisionChecker,
                       q0: Sequence[float], dq: Sequence[float],
                       t0: float, tau: float, obstacle_at: ObstacleAt,
                       samples: int = 12) -> tuple[bool, tuple[str, str, str] | None]:
    """Sampled ground-truth detection along q(s)=q0+(s/tau)*dq, s in [0,tau].

    Returns (collided, hit-record).  Not a continuous guarantee — see module
    docstring.  Used only for post-execution detection and witness checks.
    """
    if tau <= 0:
        boxes, names = obstacle_at(t0)
        hits = checker.check_state(q0, boxes, names)
        return (bool(hits), (hits[0].category, hits[0].body_a, hits[0].body_b)
                if hits else None)
    for k in range(samples + 1):
        s = k / samples
        q = np.asarray(q0, dtype=float) + s * np.asarray(dq, dtype=float)
        boxes, names = obstacle_at(t0 + s * tau)
        hits = checker.check_state(q, boxes, names)
        if hits:
            return True, (hits[0].category, hits[0].body_a, hits[0].body_b)
    return False, None


def certify_motion(spec: RobotSpec, checker: CollisionChecker,
                   q0: Sequence[float], dq: Sequence[float],
                   t0: float, tau: float, obstacle_at: ObstacleAt,
                   *, min_interval: float | None = None,
                   max_checks: int = 96) -> MotionVerdict:
    """Conservatively certify the swept motion q(s) against exact geometry.

    SAFE  — proven collision-free for the whole interval (start state exact-
            cleared, every pair's clearance exceeds the displacement bound).
    UNSAFE — an exact sampled collision was found.
    UNKNOWN — budget exhausted; callers must treat as not provably safe.
    """
    q0 = np.asarray(q0, dtype=float)
    dq = np.asarray(dq, dtype=float)
    if tau <= _EPS:
        if np.linalg.norm(dq) <= _EPS:
            boxes, names = obstacle_at(t0)
            hits = checker.check_state(q0, boxes, names)
            if hits:
                return MotionVerdict(UNSAFE, "stationary state collides", 1,
                                     (hits[0].category, hits[0].body_a, hits[0].body_b))
            return MotionVerdict(SAFE, "stationary and clear", 1)
        tau = _EPS
    if min_interval is None:
        min_interval = tau / 32.0
    counters = {"checks": 0}

    def obs_speeds(t: float) -> list[float]:
        boxes, _ = obstacle_at(t)
        # speeds are provided by the closure's obstacle model; a zero vector
        # velocity field means static obstacles.  We recompute magnitude from
        # the box metadata attached by the caller (attribute ``speed``).
        out = []
        for box in boxes:
            out.append(float(getattr(box, "speed", 0.0)))
        return out

    def recurse(q: np.ndarray, dq_seg: np.ndarray, lo: float, hi: float) -> MotionVerdict:
        if counters["checks"] >= max_checks:
            return MotionVerdict(UNKNOWN, "check budget exhausted", counters["checks"])
        counters["checks"] += 1
        boxes, names = obstacle_at(lo)
        hits = checker.check_state(q, boxes, names)
        if hits:
            return MotionVerdict(UNSAFE, "exact state collision", counters["checks"],
                                 (hits[0].category, hits[0].body_a, hits[0].body_b))
        seg = hi - lo
        link_bounds = displacement_bound(spec, q, dq_seg)
        obs_bounds = [speed * seg for speed in obs_speeds(lo)]
        link_boxes = checker.link_obbs(q)
        chamber = spec.chamber_size

        # 1. environment: every link-obstacle pair clearance >= relative bound
        env_certified = True
        for li, lbox in enumerate(link_boxes):
            for oi, obox in enumerate(boxes):
                rel = float(link_bounds[li]) + (obs_bounds[oi] if oi < len(obs_bounds) else 0.0)
                if obb_sphere_gap(lbox, obox) < rel - _EPS:
                    env_certified = False
                    break
            if not env_certified:
                break

        # 2. boundary: every corner stays >= its link bound inside each wall
        boundary_certified = all(
            corners_clear_of_walls(link_boxes[li].corners(), chamber,
                                   float(link_bounds[li]))
            for li in range(len(link_boxes)))

        # 3. self: every non-masked pair clearance >= combined bound
        self_certified = True
        n = len(link_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in checker.self_mask:
                    continue
                rel = float(link_bounds[i] + link_bounds[j])
                if obb_sphere_gap(link_boxes[i], link_boxes[j]) < rel - _EPS:
                    self_certified = False
                    break
            if not self_certified:
                break

        if env_certified and boundary_certified and self_certified:
            return MotionVerdict(SAFE, "displacement bound certified",
                                 counters["checks"])
        if seg <= min_interval * (1 + _EPS):
            return MotionVerdict(UNKNOWN, "min interval reached uncertified",
                                 counters["checks"])
        mid = 0.5 * (lo + hi)
        q_mid = q + 0.5 * dq_seg
        first = recurse(q, 0.5 * dq_seg, lo, mid)
        if first.status != SAFE:
            return first
        return recurse(q_mid, 0.5 * dq_seg, mid, hi)

    return recurse(q0, dq, t0, t0 + tau)


class MovingObstacleField:
    """Adapter binding DynamicBox obstacles to the collision APIs.

    Exposes ``obstacle_at(t)`` (exact current centers, using the same
    piecewise-constant bounce model as v1.2) and attaches a ``speed``
    attribute to each OBB for the certifier's motion bound.  The field is
    built from *observable* center/velocity/size data or from simulator
    boxes — the certifier never sees future schedule events; future obstacle
    state is always recomputed from the constant-velocity + wall-reflection
    model implied by the current snapshot.
    """

    def __init__(self, chamber_size: Sequence[float],
                 boxes: Sequence[tuple[Sequence[float], Sequence[float],
                                       Sequence[float]]]):
        """boxes: iterable of (center, size, velocity)."""
        self.chamber_size = tuple(chamber_size)
        self._spec = tuple((np.asarray(c, float), np.asarray(s, float),
                            np.asarray(v, float)) for c, s, v in boxes)

    @classmethod
    def from_dynamic(cls, chamber_size: Sequence[float],
                     obstacles) -> "MovingObstacleField":
        """Build from ``experiments_v12.dynamic_chamber.DynamicBox`` objects."""
        t0 = 0.0
        spec = []
        for ob in obstacles:
            spec.append((ob.center, ob.size, ob.velocity))
        field = cls(chamber_size, spec)
        field._dynamic = tuple(obstacles)
        return field

    def _center_in(self, pos: np.ndarray, vel: np.ndarray, size: np.ndarray,
                   t: float) -> np.ndarray:
        out = []
        for i in range(3):
            lo = size[i] / 2
            hi = self.chamber_size[i] - size[i] / 2
            span = hi - lo
            if span < _EPS:
                out.append(lo)
                continue
            phase = (pos[i] - lo + vel[i] * t) % (2 * span)
            out.append(lo + (phase if phase <= span else 2 * span - phase))
        return np.asarray(out)

    def obstacle_at(self, t: float) -> tuple[list[OBB], list[str]]:
        boxes: list[OBB] = []
        names: list[str] = []
        dynamic = getattr(self, "_dynamic", None)
        for i, (center, size, vel) in enumerate(self._spec):
            if dynamic is not None:
                ob = dynamic[i]
                if not ob.active_at(t):
                    continue
                c = np.asarray(ob.center_at(t, self.chamber_size), dtype=float)
                v = np.asarray(ob.velocity_at(t, self.chamber_size), dtype=float)
            else:
                c = self._center_in(center, vel, size, t)
                v = vel
            box = OBB.axis_aligned(c, size)
            # attach speed for the certifier bound (speed is bounce-invariant)
            object.__setattr__(box, "speed", float(np.linalg.norm(v)))
            boxes.append(box)
            names.append(f"obs{i}")
        return boxes, names

    def snapshot_speeds(self, t: float) -> list[float]:
        boxes, _ = self.obstacle_at(t)
        return [float(getattr(b, "speed", 0.0)) for b in boxes]
