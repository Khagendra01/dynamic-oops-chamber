"""Oriented bounding box collision geometry for the v1.3 arm.

Exact separating-axis-theorem (SAT) overlap tests for general OBBs (rotated
link cuboids vs oriented/axis-aligned obstacles), chamber-boundary containment,
self-collision pair masking, and conservative clearance lower bounds used by
the swept-motion certifier.

Tolerance convention: ``tolerance`` is a *safety inflation*.  Two boxes whose
surfaces are closer than ``tolerance`` count as colliding (near-touching is
conservative).  Implemented by shrinking each box by ``tolerance / 2`` before
the SAT test.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from experiments_v13.forward_kinematics import forward_pose
from experiments_v13.robot_model import RobotSpec, Vec3

_EPS = 1e-12
# rotation columns of the two boxes + 9 cross products
_TEST_AXES = (
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
)


@dataclass(frozen=True)
class OBB:
    """Oriented box: world center, per-axis half-extents, rotation columns."""

    center: np.ndarray          # (3,)
    half: np.ndarray            # (3,) positive
    rotation: np.ndarray        # (3,3) columns = local axes in world

    @classmethod
    def axis_aligned(cls, center: Sequence[float], size: Sequence[float]) -> "OBB":
        return cls(np.asarray(center, dtype=float),
                   np.asarray(size, dtype=float) / 2.0,
                   np.eye(3, dtype=float))

    def corners(self) -> np.ndarray:
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=float)
        local = signs * self.half
        return local @ self.rotation.T + self.center

    def circumsphere_radius(self) -> float:
        return float(np.linalg.norm(self.half))

    def grown(self, amount: float) -> "OBB":
        return OBB(self.center, self.half + amount, self.rotation)


def _project(half: np.ndarray, rotation: np.ndarray, axis: np.ndarray) -> float:
    return float(np.sum(half * np.abs(rotation.T @ axis)))


def obb_overlap(a: OBB, b: OBB, tolerance: float = 0.0) -> bool:
    """True when the boxes overlap allowing for surface ``tolerance``.

    Grows each box by ``tolerance / 2`` so surfaces closer than ``tolerance``
    (including exact contact at tolerance 0) report as overlapping —
    conservative for safety.
    """
    if tolerance > 0.0:
        a = a.grown(tolerance / 2.0)
        b = b.grown(tolerance / 2.0)
    t = b.center - a.center
    for i in range(3):
        ra = float(a.half[i])
        rb = _project(b.half, b.rotation, a.rotation[:, i])
        if abs(float(t @ a.rotation[:, i])) > ra + rb + _EPS:
            return False
    for i in range(3):
        ra = _project(a.half, a.rotation, b.rotation[:, i])
        rb = float(b.half[i])
        if abs(float(t @ b.rotation[:, i])) > ra + rb + _EPS:
            return False
    ra_rot, rb_rot = a.rotation, b.rotation
    for i, j in _TEST_AXES:
        axis = np.cross(ra_rot[:, i], rb_rot[:, j])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-10:
            continue  # parallel axes: no new separating direction
        axis /= norm
        ra = _project(a.half, a.rotation, axis)
        rb = _project(b.half, b.rotation, axis)
        if abs(float(t @ axis)) > ra + rb + _EPS:
            return False
    return True


def obb_sphere_gap(a: OBB, b: OBB) -> float:
    """Lower bound on surface distance via circumspheres.

    ``|c_a - c_b| - r_a - r_b <= true_surface_distance`` for any convex
    bodies, so a positive gap certifies separation (conservative).
    """
    return float(np.linalg.norm(a.center - b.center)
                 - a.circumsphere_radius() - b.circumsphere_radius())


def corners_inside_chamber(corners: np.ndarray, chamber: Sequence[float],
                           tolerance: float = 0.0) -> bool:
    """Exact containment test of a convex body in an AABB chamber.

    For a convex box vs axis-aligned chamber, the body is contained iff all
    corners are inside (separating axis must be a chamber face normal).
    """
    lo = -tolerance
    hi = np.asarray(chamber, dtype=float) + tolerance
    return bool(np.all(corners >= lo - _EPS) and np.all(corners <= hi + _EPS))


def chamber_margin(corners: np.ndarray, chamber: Sequence[float]) -> float:
    """Signed min distance from corners to the nearest wall (negative = out)."""
    hi = np.asarray(chamber, dtype=float)
    return float(min(np.min(corners), np.min(hi - corners)))


def corners_clear_of_walls(corners: np.ndarray, chamber: Sequence[float],
                           margin: float) -> bool:
    """True when every corner is at least ``margin`` inside every wall."""
    hi = np.asarray(chamber, dtype=float)
    return bool(np.all(corners >= margin - _EPS) and
                np.all(corners <= hi - margin + _EPS))


@dataclass(frozen=True)
class Collision:
    category: str          # "environment" | "self" | "boundary"
    body_a: str
    body_b: str


def default_self_mask(spec: RobotSpec) -> frozenset[tuple[int, int]]:
    """Disabled self-collision pairs: adjacent bodies sharing a joint.

    Indices refer to positions in ``spec.links``.  Bodies whose drive joints
    differ by at most one (the static base ``drive_joint=-1`` counts as its
    own index, so it is adjacent only to the column driven by joint 0) share
    modeled attachment geometry and are excluded; *all* other pairs remain
    enabled, including base-vs-upper-arm and nonadjacent wrist links.
    """
    disabled: set[tuple[int, int]] = set()
    order = list(spec.links)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j <= i:
                continue
            if abs(a.drive_joint - b.drive_joint) <= 1:
                disabled.add((i, j))
    return frozenset(disabled)


class CollisionChecker:
    """Stateless exact collision queries for one robot specification."""

    def __init__(self, spec: RobotSpec, *,
                 tolerance: float = 1e-6,
                 self_mask: Iterable[tuple[int, int]] | None = None):
        self.spec = spec
        self.tolerance = float(tolerance)
        mask = default_self_mask(spec) if self_mask is None else frozenset(self_mask)
        self.self_mask = mask
        self._link_index = {ln.name: i for i, ln in enumerate(spec.links)}

    # ------------------------------------------------------------------ helpers
    def link_obbs(self, q: Sequence[float]) -> list[OBB]:
        pose = forward_pose(self.spec, q)
        out: list[OBB] = []
        for link in self.spec.links:
            # frames[0] is the base; joint j's link frame is frames[j + 1]
            frame = pose.frames[link.drive_joint + 1 if link.drive_joint >= 0 else 0]
            center = frame[:3, :3] @ np.asarray(link.local_center) + frame[:3, 3]
            out.append(OBB(center, np.asarray(link.local_half_extents, dtype=float),
                           frame[:3, :3].copy()))
        return out

    def ee_obbs(self, q: Sequence[float]) -> list[OBB]:
        """End-link/tool geometry: the last driven link (gripper) box."""
        return self.link_obbs(q)[-1:]

    # ------------------------------------------------------------------ queries
    def check_state(self, q: Sequence[float],
                    obstacles: Sequence[OBB] = (),
                    obstacle_names: Sequence[str] | None = None,
                    chamber: Sequence[float] | None = None,
                    ) -> list[Collision]:
        """All collisions at one configuration (exact SAT + boundary)."""
        boxes = self.link_obbs(q)
        chamber = self.spec.chamber_size if chamber is None else chamber
        names = obstacle_names or [f"obs{i}" for i in range(len(obstacles))]
        hits: list[Collision] = []
        for i, link in enumerate(self.spec.links):
            corners = boxes[i].corners()
            if not corners_inside_chamber(corners, chamber, 0.0):
                hits.append(Collision("boundary", link.name, "chamber"))
            for k, obs in enumerate(obstacles):
                if obb_overlap(boxes[i], obs, self.tolerance):
                    hits.append(Collision("environment", link.name, names[k]))
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self.self_mask:
                    continue
                if obb_overlap(boxes[i], boxes[j], self.tolerance):
                    hits.append(Collision("self", self.spec.links[i].name,
                                          self.spec.links[j].name))
        return hits

    def state_free(self, q: Sequence[float], obstacles: Sequence[OBB] = (),
                   chamber: Sequence[float] | None = None) -> bool:
        return not self.check_state(q, obstacles, chamber=chamber)

    def min_obstacle_gap(self, q: Sequence[float],
                         obstacles: Sequence[OBB]) -> float:
        """Conservative lower bound on min link-to-obstacle surface distance."""
        boxes = self.link_obbs(q)
        best = math.inf
        for box in boxes:
            for obs in obstacles:
                best = min(best, obb_sphere_gap(box, obs))
        return best

    def min_self_gap(self, q: Sequence[float]) -> float:
        boxes = self.link_obbs(q)
        best = math.inf
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self.self_mask:
                    continue
                best = min(best, obb_sphere_gap(boxes[i], boxes[j]))
        return best

    def min_boundary_margin(self, q: Sequence[float],
                            chamber: Sequence[float] | None = None) -> float:
        chamber = self.spec.chamber_size if chamber is None else chamber
        best = math.inf
        for box in self.link_obbs(q):
            best = min(best, chamber_margin(box.corners(), chamber))
        return best
