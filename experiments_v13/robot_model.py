"""Six-DOF articulated robot model for experiment series v1.3.

Right-handed world frame, z up, meters and radians.  The robot is a fixed-base
serial chain of six revolute joints described by a :class:`RobotSpec`:

* joint ``i`` has an origin translation in its parent frame (identity
  rotation at zero angle), a unit rotation axis, and position/velocity/
  acceleration limits;
* each collision body is a cuboid rigidly attached to a link frame with a
  local center and local half-extents (axis-aligned *in the link frame*, so
  world poses are generally oriented);
* a tool offset transform defines the end-effector point from the last link
  frame; a base transform anchors frame 0 in the world.

Nothing here hardcodes the default geometry: a different chain, link count,
or dimensions can be substituted through the configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

Mat4 = np.ndarray
Vec3 = tuple[float, float, float]
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def identity() -> Mat4:
    return np.eye(4, dtype=float)


def transl(x: float, y: float, z: float) -> Mat4:
    out = identity()
    out[:3, 3] = (x, y, z)
    return out


def axis_angle_matrix(axis: Sequence[float], angle: float) -> Mat4:
    """Rodrigues rotation about ``axis`` (any nonzero scale) by ``angle``."""
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = a / n
    c, s, C = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    out = identity()
    out[0, 0] = c + x * x * C
    out[0, 1] = x * y * C - z * s
    out[0, 2] = x * z * C + y * s
    out[1, 0] = y * x * C + z * s
    out[1, 1] = c + y * y * C
    out[1, 2] = y * z * C - x * s
    out[2, 0] = z * x * C - y * s
    out[2, 1] = z * y * C + x * s
    out[2, 2] = c + z * z * C
    return out


def transform_point(matrix: Mat4, point: Sequence[float]) -> Vec3:
    p = np.asarray(point, dtype=float)
    out = matrix[:3, :3] @ p + matrix[:3, 3]
    return (float(out[0]), float(out[1]), float(out[2]))


def rotation_close(a: Mat4, b: Mat4, tol: float = 1e-9) -> bool:
    return bool(np.max(np.abs(a - b)) <= tol)


@dataclass(frozen=True)
class LinkSpec:
    """One collision cuboid rigidly attached to a link frame.

    ``drive_joint`` is the 0-based joint that moves this link (``-1`` for the
    static base body).  The cuboid is axis-aligned in that link's frame.
    """

    name: str
    drive_joint: int
    local_center: Vec3
    local_half_extents: Vec3

    def validate(self) -> None:
        if not self.name:
            raise ValueError("link name cannot be empty")
        if self.drive_joint < -1:
            raise ValueError("drive_joint must be >= -1")
        if any(h <= 0 for h in self.local_half_extents):
            raise ValueError(f"{self.name}: half-extents must be positive")
        if any(not math.isfinite(v) for v in (*self.local_center, *self.local_half_extents)):
            raise ValueError(f"{self.name}: non-finite geometry")


@dataclass(frozen=True)
class JointSpec:
    """One revolute joint.

    ``origin`` is the translation of the joint frame in the parent link frame
    (identity rotation at zero angle).  ``axis`` is the rotation axis in the
    joint frame.
    """

    name: str
    origin: Vec3
    axis: Vec3
    limit_lo: float
    limit_hi: float
    velocity_limit: float
    acceleration_limit: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("joint name cannot be empty")
        if self.limit_lo >= self.limit_hi:
            raise ValueError(f"{self.name}: empty position range")
        if self.velocity_limit <= 0 or self.acceleration_limit <= 0:
            raise ValueError(f"{self.name}: velocity/acceleration limits must be positive")
        n = sum(v * v for v in self.axis)
        if abs(n - 1.0) > 1e-6 or self.axis[0] * self.axis[1] != 0.0:
            # accept any unit axis; reject clearly bad input
            if abs(math.sqrt(n) - 1.0) > 1e-6:
                raise ValueError(f"{self.name}: axis must be unit length")
        if any(not math.isfinite(v) for v in (*self.origin, *self.axis,
                                              self.limit_lo, self.limit_hi)):
            raise ValueError(f"{self.name}: non-finite parameter")

    @property
    def axis_unit(self) -> Vec3:
        n = math.sqrt(sum(v * v for v in self.axis))
        return (self.axis[0] / n, self.axis[1] / n, self.axis[2] / n)


@dataclass(frozen=True)
class RobotSpec:
    """Complete kinematic + geometric description of the manipulator."""

    name: str
    joints: tuple[JointSpec, ...]
    links: tuple[LinkSpec, ...]
    tool_offset: Vec3 = (0.0, 0.0, 0.0)
    base_position: Vec3 = (0.0, 0.0, 0.0)
    chamber_size: Vec3 = (1.0, 1.0, 1.0)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if not self.name:
            raise ValueError("robot name cannot be empty")
        n = len(self.joints)
        if n < 1:
            raise ValueError("robot needs at least one joint")
        for joint in self.joints:
            joint.validate()
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")
        if not self.links:
            raise ValueError("robot needs at least one collision link")
        link_names = [ln.name for ln in self.links]
        if len(set(link_names)) != len(link_names):
            raise ValueError("link names must be unique")
        for link in self.links:
            link.validate()
            if link.drive_joint >= n:
                raise ValueError(f"{link.name}: drive_joint out of range")
        if any(v <= 0 for v in self.chamber_size):
            raise ValueError("chamber_size must be positive")

    # ------------------------------------------------------------- dimensions
    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def joint_lo(self) -> np.ndarray:
        return np.array([j.limit_lo for j in self.joints], dtype=float)

    @property
    def joint_hi(self) -> np.ndarray:
        return np.array([j.limit_hi for j in self.joints], dtype=float)

    @property
    def velocity_limits(self) -> np.ndarray:
        return np.array([j.velocity_limit for j in self.joints], dtype=float)

    @property
    def acceleration_limits(self) -> np.ndarray:
        return np.array([j.acceleration_limit for j in self.joints], dtype=float)

    @property
    def base_transform(self) -> Mat4:
        return transl(*self.base_position)

    @property
    def tool_transform(self) -> Mat4:
        return transl(*self.tool_offset)

    def zero_configuration(self) -> np.ndarray:
        return np.zeros(self.joint_count, dtype=float)

    def random_configuration(self, rng) -> np.ndarray:
        """Uniform sample inside joint limits using a ``numpy.random.Generator``."""
        lo, hi = self.joint_lo, self.joint_hi
        return rng.uniform(lo, hi)

    def clamp_configuration(self, q: Sequence[float]) -> np.ndarray:
        out = np.asarray(q, dtype=float)
        if out.shape != (self.joint_count,):
            raise ValueError(f"expected ({self.joint_count},) joint vector")
        return np.clip(out, self.joint_lo, self.joint_hi)

    def configuration_in_limits(self, q: Sequence[float], tol: float = 1e-9) -> bool:
        arr = np.asarray(q, dtype=float)
        if arr.shape != (self.joint_count,):
            return False
        return bool(np.all(arr >= self.joint_lo - tol) and
                    np.all(arr <= self.joint_hi + tol))

    def max_reach(self) -> float:
        """Conservative upper bound: sum of downstream link-origin offsets + tool."""
        total = float(np.linalg.norm(self.tool_offset))
        # distance from each joint origin to the next, plus tool chain
        origins = [np.asarray(j.origin, dtype=float) for j in self.joints]
        for i in range(len(origins) - 1, -1, -1):
            total += float(np.linalg.norm(origins[i]))
        return total

    def link_lever_arms(self) -> np.ndarray:
        """Max distance from each joint axis to any point of its downstream bodies.

        Used by the conservative swept checker: rotating joint ``j`` by
        ``|dq_j|`` moves any downstream body point by at most
        ``|dq_j| * lever[j]``.
        """
        n = self.joint_count
        levers = np.zeros(n, dtype=float)
        for j in range(n):
            worst = 0.0
            # points of interest: every downstream link's 8 cuboid corners and
            # the tool tip, expressed in joint j's frame at zero angles is not
            # required — use world-frame zero-configuration distances from the
            # joint origin to the point, which bound rotation displacement.
            fk0 = zero_pose_frames(self)  # cheap; cached per spec in hot path by caller
            joint_origin = fk0[j + 1][:3, 3]
            for link in self.links:
                if link.drive_joint < j and link.drive_joint != -1:
                    continue
                if link.drive_joint == -1 and j >= 0:
                    # base body does not move with any joint
                    continue
                frame_index = link.drive_joint + 1 if link.drive_joint >= 0 else 0
                corners = link_corners_world(link, fk0[frame_index])
                for corner in corners:
                    worst = max(worst, float(np.linalg.norm(corner - joint_origin)))
            tool_world = transform_point(fk0[n] @ self.tool_transform, (0, 0, 0))
            worst = max(worst, float(np.linalg.norm(tool_world - joint_origin)))
            levers[j] = max(worst, 1e-6)
        return levers


def link_corners_world(link: LinkSpec, link_frame: Mat4) -> np.ndarray:
    """(8,3) world-frame corners of a link cuboid."""
    hx, hy, hz = link.local_half_extents
    signs = np.array([[sx, sy, sz] for sx in (-hx, hx)
                      for sy in (-hy, hy) for sz in (-hz, hz)], dtype=float)
    local = signs + np.asarray(link.local_center, dtype=float)
    rot = link_frame[:3, :3]
    trans = link_frame[:3, 3]
    return local @ rot.T + trans


def zero_pose_frames(spec: RobotSpec) -> list[Mat4]:
    """Link-frame transforms at the all-zero joint configuration."""
    frames = [spec.base_transform.copy()]
    for joint in spec.joints:
        frames.append(frames[-1] @ transl(*joint.origin))
    return frames


# ------------------------------------------------------------------ defaults
def default_robot() -> RobotSpec:
    """Anthropomorphic six-revolute arm inside a 1 m cubic chamber.

    Base yaw (J1, z), shoulder pitch (J2, y), elbow pitch (J3, y), wrist roll
    (J4, z), wrist pitch (J5, y), wrist roll (J6, z).  Chain reach ~0.76 m
    from the shoulder at (0.5, 0.5, 0.19); the base sits at the chamber-floor
    center.  Seven collision bodies: static pedestal plus one cuboid per
    moving link.
    """
    joints = (
        JointSpec("J1_base_yaw", (0.0, 0.0, 0.09), (0.0, 0.0, 1.0),
                  -math.pi, math.pi, 1.2, 4.0),
        JointSpec("J2_shoulder_pitch", (0.0, 0.0, 0.10), (0.0, 1.0, 0.0),
                  -2.6, 2.6, 1.2, 4.0),
        JointSpec("J3_elbow_pitch", (0.0, 0.0, 0.30), (0.0, 1.0, 0.0),
                  -2.6, 2.6, 1.5, 6.0),
        JointSpec("J4_wrist_roll", (0.0, 0.0, 0.25), (0.0, 0.0, 1.0),
                  -math.pi, math.pi, 2.0, 8.0),
        JointSpec("J5_wrist_pitch", (0.0, 0.0, 0.06), (0.0, 1.0, 0.0),
                  -2.2, 2.2, 2.0, 8.0),
        JointSpec("J6_wrist_roll2", (0.0, 0.0, 0.05), (0.0, 0.0, 1.0),
                  -math.pi, math.pi, 2.5, 10.0),
    )
    links = (
        LinkSpec("L0_pedestal", -1, (0.0, 0.0, 0.045), (0.05, 0.05, 0.045)),
        LinkSpec("L1_column", 0, (0.0, 0.0, 0.05), (0.045, 0.045, 0.05)),
        LinkSpec("L2_upper_arm", 1, (0.0, 0.0, 0.15), (0.026, 0.026, 0.15)),
        LinkSpec("L3_forearm", 2, (0.0, 0.0, 0.125), (0.023, 0.023, 0.125)),
        LinkSpec("L4_wrist1", 3, (0.0, 0.0, 0.03), (0.02, 0.02, 0.03)),
        LinkSpec("L5_wrist2", 4, (0.0, 0.0, 0.025), (0.018, 0.018, 0.025)),
        LinkSpec("L6_gripper", 5, (0.0, 0.0, 0.035), (0.016, 0.016, 0.035)),
    )
    return RobotSpec(
        name="oops6_default",
        joints=joints,
        links=links,
        tool_offset=(0.0, 0.0, 0.07),
        base_position=(0.5, 0.5, 0.02),
        chamber_size=(1.0, 1.0, 1.0),
        metadata={"family": "anthropomorphic", "dof": 6},
    )


# ------------------------------------------------------------------ serialization
def spec_to_dict(spec: RobotSpec) -> dict:
    return {
        "robot": {
            "name": spec.name,
            "joint_count": spec.joint_count,
            "joint_types": ["revolute"] * spec.joint_count,
            "joints": [
                {"name": j.name, "origin": list(j.origin),
                 "axis": list(j.axis_unit),
                 "limit_lo": j.limit_lo, "limit_hi": j.limit_hi,
                 "velocity_limit": j.velocity_limit,
                 "acceleration_limit": j.acceleration_limit}
                for j in spec.joints
            ],
            "links": [
                {"name": ln.name, "drive_joint": ln.drive_joint,
                 "local_center": list(ln.local_center),
                 "local_half_extents": list(ln.local_half_extents)}
                for ln in spec.links
            ],
            "tool_offset": list(spec.tool_offset),
            "base_position": list(spec.base_position),
            "chamber_size": list(spec.chamber_size),
            "metadata": dict(spec.metadata),
        }
    }


def spec_from_dict(payload: dict) -> RobotSpec:
    data = payload.get("robot", payload)
    joints = tuple(
        JointSpec(j["name"], tuple(j["origin"]), tuple(j["axis"]),
                  float(j["limit_lo"]), float(j["limit_hi"]),
                  float(j["velocity_limit"]), float(j["acceleration_limit"]))
        for j in data["joints"])
    links = tuple(
        LinkSpec(ln["name"], int(ln["drive_joint"]), tuple(ln["local_center"]),
                 tuple(ln["local_half_extents"]))
        for ln in data["links"])
    return RobotSpec(
        name=data["name"], joints=joints, links=links,
        tool_offset=tuple(data.get("tool_offset", (0, 0, 0))),
        base_position=tuple(data.get("base_position", (0, 0, 0))),
        chamber_size=tuple(data.get("chamber_size", (1, 1, 1))),
        metadata=dict(data.get("metadata", {})))


def save_robot_config(spec: RobotSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_to_dict(spec), indent=2) + "\n", encoding="utf8")
    return path


def load_robot_config(path: str | Path) -> RobotSpec:
    return spec_from_dict(json.loads(Path(path).read_text(encoding="utf8")))
