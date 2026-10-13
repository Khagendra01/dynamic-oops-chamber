"""Forward kinematics and analytic-structure Jacobians for the v1.3 arm.

``forward_pose(spec, q)`` returns every link-frame world transform and the
end-effector pose.  Rotating a proximal joint updates every downstream frame;
this is the ground-truth kinematic engine used by the simulator, collision
system, baselines and VM primitives.  Pure NumPy, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from experiments_v13.robot_model import (Mat4, RobotSpec, Vec3, axis_angle_matrix,
                                         identity, transform_point, transl)


@dataclass(frozen=True)
class RobotPose:
    """FK result: ``frames[i]`` is the world transform of link frame ``i``
    (after joint ``i``'s motion); ``frames[0]`` equals the base transform.
    ``ee`` is the tool (end-effector point) transform; ``ee_position`` its
    translation."""

    frames: tuple[Mat4, ...]
    ee: Mat4

    @property
    def ee_position(self) -> Vec3:
        return (float(self.ee[0, 3]), float(self.ee[1, 3]), float(self.ee[2, 3]))

    @property
    def ee_rotation(self) -> Mat4:
        return self.ee[:3, :3]


def forward_pose(spec: RobotSpec, q: Sequence[float]) -> RobotPose:
    """Compute link frames and EE pose for joint configuration ``q``.

    Frame chain: ``T_0 = base``; ``T_i = T_{i-1} @ transl(origin_i) @ Rot(axis_i, q_i)``.
    EE: ``T_ee = T_n @ transl(tool_offset)``.
    """
    arr = np.asarray(q, dtype=float)
    if arr.shape != (spec.joint_count,):
        raise ValueError(f"expected joint vector of shape ({spec.joint_count},)")
    frames: list[Mat4] = [spec.base_transform.copy()]
    for joint, angle in zip(spec.joints, arr):
        origin = frames[-1] @ transl(*joint.origin)
        frames.append(origin @ axis_angle_matrix(joint.axis_unit, float(angle)))
    ee = frames[-1] @ spec.tool_transform
    return RobotPose(tuple(frames), ee)


def ee_position(spec: RobotSpec, q: Sequence[float]) -> Vec3:
    return forward_pose(spec, q).ee_position


def position_jacobian(spec: RobotSpec, q: Sequence[float],
                      eps: float = 1e-6) -> np.ndarray:
    """(3, n) geometric position Jacobian by central finite differences.

    Used by the IK baseline and reactively; the FK engine itself is exact
    closed-form.  Step size ``eps`` is validated against smaller/larger
    steps in the kinematics tests.
    """
    arr = np.asarray(q, dtype=float)
    jac = np.zeros((3, spec.joint_count), dtype=float)
    for i in range(spec.joint_count):
        step = np.zeros_like(arr)
        step[i] = eps
        plus = np.asarray(ee_position(spec, arr + step))
        minus = np.asarray(ee_position(spec, arr - step))
        jac[:, i] = (plus - minus) / (2.0 * eps)
    return jac


def link_world_corners(spec: RobotSpec, pose: RobotPose) -> dict[str, np.ndarray]:
    """Map link name -> (8,3) world corners for the given pose."""
    from experiments_v13.robot_model import link_corners_world
    out: dict[str, np.ndarray] = {}
    for link in spec.links:
        frame_index = link.drive_joint if link.drive_joint >= 0 else 0
        out[link.name] = link_corners_world(link, pose.frames[frame_index])
    return out


def link_world_centers(spec: RobotSpec, pose: RobotPose) -> dict[str, Vec3]:
    out: dict[str, Vec3] = {}
    for link in spec.links:
        frame_index = link.drive_joint if link.drive_joint >= 0 else 0
        center = transform_point(pose.frames[frame_index], link.local_center)
        out[link.name] = center
    return out
