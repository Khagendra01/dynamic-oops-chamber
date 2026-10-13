"""Gate 1 tests: six-joint kinematics, joint dynamics, configuration roundtrip."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

import numpy as np

from experiments_v13.forward_kinematics import (ee_position, forward_pose,
                                                position_jacobian)
from experiments_v13.joint_dynamics import JointController
from experiments_v13.robot_model import (axis_angle_matrix, default_robot,
                                         identity, load_robot_config,
                                         save_robot_config, spec_from_dict,
                                         spec_to_dict, transform_point)


def _mat_close(a, b, tol=1e-9):
    return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol)


class TestTransformMath(unittest.TestCase):
    def test_axis_angle_orthonormal(self):
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)):
            for angle in (0.0, 0.3, -1.7, math.pi / 2, math.pi):
                r = axis_angle_matrix(axis, angle)[:3, :3]
                self.assertTrue(_mat_close(r @ r.T, np.eye(3), 1e-12))
                self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)

    def test_axis_angle_z_90(self):
        r = axis_angle_matrix((0, 0, 1), math.pi / 2)[:3, :3]
        moved = r @ np.array([1.0, 0.0, 0.0])
        self.assertTrue(_mat_close(moved, [0.0, 1.0, 0.0], 1e-12))

    def test_axis_angle_y_90(self):
        r = axis_angle_matrix((0, 1, 0), math.pi / 2)[:3, :3]
        moved = r @ np.array([0.0, 0.0, 1.0])
        self.assertTrue(_mat_close(moved, [1.0, 0.0, 0.0], 1e-12))


class TestForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()

    def test_zero_configuration_ee(self):
        pose = forward_pose(self.spec, np.zeros(6))
        # shoulder at base + 0.09 + 0.10 = z 0.21 world (base z=0.02)
        # chain straight up: 0.30 + 0.25 + 0.06 + 0.05 + 0.07 above shoulder
        expected = np.array([0.5, 0.5, 0.02 + 0.09 + 0.10 + 0.30 + 0.25 + 0.06 + 0.05 + 0.07])
        self.assertTrue(_mat_close(pose.ee_position, expected, 1e-12))

    def test_base_yaw_rotates_ee_around_axis(self):
        q0 = np.zeros(6)
        p0 = np.asarray(ee_position(self.spec, q0))
        shoulder = np.array([0.5, 0.5, 0.02 + 0.09 + 0.10])
        q1 = q0.copy()
        q1[0] = math.pi / 2
        p1 = np.asarray(ee_position(self.spec, q1))
        v0 = p0 - shoulder
        v1 = p1 - shoulder
        self.assertAlmostEqual(float(np.linalg.norm(v0)), float(np.linalg.norm(v1)), places=10)
        # z unchanged by base yaw
        self.assertAlmostEqual(p0[2], p1[2], places=10)
        # 90° rotation maps +x offset to +y (or -y depending on sign)
        self.assertAlmostEqual(float(v0[0]), float(v1[1]), places=9)
        self.assertAlmostEqual(float(v0[1]), -float(v1[0]), places=9)

    def test_shoulder_pitch_moves_ee_in_xz(self):
        q = np.zeros(6)
        p0 = np.asarray(ee_position(self.spec, q))
        q[1] = 0.5
        p1 = np.asarray(ee_position(self.spec, q))
        # rotation about +y: x' = x cos + z sin relative to shoulder
        self.assertGreater(abs(p1[0] - p0[0]), 0.1)
        self.assertNotAlmostEqual(p1[2], p0[2], places=3)
        self.assertAlmostEqual(p0[1], p1[1], places=10)

    def test_downstream_links_move_upstream_joint(self):
        pose0 = forward_pose(self.spec, np.zeros(6))
        q = np.zeros(6)
        q[1] = 0.7
        pose1 = forward_pose(self.spec, q)
        # proximal column (driven by J1, upstream of J2) must not move
        self.assertTrue(_mat_close(pose0.frames[1], pose1.frames[1], 1e-12))
        # every downstream frame must move
        for i in (2, 3, 4, 5, 6):
            self.assertFalse(_mat_close(pose0.frames[i], pose1.frames[i], 1e-9),
                             f"frame {i} should move when J2 rotates")

    def test_wrist_joint_moves_only_wrist_and_ee(self):
        pose0 = forward_pose(self.spec, np.zeros(6))
        q = np.zeros(6)
        q[5] = 1.0
        pose1 = forward_pose(self.spec, q)
        for i in (0, 1, 2, 3, 4, 5):
            self.assertTrue(_mat_close(pose0.frames[i], pose1.frames[i], 1e-12),
                            f"frame {i} must be rigid under J6")
        self.assertFalse(_mat_close(pose0.ee, pose1.ee, 1e-9))

    def test_rotation_matrices_orthogonal(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = default_robot().random_configuration(rng)
            pose = forward_pose(self.spec, q)
            for frame in pose.frames:
                r = frame[:3, :3]
                self.assertTrue(_mat_close(r @ r.T, np.eye(3), 1e-10))
                self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=10)

    def test_combined_rotation_composition(self):
        """FK(q_combined) equals sequential application in joint order."""
        base = np.zeros(6)
        delta = np.array([0.2, -0.4, 0.3, 0.5, -0.2, 0.1])
        # stepwise: rotate one joint at a time from zero, comparing to a
        # reference that multiplies transforms left-to-right (definition).
        spec = self.spec
        t = spec.base_transform.copy()
        for joint, angle in zip(spec.joints, delta):
            from experiments_v13.robot_model import transl
            t = t @ transl(*joint.origin) @ axis_angle_matrix(joint.axis_unit, angle)
        from experiments_v13.robot_model import transl as _t
        t = t @ _t(*spec.tool_offset)
        pose = forward_pose(spec, base + delta)
        self.assertTrue(_mat_close(pose.ee, t, 1e-12))

    def test_jacobian_finite_difference_stability(self):
        rng = np.random.default_rng(11)
        for _ in range(8):
            q = self.spec.random_configuration(rng)
            j1 = position_jacobian(self.spec, q, eps=1e-5)
            j2 = position_jacobian(self.spec, q, eps=1e-6)
            j3 = position_jacobian(self.spec, q, eps=1e-7)
            self.assertLess(float(np.max(np.abs(j1 - j2))), 1e-4)
            self.assertLess(float(np.max(np.abs(j2 - j3))), 1e-4)

    def test_jacobian_matches_small_step(self):
        rng = np.random.default_rng(13)
        q = self.spec.random_configuration(rng)
        jac = position_jacobian(self.spec, q, eps=1e-6)
        dq = np.array([1e-3, -1e-3, 5e-4, 2e-4, -3e-4, 4e-4])
        p0 = np.asarray(ee_position(self.spec, q))
        p1 = np.asarray(ee_position(self.spec, q + dq))
        # first-order Taylor remainder is O(|dq|^2 * curvature) ~ 1e-6..1e-5
        self.assertLess(float(np.linalg.norm((p1 - p0) - jac @ dq)), 1e-5)

    def test_units_are_radians(self):
        """A 2*pi joint rotation returns the EE to its start."""
        rng = np.random.default_rng(17)
        q = self.spec.random_configuration(rng)
        p0 = np.asarray(ee_position(self.spec, q))
        for joint in range(6):
            q2 = q.copy()
            q2[joint] += 2 * math.pi
            if q2[joint] > self.spec.joint_hi[joint] + 1e-9:
                q2[joint] -= 4 * math.pi
            p2 = np.asarray(ee_position(self.spec, q2))
            self.assertLess(float(np.linalg.norm(p2 - p0)), 1e-9)

    def test_joint_limits_configuration(self):
        spec = self.spec
        self.assertFalse(spec.configuration_in_limits(np.full(6, 4.0)))
        self.assertTrue(spec.configuration_in_limits(spec.clamp_configuration(np.full(6, 4.0))))
        clamped = spec.clamp_configuration(np.array([4, -4, 4, -4, 4, -4], dtype=float))
        self.assertTrue(np.all(clamped <= spec.joint_hi + 1e-12))
        self.assertTrue(np.all(clamped >= spec.joint_lo - 1e-12))

    def test_zero_config_fits_chamber(self):
        """At rest every link corner is inside the 1 m chamber.

        Fully stretched configs may reach through a wall; that is a *boundary
        collision* for the collision system to detect (Gate 2), not a model
        defect.  Here we pin that the default geometry is sane at rest and in
        a folded working posture.
        """
        from experiments_v13.robot_model import link_corners_world
        spec = self.spec
        postures = {
            "zero": np.zeros(6),
            "folded": np.array([0.4, 1.2, -2.0, 0.3, -0.8, 0.0]),
            "side": np.array([math.pi / 2, 0.6, -1.2, 0.0, 0.4, 0.0]),
        }
        for label, q in postures.items():
            pose = forward_pose(spec, q)
            for link in spec.links:
                frame = pose.frames[link.drive_joint + 1 if link.drive_joint >= 0 else 0]
                corners = link_corners_world(link, frame)
                self.assertTrue(np.all(corners >= -1e-9),
                                f"{link.name} below floor at {label}")
                self.assertTrue(np.all(corners <= np.array(spec.chamber_size) + 1e-9),
                                f"{link.name} outside chamber at {label}")


class TestConfigRoundtrip(unittest.TestCase):
    def test_json_roundtrip_identical_fk(self):
        spec = default_robot()
        payload = spec_to_dict(spec)
        clone = spec_from_dict(payload)
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = spec.random_configuration(rng)
            self.assertTrue(_mat_close(ee_position(spec, q), ee_position(clone, q), 1e-12))

    def test_file_roundtrip(self):
        import tempfile
        spec = default_robot()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_robot_config(spec, Path(tmp) / "r.json")
            loaded = load_robot_config(path)
        self.assertEqual(loaded.joint_count, 6)
        self.assertEqual(len(loaded.links), 7)
        self.assertEqual(loaded.name, spec.name)


class TestJointDynamics(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()
        self.dt = 0.1
        self.ctrl = JointController(self.spec, self.dt)

    def test_zero_command_hold(self):
        state = self.ctrl.start_state(np.zeros(6))
        step = self.ctrl.step(state, np.zeros(6))
        self.assertTrue(np.allclose(step.executed_dq, 0.0))
        self.assertFalse(step.limit_clipped)
        self.assertTrue(step.stalled)

    def test_velocity_limit_clamps_command(self):
        state = self.ctrl.start_state(np.zeros(6))
        step = self.ctrl.step(state, np.full(6, 5.0))
        self.assertTrue(step.clamped_command)
        max_step = self.spec.velocity_limits * self.dt
        self.assertTrue(np.all(np.abs(step.executed_dq) <= max_step + 1e-12))

    def test_acceleration_limit_ramps_up(self):
        state = self.ctrl.start_state(np.zeros(6))
        cmd = self.spec.velocity_limits * self.dt  # full-speed step each period
        speeds = []
        for _ in range(10):
            step = self.ctrl.step(state, cmd)
            state = step.state
            speeds.append(abs(state.qd[0]))
        # first period cannot exceed amax*dt
        self.assertLessEqual(speeds[0], self.spec.acceleration_limits[0] * self.dt + 1e-12)
        # monotone ramp to steady speed
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(speeds, speeds[1:])))
        self.assertAlmostEqual(speeds[-1], self.spec.velocity_limits[0], places=6)

    def test_position_limit_enforced(self):
        q0 = np.full(6, 0.0)
        q0[0] = self.spec.joint_hi[0] - 0.01
        state = self.ctrl.start_state(q0)
        step = self.ctrl.step(state, np.array([0.5, 0, 0, 0, 0, 0]))
        self.assertTrue(step.limit_clipped)
        self.assertLessEqual(step.state.q[0], self.spec.joint_hi[0] + 1e-12)
        # velocity zeroed against the limit
        self.assertAlmostEqual(float(step.state.qd[0]), 0.0, places=9)

    def test_deterministic(self):
        rng = np.random.default_rng(5)
        cmds = rng.normal(scale=0.05, size=(20, 6))
        run1, run2 = [], []
        for store in (run1, run2):
            state = self.ctrl.start_state(np.zeros(6))
            for cmd in cmds:
                step = self.ctrl.step(state, cmd)
                state = step.state
                store.append(state.q.copy())
        self.assertTrue(np.allclose(np.array(run1), np.array(run2)))

    def test_unit_increment_matches_max(self):
        inc = self.ctrl.unit_increment(2, +1.0)
        self.assertAlmostEqual(inc[2], self.spec.velocity_limits[2] * self.dt, places=12)
        self.assertEqual(np.count_nonzero(inc), 1)


if __name__ == "__main__":
    unittest.main()
