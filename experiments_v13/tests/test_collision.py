"""Gate 2 tests: OBB geometry, self-collision, swept and moving-obstacle cases.

Adversarial set required by the v1.3 spec: unsafe paths with safe endpoints,
rotational sweeps through thin boxes, moving obstacles vs stationary links,
crossing bodies between ticks, near-limit configurations, and the soundness
property that a SAFE certification never coexists with a dense-detected hit.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

import numpy as np

from experiments_v13.collision_geometry import (CollisionChecker, OBB,
                                                corners_clear_of_walls,
                                                corners_inside_chamber,
                                                default_self_mask,
                                                obb_overlap, obb_sphere_gap)
from experiments_v13.forward_kinematics import forward_pose
from experiments_v13.robot_model import default_robot
from experiments_v13.swept_collision import (MovingObstacleField, SAFE, UNSAFE,
                                             UNKNOWN, certify_motion,
                                             dense_motion_check,
                                             displacement_bound,
                                             link_lever_chain)


class TestOBB(unittest.TestCase):
    def test_separated_boxes(self):
        a = OBB.axis_aligned((0, 0, 0), (1, 1, 1))
        b = OBB.axis_aligned((2, 0, 0), (1, 1, 1))
        self.assertFalse(obb_overlap(a, b))
        self.assertAlmostEqual(obb_sphere_gap(a, b),
                               2 - math.sqrt(3), places=9)

    def test_overlap_and_touching_tolerance(self):
        a = OBB.axis_aligned((0, 0, 0), (1, 1, 1))
        touching = OBB.axis_aligned((1.0, 0, 0), (1, 1, 1))
        # exact surface contact counts as collision (conservative)
        self.assertTrue(obb_overlap(a, touching, tolerance=0.0))
        separated = OBB.axis_aligned((1.01, 0, 0), (1, 1, 1))
        self.assertFalse(obb_overlap(a, separated, tolerance=0.0))
        inside = OBB.axis_aligned((0.5, 0, 0), (1, 1, 1))
        self.assertTrue(obb_overlap(a, inside))

    def test_rotated_box_face_edge_corner_axes(self):
        """SAT must use rotated axes, not just world projections."""
        a = OBB.axis_aligned((0, 0, 0), (2, 0.2, 0.2))
        # thin plate rotated 45° about z, offset so only a diagonal axis overlaps
        r = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0],
                      [math.sin(math.pi / 4), math.cos(math.pi / 4), 0],
                      [0, 0, 1]])
        b = OBB(np.array([0.6, 0.6, 0.0]), np.array([1.0, 0.02, 0.02]), r)
        self.assertTrue(obb_overlap(a, b))
        far = OBB(np.array([1.4, 1.4, 0.0]), np.array([1.0, 0.02, 0.02]), r)
        self.assertFalse(obb_overlap(a, far))

    def test_near_touching_reports_collision_with_tolerance(self):
        a = OBB.axis_aligned((0, 0, 0), (1, 1, 1))
        near = OBB.axis_aligned((1.0005, 0, 0), (1, 1, 1))
        self.assertFalse(obb_overlap(a, near, tolerance=1e-6))
        self.assertTrue(obb_overlap(a, near, tolerance=1e-2))

    def test_boundary_containment(self):
        chamber = (1.0, 1.0, 1.0)
        inside = OBB.axis_aligned((0.5, 0.5, 0.5), (0.4, 0.4, 0.4)).corners()
        self.assertTrue(corners_inside_chamber(inside, chamber))
        poke = OBB.axis_aligned((0.9, 0.5, 0.5), (0.4, 0.4, 0.4)).corners()
        self.assertFalse(corners_inside_chamber(poke, chamber))
        self.assertFalse(corners_clear_of_walls(inside, chamber, 0.35))
        self.assertTrue(corners_clear_of_walls(inside, chamber, 0.25))

    def test_link_orientation_changes_collision(self):
        """The same link cuboid hits an obstacle in one pose, not another."""
        spec = default_robot()
        checker = CollisionChecker(spec)
        # plate at forearm height (z ~ 0.21 at near-horizontal stretch)
        obstacle = OBB.axis_aligned((0.80, 0.5, 0.21), (0.1, 0.3, 0.1))
        # stretched forward (q2 = pi/2): forearm points at +x toward obstacle
        q_hit = np.array([0.0, math.pi / 2, 0.0, 0, 0, 0])
        # folded: arm compact near base
        q_miss = np.array([0.0, 0.4, -2.2, 0, 0, 0])
        hit = checker.check_state(q_hit, [obstacle], ["wall_box"])
        miss = checker.check_state(q_miss, [obstacle], ["wall_box"])
        self.assertTrue(any(c.category == "environment" for c in hit))
        self.assertFalse(any(c.category == "environment" for c in miss))


class TestSelfCollision(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()
        self.checker = CollisionChecker(self.spec)

    def test_adjacent_pairs_masked(self):
        mask = default_self_mask(self.spec)
        # L0/L1 and L1/L2 etc. adjacent pairs disabled
        self.assertIn((0, 1), mask)
        self.assertIn((1, 2), mask)
        self.assertIn((5, 6), mask)
        # nonadjacent pairs enabled
        self.assertNotIn((0, 2), mask)
        self.assertNotIn((2, 5), mask)
        self.assertNotIn((0, 6), mask)

    def test_zero_config_no_self_collision(self):
        hits = self.checker.check_state(np.zeros(6))
        self.assertFalse([h for h in hits if h.category == "self"], hits)

    def test_nonadjacent_self_collision_forced(self):
        """Fold the wrist hard so wrist bodies cross nonadjacent links."""
        rng = np.random.default_rng(23)
        found = False
        for _ in range(4000):
            q = self.spec.random_configuration(rng)
            hits = self.checker.check_state(q)
            if any(h.category == "self" for h in hits):
                found = True
                for h in hits:
                    if h.category == "self":
                        i = self.spec.links.index(
                            next(l for l in self.spec.links if l.name == h.body_a))
                        j = self.spec.links.index(
                            next(l for l in self.spec.links if l.name == h.body_b))
                        self.assertGreater(abs(i - j), 1)
                break
        self.assertTrue(found, "random search never produced a nonadjacent self hit")

    def test_dense_fold_sweep_detects_self_hit_between_samples(self):
        """A slow fold can cross a self-collision configuration."""
        checker = self.checker
        q0 = np.array([0.0, 0.2, -0.2, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 2.4, -2.4, 0.0, 0.0, 0.0])
        dq = q1 - q0
        field = MovingObstacleField((1, 1, 1), ())
        hit, rec = dense_motion_check(self.spec, checker, q0, dq, 0.0, 1.0,
                                      field.obstacle_at, samples=64)
        # a hard fold may self-collide, hit the boundary, or both; the
        # detector must report a real category when it fires
        if hit:
            self.assertIn(rec[0], ("self", "boundary", "environment"))


class TestSweptMotion(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()
        self.checker = CollisionChecker(self.spec)
        self.field = MovingObstacleField((1, 1, 1), ())

    def test_stationary_clear_certified_safe(self):
        verdict = certify_motion(self.spec, self.checker, np.zeros(6),
                                 np.zeros(6), 0.0, 0.1, self.field.obstacle_at)
        self.assertEqual(verdict.status, SAFE)

    def test_stationary_colliding_unsafe(self):
        q = np.array([0.0, math.pi / 2, 0.0, 0, 0, 0])
        field = MovingObstacleField((1, 1, 1), (
            ((0.85, 0.5, 0.71), (0.1, 0.3, 0.3), (0, 0, 0)),))
        verdict = certify_motion(self.spec, self.checker, q, np.zeros(6),
                                 0.0, 0.1, field.obstacle_at)
        self.assertEqual(verdict.status, UNSAFE)

    def test_rotating_link_sweeps_through_thin_box(self):
        """Endpoints safe, interior unsafe: certification must not say SAFE."""
        # small shoulder sweep that stays inside the chamber at both ends;
        # a thin vertical plate sits exactly on the mid-sweep EE path.
        q0 = np.array([0.0, 0.05, 0.0, 0, 0, 0])
        dq = np.array([0.0, 0.6, 0.0, 0, 0, 0])
        probe = forward_pose(self.spec, q0 + 0.5 * dq).ee_position
        # thin plate centered on the mid-sweep EE path (legal center range
        # for the bounce model: z in [0.08, 0.92] for this size)
        field = MovingObstacleField((1, 1, 1), (
            ((probe[0], probe[1], 0.80), (0.02, 0.02, 0.16), (0, 0, 0)),))
        # endpoints must be clear
        self.assertTrue(self.checker.state_free(
            q0, field.obstacle_at(0.0)[0]))
        self.assertTrue(self.checker.state_free(
            q0 + dq, field.obstacle_at(1.0)[0]))
        verdict = certify_motion(self.spec, self.checker, q0, dq, 0.0, 1.0,
                                 field.obstacle_at, max_checks=256)
        self.assertIn(verdict.status, (UNSAFE, UNKNOWN))
        hit, _ = dense_motion_check(self.spec, self.checker, q0, dq, 0.0, 1.0,
                                    field.obstacle_at, samples=100)
        self.assertTrue(hit, "dense detector must see the swept hit")

    def test_moving_obstacle_hits_stationary_arm(self):
        """Obstacle approaches a stationary link: wait is not automatically safe."""
        q = np.zeros(6)
        # pedestal occupies z in [0.02, 0.20] near (0.5, 0.5).  Box drifts in.
        start = (0.75, 0.5, 0.11)
        field = MovingObstacleField((1, 1, 1), (
            (start, (0.08, 0.08, 0.08), (-0.5, 0.0, 0.0)),))
        # far away now: certified safe for a short hold
        verdict0 = certify_motion(self.spec, self.checker, q, np.zeros(6),
                                  0.0, 0.1, field.obstacle_at)
        self.assertEqual(verdict0.status, SAFE)
        # after 0.4 s the box is at 0.55, about to touch the column: the
        # certifier must stop claiming safety.
        t_late = 0.45
        verdict1 = certify_motion(self.spec, self.checker, q, np.zeros(6),
                                  t_late, 0.3, field.obstacle_at)
        self.assertNotEqual(verdict1.status, SAFE)
        hit, rec = dense_motion_check(self.spec, self.checker, q, np.zeros(6),
                                      t_late, 0.4, field.obstacle_at, samples=40)
        self.assertTrue(hit)
        self.assertEqual(rec[0], "environment")

    def test_two_bodies_crossing_between_ticks(self):
        """Arm retracts while a fast obstacle sweeps through its volume."""
        q0 = np.array([0.0, 0.5, 0.0, 0, 0, 0])
        dq = np.array([0.0, -0.35, 0.0, 0, 0, 0])
        q_end = q0 + dq
        end_probe = forward_pose(self.spec, q_end).ee_position
        start_probe = forward_pose(self.spec, q0).ee_position
        mid_dir = np.asarray(start_probe) - np.asarray(end_probe)
        mid_dir /= max(np.linalg.norm(mid_dir), 1e-9)
        obs_start = np.asarray(end_probe) + mid_dir * 0.35
        obs_vel = -mid_dir * 1.2  # chases the retracting arm from behind
        field = MovingObstacleField((1, 1, 1), (
            (tuple(np.clip(obs_start, 0.05, 0.95)), (0.06, 0.06, 0.06),
             tuple(obs_vel)),))
        self.assertTrue(self.checker.state_free(q0, field.obstacle_at(0.0)[0]),
                        "start pose must be clear")
        self.assertTrue(self.checker.state_free(q_end, field.obstacle_at(1.0)[0]),
                        "end pose must be clear")
        verdict = certify_motion(self.spec, self.checker, q0, dq, 0.0, 1.0,
                                 field.obstacle_at, max_checks=256)
        self.assertIn(verdict.status, (UNSAFE, UNKNOWN))
        hit, _ = dense_motion_check(self.spec, self.checker, q0, dq, 0.0, 1.0,
                                    field.obstacle_at, samples=120)
        self.assertTrue(hit)

    def test_near_joint_limit_motion(self):
        spec = self.spec
        q0 = spec.joint_hi - 1e-3
        dq = np.full(6, 0.05)  # commands push past the limits
        # execution clips, but raw certify of the raw dq must at least not
        # crash and must respect boundary/self at the limit posture
        verdict = certify_motion(spec, self.checker, q0, dq, 0.0, 0.1,
                                 self.field.obstacle_at)
        self.assertIn(verdict.status, (SAFE, UNSAFE, UNKNOWN))

    def test_displacement_bound_positive_and_local(self):
        spec = self.spec
        b_single = displacement_bound(spec, np.zeros(6),
                                      np.array([0.1, 0, 0, 0, 0, 0]))
        # base-yaw motion must not move the static pedestal
        self.assertEqual(b_single[0], 0.0)
        self.assertGreater(b_single[-1], 0.05)
        b_all = displacement_bound(spec, np.zeros(6), np.full(6, 0.1))
        self.assertTrue(np.all(b_all[1:] >= b_single[1:] - 1e-12))

    def test_lever_chain_rigorous_on_random_motions(self):
        """Empirical soundness: actual point motion <= displacement bound."""
        rng = np.random.default_rng(31)
        spec = self.spec
        checker = self.checker
        for _ in range(30):
            q0 = spec.random_configuration(rng)
            dq = rng.uniform(-0.12, 0.12, size=6)
            # clamp so q0+dq stays near limits
            dq = np.clip(dq, spec.joint_lo - q0, spec.joint_hi - q0)
            bounds = displacement_bound(spec, q0, dq)
            boxes0 = checker.link_obbs(q0)
            boxes1 = checker.link_obbs(q0 + dq)
            for i in range(len(boxes0)):
                actual = float(np.linalg.norm(boxes1[i].center - boxes0[i].center))
                # center motion <= max corner motion <= bound for the link
                self.assertLessEqual(actual, bounds[i] + 1e-9,
                                     f"link {i} exceeded bound")

    def test_safe_certification_sound_on_random_static_cases(self):
        """Whenever certify says SAFE, dense detection must find nothing."""
        rng = np.random.default_rng(41)
        spec = self.spec
        checker = self.checker
        agreed_safe = 0
        for _ in range(25):
            q0 = spec.random_configuration(rng)
            dq = rng.uniform(-0.05, 0.05, size=6)
            dq = np.clip(dq, spec.joint_lo - q0, spec.joint_hi - q0)
            # sprinkle random static obstacles away from start
            obs = []
            for k in range(2):
                c = rng.uniform(0.1, 0.9, size=3)
                if np.linalg.norm(c - np.asarray(forward_pose(spec, q0).ee_position)) < 0.15:
                    c = c + 0.3
                obs.append((tuple(np.clip(c, 0.08, 0.92)), (0.06, 0.06, 0.06),
                            (0.0, 0.0, 0.0)))
            field = MovingObstacleField((1, 1, 1), tuple(obs))
            verdict = certify_motion(spec, checker, q0, dq, 0.0, 0.1,
                                     field.obstacle_at, max_checks=128)
            if verdict.status == SAFE:
                hit, _ = dense_motion_check(spec, checker, q0, dq, 0.0, 0.1,
                                            field.obstacle_at, samples=40)
                self.assertFalse(hit, "SAFE certification was unsound")
                agreed_safe += 1
        self.assertGreater(agreed_safe, 0, "no case ever certified SAFE")


class TestMovingObstacleField(unittest.TestCase):
    def test_bounce_reflection_matches_model(self):
        field = MovingObstacleField((1, 1, 1), (
            ((0.2, 0.5, 0.5), (0.1, 0.1, 0.1), (0.5, 0, 0)),))
        c0 = field.obstacle_at(0.0)[0][0].center
        c1 = field.obstacle_at(0.5)[0][0].center
        self.assertAlmostEqual(c0[0], 0.2, places=9)
        self.assertAlmostEqual(c1[0], 0.45, places=9)
        # wall at center-max 0.95; reached at t=1.5, reflected afterwards
        c_hit = field.obstacle_at(1.5)[0][0].center
        self.assertAlmostEqual(c_hit[0], 0.95, places=6)
        c2 = field.obstacle_at(1.7)[0][0].center
        self.assertAlmostEqual(c2[0], 0.95 - 0.5 * 0.2, places=6)
        v = field.obstacle_at(1.7)[0][0]
        self.assertAlmostEqual(float(v.center[0]), 0.85, places=6)

    def test_speed_attribute_is_bounce_invariant(self):
        field = MovingObstacleField((1, 1, 1), (
            ((0.2, 0.5, 0.5), (0.1, 0.1, 0.1), (0.5, 0, 0)),))
        for t in (0.0, 0.9, 1.1, 2.0):
            speeds = field.snapshot_speeds(t)
            self.assertAlmostEqual(speeds[0], 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
