"""Regression tests for the v1.2 dynamic chamber (geometry and determinism)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from chamber import ChamberEnv, Cuboid, example_scene
from experiments_v12.dynamic_chamber import (DynamicBox, DynamicChamber,
                                             VelocityEvent)

VELOCITY_SCRIPT = [(0.0, 0.0, 0.0, 0.0)]


def _drive(env, commands):
    """Apply the same velocity script to an env, returning per-step records."""
    env.reset()
    out = []
    for raw in commands:
        if env.done:
            break
        speed = sum(v * v for v in raw) ** 0.5
        command = raw if speed <= env.max_speed else tuple(
            v * env.max_speed / speed for v in raw)
        result = env.step(command)
        out.append((tuple(round(v, 12) for v in result.observation.position),
                    result.collision, result.safety_intervention,
                    tuple(tuple(round(c, 12) for c in o.center)
                          for o in result.observation.obstacles)))
    return out


class TestChamberEquivalence(unittest.TestCase):
    """A dynamic chamber built from plain Cuboids must equal ChamberEnv."""

    def test_example_scenes_identical(self):
        for kind, seed in [("empty", 7), ("static", 7), ("moving", 7),
                           ("crossing", 7), ("random", 3)]:
            reference = example_scene(kind, seed=seed)
            extended = DynamicChamber(
                chamber_size=reference.chamber_size, start=reference.start,
                goal=reference.goal, obstacles=reference.obstacles,
                radius=reference.radius, dt=reference.dt,
                max_speed=reference.max_speed,
                goal_tolerance=reference.goal_tolerance,
                max_steps=reference.max_steps)
            rng = random.Random(11)
            commands = []
            for _ in range(reference.max_steps):
                commands.append(tuple(
                    rng.uniform(-reference.max_speed, reference.max_speed)
                    for _ in range(3)))
            self.assertEqual(_drive(reference, commands), _drive(extended, commands),
                             msg=f"trajectory mismatch for {kind}")

    def test_motion_safe_agreement(self):
        reference = example_scene("moving", seed=7)
        extended = DynamicChamber(
            chamber_size=reference.chamber_size, start=reference.start,
            goal=reference.goal, obstacles=reference.obstacles,
            radius=reference.radius, dt=reference.dt,
            max_speed=reference.max_speed,
            goal_tolerance=reference.goal_tolerance,
            max_steps=reference.max_steps)
        rng = random.Random(5)
        for _ in range(300):
            p0 = tuple(rng.uniform(0.05, 0.95) for _ in range(3))
            p1 = tuple(rng.uniform(0.05, 0.95) for _ in range(3))
            t0 = round(rng.uniform(0.0, 4.0), 6)
            t1 = round(t0 + rng.uniform(0.0, 1.5), 6)
            self.assertEqual(reference.is_motion_safe(p0, p1, t0, t1),
                             extended.is_motion_safe(p0, p1, t0, t1),
                             msg=f"disagreement at t=[{t0},{t1}]")


class TestScheduledMotion(unittest.TestCase):
    def test_direction_change_is_continuous(self):
        chamber = (1.0, 1.0, 1.0)
        box = DynamicBox("flip", (0.5, 0.5, 0.5), (0.2, 0.2, 0.2),
                         velocity=(0.3, 0.0, 0.0),
                         events=[VelocityEvent(1.0, (0.0, -0.25, 0.0))])
        before = box.center_at(1.0 - 1e-9, chamber)
        after = box.center_at(1.0 + 1e-9, chamber)
        for i in range(3):
            self.assertAlmostEqual(before[i], after[i], places=6)
        self.assertEqual(box.velocity_at(1.5, chamber), (0.0, -0.25, 0.0))

    def test_stop_and_start(self):
        chamber = (1.0, 1.0, 1.0)
        box = DynamicBox("stopper", (0.2, 0.5, 0.5), (0.1, 0.1, 0.1),
                         velocity=(0.4, 0.0, 0.0),
                         events=[VelocityEvent(0.5, (0.0, 0.0, 0.0)),
                                 VelocityEvent(1.5, (0.4, 0.0, 0.0))])
        frozen_a = box.center_at(0.7, chamber)
        frozen_b = box.center_at(1.4, chamber)
        for i in range(3):
            self.assertAlmostEqual(frozen_a[i], frozen_b[i], places=9)
        moving = box.center_at(2.0, chamber)
        self.assertGreater(moving[0], frozen_b[0])

    def test_appearance_and_disappearance(self):
        chamber = (1.0, 1.0, 1.0)
        box = DynamicBox("late", (0.5, 0.5, 0.5), (0.2, 0.2, 0.2),
                         velocity=(0.0, 0.2, 0.0), appear_at=2.0,
                         disappear_at=4.0)
        self.assertFalse(box.active_at(1.0))
        self.assertTrue(box.active_at(2.5))
        self.assertFalse(box.active_at(4.0))
        self.assertEqual(box.velocity_at(1.0, chamber), (0.0, 0.0, 0.0))

    def test_chamber_hides_inactive_obstacle_from_observation(self):
        env = DynamicChamber(start=(0.1, 0.5, 0.5), goal=(0.9, 0.5, 0.5),
                             obstacles=[DynamicBox(
                                 "late", (0.5, 0.5, 0.5), (0.1, 0.1, 0.1),
                                 velocity=(0.0, 0.2, 0.0), appear_at=2.0)],
                             dt=0.5, max_steps=40, radius=0.035)
        self.assertEqual(len(env.observe().obstacles), 0)
        env.step_index = 4  # t = 2.0
        self.assertEqual(len(env.observe().obstacles), 1)

    def test_replay_determinism(self):
        def build():
            rng = random.Random(42)
            boxes = []
            for i in range(3):
                size = tuple(rng.uniform(0.1, 0.2) for _ in range(3))
                center = tuple(rng.uniform(size[j] / 2 + 0.05,
                                           1 - size[j] / 2 - 0.05) for j in range(3))
                boxes.append(DynamicBox(f"b{i}", center, size,
                                        velocity=tuple(rng.uniform(-.2, .2)
                                                       for _ in range(3)),
                                        events=[VelocityEvent(
                                            1.0 + 0.3 * i,
                                            tuple(rng.uniform(-.2, .2)
                                                  for _ in range(3)))]))
            return DynamicChamber(obstacles=boxes, radius=.035, dt=.5,
                                  max_steps=30, start=(.1, .5, .5),
                                  goal=(.9, .5, .5))
        rng = random.Random(9)
        commands = [tuple(rng.uniform(-.7, .7) for _ in range(3))
                    for _ in range(30)]
        self.assertEqual(_drive(build(), commands), _drive(build(), commands))


if __name__ == "__main__":
    unittest.main()
