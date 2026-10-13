"""Tests for the causal observation model used in v1.2."""
from __future__ import annotations

import math
import unittest

from chamber import norm, sub
from experiments_v12.dynamic_chamber import (DynamicBox, DynamicChamber,
                                             VelocityEvent)
from experiments_v12.dynamic_observations import (CausalObserver,
                                                  ObservationConfig)


def noisy_env():
    return DynamicChamber(
        chamber_size=(1.0, 1.0, 1.0), start=(0.1, 0.5, 0.2),
        goal=(0.1, 0.5, 0.9),
        obstacles=[DynamicBox("mover", (0.5, 0.85, 0.5), (0.1, 0.1, 0.1),
                              velocity=(0.1, 0.0, 0.0))],
        radius=0.035, dt=0.5, max_speed=0.7, goal_tolerance=0.055,
        max_steps=45)


def plain_env():
    return DynamicChamber(
        chamber_size=(1.0, 1.0, 1.0), start=(0.1, 0.5, 0.5),
        goal=(0.9, 0.5, 0.5), obstacles=[], radius=0.035, dt=0.5,
        max_speed=0.7, goal_tolerance=0.055, max_steps=45)


def gate_env():
    return DynamicChamber(
        chamber_size=(1.0, 1.0, 1.0), start=(0.1, 0.5, 0.5),
        goal=(0.9, 0.5, 0.5),
        obstacles=[DynamicBox("gate", (0.5, 0.5, 0.5), (0.08, 0.9, 0.9),
                              velocity=(0.45, 0.0, 0.0),
                              events=[VelocityEvent(1.5, (-0.45, 0.0, 0.0))])],
        radius=0.035, dt=0.5, max_speed=0.7, goal_tolerance=0.055,
        max_steps=45)


class PerfectObservationTests(unittest.TestCase):
    def test_matches_env_over_episode(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(), episode_seed=1)
        env.reset()
        for _ in range(12):
            served = observer.observe(env)
            truth = env.observe()
            self.assertEqual(served.step_index, truth.step_index)
            self.assertEqual(served.position, truth.position)
            self.assertEqual(served.obstacles, truth.obstacles)
            if env.done:
                break
            env.step((0.1, 0.0, 0.0))


class DelayTests(unittest.TestCase):
    def test_reports_older_state_and_age(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(delay_steps=2),
                                  episode_seed=1)
        env.reset()
        env.step((0.1, 0.0, 0.0))
        env.step((0.1, 0.0, 0.0))
        served = observer.observe(env)
        self.assertEqual(served.step_index, env.step_index - 2)
        self.assertAlmostEqual(served.time_s, env.time_s - 2 * env.dt)
        self.assertAlmostEqual(served.position[0], env.start[0] + 0.1)

    def test_v121_proprioception_current_exteroception_delayed(self):
        """v1.2.1 contract: pose is current, obstacles are last-seen."""
        env = gate_env()
        observer = CausalObserver(ObservationConfig(delay_steps=2),
                                  episode_seed=1)
        env.reset()
        env.step((0.1, 0.0, 0.0))
        env.step((0.1, 0.0, 0.0))
        step_before = env.step_index
        served = observer.observe(env)
        # The observer must not mutate simulator state (v1.2 hack removed).
        self.assertEqual(env.step_index, step_before)
        # Proprioception is current.
        self.assertEqual(served.position, env.position)
        self.assertEqual(served.goal, env.goal)
        # Exteroception is delayed: obstacles match truth at the old time,
        # which differs from current truth once the gate has moved.
        saved = env.step_index
        env.step_index = served.step_index
        past = env.observe()
        env.step_index = saved
        self.assertEqual(served.obstacles, past.obstacles)
        current = env.observe()
        self.assertNotEqual(served.obstacles, current.obstacles)

    def test_clamped_at_episode_start(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(delay_steps=5))
        env.reset()
        served = observer.observe(env)
        self.assertEqual(served.step_index, 0)

    def test_no_future_velocity_leak(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(delay_steps=2))
        env.reset()
        # Advance past the 1.5 s velocity reversal (step 4 == t=2.0 s).
        for _ in range(4):
            env.step((0.0, 0.0, 0.0))
        served = observer.observe(env)
        truth = env.observe()
        self.assertLess(truth.obstacles[0].velocity[0], 0.0)   # post-event
        self.assertGreater(served.obstacles[0].velocity[0], 0.0)  # delayed
        self.assertEqual(served.step_index, env.step_index - 2)
        # The delayed observation must equal ground truth at the delayed time.
        saved = env.step_index
        env.step_index = served.step_index
        past = env.observe()
        env.step_index = saved
        self.assertEqual(served.obstacles, past.obstacles)


class ServeRateTests(unittest.TestCase):
    def test_stale_between_refreshes(self):
        env = plain_env()
        observer = CausalObserver(ObservationConfig(serve_every=3))
        env.reset()
        seen = []
        for _ in range(7):
            served = observer.observe(env)
            seen.append(served.step_index)
            env.step((0.05, 0.0, 0.0))
        self.assertEqual(seen, [0, 0, 0, 3, 3, 3, 6])


class NoiseTests(unittest.TestCase):
    def test_deterministic_per_seed(self):
        def sequence(seed):
            env = noisy_env()
            observer = CausalObserver(
                ObservationConfig(position_noise=0.01, velocity_noise=0.02),
                episode_seed=seed)
            env.reset()
            out = []
            for _ in range(6):
                served = observer.observe(env)
                out.append((served.position, served.obstacles[0].velocity))
                env.step((0.0, 0.0, 0.05))
            return out

        self.assertEqual(sequence(7), sequence(7))
        self.assertNotEqual(sequence(7), sequence(8))

    def test_stays_inside_chamber(self):
        env = plain_env()
        observer = CausalObserver(ObservationConfig(position_noise=0.5),
                                  episode_seed=3)
        env.reset()
        for _ in range(10):
            served = observer.observe(env)
            for i in range(3):
                self.assertGreaterEqual(served.position[i], env.radius - 1e-9)
                self.assertLessEqual(served.position[i],
                                     env.chamber_size[i] - env.radius + 1e-9)
            env.step((0.05, 0.0, 0.0))


class VisibilityTests(unittest.TestCase):
    def test_far_obstacles_hidden(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(visibility_range=0.25))
        env.reset()
        served = observer.observe(env)
        self.assertEqual(served.obstacles, ())
        self.assertEqual(len(env.observe().obstacles), 1)

    def test_near_obstacles_kept(self):
        env = gate_env()
        observer = CausalObserver(ObservationConfig(visibility_range=0.75))
        env.reset()
        served = observer.observe(env)
        self.assertEqual(len(served.obstacles), 1)
        self.assertLess(norm(sub(served.obstacles[0].center, served.position)),
                        0.75)


class ConfigTests(unittest.TestCase):
    def test_round_trip_and_inf(self):
        cfg = ObservationConfig(visibility_range=math.inf, delay_steps=2)
        payload = cfg.to_dict()
        self.assertEqual(payload["visibility_range"], "inf")
        self.assertEqual(ObservationConfig.from_dict(payload).delay_steps, 2)
        self.assertFalse(ObservationConfig.from_dict(payload).perfect)
        self.assertTrue(ObservationConfig().perfect)


if __name__ == "__main__":
    unittest.main()
