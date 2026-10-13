"""Tests for the conventional planner baselines used in v1.2."""
from __future__ import annotations

import unittest

from chamber import norm, sub
from experiments_v12.baseline_planners import (PLANNERS, BasePlanner,
                                               make_planner)
from experiments_v12.dynamic_chamber import (DynamicBox, DynamicChamber)
from experiments_v12.motion_prediction import (Predictor, config_from_env,
                                               predict, segment_safe)
from experiments_v12.scenarios import generate
from oops_language import MotionConfig


def empty_env(max_steps: int = 45) -> DynamicChamber:
    return DynamicChamber(chamber_size=(1.0, 1.0, 1.0),
                          start=(0.1, 0.5, 0.5), goal=(0.9, 0.5, 0.5),
                          obstacles=[], radius=0.035, dt=0.5, max_speed=0.7,
                          goal_tolerance=0.055, max_steps=max_steps)


def gate_env(max_steps: int = 45) -> DynamicChamber:
    """A wall oscillating across the corridor: the agent must time its crossing.

    The box only sweeps in y at x=0.5, so waiting near the start is always
    safe and the corridor opens whenever the box is away from y=0.5.
    """
    return DynamicChamber(
        chamber_size=(1.0, 1.0, 1.0), start=(0.1, 0.5, 0.5),
        goal=(0.9, 0.5, 0.5),
        obstacles=[DynamicBox("gate", (0.5, 0.5, 0.5), (0.06, 0.30, 0.92),
                              velocity=(0.0, 0.5, 0.0))],
        radius=0.035, dt=0.5, max_speed=0.7, goal_tolerance=0.055,
        max_steps=max_steps)


def run(planner: BasePlanner, env: DynamicChamber) -> dict:
    config = MotionConfig(radius=env.radius, dt=env.dt, max_speed=env.max_speed)
    planner.reset()
    obs = env.reset()
    actions = []
    collision = False
    interventions = 0
    while not env.done and env.step_index < env.max_steps:
        velocity = planner.act(obs, config)
        actions.append(velocity)
        result = env.step(velocity)
        collision |= result.collision
        interventions += int(result.safety_intervention)
        obs = result.observation
    return {"actions": actions, "collision": collision,
            "interventions": interventions, "steps": env.step_index,
            "success": env.done and not collision
            and norm(sub(obs.position, env.goal)) <= env.goal_tolerance + 1e-9,
            "stats": planner.stats.summary(), "env": env}


class PredictionTests(unittest.TestCase):
    def test_predict_is_no_oracle(self):
        env = gate_env()
        obs = env.reset()
        shifted = predict(obs, 0.5)
        self.assertGreater(shifted.obstacles[0].center[1],
                           obs.obstacles[0].center[1])
        # Prediction depends only on the observation's own velocity field.
        from chamber import ObstacleObservation, Observation
        fake = Observation(obs.time_s, obs.step_index, obs.position, obs.goal,
                           (ObstacleObservation("gate", (0.2, 0.5, 0.5),
                                                (0.08, 0.92, 0.92),
                                                (-0.45, 0.0, 0.0)),),
                           obs.chamber_size)
        moved = predict(fake, 0.5)
        self.assertLess(moved.obstacles[0].center[0], 0.2)

    def test_segment_safe_matches_interpreter_rule(self):
        env = gate_env()
        obs = env.reset()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        # A long move through the wall must be rejected by the shared rule.
        self.assertFalse(segment_safe(obs, (0.1, 0.5, 0.5), (0.9, 0.5, 0.5),
                                      config))
        self.assertTrue(segment_safe(obs, (0.1, 0.5, 0.5), (0.2, 0.5, 0.5),
                                     config))


class PlannerContractTests(unittest.TestCase):
    def test_factory_rejects_unknown(self):
        with self.assertRaises(ValueError):
            make_planner("nope")

    def test_command_stride_respects_language_cap(self):
        for kind in PLANNERS:
            planner = make_planner(kind)
            env = empty_env()
            config = MotionConfig(radius=env.radius, dt=env.dt,
                                  max_speed=env.max_speed)
            obs = env.reset()
            for _ in range(5):
                velocity = planner.act(obs, config)
                self.assertLessEqual(norm(velocity) * config.dt,
                                     config.stride + 1e-9, kind)
                obs = env.step(velocity).observation

    def test_stop_when_planning_misses_deadline(self):
        planner = make_planner("space_time_astar", deadline_seconds=0.0)
        env = empty_env()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        obs = env.reset()
        self.assertEqual(planner.act(obs, config), (0.0, 0.0, 0.0))
        self.assertEqual(planner.stats.deadline_misses, 1)

    def test_deterministic(self):
        for kind in PLANNERS:
            first = run(make_planner(kind), empty_env())["actions"]
            second = run(make_planner(kind), empty_env())["actions"]
            self.assertEqual(first, second, kind)


class SolverTests(unittest.TestCase):
    def test_all_planners_solve_empty_chamber(self):
        for kind in PLANNERS:
            outcome = run(make_planner(kind), empty_env())
            self.assertTrue(outcome["success"], kind)
            self.assertFalse(outcome["collision"], kind)
            self.assertEqual(outcome["stats"]["deadline_misses"], 0, kind)

    def test_space_time_waits_for_a_moving_gate(self):
        outcome = run(make_planner("space_time_astar"), gate_env())
        self.assertTrue(outcome["success"], outcome["stats"])
        self.assertFalse(outcome["collision"])
        waits = sum(1 for a in outcome["actions"] if a == (0.0, 0.0, 0.0))
        self.assertGreaterEqual(waits, 1)   # it timed the gate instead of pushing
        # The static snapshot planner cannot: the corridor is closed at t=0
        # and it has no model of when it re-opens.
        static = run(make_planner("static_astar"), gate_env())
        self.assertFalse(static["success"])

    def test_rollout_solves_static_block(self):
        env = generate(1, 1000, split="train").build()
        outcome = run(make_planner("rollout_greedy"), env)
        self.assertTrue(outcome["success"], outcome["stats"])
        self.assertFalse(outcome["collision"])

    def test_infeasible_scene_reports_no_path_without_collision(self):
        spec = next(generate(1, seed, split="train")
                    for seed in range(1000, 1100)
                    if generate(1, seed, split="train").label
                    == "infeasible_static")
        env = spec.build()
        outcome = run(make_planner("space_time_astar"), env)
        self.assertFalse(outcome["success"])
        self.assertFalse(outcome["collision"])
        self.assertGreaterEqual(outcome["stats"]["no_path_steps"], 1)


class EdgeMemoTests(unittest.TestCase):
    """The planner edge memo must reuse only on identical obstacle state."""

    def test_signature_changes_with_position_noise(self):
        env = gate_env()
        obs = env.reset()
        sig = BasePlanner._scene_signature(obs)
        from chamber import ObstacleObservation, Observation
        noisy = Observation(
            obs.time_s, obs.step_index, obs.position, obs.goal,
            (ObstacleObservation("gate", (obs.obstacles[0].center[0] + 1e-6,
                                          obs.obstacles[0].center[1],
                                          obs.obstacles[0].center[2]),
                                 obs.obstacles[0].size,
                                 obs.obstacles[0].velocity),),
            obs.chamber_size)
        self.assertNotEqual(sig, BasePlanner._scene_signature(noisy))

    def test_memoized_and_uncached_planners_agree(self):
        """Caching never changes the action sequence (static and dynamic)."""

        class NoMemoPlannerMixin:
            def _edge_memo(self):
                return {}

        for kind in ("static_astar", "space_time_astar"):
            for env in (empty_env(), gate_env()):
                cached = run(make_planner(kind), env)["actions"]
                plain_cls = type(f"NoMemo_{kind}",
                                 (NoMemoPlannerMixin, type(make_planner(kind))),
                                 {})
                uncached = run(plain_cls(), env)["actions"]
                self.assertEqual(cached, uncached, kind)

    def test_memo_reuses_across_steps_on_static_scenes(self):
        planner = make_planner("static_astar")
        env = empty_env()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        planner.reset()
        obs = env.reset()
        for _ in range(4):
            obs = env.step(planner.act(obs, config)).observation
        self.assertIsNotNone(planner._edge_cache)
        self.assertGreater(len(planner._edge_cache), 0)

    def test_memo_misses_on_dynamic_scenes_per_depth(self):
        planner = make_planner("space_time_astar")
        env = gate_env()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        planner.reset()
        obs = env.reset()
        planner.act(obs, config)
        signatures = {key[0] for key in planner._edge_cache}
        self.assertGreater(len(signatures), 1,
                           "moving gate must yield distinct depth signatures")


if __name__ == "__main__":
    unittest.main()
