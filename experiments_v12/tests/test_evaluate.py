"""Tests for the v1.2 episode runner and paired statistics."""
from __future__ import annotations

import math
import unittest

from experiments_v12.baseline_planners import make_planner
from experiments_v12.evaluate import (PlannerPolicy, evaluate_method, method_row,
                                      run_episode, summarize, surface_clearance,
                                      write_rows)
from experiments_v12.dynamic_observations import ObservationConfig
from experiments_v12.policies import ProgramPolicy
from experiments_v12.scenarios import ScenarioSpec, generate, training_curriculum
from experiments_v12.statistics import (bootstrap_mean_ci, format_ci,
                                        paired_comparison,
                                        paired_rate_comparison,
                                        wilson_interval)
from oops_language import MotionConfig, Program


def free_spec(seed: int = 0) -> ScenarioSpec:
    return ScenarioSpec(id=f"free_{seed}", level=1, seed=seed,
                        chamber_size=(1.0, 1.0, 1.0),
                        start=(0.1, 0.5, 0.5), goal=(0.9, 0.5, 0.5),
                        radius=0.035, dt=0.5, max_speed=0.7,
                        goal_tolerance=0.055, max_steps=45, boxes=[],
                        label="feasible")


class EvaluateTests(unittest.TestCase):
    def test_program_episode_metrics(self):
        spec = free_spec()
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        policy = ProgramPolicy(Program(("GOAL",), "g"), {}, config=config)
        row = method_row("goal_only", spec, env, policy)
        self.assertTrue(row["success"], row)
        self.assertFalse(row["collision"])
        self.assertGreater(row["path_length_m"], 0.5)
        self.assertGreater(row["steps"], 0)
        self.assertGreater(row["decision_latency_mean_s"], 0.0)
        self.assertEqual(row["deadline_misses"], 0)
        self.assertEqual(row["extras"]["work_units"] > 0, True)
        # Obstacle-free scene: no clearance statistic is available (-1.0).
        self.assertEqual(row["min_clearance_m"], -1.0)

    def test_planner_policy_adapter(self):
        spec = free_spec()
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        policy = PlannerPolicy(make_planner("static_astar"), config)
        row = method_row("static_astar", spec, env, policy)
        self.assertTrue(row["success"], row)
        self.assertIn("replans", row["extras"])
        self.assertEqual(row["deadline_misses"], 0)

    def test_observer_is_used_when_configured(self):
        spec = free_spec()
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        policy = ProgramPolicy(Program(("GOAL",), "g"), {}, config=config)
        row = method_row("goal_only", spec, env, policy,
                         observation=ObservationConfig(position_noise=0.005))
        self.assertIn("success", row)

    def test_evaluate_method_and_summarize(self):
        scenarios = [free_spec(seed) for seed in (0, 1, 2)]
        config = MotionConfig()

        def factory():
            return ProgramPolicy(Program(("GOAL",), "g"), {}, config=config)

        rows = evaluate_method(scenarios, "goal_only", factory)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["method"] == "goal_only" for r in rows))
        summary = summarize(rows)
        self.assertEqual(summary[0]["n"], 3)
        self.assertIn("success_rate", summary[0])

    def test_write_rows_json_and_csv(self):
        import json
        import tempfile
        from pathlib import Path
        rows = [{"method": "m", "success": 1, "extras": {"a": 1}}]
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "rows.json"
            csv_path = Path(tmp) / "rows.csv"
            write_rows(rows, json_path)
            write_rows(rows, csv_path)
            self.assertEqual(json.loads(json_path.read_text())[0]["method"], "m")
            self.assertIn("method", csv_path.read_text().splitlines()[0])

    def test_clearance_reports_gap_to_obstacles(self):
        spec = generate(1, 1000, split="train")
        env = spec.build()
        clearance = surface_clearance(env, env.start, 0.0)
        self.assertTrue(math.isfinite(clearance))
        self.assertGreaterEqual(clearance, 0.0)

    def test_curriculum_is_deterministic(self):
        first = training_curriculum()
        second = training_curriculum()
        self.assertEqual([name for name, _ in first],
                         [name for name, _ in second])
        self.assertGreaterEqual(len(first), 20)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        values = [1.0, -1.0, 0.5, 0.0, -0.25]
        a = bootstrap_mean_ci(values, n_samples=500)
        b = bootstrap_mean_ci(values, n_samples=500)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_low"], a["mean"])
        self.assertGreaterEqual(a["ci_high"], a["mean"])

    def test_paired_comparison(self):
        left = [1, 1, 0, 1, 0]
        right = [1, 0, 0, 0, 0]
        result = paired_rate_comparison(left, right, n_samples=500)
        self.assertAlmostEqual(result["mean"], 0.4)
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 0)
        self.assertEqual(result["both"], 1)
        self.assertIn("left_wilson", result)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_comparison([1.0], [1.0, 2.0])

    def test_wilson_bounds(self):
        result = wilson_interval(0, 10)
        self.assertEqual(result["rate"], 0.0)
        self.assertGreaterEqual(result["ci_low"], 0.0)
        self.assertLessEqual(result["ci_high"], 1.0)

    def test_format_ci(self):
        text = format_ci({"mean": 0.1234, "ci_low": 0.0, "ci_high": 0.2})
        self.assertIn("[0.000, 0.200]", text)


if __name__ == "__main__":
    unittest.main()
