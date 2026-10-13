"""Tests for the v1.2 C1-C5 ablation matrix and its implementation switches."""
from __future__ import annotations

import types
import unittest

from chamber import example_scene
from experiments_v12.ablation_matrix import (C1, C2, C3, C4, C5, CANONICAL,
                                             FACTORIAL, PRIMARY_CONTRASTS,
                                             validate_matrix)
from experiments_v12.policies import make_policy, ProgramPolicy
from experiments_v12.synthesis import (AblationPrefixOOPS, deployment_library,
                                       train_variant)
from oops_language import MotionConfig, Program
from oops_prefix import Task


def tiny_learner(**kwargs):
    params = dict(reuse=False, cumulative=True, max_depth=2,
                  starting_budget=1, max_phase_budget=1,
                  max_prefixes_per_phase=8)
    params.update(kwargs)
    return AblationPrefixOOPS(**params)


def empty_task(name: str, **kwargs) -> Task:
    return Task(name, lambda: example_scene("empty"), **kwargs)


class MatrixTests(unittest.TestCase):
    def test_spec_table(self):
        validate_matrix()
        self.assertEqual(C1.bits(), (False, False, False))
        self.assertEqual(C2.bits(), (False, True, True))
        self.assertEqual(C3.bits(), (True, False, True))
        self.assertEqual(C4.bits(), (True, True, False))
        self.assertEqual(C5.bits(), (True, True, True))

    def test_factorial_is_complete_and_distinct(self):
        self.assertEqual(len(FACTORIAL), 8)
        self.assertEqual(len({v.bits() for v in FACTORIAL}), 8)
        for canonical in CANONICAL:
            self.assertIn(canonical, FACTORIAL)

    def test_primary_contrasts_are_single_factor(self):
        for left, right, label in PRIMARY_CONTRASTS:
            a = next(v for v in FACTORIAL if v.id == left)
            b = next(v for v in FACTORIAL if v.id == right)
            differing = [name for name in ("reuse", "cumulative", "memory")
                         if getattr(a, name) != getattr(b, name)]
            expected = 3 if (left, right) == ("C1", "C5") else 1
            self.assertEqual(len(differing), expected, label)


class SynthesisSwitchTests(unittest.TestCase):
    def _capture_lanes(self, learner):
        calls = []

        def fake(self, lane, budget, stage_tasks, prev):
            calls.append({"lane": lane, "tasks": [t.name for t in stage_tasks]})
            return dict(found=None, spent=0, visits=0, tested=0)

        learner._run_lane = types.MethodType(fake, learner)
        return calls

    def _seeded(self, cumulative):
        # Simulate a completed first stage without spending search budget.
        learner = tiny_learner(reuse=False, cumulative=cumulative)
        learner.tasks = [empty_task("t1")]
        learner.frozen["P001"] = Program(("GOAL",), "P001")
        return learner, self._capture_lanes(learner)

    def test_cumulative_on_sees_all_tasks(self):
        learner, calls = self._seeded(cumulative=True)
        learner.train_task(empty_task("t2"))
        self.assertTrue(calls)
        self.assertEqual(calls[0]["tasks"], ["t1", "t2"])

    def test_cumulative_off_sees_only_current_task(self):
        learner, calls = self._seeded(cumulative=False)
        learner.train_task(empty_task("t2"))
        self.assertTrue(calls)
        self.assertEqual(calls[0]["tasks"], ["t2"])

    def test_history_is_kept_but_not_used_for_acceptance(self):
        learner, calls = self._seeded(cumulative=False)
        learner.train_task(empty_task("t2"))
        # Nothing new was discovered, but the log still holds stage one.
        self.assertEqual([t.name for t in learner.tasks], ["t1"])
        self.assertEqual(calls[0]["tasks"], ["t2"])

    def test_reuse_on_opens_extend_lane_and_call(self):
        learner = tiny_learner(reuse=True)
        learner.frozen["P001"] = Program(("GOAL",), "P001")
        learner.tasks = [empty_task("t1")]
        calls = self._capture_lanes(learner)
        # require_explicit_wait guarantees the reuse shortcut does not fire.
        learner.train_task(empty_task("t2", require_explicit_wait=True))
        self.assertEqual({c["lane"] for c in calls}, {"extend", "fresh"})
        tokens = [name for name, _ in learner._vocabulary("fresh")]
        self.assertIn("CALL:P001", tokens)

    def test_reuse_off_is_fresh_lane_only_without_call(self):
        learner = tiny_learner(reuse=False)
        learner.frozen["P001"] = Program(("GOAL",), "P001")
        learner.tasks = [empty_task("t1")]
        calls = self._capture_lanes(learner)
        learner.train_task(empty_task("t2"))
        self.assertEqual({c["lane"] for c in calls}, {"fresh"})
        tokens = [name for name, _ in learner._vocabulary("fresh")]
        self.assertFalse([t for t in tokens if t.startswith("CALL:")])


class DeploymentSwitchTests(unittest.TestCase):
    def test_deployment_library_follows_reuse_bit(self):
        learner = tiny_learner(reuse=True)
        learner.frozen["P001"] = Program(("GOAL",), "P001")
        self.assertEqual(set(deployment_library(learner, C5)), {"P001"})
        self.assertEqual(deployment_library(learner, C2), {})
        self.assertEqual(deployment_library(learner, C1), {})

    def test_variants_construct_distinct_configurations(self):
        program = Program(("GOAL",), "P")
        library = {"P": program}
        built = []
        for variant in (C1, C2, C3, C4, C5):
            learner = AblationPrefixOOPS(reuse=variant.reuse,
                                         cumulative=variant.cumulative,
                                         starting_budget=1, max_phase_budget=1)
            if variant.reuse:
                learner.frozen["P001"] = program
            policy = make_policy(variant, program, library, MotionConfig())
            built.append((variant.id, learner.reuse, learner.cumulative,
                          policy.persistent_memory,
                          sorted(deployment_library(learner, variant))))
        self.assertEqual(built, [
            ("C1", False, False, False, []),
            ("C2", False, True, True, []),
            ("C3", True, False, True, ["P001"]),
            ("C4", True, True, False, ["P001"]),
            ("C5", True, True, True, ["P001"]),
        ])

    def test_memory_off_resets_registers_every_step(self):
        env = example_scene("empty")
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        program = Program(("GOAL",), "drive")
        persistent = ProgramPolicy(program, {}, config=config,
                                   persistent_memory=True)
        stateless = ProgramPolicy(program, {}, config=config,
                                  persistent_memory=False)
        obs = env.reset()
        persistent.act(obs)
        stateless.act(obs)
        # Persistent memory keeps one register across observations; the
        # stateless variant starts every decision from a fresh VMState.
        kept = persistent.memory
        fresh = stateless.memory
        self.assertNotEqual(kept.last_direction, (0.0, 0.0, 0.0))
        persistent.act(obs)
        stateless.act(obs)
        self.assertIs(persistent.memory, kept)
        self.assertIsNot(stateless.memory, fresh)
        self.assertEqual(persistent.counters.steps, 2)
        self.assertEqual(stateless.counters.steps, 2)

    def test_memory_changes_axis_choice_at_a_score_tie(self):
        # Scene engineered so step 2 has an exact X/Y alignment tie: the
        # persistence register then decides which axis is emitted.
        env = tie_env()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        program = Program(("CHOOSE_AXIS",), "axis")
        persistent = ProgramPolicy(program, {}, config=config,
                                   persistent_memory=True)
        stateless = ProgramPolicy(program, {}, config=config,
                                  persistent_memory=False)
        obs_p, env_s = env.reset(), tie_env()
        obs_s = env_s.reset()
        first_p, first_s = persistent.act(obs_p), stateless.act(obs_s)
        self.assertEqual(first_p, first_s)          # memory empty on step 1
        obs_p = env.step(first_p).observation
        obs_s = env_s.step(first_s).observation
        second_p, second_s = persistent.act(obs_p), stateless.act(obs_s)
        self.assertNotEqual(second_p, second_s)
        self.assertGreater(second_p[1], 0.0)        # persistent keeps +Y
        self.assertGreater(second_s[0], 0.0)        # stateless picks +X

    def test_conditionals_identical_across_variants(self):
        env = example_scene("empty")
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        # X_NEG is unsafe at the start wall, so the conditional branch matters.
        program = Program(("X_NEG", "GOAL"), "branchy")
        library = {"P": program}
        obs = env.reset()
        sequences = {}
        for variant in (C1, C2, C3, C4, C5):
            policy = make_policy(variant, program, library, config)
            actions = [policy.act(obs) for _ in range(4)]
            self.assertEqual(policy.counters.steps, 4)  # live replanning
            sequences[variant.id] = actions
        self.assertEqual(len({tuple(map(tuple, actions))
                              for actions in sequences.values()}), 1)
        self.assertTrue(all(actions[0][0] > 0.0 for actions in sequences.values()))

    def test_train_variant_uses_identical_budgets(self):
        curriculum = [("s1", _spec())]
        outcome_a = train_variant(C1, curriculum, max_depth=2,
                                  starting_budget=1, max_phase_budget=1,
                                  max_prefixes_per_phase=4)
        outcome_b = train_variant(C5, curriculum, max_depth=2,
                                  starting_budget=1, max_phase_budget=1,
                                  max_prefixes_per_phase=4)
        self.assertEqual(outcome_a.variant["id"], "C1")
        self.assertEqual(outcome_b.variant["id"], "C5")
        self.assertIn("wall_time_s", outcome_a.totals)
        self.assertIn("search_work_units", outcome_a.totals)


def tie_env():
    """Symmetric diagonal scene with one axis blocked on the first step."""
    from experiments_v12.dynamic_chamber import DynamicBox, DynamicChamber
    return DynamicChamber(
        chamber_size=(1.0, 1.0, 1.0), start=(0.5, 0.1, 0.5),
        goal=(0.9, 0.62, 0.5),
        obstacles=[DynamicBox("block_x", (0.62, 0.1, 0.5),
                              (0.08, 0.08, 0.08))],
        radius=0.035, dt=0.5, max_speed=0.7, goal_tolerance=0.055,
        max_steps=20)


def _spec():
    from experiments_v12.scenarios import generate
    return generate(1, 1000, split="train")


if __name__ == "__main__":
    unittest.main()
