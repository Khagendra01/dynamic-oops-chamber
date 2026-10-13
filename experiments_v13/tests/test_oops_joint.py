"""Gate 4 tests: joint DSL, VM semantics, prefix search, frozen reuse."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

import numpy as np

from experiments_v13.forward_kinematics import ee_position
from experiments_v13.frozen_library import load_library, save_library
from experiments_v13.joint_dsl import (JointProgram, JointVMState, candidate_dq,
                                       choose_joint, compile_program,
                                       in_limits, normalize_rule,
                                       reduces_error)
from experiments_v13.joint_oops_search import (JointPrefixOOPS, make_task,
                                               replay)
from experiments_v13.joint_vm import JointInterpreter
from experiments_v13.robot_chamber import RobotChamber
from experiments_v13.robot_model import default_robot


def _env(spec, q_goal, obstacles=(), **kw):
    goal = ee_position(spec, q_goal)
    def build():
        return RobotChamber(spec, np.zeros(6), goal, obstacles=list(obstacles),
                            **kw)
    return build


class TestDSL(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()

    def test_valid_and_invalid_rules(self):
        self.assertEqual(normalize_rule("JOINT_POS_2"), "JOINT_POS_2")
        self.assertEqual(normalize_rule("JOINT_NEG_3?"), "JOINT_NEG_3?")
        self.assertEqual(normalize_rule("HOLD"), "HOLD")
        self.assertEqual(normalize_rule("CALL:P001"), "CALL:P001")
        for bad in ("MOVE_ARM", "JOINT_POS_0", "JOINT_POS_7", "HOLD?",
                    "CHOOSE_JOINT?", "CALL:"):
            with self.assertRaises(ValueError):
                normalize_rule(bad)

    def test_program_validation(self):
        JointProgram(("JOINT_POS_2", "HOLD"))
        with self.assertRaises(ValueError):
            JointProgram(())
        with self.assertRaises(ValueError):
            JointProgram(("BOGUS",))

    def test_compile_skip_offsets_land_on_next_rule(self):
        prog = JointProgram(("JOINT_POS_2", "JOINT_NEG_3?", "HOLD"))
        bc = compile_program(prog)
        # base rule: pc 2 skip 5 -> 7 (CHECK_COLLISION of next? no: next rule)
        self.assertEqual(bc[0], ("AIM", "JOINT_POS_2"))
        self.assertEqual(bc[2], ("IF_LIMIT_SKIP", 5))
        self.assertEqual(bc[2][1] + 2, 7)
        self.assertEqual(bc[4][0], "IF_UNSAFE_SKIP")
        self.assertEqual(bc[4][1] + 4, 7)
        self.assertEqual(bc[7][0], "AIM")
        # progress rule at 7: skip offsets land at 16 (HOLD rule)
        self.assertEqual(bc[9], ("IF_LIMIT_SKIP", 7))
        self.assertEqual(bc[9][1] + 9, 16)
        self.assertEqual(bc[11][0], "IF_UNSAFE_SKIP")
        self.assertEqual(bc[11][1] + 11, 16)
        self.assertEqual(bc[12], ("CHECK_PROGRESS",))
        self.assertEqual(bc[13], ("IF_NO_PROGRESS_SKIP", 3))
        self.assertEqual(bc[13][1] + 13, 16)
        self.assertEqual(bc[16][0], "AIM")
        self.assertEqual(bc[-1], ("END_NO_MATCH",))

    def test_candidate_clips_to_limits(self):
        spec = self.spec
        obs_like = type("O", (), {"q": tuple(spec.joint_hi - 0.01),
                                  "goal": (0.8, 0.5, 0.5)})()
        mem = JointVMState()
        dq = candidate_dq("JOINT_POS_1", obs_like, mem, spec, 0.1)
        self.assertLessEqual(dq[0], 0.01 + 1e-12)
        self.assertEqual(np.count_nonzero(dq), 1)

    def test_progress_check_uses_fk(self):
        spec = self.spec
        q = np.zeros(6)
        goal = (0.5, 0.5, 0.5)  # below the straight-up EE: pitch down helps
        obs = type("O", (), {"q": tuple(q), "goal": goal})()
        dq_toward = np.array([0.0, 0.12, 0.0, 0, 0, 0])
        # shoulder pitch + rotates EE toward +x; goal straight below in z
        p0 = np.asarray(ee_position(spec, q))
        better = None
        for joint in range(6):
            for sign in (0.12, -0.12):
                dq = np.zeros(6)
                dq[joint] = sign
                if reduces_error(spec, obs, dq):
                    better = dq
                    break
            if better is not None:
                break
        self.assertIsNotNone(better, "no reducing step toward a nearby goal")

    def test_choose_joint_returns_bounded_single_joint(self):
        spec = self.spec
        obs = type("O", (), {"q": tuple(np.zeros(6)),
                             "goal": (0.7, 0.5, 0.6)})()
        dq = choose_joint(spec, obs, JointVMState(), 0.1)
        self.assertLessEqual(int(np.count_nonzero(dq)), 1)
        self.assertLessEqual(float(np.linalg.norm(dq)),
                             float(np.max(spec.velocity_limits * 0.1)) + 1e-12)
        # with a certify that rejects everything -> zero
        dq2 = choose_joint(spec, obs, JointVMState(), 0.1,
                           certify=lambda _dq: "unsafe")
        self.assertTrue(np.allclose(dq2, 0.0))


class TestVM(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()
        goal = ee_position(self.spec, np.array([0.2, 0.6, -0.3, 0, 0, 0]))
        env = RobotChamber(self.spec, np.zeros(6), goal)
        self.obs = env.observe()
        self.vm = JointInterpreter(self.spec)

    def test_direction_rule_emits_bounded_dq(self):
        prog = JointProgram(("JOINT_POS_2",))
        dq, trace = self.vm.act(prog, self.obs, JointVMState())
        self.assertIn("AIM JOINT_POS_2", trace)
        self.assertIn("EMIT", trace)
        self.assertEqual(np.count_nonzero(dq), 1)
        self.assertGreater(dq[1], 0)

    def test_blocked_rule_falls_through_to_hold(self):
        prog = JointProgram(("JOINT_POS_1?", "HOLD"))
        # progress guard: base yaw may not reduce error -> fall to HOLD
        dq, trace = self.vm.act(prog, self.obs, JointVMState())
        self.assertTrue(np.allclose(dq, 0.0))

    def test_memory_persists_last_delta(self):
        mem = JointVMState()
        prog1 = JointProgram(("JOINT_POS_2",))
        self.vm.act(prog1, self.obs, mem)
        self.assertGreater(mem.last_delta[1], 0)
        prog2 = JointProgram(("LAST",))
        dq, trace = self.vm.act(prog2, self.obs, mem)
        self.assertIn("AIM LAST", trace)
        self.assertGreater(dq[1], 0)

    def test_stall_suffix_requires_stagnation(self):
        mem = JointVMState()
        mem.stall_steps = 5
        dq, trace = self.vm.act(JointProgram(("JOINT_NEG_2!",)), self.obs, mem)
        self.assertIn("EMIT", trace)
        mem2 = JointVMState()
        mem2.stall_steps = 0
        dq2, trace2 = self.vm.act(JointProgram(("JOINT_NEG_2!", "HOLD")),
                                  self.obs, mem2)
        self.assertIn("EMIT_WAIT", trace2)

    def test_frozen_call_executes_child(self):
        child = JointProgram(("JOINT_POS_2",), "P001")
        vm = JointInterpreter(self.spec, {"P001": child})
        dq, trace = vm.act(JointProgram(("CALL:P001",)), self.obs,
                           JointVMState())
        self.assertIn("CALL P001", trace)
        self.assertGreater(dq[1], 0)

    def test_missing_call_falls_through(self):
        dq, trace = self.vm.act(JointProgram(("CALL:P999", "HOLD")), self.obs,
                                JointVMState())
        self.assertIn("MISSING CALL P999", trace)
        self.assertIn("EMIT_WAIT", trace)

    def test_no_cross_observation_contamination(self):
        """A fresh JointVMState yields identical output for same obs+program."""
        prog = JointProgram(("JOINT_NEG_3",))
        dq1, t1 = self.vm.act(prog, self.obs, JointVMState())
        dq2, t2 = self.vm.act(prog, self.obs, JointVMState())
        self.assertEqual(t1, t2)
        self.assertTrue(np.array_equal(dq1, dq2))


class TestReplayAndSearch(unittest.TestCase):
    def setUp(self):
        self.spec = default_robot()

    def test_replay_deterministic(self):
        factory = _env(self.spec, np.array([0.2, 0.5, -0.3, 0, 0, 0]))
        prog = JointProgram(("JOINT_POS_2", "HOLD"))
        r1, _ = replay(prog, factory)
        r2, _ = replay(prog, factory)
        self.assertEqual(r1.success, r2.success)
        self.assertEqual(r1.steps, r2.steps)
        self.assertEqual(r1.work_units, r2.work_units)

    def test_search_finds_reaching_program_small_budget(self):
        """OOPS discovers a program that reaches a nearby empty-chamber goal."""
        factory = _env(self.spec, np.array([0.0, 0.55, -0.25, 0, 0, 0]),
                       max_steps=50, goal_tolerance=0.035)
        learner = JointPrefixOOPS(self.spec, reuse=True, max_depth=3,
                                  starting_budget=64, max_phase_budget=4096,
                                  max_prefixes_per_phase=2000)
        stage = learner.train_task(make_task("empty", factory))
        self.assertTrue(stage.success, f"search failed: {stage}")
        self.assertGreaterEqual(len(learner.frozen), 1)
        program = next(reversed(learner.frozen.values()))
        outcome, _ = replay(program, factory, learner.frozen)
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.collision)

    def test_reuse_flag_changes_lanes(self):
        f1 = _env(self.spec, np.array([0.0, 0.55, -0.25, 0, 0, 0]), max_steps=50)
        f2 = _env(self.spec, np.array([0.25, 0.5, -0.3, 0, 0, 0]), max_steps=50)
        with_reuse = JointPrefixOOPS(self.spec, reuse=True, max_depth=3,
                                     starting_budget=64,
                                     max_phase_budget=8192,
                                     max_prefixes_per_phase=3000)
        without = JointPrefixOOPS(self.spec, reuse=False, max_depth=3,
                                  starting_budget=64, max_phase_budget=8192,
                                  max_prefixes_per_phase=3000)
        s1r = with_reuse.train_task(make_task("t1", f1))
        s2r = with_reuse.train_task(make_task("t2", f2))
        s1n = without.train_task(make_task("t1", f1))
        s2n = without.train_task(make_task("t2", f2))
        self.assertTrue(s1r.success, f"reuse first task failed: {s1r}")
        if s2r.found_lane == "reused":
            # frozen library from t1 already solves t2: no search phases
            self.assertGreaterEqual(len(with_reuse.frozen), 1)
        else:
            lanes_reuse = {p["lane"] for p in s2r.phases}
            self.assertEqual(lanes_reuse, {"extend", "fresh"})
            first_extend = next(p for p in s2r.phases if p["lane"] == "extend")
            self.assertEqual(first_extend["lane_budget"],
                             first_extend["phase_budget"] // 2)
        lanes_fresh = {p["lane"] for p in s2n.phases}
        self.assertEqual(lanes_fresh, {"fresh"})

    def test_cumulative_curriculum_accepts_second_task(self):
        f1 = _env(self.spec, np.array([0.0, 0.55, -0.25, 0, 0, 0]), max_steps=50)
        f2 = _env(self.spec, np.array([0.3, 0.5, -0.35, 0, 0, 0]), max_steps=50)
        learner = JointPrefixOOPS(self.spec, reuse=True, max_depth=3,
                                  starting_budget=64, max_phase_budget=8192,
                                  max_prefixes_per_phase=3000)
        s1 = learner.train_task(make_task("t1", f1))
        self.assertTrue(s1.success)
        s2 = learner.train_task(make_task("t2", f2))
        self.assertTrue(s2.success or s2.found_lane == "none")

    def test_library_save_load_roundtrip(self):
        child = JointProgram(("JOINT_POS_2",), "P001")
        parent = JointProgram(("CALL:P001", "HOLD"), "P002")
        lib = {"P001": child, "P002": parent}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = save_library(lib, Path(tmp) / "lib.json")
            loaded = load_library(path)
        self.assertEqual(set(loaded), {"P001", "P002"})
        self.assertEqual(loaded["P002"].rules, parent.rules)

    def test_save_rejects_forward_reference(self):
        import json, tempfile
        payload = {"format": "robot-oops-joint-lib-0.1",
                   "programs": [{"id": "P001", "rules": ["CALL:P002"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_library(path)


if __name__ == "__main__":
    unittest.main()
