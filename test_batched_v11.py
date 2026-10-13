"""Parity: batched job evaluation must reproduce oops_training.replay exactly."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from chamber import example_scene
from oops_language import Program
from oops_scenarios import north_blocked, south_blocked, moving_gate
from oops_training import extended_curriculum, replay
from oops_prefix import standard_tasks
from torch_chamber import TorchChamberBatch
from torch_job_policy import TorchJobPolicy, compile_program

PROGRAM_WIDTH = 16


def task_templates():
    return [factory() for _, factory, *_ in extended_curriculum()]


TEMPLATES = task_templates()
TASK_INDEX = {name: i for i, (name, *_rest) in enumerate(extended_curriculum())}


def batched_episodes(pairs, frozen, *, width=PROGRAM_WIDTH, device="cpu",
                     dtype=torch.float64):
    """Evaluate (rules, task_name) pairs and return EpisodeResult-like dicts."""
    task_ids = torch.tensor([TASK_INDEX[name] for _rules, name in pairs],
                            dtype=torch.long)
    n = len(pairs)
    env = TorchChamberBatch.from_templates(TEMPLATES, task_ids, device=device,
                                           dtype=dtype)
    programs = [compile_program(Program(tuple(rules)), frozen)
                for rules, _name in pairs]
    policy = TorchJobPolicy(env, programs, program_width=width)
    policy.run(max_steps=int(env.max_steps_t.max()))
    if bool((policy.max_instructions > policy.instruction_limit).any()):
        raise AssertionError("instruction limit exceeded")
    return {
        "success": (env.done & ~env.collision &
                    (torch.linalg.vector_norm(env.position - env.goal, dim=-1)
                     <= env.tolerance_t)),
        "collision": env.collision,
        "interventions": policy.interventions,
        "steps": env.step_index,
        "fallback_waits": policy.fallback_waits,
        "explicit_waits": policy.explicit_waits,
        "continued_actions": policy.continued,
        "directional_choices": policy.directional,
        "trace_events": policy.trace_events,
        "work_units": policy.work_units,
    }


FIELDS = ("success", "collision", "interventions", "steps", "fallback_waits",
          "explicit_waits", "continued_actions", "directional_choices",
          "trace_events", "work_units")


def _one(rules, task_name, frozen):
    tasks = {name: factory for name, factory, *_ in extended_curriculum()}
    expected, _ = replay(Program(tuple(rules)), tasks[task_name], frozen)
    got = batched_episodes([(rules, task_name)], frozen)
    for field in FIELDS:
        want = getattr(expected, field)
        have = got[field][0].item() if torch.is_tensor(got[field]) else got[field]
        if isinstance(want, bool):
            want = int(want)
        if isinstance(have, bool):
            have = int(have)
        yield field, want, int(have)


SAMPLES = [
    (("GOAL",), "empty"),
    (("CHOOSE_AXIS",), "static"),
    (("WAIT_GAP",), "moving"),
    (("CONTINUE",), "empty"),
    (("LAST",), "crossing"),
    (("X_POS",), "empty"),
    (("GOAL",), "north_blocked"),
    (("WAIT_GAP",), "moving_gate"),
    (("GOAL", "CHOOSE_AXIS"), "moving"),
    (("WAIT_GAP", "CALL:P001"), "moving_gate"),
    (("CONTINUE", "CALL:P003"), "crossing"),
    (("GOAL", "CHOOSE_AXIS", "WAIT_GAP", "CONTINUE"), "south_blocked"),
]


class ParityTests(unittest.TestCase):
    def test_samples(self):
        frozen = {"P001": Program(("GOAL",), "P001"),
                  "P002": Program(("GOAL", "CHOOSE_AXIS"), "P002"),
                  "P003": Program(("WAIT_GAP", "CALL:P002"), "P003")}
        failures = []
        for rules, task in SAMPLES:
            for field, want, have in _one(rules, task, frozen):
                if want != have:
                    failures.append(f"{rules} {task} {field}: cpu={want} gpu={have}")
        if failures:
            self.fail("\n".join(failures))

    def test_batch_matches_singles(self):
        frozen = {"P001": Program(("GOAL",), "P001")}
        pairs = [(rules, task) for rules, task in SAMPLES]
        got = batched_episodes(pairs, frozen)
        for i, (rules, task) in enumerate(pairs):
            tasks = {name: factory for name, factory, *_ in extended_curriculum()}
            expected, _ = replay(Program(tuple(rules)), tasks[task], frozen)
            for field in FIELDS:
                want = getattr(expected, field)
                have = got[field][i].item()
                self.assertEqual(int(bool(want)) if isinstance(want, bool)
                                 else int(want), int(have),
                                 f"{rules} {task} {field}")


if __name__ == "__main__":
    unittest.main()
