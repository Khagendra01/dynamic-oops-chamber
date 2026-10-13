"""OOPS prefix-search adapter for the joint-space environment.

Reuses the unchanged v1.2.1 ``PrefixOOPS`` engine (probability-budgeted
executable-prefix DFS, extend/fresh lanes, cumulative curriculum validation,
frozen library, phase doubling) through a thin adapter:

* ``_evaluate`` replays a candidate program on a :class:`RobotTask` factory
  with the joint interpreter instead of the point-agent interpreter;
* ``_vocabulary`` supplies joint-space priors plus ``CALL`` of the latest
  frozen program;
* ``train_task`` is copied from the parent with a single change: programs
  are ``JointProgram`` (the parent hardcodes the point-agent ``Program``,
  which validates against the v1.2 vocabulary).  All search mechanics —
  lane budgets, prefix testing, cache, acceptance shortcut — are the same
  code path as v1.2.1.

``JointEpisodeResult`` subclasses the v1.2.1 ``EpisodeResult`` so
``oops_prefix.Task.accepts`` works unchanged; extra arm-specific fields are
carried for reporting only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from experiments_v13.joint_dsl import (JointProgram, JointVMState,
                                       compile_program, instruction_cost)
from experiments_v13.joint_vm import JointInterpreter
from experiments_v13.robot_chamber import RobotChamber
from oops_prefix import PrefixOOPS, StageResult, Task
from oops_training import EpisodeResult


@dataclass(frozen=True)
class JointEpisodeResult(EpisodeResult):
    """Episode outcome; ``EpisodeResult`` fields drive Task.accepts."""

    limit_clipped_steps: int = 0
    collision_category: str = ""
    min_ee_error: float = -1.0
    final_ee_error: float = -1.0
    joint_travel: float = 0.0


def replay(program: JointProgram, make_env: Callable[[], RobotChamber],
           library: dict[str, JointProgram] | None = None,
           with_trace: bool = False) -> tuple[JointEpisodeResult, list[dict]]:
    """Run one program on one fresh scene; the synthesis workhorse."""
    env = make_env()
    vm = JointInterpreter(env.spec, library, dt=env.dt)
    memory = JointVMState()
    obs = env.reset()
    traces: list[dict] = []
    interventions = 0
    fallback_waits = 0
    collision = False
    limit_steps = 0
    trace_events = 0
    work_units = 0
    joint_travel = 0.0
    min_error = float("inf")
    for _ in range(env.max_steps):
        if env.done:
            break
        dq, events = vm.act(program, obs, memory)
        trace_events += len(events)
        work_units += vm.program_work_units(events)
        fallback_waits += int(events[-1] == "FALLBACK WAIT")
        result = env.step(dq)
        interventions += int(result.intervention)
        limit_steps += int(result.limit_clipped)
        joint_travel += float(np.sum(np.abs(result.executed_dq)))
        collision |= result.collision
        error = env.ee_error_at()
        min_error = min(min_error, error)
        if with_trace:
            traces.append({
                "step": obs.step_index,
                "q": list(obs.q),
                "dq": [float(v) for v in result.executed_dq],
                "vm": events,
                "ee_error": error,
                "collision": result.collision,
                "shield_intervention": result.intervention,
            })
        obs = result.observation
        if result.done:
            break
    success = bool(env.done and not collision
                   and env.ee_error_at() <= env.goal_tolerance + 1e-9)
    outcome = JointEpisodeResult(
        success=success, collision=collision, interventions=interventions,
        steps=env.step_index, fallback_waits=fallback_waits,
        explicit_waits=memory.explicit_waits,
        continued_actions=0,
        directional_choices=sum(1 for t in traces for e in t["vm"]
                                if e.startswith("SELECTED:JOINT_"))
        if with_trace else 0,
        trace_events=trace_events, work_units=work_units,
        limit_clipped_steps=limit_steps,
        collision_category=env.collision_category if collision else "",
        min_ee_error=(min_error if min_error < float("inf") else -1.0),
        final_ee_error=env.ee_error_at(),
        joint_travel=joint_travel)
    return outcome, traces


def make_task(name: str, factory: Callable[[], RobotChamber], *,
              require_explicit_wait: bool = False,
              forbid_fallback: bool = False,
              max_trace_events: int | None = None,
              max_work_units: int | None = None) -> Task:
    """Build a v1.2.1 ``Task`` whose factory yields a RobotChamber."""
    return Task(name, factory, require_explicit_wait, forbid_fallback,
                max_trace_events, max_work_units)


class JointPrefixOOPS(PrefixOOPS):
    """``PrefixOOPS`` with a joint-space evaluator and joint vocabulary."""

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

    # ------------------------------------------------------------ vocabulary
    def _vocabulary(self, lane: str) -> list[tuple[str, float]]:
        from experiments_v13.joint_dsl import BASE_WEIGHTS_JOINT
        weights = dict(BASE_WEIGHTS_JOINT)
        if self.reuse and self.frozen and lane == "fresh":
            weights["CALL:" + next(reversed(self.frozen))] = 0.27
        total = sum(weights.values())
        return sorted(((name, weight / total) for name, weight in weights.items()),
                      key=lambda row: (-row[1], row[0]))

    # ------------------------------------------------------------- evaluator
    def _evaluate(self, rules: tuple[str, ...], task: Task):
        key = (tuple((pid, p.rules) for pid, p in self.frozen.items()),
               rules, task.name)
        if key not in self._cache:
            outcome, _ = replay(JointProgram(rules), task.factory, self.frozen)
            self.simulated_steps += outcome.steps
            self._cache[key] = (outcome, max(1, outcome.work_units + len(rules)))
            self.real_replay_calls += 1
        return self._cache[key]

    # ------------------------------------------------------------- training
    def train_task(self, task: Task) -> StageResult:
        """Parent logic with JointProgram construction (see module docstring)."""
        targets = [*self.tasks, task]
        old = next(reversed(self.frozen.values())) if self.frozen else None
        self._cache = {}
        self.simulated_steps = 0
        self.reused_steps = 0
        self.real_replay_calls = 0
        if self.reuse and old is not None:
            existing, cost = self._evaluate(old.rules, task)
            if task.accepts(existing):
                self.tasks.append(task)
                result = StageResult(task.name, True, old.id, list(old.rules),
                                     "reused", 0, cost, 1, 1, 1, [])
                self.history.append(result)
                return result
        phases: list[dict] = []
        total_work = 0
        visits = 0
        tested = 0
        phase = self.starting_budget
        while phase <= self.max_phase_budget:
            lanes = ([("extend", phase // 2), ("fresh", phase - phase // 2)]
                     if self.reuse and old else [("fresh", phase)])
            for lane, lane_budget in lanes:
                attempt = self._run_lane(lane, lane_budget, targets, old)
                total_work += attempt["spent"]
                visits += attempt["visits"]
                tested += attempt["tested"]
                phases.append({"phase_budget": phase, "lane": lane,
                               "lane_budget": lane_budget,
                               "work_units": attempt["spent"],
                               "prefix_visits": attempt["visits"],
                               "prefixes_tested": attempt["tested"],
                               "found": attempt["found"] is not None})
                if attempt["found"] is not None:
                    pid = f"P{len(self.frozen) + 1:03d}"
                    prog = JointProgram(attempt["found"], pid)
                    self.frozen[pid] = prog
                    self.tasks.append(task)
                    result = StageResult(task.name, True, pid, list(prog.rules),
                                         lane, phase, total_work,
                                         self.real_replay_calls, tested,
                                         visits, phases)
                    self.history.append(result)
                    return result
            phase *= 2
        result = StageResult(task.name, False, None, [], "none", 0, total_work,
                             self.real_replay_calls, tested, visits, phases)
        self.history.append(result)
        return result

    def save(self, path: str | Path) -> None:
        from dataclasses import asdict
        payload = {
            "format": "robot-oops-joint-0.1",
            "disclaimer": ("Bounded macro-prefix search with replay reset over "
                           "a joint-space DSL; not full original universal "
                           "OOPS or bias optimal."),
            "params": {"reuse": self.reuse, "max_depth": self.max_depth,
                       "starting_budget": self.starting_budget,
                       "max_phase_budget": self.max_phase_budget,
                       "max_prefixes_per_phase": self.max_prefixes_per_phase},
            "programs": [p.to_dict() for p in self.frozen.values()],
            "stages": [asdict(x) for x in self.history],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
