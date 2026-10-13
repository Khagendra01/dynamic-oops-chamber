"""Per-variant synthesis for the v1.2 ablation matrix (experiment C).

``AblationPrefixOOPS`` extends the unchanged ``PrefixOOPS`` search engine with
the single missing switch: whether a candidate program must satisfy *all*
previously seen tasks (cumulative validation, ON) or only the task being
trained right now (OFF).  Every other search mechanism -- prefix DFS,
probability-weighted budget allocation, extend/fresh lanes, frozen library --
is the same code path for every variant.

* ``reuse=False`` disables the extend lane, the ``CALL`` vocabulary entry, and
  the previous-program acceptance shortcut; ordinary program search is intact.
* ``reuse=True`` at deployment additionally means the frozen library is passed
  to the interpreter so ``CALL`` targets resolve.  With ``reuse=False`` the
  deployment library is emptied, so previously discovered code is not callable
  even if a stray rule referenced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping

from chamber import ChamberEnv
from experiments_v12.ablation_matrix import Variant
from oops_language import Program
from oops_prefix import PrefixOOPS, StageResult, Task
from oops_training import EpisodeResult


class AblationPrefixOOPS(PrefixOOPS):
    """``PrefixOOPS`` with an explicit cumulative-validation switch."""

    def __init__(self, *, cumulative: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cumulative = bool(cumulative)

    def train_task(self, task: Task) -> StageResult:
        if self.cumulative:
            return super().train_task(task)
        # Cumulative validation OFF: search sees only the current task.  The
        # accepted-task log still records history for inspection, but it never
        # widens the acceptance test.
        history = self.tasks
        self.tasks = []
        try:
            return super().train_task(task)
        finally:
            discovered = self.tasks
            self.tasks = history
            self.tasks.extend(discovered)


def deployment_library(learner: PrefixOOPS, variant: Variant) -> dict[str, Program]:
    """Frozen code that deployment may call (empty when reuse is OFF)."""
    if not variant.reuse:
        return {}
    return dict(learner.frozen)


def tasks_from_curriculum(curriculum: list[tuple[str, "ScenarioSpec"]]) -> list[Task]:
    return [Task(name, _factory(spec)) for name, spec in curriculum]


def _factory(spec):
    def build() -> ChamberEnv:
        return spec.build()
    build.__name__ = f"scenario_{spec.id}"
    return build


@dataclass
class StageRecord:
    name: str
    success: bool
    program_id: str | None
    rules: list[str]
    selected_lane: str
    phase_budget: int
    search_work_units: int
    replay_calls: int
    prefixes_tested: int
    total_prefix_visits: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class VariantOutcome:
    variant: dict
    library: dict[str, Program]
    deployed_program_id: str | None
    deployed_rules: list[str]
    stages: list[StageRecord] = field(default_factory=list)
    totals: dict = field(default_factory=dict)

    def to_dict(self, include_library: bool = True) -> dict:
        payload = {
            "variant": self.variant,
            "deployed_program_id": self.deployed_program_id,
            "deployed_rules": list(self.deployed_rules),
            "stages": [s.to_dict() for s in self.stages],
            "totals": dict(self.totals),
        }
        if include_library:
            payload["library"] = [p.to_dict() for p in self.library.values()]
        return payload


def _record(stage: StageResult) -> StageRecord:
    return StageRecord(stage.name, stage.success, stage.program_id,
                       list(stage.rules), stage.selected_lane,
                       stage.phase_budget, stage.search_work_units,
                       stage.replay_calls, stage.prefixes_tested,
                       stage.total_prefix_visits)


def train_variant(variant: Variant, curriculum: list[tuple[str, "ScenarioSpec"]],
                  *, max_depth: int = 4, starting_budget: int = 256,
                  max_phase_budget: int = 1048576,
                  max_prefixes_per_phase: int = 30000,
                  continuable: bool = False) -> VariantOutcome:
    """Run one synthesis configuration over the identical curriculum."""
    learner = AblationPrefixOOPS(
        reuse=variant.reuse, cumulative=variant.cumulative,
        max_depth=max_depth, starting_budget=starting_budget,
        max_phase_budget=max_phase_budget,
        max_prefixes_per_phase=max_prefixes_per_phase, continuable=continuable)
    stages: list[StageRecord] = []
    started = time.perf_counter()
    for name, spec in curriculum:
        stage = learner.train_task(Task(name, _factory(spec)))
        stages.append(_record(stage))
        if not stage.success:
            break
    wall = time.perf_counter() - started
    library = dict(learner.frozen)
    deployed = next(reversed(library.values())) if library else None
    outcome = VariantOutcome(
        variant=variant.to_dict(), library=library,
        deployed_program_id=deployed.id if deployed else None,
        deployed_rules=list(deployed.rules) if deployed else [],
        stages=stages,
        totals={
            "wall_time_s": wall,
            "search_work_units": sum(s.search_work_units for s in stages),
            "replay_calls": sum(s.replay_calls for s in stages),
            "prefixes_tested": sum(s.prefixes_tested for s in stages),
            "total_prefix_visits": sum(s.total_prefix_visits for s in stages),
            "simulated_steps": learner.simulated_steps,
            "reused_steps": learner.reused_steps,
            "stages_completed": len(stages),
            "stages_succeeded": sum(int(s.success) for s in stages),
            "final_program_rules": list(deployed.rules) if deployed else [],
        })
    outcome.learner = learner  # type: ignore[attr-defined]
    return outcome


def save_outcome(outcome: VariantOutcome, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"synthesis_{outcome.variant['id']}.json"
    path.write_text(json.dumps(outcome.to_dict(), indent=2) + "\n")
    return path


def program_ids(outcome: VariantOutcome) -> dict[str, str]:
    """Map stage name -> discovered program id (None when a stage failed)."""
    return {stage.name: (stage.program_id or "") for stage in outcome.stages}
