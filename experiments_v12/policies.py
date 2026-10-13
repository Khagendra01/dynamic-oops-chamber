"""Deployment policies for experiment series v1.2.

``ProgramPolicy`` executes a frozen program with the *unchanged* interpreter,
so conditional geometric branching (``CHECK_SAFE``, ``IF_UNSAFE_SKIP``,
``CHECK_WAIT_GAP``) is identical in every ablation variant.  It differs from
``oops_training.replay`` in one respect that the ablation matrix needs: the
persistent ``VMState`` can be reset at every observation (variant ``memory``
OFF), which clears user-defined interpreter memory only -- the physical state,
the sensor observation and the environment dynamics are untouched.

Every policy re-decides from the current observation each control period
("live replanning" is a controlled constant in the C-series).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from chamber import Observation, Vec3
from experiments_v12.ablation_matrix import Variant
from oops_language import Interpreter, MotionConfig, Program, VMState


@dataclass
class PolicyCounters:
    steps: int = 0
    fallback_waits: int = 0
    explicit_waits: int = 0
    continued_actions: int = 0
    directional_choices: int = 0
    trace_events: int = 0
    work_units: int = 0
    zero_commands: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def trace_cost(trace: list[str]) -> int:
    """Deterministic search-cost proxy used by the v0.x studies (NOT latency)."""
    return sum(8 if x == "AIM CHOOSE_AXIS"
               else 4 if x in ("AIM GOAL", "GAP_WAIT_READY", "GAP_WAIT_NOT_NEEDED")
               else 1 for x in trace)


class ProgramPolicy:
    """Run one frozen program against live observations."""

    name = "program"

    def __init__(self, program: Program,
                 library: Mapping[str, Program] | None = None, *,
                 config: MotionConfig, persistent_memory: bool = True,
                 instruction_limit: int = 128):
        self.program = program
        self.library = dict(library or {})
        self.config = config
        self.persistent_memory = bool(persistent_memory)
        self.vm = Interpreter(frozen=self.library, config=config,
                              instruction_limit=instruction_limit)
        self.memory = VMState()
        self.counters = PolicyCounters()

    def reset(self) -> None:
        self.memory = VMState()
        self.counters = PolicyCounters()

    def act(self, obs: Observation) -> Vec3:
        if not self.persistent_memory:
            # User-defined interpreter memory only. The observation itself and
            # the physical state are never reset here.
            self.memory = VMState()
        velocity, trace = self.vm.act(self.program, obs, self.memory)
        self.counters.steps += 1
        self.counters.trace_events += len(trace)
        self.counters.work_units += trace_cost(trace)
        self.counters.fallback_waits += int(trace[-1] == "FALLBACK WAIT")
        self.counters.explicit_waits += int("EMIT_WAIT_GAP" in trace)
        self.counters.continued_actions += int("SELECTED:CONTINUE" in trace)
        self.counters.directional_choices += int("SELECTED:CHOOSE_AXIS" in trace)
        self.counters.zero_commands += int(
            velocity[0] == 0.0 and velocity[1] == 0.0 and velocity[2] == 0.0)
        return velocity


def make_policy(variant: Variant, program: Program,
                library: Mapping[str, Program] | None, config: MotionConfig,
                **kwargs) -> ProgramPolicy:
    """Build the deployment policy implied by a variant's bit vector."""
    return ProgramPolicy(program, library if variant.reuse else {},
                         config=config, persistent_memory=variant.memory,
                         **kwargs)


def full_system_policy(program: Program,
                       library: Mapping[str, Program],
                       config: MotionConfig) -> ProgramPolicy:
    """Convenience constructor for the C5 reference system."""
    from experiments_v12.ablation_matrix import C5
    return make_policy(C5, program, library, config)
