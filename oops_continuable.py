"""v0.5 transactional, continuable rule-prefix episode evaluator.

An extension of a fall-through rule program differs from its parent ONLY at the
first END_NO_MATCH. We checkpoint the exact chamber + VM + counters immediately
before this instruction, then resume a child program from the checkpoint.
No earlier environment steps are replayed. Rejected branches leave the parent
checkpoint untouched. This is exact under the current deterministic, pure
observation/action DSL; it is *not* the original OOPS universal Try VM.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Mapping
from chamber import ChamberEnv, norm, sub
from oops_language import Program, VMState, MotionConfig, Interpreter
from oops_training import EpisodeResult, replay

@dataclass
class Checkpoint:
    env: ChamberEnv
    mem: VMState
    interventions: int = 0
    collisions: bool = False
    waits: int = 0
    continued: int = 0
    directional: int = 0
    trace_events: int = 0
    work_units: int = 0

@dataclass
class ContinuationResult:
    outcome: EpisodeResult
    first_fallback: Checkpoint | None
    simulated_steps: int
    reused_steps: int


def _work(trace):
    return sum(8 if x == 'AIM CHOOSE_AXIS' else 4 if x in
               ('AIM GOAL','GAP_WAIT_READY','GAP_WAIT_NOT_NEEDED') else 1 for x in trace)


class ContinuableEvaluator:
    def __init__(self, library: Mapping[str,Program]|None=None):
        self.library=dict(library or {})
        self.checkpoints: dict[tuple[str,tuple[str,...]],Checkpoint|None]={}
        self.outcomes: dict[tuple[str,tuple[str,...]],EpisodeResult]={}
        self.steps_executed=0
        self.steps_reused=0

    def evaluate(self, rules:tuple[str,...], name:str,
                 factory:Callable[[],ChamberEnv]):
        # Use longest already evaluated proper ancestor; cached checkpoint is
        # copied so neither child nor failed branch can mutate it.
        ancestor=None
        for depth in range(len(rules)-1,0,-1):
            key=(name,rules[:depth])
            if key in self.checkpoints:
                ancestor=rules[:depth]
                break
        if ancestor is not None:
            prefix_checkpoint=self.checkpoints[(name,ancestor)]
            if prefix_checkpoint is None:
                # Parent never fell through; suffix cannot affect the episode.
                parent_result=self.outcomes[(name,ancestor)]
                self.checkpoints[(name,rules)]=None
                self.outcomes[(name,rules)]=parent_result
                return ContinuationResult(parent_result,None,0,0)
            state=deepcopy(prefix_checkpoint)
            reused=state.env.step_index
        else:
            state=Checkpoint(factory(),VMState())
            reused=0
        env=state.env
        config=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed)
        vm=Interpreter(frozen=self.library,config=config)
        program=Program(rules)
        first_fallback=None
        simulated=0
        while not env.done:
            obs=env.observe()
            # Snapshot BEFORE executing the earliest fall-through instruction.
            # A proper child shares the same state before this point.
            pre=deepcopy(state)
            action,trace=vm.act(program,obs,state.mem)
            if first_fallback is None and trace[-1]=='FALLBACK WAIT':
                first_fallback=pre
            state.trace_events+=len(trace)
            state.work_units+=_work(trace)
            state.waits+=int(trace[-1]=='FALLBACK WAIT')
            state.continued+=int('SELECTED:CONTINUE' in trace)
            state.directional+=int('SELECTED:CHOOSE_AXIS' in trace)
            result=env.step(action)
            state.interventions+=int(result.safety_intervention)
            state.collisions|=result.collision
            simulated+=1
        obs=env.observe()
        success=not state.collisions and norm(sub(obs.position,env.goal))<=env.goal_tolerance
        outcome=EpisodeResult(success,state.collisions,state.interventions,
                       env.step_index,state.waits,state.mem.explicit_waits,
                       state.continued,state.directional,state.trace_events,state.work_units)
        self.checkpoints[(name,rules)]=first_fallback
        self.outcomes[(name,rules)]=outcome
        self.steps_executed+=simulated
        self.steps_reused+=reused
        return ContinuationResult(outcome,first_fallback,simulated,reused)
