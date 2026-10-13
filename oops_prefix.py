"""Research implementation of OOPS-style probability-budgeted *prefix* search.

Compared with v0.3, each DSL-rule prefix is executed/tested as soon as it is
constructed, in depth-first program-space search. The task curriculum is
ordered, discovered programs are frozen, a literal frozen program rule PREFIX
is searched in the extension lane, a separate fresh lane may call previously
frozen programs, and search phases double their per-lane work budgets.

Limits relative to Schmidhuber's 2002 OOPS:
- Search language is a bounded, non-universal macro-DSL (not self-delimiting
  universal instructions or writable search-bias programs).
- Candidate partial programs fall through to implicit wait; no VM instruction
  explicitly requests its next code symbol.
- We restore state by deterministic episode replay, not the original `Try`
  stack-based fine-grained undo of partially executed program states.
- Synthetic VM trace-cost proxy measures allocations, not wall-clock runtime;
  no bias-optimality or hardware real-time guarantee is claimed.
- An active two-lane budget is allocated 50/50 in work units, with an explicit
  global depth/node cap for finite experiments.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import json
from chamber import ChamberEnv
from oops_language import Program
from oops_training import replay, EpisodeResult, extended_curriculum
from oops_continuable import ContinuableEvaluator

BASE_WEIGHTS={
    'GOAL':.26, 'CHOOSE_AXIS':.22, 'WAIT_GAP':.15, 'CONTINUE':.14,
    'Y_POS':.065, 'Y_NEG':.055, 'Z_POS':.045, 'Z_NEG':.035,
    'X_POS':.02, 'X_NEG':.01,
}

@dataclass(frozen=True)
class Task:
    name: str
    factory: Callable[[],ChamberEnv]
    require_explicit_wait: bool=False
    forbid_fallback: bool=False
    max_trace_events: int | None=None
    max_work_units: int | None=None

    def accepts(self,r:EpisodeResult):
        return (r.success and not r.collision and r.interventions==0
            and (not self.require_explicit_wait or r.explicit_waits>0)
            and (not self.forbid_fallback or r.fallback_waits==0)
            and (self.max_trace_events is None or r.trace_events<=self.max_trace_events)
            and (self.max_work_units is None or r.work_units<=self.max_work_units))

@dataclass
class StageResult:
    name:str
    success:bool
    program_id:str|None
    rules:list[str]
    selected_lane:str
    phase_budget:int
    search_work_units:int
    replay_calls:int
    prefixes_tested:int
    total_prefix_visits:int
    phases:list[dict]

class PrefixOOPS:
    """Bounded prefix-search approximation; see module docstring for limits."""
    def __init__(self, *, reuse:bool=True, max_depth:int=4,
                 starting_budget:int=256, max_phase_budget:int=1048576,
                 max_prefixes_per_phase:int=30000, continuable:bool=False):
        self.reuse=reuse
        self.continuable=continuable
        self.continuation=None
        self.simulated_steps=0
        self.reused_steps=0
        self.max_depth=max_depth
        self.starting_budget=starting_budget
        self.max_phase_budget=max_phase_budget
        self.max_prefixes_per_phase=max_prefixes_per_phase
        self.frozen:dict[str,Program]={}
        self.tasks:list[Task]=[]
        self.history:list[StageResult]=[]
        self._cache:dict[tuple,tuple[EpisodeResult,int]]={}
        self.real_replay_calls=0

    def _evaluate(self, rules:tuple[str,...], task:Task) -> tuple[EpisodeResult,int]:
        # Previous frozen library does not change within a search stage.
        # Cache across doubling phases; count virtual work as if replayed.
        key=(tuple((id,p.rules) for id,p in self.frozen.items()),rules,task.name)
        if key not in self._cache:
            if self.continuable:
                if self.continuation is None:
                    self.continuation=ContinuableEvaluator(self.frozen)
                result=self.continuation.evaluate(rules,task.name,task.factory)
                outcome=result.outcome
                self.simulated_steps+=result.simulated_steps
                self.reused_steps+=result.reused_steps
            else:
                outcome,_=replay(Program(rules),task.factory,self.frozen)
                self.simulated_steps+=outcome.steps
            self._cache[key]=(outcome,max(1,outcome.work_units+len(rules)))
            self.real_replay_calls+=1
        return self._cache[key]

    def _vocabulary(self, lane:str) -> list[tuple[str,float]]:
        weights=dict(BASE_WEIGHTS)
        if self.reuse and self.frozen and lane=='fresh':
            # In original OOPS, fresh code can execute earlier frozen code.
            # Here CALL is a bounded DSL instruction for that purpose.
            weights['CALL:'+next(reversed(self.frozen))]=.27
        total=sum(weights.values())
        return sorted(((name,weight/total) for name,weight in weights.items()),
                      key=lambda row:(-row[1],row[0]))

    def _run_lane(self, lane:str, budget:int, stage_tasks:list[Task],
                  prev:Program|None) -> dict:
        """Probability-allocated recursive DFS over *executable prefixes*.

        Crucial: each subtree receives its own bounded portion of its parent
        search budget. An unproductive high-prior GOAL subtree must not starve
        lower-prior branches like CONTINUE. This corrects naive flat DFS with
        one shared budget counter (which is NOT probability-proportional).
        """
        base=prev.rules if lane=='extend' and prev is not None else ()
        if len(base)>self.max_depth:
            return dict(found=None,spent=0,visits=0,tested=0)
        vocab=self._vocabulary(lane)
        visited=0;tested=0
        found=None

        def visit(prefix:tuple[str,...],allocation:float,depth:int) -> float:
            nonlocal visited,tested,found
            if found is not None or visited>=self.max_prefixes_per_phase or allocation<1:
                return 0.0
            visited+=1
            consumed=0.0
            if prefix:
                # Prefixes are executable complete fall-through programs. We
                # test as soon as this partial code has been constructed.
                cost=1;accepted=True
                for task in reversed(stage_tasks):
                    result,execution_cost=self._evaluate(prefix,task)
                    cost+=execution_cost
                    if not task.accepts(result):
                        accepted=False;break
                # A program only qualifies when it can be fully evaluated
                # within its assigned share of search time.
                tested+=1
                consumed=min(float(cost),allocation)
                if accepted and cost<=allocation:
                    found=prefix
                    return consumed
                if cost>allocation:
                    # In a preemptible VM execution would stop at the budget.
                    # Here deterministic replay executes fully and we account
                    # only the allocated fraction. This is an approximation.
                    return consumed
            else:
                consumed=1.0  # root program-generation overhead
            if depth>=self.max_depth-len(base) or found is not None:
                return consumed
            remaining=max(0.0,allocation-consumed)
            # Each child inherits its conditional *fraction* of remaining
            # budget. No branch can consume more than its allocation.
            for token,weight in vocab:
                if found is not None:break
                if token in prefix:continue
                child_allocation=remaining*weight
                if child_allocation<1:continue
                consumed+=visit(prefix+(token,),child_allocation,depth+1)
            return min(allocation,consumed)

        spent=visit(base,float(budget),0)
        return dict(found=found,spent=int(round(spent)),visits=visited,tested=tested)

    def train_task(self, task:Task) -> StageResult:
        targets=[*self.tasks,task]
        old=next(reversed(self.frozen.values())) if self.frozen else None
        self._cache={}
        self.continuation=ContinuableEvaluator(self.frozen) if self.continuable else None
        self.simulated_steps=0
        self.reused_steps=0
        self.real_replay_calls=0
        if self.reuse and old is not None:
            existing,cost=self._evaluate(old.rules,task)
            if task.accepts(existing):
                self.tasks.append(task)
                result=StageResult(task.name,True,old.id,list(old.rules),'reused',0,
                                   cost,1,1,1,[])
                self.history.append(result)
                return result
        phases=[];total_work=0;visits=0;tested=0
        phase=self.starting_budget
        while phase<=self.max_phase_budget:
            # A 50/50 *budget* split, not an equal count of candidate programs.
            lanes=[('extend',phase//2),('fresh',phase-phase//2)] if self.reuse and old else [('fresh',phase)]
            for lane,lane_budget in lanes:
                attempt=self._run_lane(lane,lane_budget,targets,old)
                total_work+=attempt['spent'];visits+=attempt['visits'];tested+=attempt['tested']
                phases.append({'phase_budget':phase,'lane':lane,'lane_budget':lane_budget,
                               'work_units':attempt['spent'],'prefix_visits':attempt['visits'],
                               'prefixes_tested':attempt['tested'],
                               'found':attempt['found'] is not None})
                if attempt['found'] is not None:
                    pid=f'P{len(self.frozen)+1:03d}'
                    prog=Program(attempt['found'],pid)
                    self.frozen[pid]=prog;self.tasks.append(task)
                    result=StageResult(task.name,True,pid,list(prog.rules),lane,phase,
                                       total_work,self.real_replay_calls,tested,visits,phases)
                    self.history.append(result)
                    return result
            phase*=2
        result=StageResult(task.name,False,None,[],'none',0,total_work,
                           self.real_replay_calls,tested,visits,phases)
        self.history.append(result)
        return result

    def save(self,path:str|Path):
        payload={'format':'chamber-prefix-search-0.4',
                 'disclaimer':'Bounded macro-prefix search with replay reset; not full original universal OOPS or bias optimal.',
                 'params':{'reuse':self.reuse,'max_depth':self.max_depth,
                     'starting_budget':self.starting_budget,
                     'max_phase_budget':self.max_phase_budget,
                     'max_prefixes_per_phase':self.max_prefixes_per_phase, 'continuable':self.continuable},
                 'programs':[p.to_dict() for p in self.frozen.values()],
                 'stages':[asdict(x) for x in self.history]}
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2)+'\n')


def standard_tasks():
    return [Task(name,scene,wait,nofallback,events,work)
            for name,scene,wait,nofallback,events,work in extended_curriculum()]


def load_prefix_library(path:str|Path) -> dict[str,Program]:
    """Load v0.4 frozen code, rejecting forward and missing CALL references."""
    obj=json.loads(Path(path).read_text())
    if obj.get('format')!='chamber-prefix-search-0.4':
        raise ValueError('Unsupported prefix library format')
    frozen={}
    for entry in obj['programs']:
        prog=Program(tuple(entry['rules']),entry['id'])
        if prog.id in frozen:
            raise ValueError('Duplicate frozen program ID')
        for token in prog.rules:
            if token.startswith('CALL:') and token[5:] not in frozen:
                raise ValueError('Unknown/forward frozen program reference')
        frozen[prog.id]=prog
    return frozen
