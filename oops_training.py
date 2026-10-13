"""Bounded incremental synthesis for the chamber DSL.

Not the full original OOPS algorithm. This v0.2 prototype keeps these useful
properties: probability-ranked program candidates, ordered tasks, deterministic
replay, frozen prior programs, explicit code calls, and distinct extension/fresh
search lanes. It does NOT implement OOPS's full prefix-testing machine or
bias-optimality theorem. Those remain future research work.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Callable

from chamber import ChamberEnv, example_scene
from oops_language import Program, MotionConfig, Interpreter, VMState, ACTIONS

# Priors affect search ordering, not trajectory collision probability.
PROBABILITIES = {"GOAL": .35, "Y_POS": .13, "Y_NEG": .10,
                 "Z_POS": .10, "Z_NEG": .08, "X_POS": .07,
                 "X_NEG": .07, "LAST": .10,
                 "CHOOSE_AXIS": .16, "CONTINUE": .17, "WAIT_GAP": .12}


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    collision: bool
    interventions: int
    steps: int
    fallback_waits: int
    explicit_waits: int = 0
    continued_actions: int = 0
    directional_choices: int = 0
    trace_events: int = 0
    work_units: int = 0


def replay(program: Program, make_scene: Callable[[], ChamberEnv],
           library: dict[str, Program] | None = None,
           with_trace: bool = False) -> tuple[EpisodeResult, list[dict]]:
    env = make_scene()
    config = MotionConfig(radius=env.radius, dt=env.dt,
                          max_speed=env.max_speed)
    vm = Interpreter(frozen=library, config=config)
    mem = VMState()
    obs = env.reset()
    traces = []
    interventions = 0
    waits = 0
    collision = False
    continued = 0
    directional = 0
    trace_events = 0
    work_units = 0
    for _ in range(env.max_steps):
        if env.done:
            break
        action, instruction_trace = vm.act(program, obs, mem)
        trace_events += len(instruction_trace)
        # Deliberately coarse, deterministic search-cost proxy. NOT latency.
        work_units += sum(8 if x == 'AIM CHOOSE_AXIS'
                          else 4 if x in ('AIM GOAL','GAP_WAIT_READY','GAP_WAIT_NOT_NEEDED')
                          else 1 for x in instruction_trace)
        waits += int(instruction_trace[-1] == "FALLBACK WAIT")
        continued += int("SELECTED:CONTINUE" in instruction_trace)
        directional += int("SELECTED:CHOOSE_AXIS" in instruction_trace)
        result = env.step(action)
        interventions += int(result.safety_intervention)
        collision |= result.collision
        if with_trace:
            traces.append({"step": obs.step_index, "position": list(obs.position),
                           "velocity": list(action), "vm": instruction_trace,
                           "new_position": list(result.observation.position),
                           "collision": result.collision,
                           "shield_intervention": result.safety_intervention})
        obs = result.observation
        if result.done:
            break
    success = env.done and not collision and (sum((obs.position[i]-env.goal[i])**2
                                                      for i in range(3))**.5 <= env.goal_tolerance)
    return EpisodeResult(success, collision, interventions,
                         env.step_index, waits, mem.explicit_waits, continued,
                         directional, trace_events, work_units), traces


def _candidates(max_rules: int, frozen_id: str | None = None):
    """Deterministic probability priority; exhaustive only up to max_rules."""
    vocabulary = list(ACTIONS)
    if frozen_id is not None:
        vocabulary.insert(0, "CALL:" + frozen_id)
    scored: list[tuple[float, tuple[str,...]]] = []
    for n in range(1, max_rules+1):
        for rules in itertools.product(vocabulary, repeat=n):
            # Avoid unproductive repetitions, recursive calls, and trivial no-ops.
            if len(set(rules)) < len(rules):
                continue
            score = 1.0
            for r in rules:
                score *= (.45 if r.startswith("CALL:") else PROBABILITIES[r])
            scored.append((score, rules))
    scored.sort(key=lambda x:(-x[0],len(x[1]),x[1]))
    for probability, rules in scored:
        yield Program(rules), probability


@dataclass
class Discovery:
    stage: str
    program: Program | None
    tested: int
    extension_tested: int
    fresh_tested: int
    replayed_episodes: int
    found_lane: str


class IncrementalSynthesizer:
    """Searches an ordered curriculum and freezes programs that solve all tasks.

    Half the candidate search budget is assigned to programs that CALL previous
    frozen code, and half to fresh candidates. This is a bounded approximation,
    NOT the literal OOPS 50/50 allocation of execution time to prefix searches.
    """
    def __init__(self, max_rules: int = 3, budget_per_stage: int = 2000):
        self.max_rules = max_rules
        self.budget_per_stage = budget_per_stage
        self.frozen: dict[str,Program] = {}
        self.tasks: list[Callable[[],ChamberEnv]] = []
        self.requirements: list[tuple[bool,bool,int | None,int | None]] = []
        # (explicit wait, no fallback, maximum trace events, max work units)
        self.discoveries: list[Discovery] = []

    def train_stage(self, name: str, scene_factory: Callable[[], ChamberEnv],
                    *, require_explicit_wait: bool = False,
                    forbid_fallback: bool = False,
                    max_trace_events: int | None = None,
                    max_work_units: int | None = None) -> Discovery:
        tasks = [*self.tasks, scene_factory]
        requirements = [*self.requirements,(require_explicit_wait,forbid_fallback,
                                            max_trace_events,max_work_units)]
        def acceptable(result, req):
            return (result.success and not result.collision and result.interventions == 0
                    and (not req[0] or result.explicit_waits > 0)
                    and (not req[1] or result.fallback_waits == 0)
                    and (req[2] is None or result.trace_events <= req[2])
                    and (req[3] is None or result.work_units <= req[3]))
        latest_id = next(reversed(self.frozen)) if self.frozen else None
        if latest_id is not None:
            # Don't "learn" a redundant CALL wrapper when old frozen code
            # already handles the new task. Still retain the task in curriculum.
            existing=self.frozen[latest_id]
            prior,_=replay(existing,scene_factory,self.frozen)
            if acceptable(prior,requirements[-1]):
                self.tasks.append(scene_factory)
                self.requirements.append(requirements[-1])
                discovery=Discovery(name,existing,0,0,0,1,'reused')
                self.discoveries.append(discovery)
                return discovery
        lanes: list[tuple[str, object, int]] = []
        if latest_id is not None:
            extension = ((p,w) for p,w in _candidates(self.max_rules, latest_id)
                         if any(rule.startswith('CALL:') for rule in p.rules))
            lanes = [('extend', iter(extension), self.budget_per_stage // 2),
                     ('fresh', iter(_candidates(self.max_rules)), self.budget_per_stage - self.budget_per_stage//2)]
        else:
            lanes = [('fresh', iter(_candidates(self.max_rules)), self.budget_per_stage)]
        tested=0; extension_tested=0; fresh_tested=0; episodes=0
        # Interleave the lanes so both receive opportunities, keeping deterministic order.
        counters = {label:0 for label,_,_ in lanes}
        while True:
            active = False
            for label, generator, quota in lanes:
                if counters[label] >= quota:
                    continue
                try:
                    candidate, _p = next(generator)
                except StopIteration:
                    counters[label] = quota
                    continue
                active=True
                counters[label] += 1
                tested += 1
                if label == 'extend': extension_tested+=1
                else: fresh_tested+=1
                if any(rule.startswith('CALL:') and rule.split(':',1)[1] not in self.frozen
                       for rule in candidate.rules):
                    continue
                # Each candidate must solve EVERY stage in the cumulative task set.
                # The newest environment is tested first for fast pruning.
                accepted=True
                for task, requirement in reversed(list(zip(tasks, requirements))):
                    result,_ = replay(candidate,task,self.frozen)
                    episodes += 1
                    if not acceptable(result,requirement):
                        accepted=False
                        break
                if accepted:
                    pid=f'P{len(self.frozen)+1:03d}'
                    found=Program(candidate.rules,pid)
                    self.frozen[pid]=found
                    self.tasks.append(scene_factory)
                    self.requirements.append(requirements[-1])
                    discovery=Discovery(name,found,tested,extension_tested,
                                        fresh_tested,episodes,label)
                    self.discoveries.append(discovery)
                    return discovery
            if not active:
                break
        discovery=Discovery(name,None,tested,extension_tested,fresh_tested,episodes,'none')
        self.discoveries.append(discovery)
        # Crucial: don't pretend to have learned an unsolved stage.
        return discovery

    def save(self, path: str | Path):
        data={"format":"chamber-dsl-0.3", "limits":{"max_rules":self.max_rules,
              "budget_per_stage":self.budget_per_stage},
              "frozen":[p.to_dict() for p in self.frozen.values()],
              "stages":[{"stage":d.stage, "program_id":d.program.id if d.program else None,
                         "tested":d.tested,"extension_tested":d.extension_tested,
                         "fresh_tested":d.fresh_tested,
                         "replayed_episodes":d.replayed_episodes,
                         "found_lane":d.found_lane} for d in self.discoveries]}
        Path(path).write_text(json.dumps(data,indent=2)+"\n",encoding='utf8')


def curriculum():
    """Backwards-compatible core curriculum; no explicit-wait criterion."""
    return [
      ('empty', lambda: example_scene('empty')),
      ('static', lambda: example_scene('static')),
      ('crossing', lambda: example_scene('crossing')),
      ('moving', lambda: example_scene('moving')),
    ]


def extended_curriculum():
    """Stages: name, factory, explicit-wait, no-fallback, max trace, max work."""
    from oops_scenarios import north_blocked, south_blocked, moving_gate
    return [
        ('empty', lambda: example_scene('empty'), False, False, None, None),
        ('static', lambda: example_scene('static'), False, False, None, None),
        ('north_blocked', north_blocked, False, False, None, None),
        ('south_blocked', south_blocked, False, False, None, None),
        ('crossing', lambda: example_scene('crossing'), True, True, None, None),
        ('moving_gate', moving_gate, True, True, None, None),
        ('moving', lambda: example_scene('moving'), False, False, None, None),
        # Efficiency target, not a mandated CONTINUE token. Synthesis is free
        # to use any program that reduces the measured VM trace cost.
        ('efficient_empty', lambda: example_scene('empty'), False, False, None, 70),
    ]


def load_library(path: str | Path) -> dict[str, Program]:
    """Load frozen bytecode definitions and reject unknown/cyclic references."""
    data=json.loads(Path(path).read_text(encoding='utf8'))
    if data.get('format') not in ('chamber-dsl-0.2', 'chamber-dsl-0.3'):
        raise ValueError('Not a compatible frozen program library')
    library={}
    for item in data['frozen']:
        program=Program(tuple(item['rules']),item['id'])
        if program.id in library:
            raise ValueError('Duplicate program ID')
        for rule in program.rules:
            if rule.startswith('CALL:') and rule.split(':',1)[1] not in library:
                raise ValueError('Unknown or forward frozen program reference')
        library[program.id]=program
    return library
