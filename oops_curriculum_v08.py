"""v0.8 integration of v0.7 instruction-boundary branching with ordered curriculum.

This retains the bounded macro-prefix OOPS approximation in oops_prefix.py.
Search allocations, two lanes, and freezing are inherited unchanged.
Per-task branch checkpoints are stored at unmatched END_NO_MATCH; sibling
candidate programs roll back to this exact state and resume the episode.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from oops_prefix import PrefixOOPS
from oops_language import Program
from oops_transactional_v07 import EpisodeMachine, UndoLog

@dataclass
class Branch:
    machine: EpisodeMachine
    checkpoint: object | None
    journal: UndoLog | None
    mark: int | None
    base_steps: int
    base_instructions: int
    terminated: bool

class TransactionalCurriculum(PrefixOOPS):
    def __init__(self, *, backend='undo', **kwargs):
        if backend not in ('undo','snapshot','replay'):
            raise ValueError(backend)
        super().__init__(**kwargs)
        self.backend=backend
        self.branch_cache={}
        self.physics_steps=0
        self.vm_instructions=0
        self.branch_hits=0
        self.snapshot_restores=0
        self.undo_rollbacks=0

    def _branch(self, parent, task):
        key=(task.name,parent)
        if key not in self.branch_cache:
            machine=EpisodeMachine(task.factory,self.frozen)
            journal=UndoLog() if self.backend=='undo' else None
            status=machine.advance(Program(parent),True,journal)
            self.physics_steps+=machine.stats.executed_steps
            self.vm_instructions+=machine.stats.executed_instructions
            self.branch_cache[key]=Branch(machine,
                machine.snapshot() if self.backend=='snapshot' and status=='branch' else None,
                journal,journal.mark() if journal and status=='branch' else None,
                machine.stats.executed_steps,machine.stats.executed_instructions,
                status!='branch')
        return self.branch_cache[key]

    def _evaluate(self,rules,task):
        rules=tuple(rules)
        key=(tuple((k,p.rules) for k,p in self.frozen.items()),rules,task.name)
        if key in self._cache:return self._cache[key]
        if self.backend=='replay' or len(rules)<=1:
            machine=EpisodeMachine(task.factory,self.frozen)
            machine.advance(Program(rules))
            result=machine.outcome()
            self.physics_steps+=machine.stats.executed_steps
            self.vm_instructions+=machine.stats.executed_instructions
        else:
            # Longest strict parent provides maximal shared execution.
            parent=rules[:-1]
            b=self._branch(parent,task)
            if b.terminated:
                # Parent emitted an action before the tail; evaluate independently.
                machine=EpisodeMachine(task.factory,self.frozen)
                machine.advance(Program(rules))
                result=machine.outcome()
                self.physics_steps+=machine.stats.executed_steps
                self.vm_instructions+=machine.stats.executed_instructions
            else:
                self.branch_hits+=1
                machine=b.machine
                if self.backend=='snapshot':
                    machine.restore(b.checkpoint)
                    self.snapshot_restores+=1
                else:
                    b.journal.rollback(machine,b.mark)
                    self.undo_rollbacks+=1
                    b.journal.before(machine)
                machine.vm.replace_unmatched_tail(Program(rules))
                machine.advance(Program(rules),journal=b.journal if self.backend=='undo' else None)
                result=machine.outcome()
                self.physics_steps+=machine.stats.executed_steps-b.base_steps
                self.vm_instructions+=machine.stats.executed_instructions-b.base_instructions
        self._cache[key]=(result,max(1,result.work_units+len(rules)))
        self.real_replay_calls+=1
        return self._cache[key]

    def train_task(self,task):
        self.branch_cache.clear()
        result=super().train_task(task)
        self.branch_cache.clear()
        return result
