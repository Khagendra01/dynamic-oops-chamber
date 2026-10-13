"""v0.9: lazy, bounded persistent prefix-trie branch reuse.

The reusable search-state belongs to a parent program *prefix*, not to an
individual child candidate. A parent is promoted to a live branch point only
after enough DISTINCT child programs have been evaluated to justify the setup.
After promotion, arbitrary sibling candidates resume from the same exact
instruction boundary. Snapshot and selective undo-log modes share identical
candidate-search order, probability budgets, task acceptance, and frozen code.

This is still a bounded domain-specific OOPS approximation, not universal Try.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from oops_prefix import PrefixOOPS
from oops_language import Program
from oops_transactional_v07 import EpisodeMachine, UndoLog


@dataclass
class PrefixNode:
    machine: EpisodeMachine
    base_steps: int
    base_instructions: int
    snapshot: object | None = None
    journal: UndoLog | None = None
    mark: int = 0


class LazyPrefixCurriculum(PrefixOOPS):
    """Bounded prefix search with lazily materialized shared branch nodes.

    promote_after=2 means evaluate the FIRST TWO distinct siblings by regular
    replay, and create one shared checkpoint only on the THIRD sibling. This
    avoids eager checkpointing in small/low-probability search subtrees.
    """
    def __init__(self, *, backend='undo', promote_after=2, max_nodes=96, max_probe_steps=4, **kwargs):
        if backend not in ('replay','snapshot','undo'):
            raise ValueError('backend must be replay, snapshot or undo')
        if promote_after < 0 or max_nodes < 1 or max_probe_steps < 1:
            raise ValueError('promote_after must be >=0 and max_nodes >=1')
        super().__init__(**kwargs)
        self.backend=backend
        self.promote_after=int(promote_after)
        self.max_nodes=int(max_nodes)
        self.max_probe_steps=int(max_probe_steps)
        self.nodes:OrderedDict[tuple,PrefixNode|None]=OrderedDict()
        self.siblings_seen=defaultdict(set)
        self.vm_instructions=0
        self.physics_steps=0
        self.branch_hits=0
        self.created_nodes=0
        self.unusable_nodes=0
        self.node_evictions=0
        self.peak_nodes=0
        self.snapshot_restores=0
        self.undo_rollbacks=0
        self.extra_parent_physics=0
        self.extra_parent_instructions=0

    def _key(self, task, rules):
        # task's factory identity disambiguates scenes sharing human-readable names
        return (task.name,id(task.factory),tuple(rules))

    def _remember(self,key,node):
        self.nodes[key]=node
        self.nodes.move_to_end(key)
        while len(self.nodes)>self.max_nodes:
            self.nodes.popitem(last=False)
            self.node_evictions+=1
        self.peak_nodes=max(self.peak_nodes,len(self.nodes))

    def _make_node(self,parent,task):
        """Execute a shared prefix once, stopping before fallthrough bytecode."""
        runner=EpisodeMachine(task.factory,self.frozen)
        status=runner.advance(Program(parent),stop_on_fallthrough=True,probe_steps=self.max_probe_steps)
        self.physics_steps+=runner.stats.executed_steps
        self.vm_instructions+=runner.stats.executed_instructions
        self.extra_parent_physics+=runner.stats.executed_steps
        self.extra_parent_instructions+=runner.stats.executed_instructions
        if status!='branch':
            self.unusable_nodes+=1
            return None
        if self.backend=='snapshot':
            node=PrefixNode(runner,runner.stats.executed_steps,
                            runner.stats.executed_instructions,snapshot=runner.snapshot())
        else:
            node=PrefixNode(runner,runner.stats.executed_steps,
                            runner.stats.executed_instructions,journal=UndoLog())
        self.created_nodes+=1
        return node

    def _run_from_node(self,node,child):
        runner=node.machine
        if self.backend=='snapshot':
            runner.restore(node.snapshot)
            self.snapshot_restores+=1
            runner.vm.replace_unmatched_tail(Program(child))
            runner.advance(Program(child))
        else:
            node.journal.rollback(runner,node.mark)
            self.undo_rollbacks+=1
            node.journal.before(runner)  # mutation of compiled frame instructions
            runner.vm.replace_unmatched_tail(Program(child))
            runner.advance(Program(child),journal=node.journal)
        self.branch_hits+=1
        self.physics_steps+=runner.stats.executed_steps-node.base_steps
        self.vm_instructions+=runner.stats.executed_instructions-node.base_instructions
        return runner.outcome()

    def _evaluate(self,rules,task):
        rules=tuple(rules)
        cachekey=(tuple((k,p.rules) for k,p in self.frozen.items()),rules,task.name,id(task.factory))
        if cachekey in self._cache:
            return self._cache[cachekey]
        parent=rules[:-1]
        node=None
        if self.backend!='replay' and parent:
            key=self._key(task,parent)
            if key in self.nodes:
                self.nodes.move_to_end(key)
                node=self.nodes[key]
            else:
                observed=self.siblings_seen[key]
                observed.add(rules[-1])
                if len(observed)>self.promote_after:
                    node=self._make_node(parent,task)
                    self._remember(key,node)
        if node is None:
            runner=EpisodeMachine(task.factory,self.frozen)
            runner.advance(Program(rules))
            result=runner.outcome()
            self.physics_steps+=runner.stats.executed_steps
            self.vm_instructions+=runner.stats.executed_instructions
        else:
            result=self._run_from_node(node,rules)
        self.real_replay_calls+=1
        self._cache[cachekey]=(result,max(1,result.work_units+len(rules)))
        return self._cache[cachekey]

    def train_task(self,task):
        # No state from a previous frozen library is allowed in the next stage.
        self.nodes.clear()
        self.siblings_seen.clear()
        result=super().train_task(task)
        self.nodes.clear()
        self.siblings_seen.clear()
        return result
