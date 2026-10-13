"""v0.7: Multi-step instruction-level continuation, with two rollback engines.

The prefix branch point is top-level END_NO_MATCH, before its implicit WAIT.
Snapshot backend copies the whole state. Journal backend records mutable-field
preimages for each instruction/physics update (not a universal OOPS Try).
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from chamber import norm, sub
from oops_language import Program, MotionConfig, VMState
from oops_instruction_vm import InstructionVM
from oops_training import EpisodeResult, replay

@dataclass
class Counters:
    interventions:int=0
    collision:bool=False
    fallback:int=0
    continued:int=0
    directional:int=0
    events:int=0
    work:int=0
    executed_steps:int=0
    executed_instructions:int=0


def _cost(trace):
    return sum(8 if s=='AIM CHOOSE_AXIS' else 4 if s in
               ('AIM GOAL','GAP_WAIT_READY','GAP_WAIT_NOT_NEEDED') else 1 for s in trace)

class EpisodeMachine:
    def __init__(self, factory, frozen=None):
        self.env=factory()
        self.frozen=dict(frozen or {})
        self.cfg=MotionConfig(radius=self.env.radius,dt=self.env.dt,max_speed=self.env.max_speed)
        self.mem=VMState()
        self.vm=None
        self.stats=Counters()

    def _start(self,program):
        self.vm=InstructionVM(self.frozen,self.cfg).start(program,self.env.observe(),self.mem)

    def _finish_decision(self):
        assert self.vm.done
        action=self.vm.output
        trace=self.vm.trace
        self.stats.events+=len(trace)
        self.stats.work+=_cost(trace)
        self.stats.fallback+=int(trace[-1]=='FALLBACK WAIT')
        self.stats.continued+=int('SELECTED:CONTINUE' in trace)
        self.stats.directional+=int('SELECTED:CHOOSE_AXIS' in trace)
        self.mem=deepcopy(self.vm.memory)
        outcome=self.env.step(action)
        self.stats.interventions+=int(outcome.safety_intervention)
        self.stats.collision|=outcome.collision
        self.stats.executed_steps+=1
        self.vm=None

    def advance(self, program, stop_on_fallthrough=False, journal=None, probe_steps=None):
        """Advance complete multi-step episode; optionally stop inside current bytecode."""
        while not self.env.done:
            if stop_on_fallthrough and probe_steps is not None and self.stats.executed_steps>=probe_steps:
                return 'probe_limit'
            if self.vm is None:
                if journal: journal.before(self)
                self._start(program)
            while not self.vm.done:
                f=self.vm.frames[-1]
                if (stop_on_fallthrough and len(self.vm.frames)==1 and
                    f.bytecode[f.pc][0]=='END_NO_MATCH'):
                    return 'branch'
                if journal: journal.before(self)
                self.vm.step_instruction()
                self.stats.executed_instructions+=1
            if journal: journal.before(self)
            self.vm.run(max_instructions=0) # finalizes per-observation memory fields
            self._finish_decision()
        return 'done'

    def outcome(self):
        env=self.env; s=self.stats
        return EpisodeResult(not s.collision and norm(sub(env.position,env.goal))<=env.goal_tolerance,
                             s.collision,s.interventions,env.step_index,s.fallback,
                             self.mem.explicit_waits,s.continued,s.directional,s.events,s.work)

    def snapshot(self):
        return deepcopy((self.env,self.mem,self.vm,self.stats))

    def restore(self,cp):
        self.env,self.mem,self.vm,self.stats=deepcopy(cp)

class UndoLog:
    """Selective reversible write-ahead preimages. No chamber or VM deep copy.

    Captures field values that may be changed at an instruction/physics step;
    frame object metadata is saved shallowly, since frame fields are reset by
    recorded scalar values. Newly pushed frames are removed by restoring list.
    Immutable observation and compiled bytecode are shared.
    """
    def __init__(self):self.entries=[];self.writes=0
    def mark(self):return len(self.entries)
    def before(self,m):
        vm=m.vm
        oldvm=vm
        if vm is None:
            v=None
        else:
            v=(tuple((f, f.program,f.bytecode,f.pc,f.candidate,f.safe,f.last_action) for f in vm.frames),
               tuple(vm.frames),vm.memory.last_direction,vm.memory.emissions,
               vm.memory.last_position,vm.memory.last_mode,vm.memory.explicit_waits,
               vm.remaining,vm.output,vm.done,vm.instructions,len(vm.trace))
        mem=m.mem
        values=(m.env.position,m.env.step_index,m.env.done,
                mem,mem.last_direction,mem.emissions,mem.last_position,mem.last_mode,mem.explicit_waits,
                m.stats,tuple(vars(m.stats).items()),oldvm,v)
        self.entries.append(values);self.writes+=1
    def rollback(self,m,mark):
        while len(self.entries)>mark:
            (pos,step,done,mem,ld,emis,lp,lm,ew,stats,sv,oldvm,v)=self.entries.pop()
            m.env.position=pos;m.env.step_index=step;m.env.done=done
            m.mem=mem;mem.last_direction=ld;mem.emissions=emis
            mem.last_position=lp;mem.last_mode=lm;mem.explicit_waits=ew
            m.stats=stats
            for k,val in sv:setattr(stats,k,val)
            m.vm=oldvm
            if v is not None:
                frames,old_frames,direction,emissions,last_pos,last_mode,waits,remaining,output,finished,instructions,tlen=v
                oldvm.frames=list(old_frames)
                for f,p,code,pc,candidate,safe,action in frames:
                    f.program=p;f.bytecode=code;f.pc=pc;f.candidate=candidate
                    f.safe=safe;f.last_action=action
                oldvm.memory.last_direction=direction
                oldvm.memory.emissions=emissions
                oldvm.memory.last_position=last_pos
                oldvm.memory.last_mode=last_mode
                oldvm.memory.explicit_waits=waits
                oldvm.remaining=remaining;oldvm.output=output;oldvm.done=finished
                oldvm.instructions=instructions;del oldvm.trace[tlen:]

def branch_compare(parent, children, factory, frozen=None, mode='snapshot'):
    """Replay OR checkpoint at first fall-through and finish entire episode."""
    beginning=perf_counter();results=[];instructions=0;steps=0
    journal=UndoLog() if mode=='undo' else None
    if mode=='replay':
        for child in children:
            machine=EpisodeMachine(factory,frozen)
            machine.advance(child)
            results.append(machine.outcome())
            instructions+=machine.stats.executed_instructions
            steps+=machine.stats.executed_steps
    else:
        machine=EpisodeMachine(factory,frozen)
        parent_stop=machine.advance(parent,True,journal)
        if parent_stop=='branch':
            prefix_instructions=machine.stats.executed_instructions
            prefix_steps=machine.stats.executed_steps
            cp=machine.snapshot() if mode=='snapshot' else None
            mark=journal.mark() if journal else None
            for child in children:
                if mode=='snapshot':machine.restore(cp)
                else:journal.rollback(machine,mark)
                if journal:journal.before(machine)
                machine.vm.replace_unmatched_tail(child)
                machine.advance(child,journal=journal)
                results.append(machine.outcome())
                instructions+=machine.stats.executed_instructions-prefix_instructions
                steps+=machine.stats.executed_steps-prefix_steps
            instructions+=prefix_instructions;steps+=prefix_steps
        else:
            results=[machine.outcome() for _ in children]
            instructions=machine.stats.executed_instructions
            steps=machine.stats.executed_steps
    return {'mode':mode,'results':results,'instructions':instructions,'steps':steps,
            'seconds':perf_counter()-beginning,
            'journal_entries':journal.writes if journal else 0}
