"""v0.6 instruction-granular, reversible bytecode VM for the chamber DSL.

A frame includes compiled instructions, pc, candidate registers, last-action and
caller identity. Snapshots capture the entire continuation, observation and
persistent memory. restore() returns an independent copy: failed descendants
cannot mutate a shared parent. This is a local research VM, NOT universal OOPS Try.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping
from chamber import Observation, Vec3, norm
from oops_language import (Program, VMState, MotionConfig, compile_program,
                           candidate_velocity, predicted_safe, _moving_gap_wait)

@dataclass
class Frame:
    program: Program
    bytecode: tuple[tuple, ...]
    pc: int=0
    candidate: Vec3=(0.,0.,0.)
    safe: bool=False
    last_action: str=''

@dataclass
class VMCheckpoint:
    frames: list[Frame]
    memory: VMState
    trace: list[str]
    observation: Observation
    remaining: int
    output: Vec3|None
    done: bool
    instructions: int

class InstructionVM:
    def __init__(self, frozen: Mapping[str,Program]|None=None,
                 config: MotionConfig=MotionConfig(), instruction_limit:int=128,
                 max_call_depth:int=6):
        self.frozen=dict(frozen or {})
        self.config=config
        self.instruction_limit=instruction_limit
        self.max_call_depth=max_call_depth
        self.frames=[]
        self.memory=VMState()
        self.trace=[]
        self.observation=None
        self.remaining=instruction_limit
        self.output=None
        self.done=False
        self.instructions=0

    def start(self,program:Program,obs:Observation,memory:VMState|None=None):
        self.frames=[Frame(program,compile_program(program))]
        self.memory=deepcopy(memory) if memory is not None else VMState()
        self.trace=[]; self.observation=deepcopy(obs)
        self.remaining=self.instruction_limit; self.output=None
        self.done=False; self.instructions=0
        return self

    def checkpoint(self)->VMCheckpoint:
        return VMCheckpoint(deepcopy(self.frames),deepcopy(self.memory),
                            list(self.trace),deepcopy(self.observation),
                            self.remaining,self.output,self.done,self.instructions)

    def restore(self,snapshot:VMCheckpoint):
        cp=deepcopy(snapshot)
        self.frames=cp.frames; self.memory=cp.memory; self.trace=cp.trace
        self.observation=cp.observation; self.remaining=cp.remaining
        self.output=cp.output; self.done=cp.done; self.instructions=cp.instructions
        return self

    def _return(self, value:Vec3|None):
        self.frames.pop()
        if value is not None:
            self.frames.clear();self.output=value;self.done=True
        elif not self.frames:
            self.output=(0.,0.,0.);self.trace.append('FALLBACK WAIT');self.done=True

    def step_instruction(self):
        if self.done: return None
        if self.remaining<=0:
            self.trace.append('INSTRUCTION LIMIT')
            self.frames.clear();self.output=(0.,0.,0.);self.done=True
            return 'LIMIT'
        f=self.frames[-1]
        inst=f.bytecode[f.pc];op=inst[0]
        self.remaining-=1;self.instructions+=1
        obs=self.observation;mem=self.memory;cfg=self.config
        advance=True
        if op=='AIM':
            f.last_action=inst[1]
            f.candidate=candidate_velocity(inst[1],obs,cfg,mem)
            self.trace.append('AIM '+inst[1])
        elif op=='CHECK_SAFE':
            origin=f.bytecode[f.pc-1][1] if f.pc>0 and f.bytecode[f.pc-1][0]=='AIM' else ''
            undefined=(origin in ('LAST','CONTINUE') and norm(mem.last_direction)<1e-12)
            undefined|=(origin in ('CHOOSE_AXIS','CONTINUE') and norm(f.candidate)<1e-12)
            f.safe=not undefined and predicted_safe(obs,f.candidate,cfg)
            self.trace.append('SAFE' if f.safe else 'BLOCKED')
        elif op=='CHECK_WAIT_GAP':
            f.safe=_moving_gap_wait(obs,cfg)
            self.trace.append('GAP_WAIT_READY' if f.safe else 'GAP_WAIT_NOT_NEEDED')
        elif op=='IF_UNSAFE_SKIP':
            if not f.safe:
                f.pc+=inst[1];advance=False
        elif op=='STORE_DIRECTION':
            if norm(f.candidate)>1e-12:mem.last_direction=f.candidate
        elif op=='EMIT':
            mem.last_mode=f.last_action
            self.trace+=['SELECTED:'+f.last_action,'EMIT']
            self._return(f.candidate);advance=False
        elif op=='EMIT_WAIT':
            mem.last_mode='WAIT_GAP';mem.explicit_waits+=1
            self.trace+=['SELECTED:WAIT_GAP','EMIT_WAIT_GAP']
            self._return((0.,0.,0.));advance=False
        elif op=='CALL':
            child=self.frozen.get(inst[1])
            if child is None:
                self.trace.append('MISSING CALL '+inst[1])
            else:
                self.trace.append('CALL '+inst[1])
                ancestry=[frame.program.id for frame in self.frames]
                if len(self.frames)>=self.max_call_depth:
                    self.trace.append('CALL DEPTH LIMIT')
                elif child.id in ancestry:
                    self.trace.append('RECURSIVE CALL BLOCKED')
                else:
                    f.pc+=1;advance=False
                    self.frames.append(Frame(child,compile_program(child)))
        elif op=='END_NO_MATCH':
            self._return(None);advance=False
        else:raise ValueError('Unknown opcode '+str(op))
        if advance and not self.done:f.pc+=1
        return op

    def run(self,max_instructions:int|None=None):
        count=0
        while not self.done and (max_instructions is None or count<max_instructions):
            self.step_instruction();count+=1
        if self.done:
            self.memory.emissions+=1
            self.memory.last_position=self.observation.position
        return self.output if self.done else None

    def replace_unmatched_tail(self, child:Program):
        """Patch top-level END_NO_MATCH to newly extended bytecode at same PC.

        Only valid when stopped BEFORE top-level END_NO_MATCH and the child
        extends the top-level parent rule tuple. No prior instructions rerun.
        """
        assert not self.done and len(self.frames)==1
        f=self.frames[0]
        assert f.bytecode[f.pc][0]=='END_NO_MATCH'
        assert child.rules[:len(f.program.rules)]==f.program.rules
        f.program=child;f.bytecode=compile_program(child)
        assert f.bytecode[f.pc][0]!='END_NO_MATCH'

class InstructionPrefixBranchSearch:
    """Depth-one executable prefix search at a single observation.

    The shared prefix is run once up to END_NO_MATCH. Each alternative compiled
    continuation resumes from the SAME instruction snapshot. This is a genuine
    instruction-level search primitive, not full curriculum-level OOPS `Try`.
    """
    def __init__(self, frozen=None, config=MotionConfig()):
        self.frozen=dict(frozen or {})
        self.config=config
        self.instructions_executed=0

    def alternatives(self, parent:Program, children:list[Program], obs:Observation,
                     memory:VMState|None=None):
        vm=InstructionVM(self.frozen,self.config).start(parent,obs,memory)
        while not vm.done and not (len(vm.frames)==1 and
            vm.frames[0].bytecode[vm.frames[0].pc][0]=='END_NO_MATCH'):
            vm.step_instruction()
        self.instructions_executed+=vm.instructions
        if vm.done:
            return [(child,vm.output,list(vm.trace),deepcopy(vm.memory)) for child in children]
        snap=vm.checkpoint()
        results=[]
        for child in children:
            vm.restore(snap)
            vm.replace_unmatched_tail(child)
            vm.run()
            self.instructions_executed+=vm.instructions-snap.instructions
            results.append((child,vm.output,list(vm.trace),deepcopy(vm.memory)))
        return results
