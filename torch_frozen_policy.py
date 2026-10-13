"""Batched frozen-DSL program execution using torch geometric operations.

Only existing frozen *rule* programs are executed; no new OOPS code is
synthesized by this module. CALL is flattened with recursion checks; the
instruction semantics mirror oops_language.py, and correctness is checked
against CPU Interpreter on held-out deterministic episodes.
"""
from __future__ import annotations
import torch
from chamber import Vec3
from oops_language import Program
from torch_chamber import TorchChamberBatch

DIRECTIONS={
    'X_NEG':(-1,0,0),'X_POS':(1,0,0),
    'Y_NEG':(0,-1,0),'Y_POS':(0,1,0),
    'Z_NEG':(0,0,-1),'Z_POS':(0,0,1),
}

class TorchFrozenPolicy:
    def __init__(self,env:TorchChamberBatch, program:Program, frozen:dict[str,Program],
                 stride:float=.12, max_depth:int=6):
        self.env=env
        self.stride=stride
        self.rules=self._flatten(program,frozen,(),max_depth)
        self.last_direction=torch.zeros((env.B,3),dtype=env.dtype,device=env.device)
        self.explicit_waits=torch.zeros(env.B,device=env.device,dtype=torch.int64)
        self.emissions=torch.zeros(env.B,device=env.device,dtype=torch.int64)

    @classmethod
    def _flatten(cls,program:Program,frozen:dict[str,Program],stack:tuple[str,...],limit:int):
        if len(stack)>=limit or program.id in stack:
            return ()
        result=[]
        for rule in program.rules:
            if rule.startswith('CALL:'):
                sub=frozen.get(rule.split(':',1)[1])
                if sub:result.extend(cls._flatten(sub,frozen,stack+(program.id,),limit))
            else:result.append(rule)
        return tuple(result)

    def reset(self):
        self.last_direction.zero_()
        self.explicit_waits.zero_()
        self.emissions.zero_()

    def _velocity(self,direction, *,cap=False):
        length=torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
        dist=min(self.stride,self.env.max_speed*self.env.dt)
        if cap:dist=torch.minimum(length,torch.full_like(length,dist))
        return direction/length.clamp_min(1e-12)*dist/self.env.dt

    def _safe(self,v, *,pos=None, future=0):
        if pos is None:pos=self.env.position
        t=(self.env.step_index.to(self.env.dtype)+future)*self.env.dt
        return self.env.motion_safe(pos,pos+v*self.env.dt,t)

    def _choose_axis(self):
        options=sorted(DIRECTIONS)
        scores=[];velocities=[]
        goal_delta=self.env.goal-self.env.position
        dist0=torch.linalg.vector_norm(goal_delta,dim=-1)
        mem_len=torch.linalg.vector_norm(self.last_direction,dim=-1).clamp_min(1e-12)
        for action in options:
            d=torch.tensor(DIRECTIONS[action],device=self.env.device,dtype=self.env.dtype).expand(self.env.B,-1)
            v=self._velocity(d)
            safe=self._safe(v)
            after=self.env.position+v*self.env.dt
            goal_v=self._velocity(self.env.goal-after,cap=True)
            next_safe=self._safe(goal_v,pos=after,future=1)
            align=(dist0-torch.linalg.vector_norm(self.env.goal-after,dim=-1))/max(self.stride,1e-12)
            momentum=.18*(v*self.last_direction).sum(dim=-1)/(torch.linalg.vector_norm(v,dim=-1).clamp_min(1e-12)*mem_len)
            score=align+next_safe.to(self.env.dtype)*.65+momentum
            scores.append(torch.where(safe,score,torch.full_like(score,-float('inf'))))
            velocities.append(v)
        stacked=torch.stack(scores,dim=-1)
        maxscore=stacked.max(dim=-1,keepdim=True).values
        # Numerical ties are common in symmetric 3D chambers. Prefer lexical
        # direction order to avoid device-dependent sub-ULP differences.
        near=(stacked>=maxscore-1e-12)&torch.isfinite(stacked)
        indices=torch.argmax(near.to(torch.int64),dim=-1)
        opts=torch.stack(velocities,dim=1)
        best=opts[torch.arange(self.env.B,device=self.env.device),indices]
        any_possible=torch.isfinite(stacked).any(dim=-1)
        return torch.where(any_possible[:,None],best,torch.zeros_like(best)),any_possible

    def _wait_gap(self):
        direct=self._velocity(self.env.goal-self.env.position,cap=True)
        direct_blocked=~self._safe(direct)
        standing_safe=self._safe(torch.zeros_like(direct))
        moving=((self.env.vel0.abs()>1e-9).any(dim=-1)&self.env.valid).any(dim=-1)
        opens=torch.zeros(self.env.B,dtype=torch.bool,device=self.env.device)
        for ahead in range(1,5):
            opens|=self._safe(direct,future=ahead)
        return direct_blocked&standing_safe&moving&opens

    @torch.no_grad()
    def act(self):
        unresolved=~self.env.done
        velocity=torch.zeros((self.env.B,3),dtype=self.env.dtype,device=self.env.device)
        goal_direction=self.env.goal-self.env.position
        for rule in self.rules:
            if not bool(unresolved.any().item()):break
            valid=unresolved
            if rule=='WAIT_GAP':
                valid=unresolved&self._wait_gap()
                self.explicit_waits+=valid.to(torch.int64)
                proposal=torch.zeros_like(velocity)
            elif rule=='GOAL':
                proposal=self._velocity(goal_direction,cap=True)
                valid=unresolved&self._safe(proposal)
            elif rule in DIRECTIONS:
                d=torch.tensor(DIRECTIONS[rule],device=self.env.device,dtype=self.env.dtype).expand(self.env.B,-1)
                proposal=self._velocity(d)
                valid=unresolved&self._safe(proposal)
            elif rule in ('LAST','CONTINUE'):
                prior_valid=torch.linalg.vector_norm(self.last_direction,dim=-1)>1e-12
                proposal=self._velocity(self.last_direction)
                if rule=='CONTINUE':
                    target_distance=torch.linalg.vector_norm(self.env.goal-self.env.position,dim=-1)
                    next_distance=torch.linalg.vector_norm(self.env.goal-(self.env.position+proposal*self.env.dt),dim=-1)
                    prior_valid&=next_distance<=target_distance+.02
                valid=unresolved&prior_valid&self._safe(proposal)
            elif rule=='CHOOSE_AXIS':
                proposal,found=self._choose_axis()
                valid=unresolved&found&self._safe(proposal)
            else:
                raise ValueError('Unsupported frozen DSL primitive: '+rule)
            velocity=torch.where(valid[:,None],proposal,velocity)
            if rule!='WAIT_GAP':
                self.last_direction=torch.where(valid[:,None],proposal,self.last_direction)
            unresolved=unresolved&(~valid)
        self.emissions+=(~self.env.done).to(torch.int64)
        return velocity
