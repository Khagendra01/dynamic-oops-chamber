"""GPU-ready vectorized deterministic chamber, numerical companion to chamber.py.

B independent scenes, K padded AABB obstacles. All obstacle geometry and moving
sphere swept checks are in torch on the chosen device. No renders or Isaac SDK.

The shape approximation matches chamber.py: inflated AABBs conservatively
approximate a sphere. Moving AABBs bounce at walls, and sweeps are split at
all bounce times within the step (at most one bounce/axis/step is enforced).

Use float64 for CPU parity tests; float32 is recommended for GPU throughput.
No gradients; planning/rollout kernel only, not a neural policy or real OOPS.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from chamber import ChamberEnv

EPS = 1e-10

@dataclass
class TorchStepResult:
    observation_positions: torch.Tensor
    reached_goal: torch.Tensor
    collision: torch.Tensor
    intervention: torch.Tensor
    done: torch.Tensor

class TorchChamberBatch:
    def __init__(self, envs: Sequence[ChamberEnv], *, device: str = 'cpu',
                 dtype: torch.dtype = torch.float64):
        if not envs:
            raise ValueError('At least one ChamberEnv required')
        self.device = torch.device(device)
        self.dtype = dtype
        ref = envs[0]
        for env in envs:
            if (env.chamber_size != ref.chamber_size or env.radius != ref.radius or
                env.dt != ref.dt or env.max_speed != ref.max_speed):
                raise ValueError('All environments must share geometry and dynamics settings')
            # One reflection per axis per step, guaranteeing 4 subsegments suffice.
            for ob in env.obstacles:
                if any(abs(ob.velocity[d])*env.dt > env.chamber_size[d]-ob.size[d]+EPS
                       for d in range(3)):
                    raise ValueError('Obstacle can reflect multiple times per control step')
        self.B = len(envs)
        self.K = max(1, max(len(env.obstacles) for env in envs))
        self.dt = ref.dt
        self.radius = ref.radius
        self.max_speed = ref.max_speed
        # Per-environment task limits: curriculum stages differ in horizon and
        # goal tolerance. Scalars remain the common value (or the maximum for
        # max_steps) so single-task batches behave exactly as before.
        self.max_steps = max(env.max_steps for env in envs)
        self.tolerance = ref.goal_tolerance
        self.size = torch.tensor(ref.chamber_size, device=self.device, dtype=dtype)
        self.start = torch.tensor([env.start for env in envs],device=self.device,dtype=dtype)
        self.goal = torch.tensor([env.goal for env in envs],device=self.device,dtype=dtype)
        self.tolerance_t = torch.tensor([env.goal_tolerance for env in envs],
                                        device=self.device, dtype=dtype)
        self.max_steps_t = torch.tensor([env.max_steps for env in envs],
                                        device=self.device, dtype=torch.int64)
        centers = torch.zeros((self.B,self.K,3),device=self.device,dtype=dtype)
        dims = torch.full((self.B,self.K,3), .01,device=self.device,dtype=dtype)
        velocities = torch.zeros_like(centers)
        valid = torch.zeros((self.B,self.K),device=self.device,dtype=torch.bool)
        for b,env in enumerate(envs):
            for k,ob in enumerate(env.obstacles):
                centers[b,k] = torch.tensor(ob.center,device=self.device,dtype=dtype)
                dims[b,k] = torch.tensor(ob.size,device=self.device,dtype=dtype)
                velocities[b,k] = torch.tensor(ob.velocity,device=self.device,dtype=dtype)
                valid[b,k] = True
        self.center0, self.dims, self.vel0, self.valid = centers,dims,velocities,valid
        self.reset()

    @classmethod
    def from_templates(cls, templates: Sequence["ChamberEnv"], task_ids: torch.Tensor,
                       *, device: str = "cpu",
                       dtype: torch.dtype = torch.float64,
                       base: "TorchChamberBatch | None" = None) -> "TorchChamberBatch":
        """Build a batch whose rows share a small set of template environments.

        Templates are simulated once (validating the usual geometry/dynamics
        constraints); every job row then gathers its own scene, task horizon and
        goal tolerance. This keeps batch construction out of the timed region
        for large candidate sweeps. `base` reuses an already-simulated template
        batch so repeated builds cost only the row gather.
        """
        if not len(task_ids):
            raise ValueError("At least one job is required")
        if base is None:
            base = cls(list(templates), device=device, dtype=dtype)
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        idx = task_ids.to(device=self.device, dtype=torch.long)
        self.B = int(idx.numel())
        self.start = base.start[idx]
        self.goal = base.goal[idx]
        self.tolerance_t = base.tolerance_t[idx]
        self.max_steps_t = base.max_steps_t[idx]
        self.center0 = base.center0[idx]
        self.dims = base.dims[idx]
        self.vel0 = base.vel0[idx]
        self.valid = base.valid[idx]
        self.reset()
        return self

    @torch.no_grad()
    def reset(self):
        self.position = self.start.clone()
        self.step_index = torch.zeros(self.B,device=self.device,dtype=torch.int64)
        self.done = torch.linalg.vector_norm(self.start-self.goal,dim=-1) <= self.tolerance_t
        self.collision = torch.zeros_like(self.done)
        return self.observe()

    def _centers_at(self, times: torch.Tensor, *, center0: torch.Tensor | None = None,
                    dims: torch.Tensor | None = None,
                    vel0: torch.Tensor | None = None) -> torch.Tensor:
        """times: [B,K] seconds; results [B,K,3]. Geometry may be supplied for
        query batches whose rows map onto several jobs of the same chamber."""
        center0 = self.center0 if center0 is None else center0
        dims = self.dims if dims is None else dims
        vel0 = self.vel0 if vel0 is None else vel0
        span = self.size - dims
        safe_span = span.clamp_min(EPS)
        u = center0 - dims/2 + vel0*times.unsqueeze(-1)
        phase = torch.remainder(u,2*safe_span)
        x = torch.minimum(phase,2*safe_span-phase) + dims/2
        return torch.where(span>EPS,x,dims/2)

    @torch.no_grad()
    def observe(self):
        times = self.step_index.to(self.dtype)*self.dt
        return {
            'position': self.position.clone(),
            'goal': self.goal,
            'obstacle_centers': self._centers_at(times[:,None].expand(-1,self.K)),
            'obstacle_sizes': self.dims,
            'obstacle_valid': self.valid,
            'time_s': times,
            'done': self.done.clone(),
        }

    def _bounce_offsets(self, t0: torch.Tensor, *, center0: torch.Tensor | None = None,
                        dims: torch.Tensor | None = None,
                        vel0: torch.Tensor | None = None) -> torch.Tensor:
        """First interior bounce time per XYZ obstacle axis: [B,K,3]."""
        center0 = self.center0 if center0 is None else center0
        dims = self.dims if dims is None else dims
        vel0 = self.vel0 if vel0 is None else vel0
        span = self.size-dims
        safe_span=span.clamp_min(EPS)
        u0=center0-dims/2+vel0*t0[:,None,None]
        ratio=u0/safe_span
        # Exclude current-boundary events; the next transition is toward + or - infinity.
        k=torch.where(vel0>0,torch.floor(ratio)+1,torch.ceil(ratio)-1)
        safe_v=torch.where(vel0.abs()>EPS,vel0,torch.ones_like(vel0))
        offset=(k*safe_span-u0)/safe_v
        interior=(offset>EPS)&(offset<self.dt-EPS)&(vel0.abs()>EPS)&(span>EPS)
        return torch.where(interior,offset,torch.full_like(offset,self.dt))

    @staticmethod
    def _swept_intersects(rel0,rel1,half):
        """Slab test for relative movement of point vs expanded AABB [B,K,3]."""
        delta=rel1-rel0
        fixed=delta.abs()<EPS
        valid_fixed=(~fixed)|(rel0.abs()<=half+EPS)
        safe_delta=torch.where(fixed,torch.ones_like(delta),delta)
        a=(-half-rel0)/safe_delta
        b=(half-rel0)/safe_delta
        enter=torch.where(fixed,torch.full_like(a,-float('inf')),torch.minimum(a,b))
        leave=torch.where(fixed,torch.full_like(a,float('inf')),torch.maximum(a,b))
        return (valid_fixed.all(dim=-1)&
                (torch.maximum(enter.max(dim=-1).values,torch.zeros_like(enter[...,0]))<=
                 torch.minimum(leave.min(dim=-1).values,torch.ones_like(leave[...,0]))+EPS))

    @torch.no_grad()
    def motion_safe(self, start: torch.Tensor, end: torch.Tensor,
                    t0: torch.Tensor, *, job_index: torch.Tensor | None = None) -> torch.Tensor:
        """Check Q proposed agent motions over [t0,t0+dt], including moving boxes.

        job_index maps each query row onto a row of this batch, which lets the
        policy stack every candidate move of a step into a single call.
        """
        if job_index is None:
            centers, dims, vel0, valid = self.center0, self.dims, self.vel0, self.valid
        else:
            idx = job_index
            centers, dims, vel0, valid = (self.center0[idx], self.dims[idx],
                                          self.vel0[idx], self.valid[idx])
        B = start.shape[0]
        K = centers.shape[1]
        inside=((start>=self.radius-EPS)&(start<=self.size-self.radius+EPS)&
                (end>=self.radius-EPS)&(end<=self.size-self.radius+EPS)).all(dim=-1)
        event_offsets = self._bounce_offsets(t0, center0=centers, dims=dims, vel0=vel0) # B,K,3
        sorted_events = torch.sort(event_offsets,dim=-1).values
        times = torch.cat([torch.zeros((B,K,1),dtype=self.dtype,device=self.device),
                           sorted_events,
                           torch.full((B,K,1),self.dt,dtype=self.dtype,device=self.device)],dim=-1)
        collided=torch.zeros((B,K),dtype=torch.bool,device=self.device)
        half=dims/2+self.radius
        for i in range(4):
            left,right=times[...,i],times[...,i+1]
            # Empty segment when multiple axis bounces coincide -> no interval.
            active_interval=right>left+EPS
            a0=start[:,None,:]+(end-start)[:,None,:]*(left/self.dt).unsqueeze(-1)
            a1=start[:,None,:]+(end-start)[:,None,:]*(right/self.dt).unsqueeze(-1)
            c0=self._centers_at(t0[:,None]+left,center0=centers,dims=dims,vel0=vel0)
            c1=self._centers_at(t0[:,None]+right,center0=centers,dims=dims,vel0=vel0)
            collided|=(active_interval&self._swept_intersects(a0-c0,a1-c1,half)&valid)
        return inside&(~collided.any(dim=-1))

    @torch.no_grad()
    def step(self, velocities: torch.Tensor, *, check_speed: bool = True) -> TorchStepResult:
        if velocities.shape!=(self.B,3):
            raise ValueError(f'Expected {(self.B,3)} velocity array')
        velocities=velocities.to(device=self.device,dtype=self.dtype)
        # check_speed=False avoids a host sync so the step can be captured by
        # CUDA graphs / torch.compile; callers must guarantee the speed bound.
        if check_speed and bool((torch.linalg.vector_norm(velocities,dim=-1)>self.max_speed+EPS).any().item()):
            raise ValueError('Velocity exceeds max speed')
        was_active=~self.done
        requested=self.position+velocities*self.dt
        t0=self.step_index.to(self.dtype)*self.dt
        # Shield and standing-still checks share one fused query batch: kernel
        # launches (not arithmetic) dominate these small per-step queries.
        q_start=torch.cat([self.position,self.position],dim=0)
        q_end=torch.cat([requested,self.position],dim=0)
        q_t0=torch.cat([t0,t0],dim=0)
        q_job=torch.cat([torch.arange(self.B,device=self.device),
                         torch.arange(self.B,device=self.device)],dim=0)
        both=self.motion_safe(q_start,q_end,q_t0,job_index=q_job)
        accepted=both[:self.B]
        held_safe=both[self.B:]
        intervention=was_active&(~accepted)
        collision=intervention&(~held_safe)
        self.position=torch.where((was_active&accepted)[:,None],requested,self.position)
        self.step_index+=was_active.to(torch.int64)
        reached=torch.linalg.vector_norm(self.position-self.goal,dim=-1)<=self.tolerance_t
        self.done=self.done|reached|collision|(self.step_index>=self.max_steps_t)
        self.collision|=collision
        return TorchStepResult(self.position.clone(),reached,collision,intervention,self.done.clone())

    @torch.no_grad()
    def greedy_action(self,stride: float=.12)->torch.Tensor:
        direction=self.goal-self.position
        distance=torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(1e-12)
        delta=torch.minimum(distance,torch.full_like(distance,stride))/self.dt
        cmd=direction/distance*delta
        return torch.where(self.done[:,None],torch.zeros_like(cmd),cmd)
