"""Compare PyTorch batch against the frozen CPU baseline (no Isaac SDK)."""
import argparse, json, time
from pathlib import Path
import torch
from chamber import example_scene
from oops_language import Program
from oops_training import replay
from torch_chamber import TorchChamberBatch
from torch_frozen_policy import TorchFrozenPolicy
from run_torch_v10 import library


def check(device='cpu',count=100,start_seed=100,dtype=torch.float64):
    frozen,program=library()
    envs=[example_scene('random',seed) for seed in range(start_seed,start_seed+count)]
    batched=TorchChamberBatch(envs,device=device,dtype=dtype)
    policy=TorchFrozenPolicy(batched,program,frozen)
    for _ in range(batched.max_steps):
        velocities=policy.act()
        batched.step(velocities)
    if device=='cuda':torch.cuda.synchronize()
    mismatches=[]
    success=0
    for j,seed in enumerate(range(start_seed,start_seed+count)):
        cpu,_=replay(program,lambda s=seed:example_scene('random',s),frozen)
        batch_succ=bool((torch.linalg.vector_norm(batched.position[j]-batched.goal[j])<=batched.tolerance).item()) and not bool(batched.collision[j].item())
        if cpu.success!=batch_succ or cpu.collision!=bool(batched.collision[j].item()) or cpu.steps!=int(batched.step_index[j].item()):
            mismatches.append({'seed':seed,'cpu_success':cpu.success,'torch_success':batch_succ,'cpu_steps':cpu.steps,'torch_steps':int(batched.step_index[j].item())})
        success+=batch_succ
    return {'device':device,'dtype':str(dtype),'scenes':count,'success':success,'mismatches':mismatches,
            'equivalent':not mismatches}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--count',type=int,default=100)
    ap.add_argument('--out',default='parity_v10.json')
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA not available')
    out=check(device=args.device,count=args.count)
    Path(args.out).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='mismatches'},indent=2))
    if out['mismatches']:
        print('First mismatches:',out['mismatches'][:5]);raise SystemExit(1)
