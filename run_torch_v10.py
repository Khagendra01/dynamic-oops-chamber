"""Run GPU/CPU batched frozen-program rollouts; record real elapsed time.

USAGE: python run_torch_v10.py --device cuda --batch-size 512 --scenes 100 --out gpu_run.json
CUDA requires an NVIDIA GPU. The CPU test may be slow in small batches due to
many Torch kernel launches; this is not a speed comparison with cuRobo/Isaac.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch
from chamber import example_scene
from oops_language import Program
from torch_chamber import TorchChamberBatch
from torch_frozen_policy import TorchFrozenPolicy

LIBRARY_PATH=Path(__file__).resolve().parent/'synthesis_results_v03/frozen_library.json'

def library():
    data=json.loads(LIBRARY_PATH.read_text())
    frozen={p['id']:Program(tuple(p['rules']),p['id']) for p in data['frozen']}
    return frozen, frozen['P004']

def simulate(*,device,batch_size,scenes,dtype,seed=100):
    frozen,program=library()
    statistics_list=[]
    for off in range(0,scenes,batch_size):
        envs=[example_scene('random',seed+i) for i in range(off,min(off+batch_size,scenes))]
        batch=TorchChamberBatch(envs,device=device,dtype=dtype)
        policy=TorchFrozenPolicy(batch,program,frozen)
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        interventions=torch.zeros(batch.B,device=batch.device,dtype=torch.int64)
        for _ in range(batch.max_steps):
            velocity=policy.act()
            outcome=batch.step(velocity)
            interventions+=outcome.intervention.to(torch.int64)
        if device=='cuda':torch.cuda.synchronize()
        duration=time.perf_counter()-start
        success=((torch.linalg.vector_norm(batch.position-batch.goal,dim=-1)<=batch.tolerance)&(~batch.collision)).sum().item()
        statistics_list.append({'batch_scenes':batch.B,'success':int(success),
                                'collisions':int(batch.collision.sum().item()),
                                'interventions':int(interventions.sum().item()),
                                'simulated_steps':int(batch.step_index.sum().item()),
                                'elapsed_s':round(duration,6)})
    return statistics_list

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--dtype',choices=['float32','float64'],default='float64')
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--scenes',type=int,default=100)
    ap.add_argument('--seed',type=int,default=100)
    ap.add_argument('--out',default='gpu_v10_summary.json')
    args=ap.parse_args()
    if args.batch_size<=0 or args.scenes<=0:ap.error('batch-size and scenes must be positive')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested, but torch.cuda.is_available() is False')
    dtype={'float32':torch.float32,'float64':torch.float64}[args.dtype]
    rows=simulate(device=device,batch_size=args.batch_size,scenes=args.scenes,dtype=dtype,seed=args.seed)
    report={'version':'v1.0','device':device,'gpu_name':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'pytorch':torch.__version__,'dtype':args.dtype,'num_scenes':args.scenes,'seed_start':args.seed,
            'num_batches':len(rows),'elapsed_s':round(sum(x['elapsed_s'] for x in rows),6),
            'success':sum(x['success'] for x in rows),'collisions':sum(x['collisions'] for x in rows),
            'interventions':sum(x['interventions'] for x in rows),
            'simulated_steps':sum(x['simulated_steps'] for x in rows),
            'runs':rows,
            'notes':['Elapsed time excludes scene construction and TorchChamberBatch initialization.',
                     'Contains interpreted Torch primitives rather than original OOPS program synthesis.',
                     'Simulation uses exact geometric state and conservative sphere/AABB collision geometry.']}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['runs','notes']},indent=2))
    print('Saved:',args.out)

if __name__=='__main__':main()
