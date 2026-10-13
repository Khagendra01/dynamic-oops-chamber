"""Evaluate frozen programs on scenes withheld from the synthesis curriculum.

The random chamber's exact obstacle locations/motions are unseen by training.
The north-blocked scene is a designed challenge to expose a directional-bias
failure; it is not used as a task in the stored frozen library.
"""
import argparse
import csv
from pathlib import Path
from chamber import Cuboid, ChamberEnv, example_scene
from oops_training import replay, load_library


def harder_scene():
    return ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5),
        obstacles=[Cuboid('central',(.5,.5,.5),(.18,.20,.24)),
                   Cuboid('north_block',(.25,.69,.5),(.32,.20,.30))],
        radius=.035,dt=.5,max_steps=55)


def evaluate(library_path='synthesis_results/frozen_library.json',
             count=20, output='synthesis_results/heldout_results.csv'):
    library=load_library(library_path)
    if not library:
        raise ValueError('No frozen program in library')
    program=next(reversed(library.values()))
    scenes=[(f'random_{seed}',lambda s=seed:example_scene('random',s))
            for seed in range(100,100+count)]
    scenes.append(('north_blocked_ood',harder_scene))
    rows=[]
    for name,factory in scenes:
        outcome,_=replay(program,factory,library)
        rows.append({'scene':name,'program':program.id,
                     'success':int(outcome.success),'collision':int(outcome.collision),
                     'interventions':outcome.interventions,'steps':outcome.steps,
                     'fallback_waits':outcome.fallback_waits})
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    randomized=rows[:-1]
    print(f'Frozen program: {program.id} {program.rules}')
    print(f'Random held-out scenes: {sum(r["success"] for r in randomized)}/{len(randomized)} solved')
    print(f'Structured OOD scene: {rows[-1]["success"]} success (0 means failure)')
    print(f'Saved per-scene results: {output}')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',default='synthesis_results/frozen_library.json')
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--output',default='synthesis_results/heldout_results.csv')
    a=p.parse_args()
    evaluate(a.library,a.count,a.output)
