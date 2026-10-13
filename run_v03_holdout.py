"""Paired held-out evaluation of original 0.2 frozen code and 0.3 programs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from chamber import example_scene
from oops_training import replay,load_library


def evaluate(count=100, first_seed=100, output='synthesis_results_v03/heldout_comparison.csv',
             baseline='baselines/v02_frozen_library.json',
             new='synthesis_results_v03/frozen_library.json'):
    old=load_library(baseline)
    learned=load_library(new)
    models=[('v02',next(reversed(old.values())),old),
            ('v03',next(reversed(learned.values())),learned)]
    rows=[]
    for seed in range(first_seed,first_seed+count):
        make_scene=lambda seed=seed: example_scene('random',seed)
        for version,program,library in models:
            result,_=replay(program,make_scene,library)
            rows.append({'seed':seed,'method':version,'program':program.id,
                         'success':int(result.success),'collision':int(result.collision),
                         'shield_interventions':result.interventions,
                         'steps':result.steps,'fallback_waits':result.fallback_waits,
                         'explicit_waits':result.explicit_waits,
                         'directional_choices':result.directional_choices,
                         'continued_actions':result.continued_actions,
                         'trace_events':result.trace_events,
                         'work_units':result.work_units})
    out=Path(output)
    out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary={}
    for method,_,_ in models:
        group=[r for r in rows if r['method']==method]
        solved=[r for r in group if r['success']]
        summary[method]={
            'episodes':len(group),'successes':len(solved),
            'collisions':sum(r['collision'] for r in group),
            'shield_interventions':sum(r['shield_interventions'] for r in group),
            'mean_steps_success_only':(sum(r['steps'] for r in solved)/len(solved) if solved else None),
            'mean_work_units_all':sum(r['work_units'] for r in group)/len(group),
            'fallback_waits_all':sum(r['fallback_waits'] for r in group),
            'explicit_waits_all':sum(r['explicit_waits'] for r in group),
            'continued_actions_all':sum(r['continued_actions'] for r in group),
        }
        print(method, summary[method])
    out.with_name('heldout_summary.json').write_text(json.dumps({
        'seeds':[first_seed,first_seed+count-1], 'methods':summary,
        'limitations':'Small deterministic chambers; perfect observations; same 0.3 interpreter for old and new rule lists; no real-time timing claims.'
        },indent=2)+'\n')
    print('saved',out)
    return rows,summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--first-seed',type=int,default=100)
    p.add_argument('--output',default='synthesis_results_v03/heldout_comparison.csv')
    a=p.parse_args()
    evaluate(a.count,a.first_seed,a.output)
