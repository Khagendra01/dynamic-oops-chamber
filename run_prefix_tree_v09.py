"""Reproducible v0.9 lazy-trie curriculum and balanced timing comparison."""
from __future__ import annotations
import argparse, csv, json, platform, statistics, sys, tracemalloc
from pathlib import Path
from time import perf_counter
from oops_prefix_tree_v09 import LazyPrefixCurriculum
from oops_prefix import standard_tasks
from chamber import example_scene
from oops_training import replay


def run_once(backend, promote_after=2, max_nodes=96, holdout=100, max_probe_steps=4):
    model=LazyPrefixCurriculum(backend=backend,promote_after=promote_after,
        max_nodes=max_nodes,max_probe_steps=max_probe_steps,reuse=True,max_depth=4,max_phase_budget=1048576)
    t0=perf_counter()
    for task in standard_tasks():
        if not model.train_task(task).success:
            break
    train_seconds=perf_counter()-t0
    last=next(reversed(model.frozen.values())) if model.frozen else None
    successes=collisions=0
    if last:
        for s in range(100,100+holdout):
            result,_=replay(last,lambda s=s:example_scene('random',s),model.frozen)
            successes+=result.success;collisions+=result.collision
    return dict(backend=backend,promote_after=promote_after,
        stages=sum(x.success for x in model.history),
        frozen_programs=[dict(id=p.id,rules=list(p.rules)) for p in model.frozen.values()],
        search_work_units=sum(x.search_work_units for x in model.history),
        candidates_evaluated=sum(x.replay_calls for x in model.history),
        instructions=model.vm_instructions,steps=model.physics_steps,
        nodes_built=model.created_nodes,checkpoint_hits=model.branch_hits,
        node_evictions=model.node_evictions,peak_nodes=model.peak_nodes,
        extra_parent_steps=model.extra_parent_physics,
        restore_calls=model.snapshot_restores+model.undo_rollbacks,
        train_seconds=train_seconds,heldout_success=successes,
        heldout_collisions=collisions,heldout_total=holdout)


def run(output='synthesis_results_v09',repeats=5,holdout=100,
        variants=None,promote_after=2,max_nodes=96,max_probe_steps=4):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if variants is None:variants=['replay','snapshot','undo']
    rows=[]
    # rotating backend order mitigates systematic timing order effects.
    for r in range(repeats):
        for backend in variants[r%len(variants):]+variants[:r%len(variants)]:
            result=run_once(backend,promote_after,max_nodes,holdout,max_probe_steps)
            result['trial']=r
            rows.append(result)
            print(f"trial={r} {backend:8} {result['train_seconds']:.3f}s "
                  f"steps={result['steps']} checkpoints={result['nodes_built']} "
                  f"reuse={result['checkpoint_hits']}",flush=True)
    assert len({json.dumps(x['frozen_programs'],sort_keys=True) for x in rows})==1, \
        'Mismatch in discovered frozen programs'
    assert len({(x['stages'],x['heldout_success'],x['heldout_collisions']) for x in rows})==1, \
        'Mismatch in trained outcomes'
    summary={}
    for name in variants:
        entries=[r for r in rows if r['backend']==name]
        summary[name]={k:entries[0][k] for k in entries[0] if k not in
                       ('frozen_programs','trial','backend','train_seconds')}
        summary[name]['median_train_seconds']=statistics.median(x['train_seconds'] for x in entries)
        summary[name]['min_train_seconds']=min(x['train_seconds'] for x in entries)
        summary[name]['max_train_seconds']=max(x['train_seconds'] for x in entries)
    output_json={'environment':{'python':sys.version.split()[0],
          'platform':platform.platform()},'repeats':repeats,
          'parameters':{'holdout':holdout,'promote_after':promote_after,'max_nodes':max_nodes,'max_probe_steps':max_probe_steps},
          'programs':rows[0]['frozen_programs'],'summary':summary}
    (folder/'summary.json').write_text(json.dumps(output_json,indent=2)+'\n')
    clean=[{k:v for k,v in row.items() if k!='frozen_programs'} for row in rows]
    with (folder/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(clean[0]));writer.writeheader();writer.writerows(clean)
    print(json.dumps(summary,indent=2))
    return output_json

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='synthesis_results_v09')
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--holdout',type=int,default=100)
    ap.add_argument('--promote-after',type=int,default=2)
    ap.add_argument('--max-nodes',type=int,default=96)
    ap.add_argument('--max-probe-steps',type=int,default=4)
    a=ap.parse_args()
    run(a.output,a.repeats,a.holdout,promote_after=a.promote_after,max_nodes=a.max_nodes,max_probe_steps=a.max_probe_steps)
