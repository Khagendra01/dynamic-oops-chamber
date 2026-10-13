"""Compare full replay, snapshots and undo during ordered OOPS-style training."""
import argparse,csv,json
from pathlib import Path
from time import perf_counter
from oops_curriculum_v08 import TransactionalCurriculum
from oops_prefix import standard_tasks
from chamber import example_scene
from oops_training import replay

def run(output='synthesis_results_v08',backends=('replay','snapshot','undo')):
    path=Path(output);path.mkdir(exist_ok=True,parents=True)
    rows=[];summary={}
    for backend in backends:
        model=TransactionalCurriculum(backend=backend,reuse=True,max_depth=4,max_phase_budget=1048576)
        t0=perf_counter()
        for task in standard_tasks():
            r=model.train_task(task)
            rows.append(dict(backend=backend,stage=r.name,success=r.success,program=r.program_id,
                rules=' '.join(r.rules),lane=r.selected_lane,phase_budget=r.phase_budget,
                work_units=r.search_work_units,replay_calls=r.replay_calls,
                tested=r.prefixes_tested))
            if not r.success:break
        elapsed=perf_counter()-t0
        last=next(reversed(model.frozen.values())) if model.frozen else None
        heldout=sum(replay(last,lambda s=s:example_scene('random',s),model.frozen)[0].success for s in range(100,120)) if last else 0
        summary[backend]=dict(stages=sum(x.success for x in model.history),final_rules=list(last.rules) if last else None,
            total_work=sum(x.search_work_units for x in model.history),total_evaluations=sum(x.replay_calls for x in model.history),
            physics_steps=model.physics_steps,vm_instructions=model.vm_instructions,
            checkpoint_hits=model.branch_hits,undo_rollbacks=model.undo_rollbacks,
            snapshot_restores=model.snapshot_restores,seconds=round(elapsed,4),heldout_20=heldout)
        model.save(path/f'{backend}_library.json')
    with (path/'curriculum_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='synthesis_results_v08')
    a=ap.parse_args();run(a.output)
