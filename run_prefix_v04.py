"""Paired fresh-vs-incremental frozen-code OOPS-style prefix-search study."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from oops_prefix import PrefixOOPS,standard_tasks
from oops_training import replay
from chamber import example_scene


def run(max_depth=4,max_budget=1048576,max_nodes=30000,output='synthesis_results_v04'):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    rows=[];models={}
    for method,reuse in [('incremental_reuse',True),('fresh_search',False)]:
        learner=PrefixOOPS(reuse=reuse,max_depth=max_depth,
            max_phase_budget=max_budget,max_prefixes_per_phase=max_nodes)
        for task in standard_tasks():
            found=learner.train_task(task)
            row={'method':method,'stage':found.name,'success':int(found.success),
                 'program_id':found.program_id,'rules':' '.join(found.rules),
                 'selected_lane':found.selected_lane,'phase_budget':found.phase_budget,
                 'search_work_units':found.search_work_units,
                 'real_replay_calls':found.replay_calls,
                 'prefixes_tested':found.prefixes_tested,
                 'total_prefix_visits':found.total_prefix_visits}
            rows.append(row);print(row,flush=True)
            if not found.success:break
        learner.save(path/(method+'_prefix_trace.json'))
        models[method]=learner
    with (path/'prefix_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    results=[]
    for method,model in models.items():
        if not model.frozen:continue
        final=next(reversed(model.frozen.values()))
        for seed in range(100,200):
            result,_=replay(final,lambda seed=seed:example_scene('random',seed),model.frozen)
            results.append({'method':method,'seed':seed,'success':int(result.success),
                            'collision':int(result.collision),'work_units':result.work_units,
                            'steps':result.steps})
    with (path/'prefix_holdout.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={}
    for method,model in models.items():
        subset=[r for r in results if r['method']==method]
        summary[method]={'stages_solved':sum(s.success for s in model.history),
            'stages_attempted':len(model.history),
            'total_search_work_units':sum(s.search_work_units for s in model.history),
            'total_real_replay_calls':sum(s.replay_calls for s in model.history),
            'total_prefixes_tested':sum(s.prefixes_tested for s in model.history),
            'final_program':list(next(reversed(model.frozen.values())).rules) if model.frozen else None,
            'heldout_successes':sum(r['success'] for r in subset),
            'heldout_collisions':sum(r['collision'] for r in subset),
            'heldout_count':len(subset)}
    (path/'prefix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-depth',type=int,default=4)
    p.add_argument('--max-budget',type=int,default=1048576)
    p.add_argument('--max-nodes',type=int,default=30000)
    a=p.parse_args();run(a.max_depth,a.max_budget,a.max_nodes)
