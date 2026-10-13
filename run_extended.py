"""Discover 0.3 reactive programs under a reproducible staged curriculum."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from oops_training import IncrementalSynthesizer, extended_curriculum, replay


def main(output='synthesis_results_v03', max_rules=3, budget=1500):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    learner=IncrementalSynthesizer(max_rules=max_rules,budget_per_stage=budget)
    rows=[]
    for name,scene,require_wait,no_fallback,max_events,max_work in extended_curriculum():
        found=learner.train_stage(name,scene,
                                 require_explicit_wait=require_wait,
                                 forbid_fallback=no_fallback,
                                 max_trace_events=max_events,
                                 max_work_units=max_work)
        print(f'{name:17} {found.program.id if found.program else "FAILED":7} '
              f'{str(found.program.rules) if found.program else "":45} '
              f'lane={found.found_lane} tested={found.tested}')
        rows.append({'stage':name,'program':found.program.id if found.program else None,
                     'found_lane':found.found_lane,'tested':found.tested,
                     'require_explicit_wait':require_wait,'forbid_fallback':no_fallback,
                     'max_trace_events':max_events,'max_work_units':max_work})
        if found.program is None:
            break
    learner.save(out/'frozen_library.json')
    (out/'curriculum_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    if learner.frozen:
        learned=next(reversed(learner.frozen.values()))
        for name,scene,*_ in extended_curriculum():
            outcome,trace=replay(learned,scene,learner.frozen,True)
            (out/f'{name}_trace.json').write_text(json.dumps({
                'outcome':vars(outcome),'trace':trace},indent=2)+'\n')
            print(f'REPLAY {name:17} success={outcome.success} '
                  f'explicit_waits={outcome.explicit_waits} '
                  f'continue={outcome.continued_actions} '
                  f'selected_axis={outcome.directional_choices} '
                  f'work_units={outcome.work_units}')
    return learner


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='synthesis_results_v03')
    p.add_argument('--max-rules',type=int,default=3)
    p.add_argument('--budget',type=int,default=1500)
    a=p.parse_args()
    main(a.output,a.max_rules,a.budget)
