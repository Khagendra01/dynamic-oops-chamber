"""Deterministic paired search evaluation of replay vs checkpoint continuation."""
import json
from pathlib import Path
from oops_prefix import PrefixOOPS, standard_tasks

def run(mode):
    engine=PrefixOOPS(continuable=mode)
    rows=[]
    for task in standard_tasks():
        r=engine.train_task(task)
        rows.append(dict(task=task.name,success=r.success,rules=r.rules,
                        simulated_steps=engine.simulated_steps,
                        reused_steps=engine.reused_steps,
                        evaluations=engine.real_replay_calls,
                        modeled_work=r.search_work_units))
        if not r.success:break
    return rows

def main():
    replay=run(False)
    continuation=run(True)
    for a,b in zip(replay,continuation):
        assert a['success']==b['success'] and a['rules']==b['rules'],(a,b)
    output={'full_replay':replay,'continuation':continuation,
            'simulated_steps_full_replay':sum(v['simulated_steps'] for v in replay),
            'simulated_steps_continuation':sum(v['simulated_steps'] for v in continuation),
            'steps_reused':sum(v['reused_steps'] for v in continuation)}
    dest=Path('synthesis_results_v05/continuation_comparison.json')
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if not isinstance(v,list)},indent=2))
    print('matched_curriculum_stages',len(continuation))
if __name__=='__main__':main()
