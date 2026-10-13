"""Controlled *deployment* ablations for frozen v0.3 reactive programs.

Ablations remove a single DSL behavior from every frozen program (not just
from the top-level caller). Saved bytecode and interpreter stay unchanged.
Separately, 'reset_each_step' evaluates persistent VM memory vs stateless
re-execution of exactly the same program. That is not fresh OOPS synthesis.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from dataclasses import replace
from chamber import example_scene
from oops_language import Program, Interpreter, VMState, MotionConfig
from oops_training import load_library, replay
from oops_scenarios import moving_gate, north_blocked, south_blocked


def ablate(library: dict[str,Program], removed: str) -> dict[str,Program]:
    new={}
    for id,prog in library.items():
        rules=tuple(x for x in prog.rules if x!=removed)
        # Preserve program identifiers and CALL linking. If removal yields an
        # empty program, explicit GOAL would distort the ablation: reject it.
        if not rules: raise ValueError(f'Ablation empties {id}')
        new[id]=Program(rules, id)
    return new


def replay_stateless(program, make_scene, library):
    """Identical program but reinitialize VMState every observation."""
    env=make_scene(); obs=env.reset()
    cfg=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed)
    vm=Interpreter(frozen=library,config=cfg)
    collision=False; shield=0; fallback=0; wait=0; continued=0; chosen=0; work=0
    for _ in range(env.max_steps):
        if env.done: break
        action,trace=vm.act(program,obs,VMState())
        work+=sum(8 if x=='AIM CHOOSE_AXIS' else 4 if x in ('AIM GOAL','GAP_WAIT_READY','GAP_WAIT_NOT_NEEDED') else 1 for x in trace)
        fallback+=int(trace[-1]=='FALLBACK WAIT')
        wait+=int('EMIT_WAIT_GAP' in trace)
        continued+=int('SELECTED:CONTINUE' in trace)
        chosen+=int('SELECTED:CHOOSE_AXIS' in trace)
        step=env.step(action); collision|=step.collision
        shield+=int(step.safety_intervention);obs=step.observation
    from chamber import norm, sub
    return dict(success=int(env.done and not collision and norm(sub(obs.position,env.goal))<=env.goal_tolerance),
        collision=int(collision),shield_interventions=shield,steps=env.step_index,
        fallback_waits=fallback, explicit_waits=wait,continued_actions=continued,
        directional_choices=chosen,work_units=work)


def asrow(res):
    if isinstance(res,dict):return res
    return dict(success=int(res.success),collision=int(res.collision),
        shield_interventions=res.interventions,steps=res.steps,
        fallback_waits=res.fallback_waits,explicit_waits=res.explicit_waits,
        continued_actions=res.continued_actions,directional_choices=res.directional_choices,
        work_units=res.work_units)


def run(count=100,first_seed=100,output='synthesis_results_v04/ablation_results.csv'):
    baseline=load_library('synthesis_results_v03/frozen_library.json')
    variants={'complete':baseline,'no_axis':ablate(baseline,'CHOOSE_AXIS'),
        'no_gap_wait':ablate(baseline,'WAIT_GAP'),
        'no_continuation':ablate(baseline,'CONTINUE')}
    scenes=[(f'random_{seed}',lambda seed=seed:example_scene('random',seed)) for seed in range(first_seed,first_seed+count)]
    scenes.extend([('moving_gate',moving_gate),('north_blocked',north_blocked),('south_blocked',south_blocked),('empty',lambda:example_scene('empty'))])
    rows=[]
    for name,make_scene in scenes:
        for variant,lib in variants.items():
            program=lib[next(reversed(lib))]
            res,_=replay(program,make_scene,lib)
            rows.append({'scene':name,'variant':variant,**asrow(res)})
        res=replay_stateless(baseline[next(reversed(baseline))],make_scene,baseline)
        rows.append({'scene':name,'variant':'reset_memory_each_step',**res})
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}
    for variant in list(variants)+['reset_memory_each_step']:
        subset=[r for r in rows if r['variant']==variant and r['scene'].startswith('random_')]
        wins=[r for r in subset if r['success']]
        summaries[variant]={'episodes':len(subset),'successes':len(wins),
            'collisions':sum(r['collision'] for r in subset),
            'mean_steps_successes':sum(r['steps'] for r in wins)/len(wins) if wins else None,
            'mean_work_units':sum(r['work_units'] for r in subset)/len(subset),
            'mean_fallback_waits':sum(r['fallback_waits'] for r in subset)/len(subset),
            'mean_explicit_waits':sum(r['explicit_waits'] for r in subset)/len(subset),
            'mean_continued_actions':sum(r['continued_actions'] for r in subset)/len(subset)}
    diagnostics={name:{r['variant']:{'success':r['success'],'collision':r['collision'],
        'explicit_waits':r['explicit_waits'],'continued_actions':r['continued_actions'],
        'work_units':r['work_units']} for r in rows if r['scene']==name}
        for name,_ in scenes[-4:]}
    summary={'setup':{'random_seed_range':[first_seed,first_seed+count-1],
        'all_variants_run_same_scenes':True,'ablation_type':'remove opcode recursively from frozen library',
        'note':'Memory reset each observation is distinct from fresh synthesis. Hand-designed primitives remain.'},
        'random_scenes':summaries,'targeted_scenes':diagnostics}
    target.with_name('ablation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=100)
    p.add_argument('--first-seed',type=int,default=100)
    a=p.parse_args();result=run(a.count,a.first_seed)
    for name,d in result['random_scenes'].items():print(name,d)
    print('targeted',json.dumps(result['targeted_scenes'],indent=2))
