"""CLI demonstration of bounded incremental program search and frozen-code replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from chamber import example_scene
from oops_language import Interpreter, MotionConfig, VMState
from oops_training import IncrementalSynthesizer, curriculum, replay


def main(max_rules=3, budget=2000, output='synthesis_results'):
    out=Path(output); out.mkdir(parents=True, exist_ok=True)
    engine=IncrementalSynthesizer(max_rules=max_rules,budget_per_stage=budget)
    for name,factory in curriculum():
        d=engine.train_stage(name,factory)
        if d.program is None:
            print(f'{name:10s} NOT FOUND  {d.tested} candidates, {d.replayed_episodes} replays')
            break
        print(f'{name:10s} {d.program.id} rules={d.program.rules} '
              f'lane={d.found_lane}  candidates={d.tested}  replays={d.replayed_episodes}')
    engine.save(out/'frozen_library.json')
    if engine.frozen:
        latest=next(reversed(engine.frozen.values()))
        for scene in ['empty','crossing','static','moving']:
            result,trace=replay(latest,lambda s=scene:example_scene(s),engine.frozen,True)
            (out/f'{scene}_trace.json').write_text(json.dumps({'episode':result.__dict__,
                                                                'trace':trace},indent=2)+'\n')
            print(f'REPLAY {scene:10s} success={result.success} collision={result.collision} '
                  f'interventions={result.interventions} steps={result.steps}')
    return engine


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-rules',type=int,default=3)
    parser.add_argument('--budget',type=int,default=2000)
    parser.add_argument('--output',type=str,default='synthesis_results')
    args=parser.parse_args()
    main(args.max_rules,args.budget,args.output)
