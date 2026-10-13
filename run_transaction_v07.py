"""Repeatable v0.7 multi-scene, same branches, three backend benchmark."""
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter
import csv,json, argparse
from chamber import example_scene
from oops_language import Program
from oops_transactional_v07 import branch_compare

PARENT=Program(('GOAL',))
CHILDREN=[Program(('GOAL',r)) for r in ('Y_POS','Y_NEG','CHOOSE_AXIS','WAIT_GAP','CONTINUE','Z_POS','Z_NEG')]

def run(repeats=5):
    rows=[]
    for scene in ('empty','static','moving','crossing'):
        reference=None
        for mode in ('replay','snapshot','undo'):
            times=[];last=None
            for _ in range(repeats):
                out=branch_compare(PARENT,CHILDREN,lambda scene=scene:example_scene(scene),mode=mode)
                if reference is None:reference=out['results']
                assert out['results']==reference,(scene,mode)
                times.append(out['seconds']);last=out
            rows.append({'scene':scene,'mode':mode,'repeat_count':repeats,
                'median_seconds':round(median(times),7),'instructions':last['instructions'],
                'physics_steps':last['steps'],'journal_entries':last['journal_entries'],
                'successes':sum(x.success for x in last['results'])})
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args();data=run(args.repeats)
    out=Path('synthesis_results_v07');out.mkdir(exist_ok=True)
    with (out/'transaction_comparison.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=data[0]);wr.writeheader();wr.writerows(data)
    (out/'transaction_comparison.json').write_text(json.dumps(data,indent=2))
    for r in data:print(r)
