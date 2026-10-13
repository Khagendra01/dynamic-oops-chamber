"""Heavy sibling-branch stress test (NOT representative of curriculum distribution)."""
import argparse, json, statistics
from pathlib import Path
from oops_language import Program
from chamber import example_scene
from oops_transactional_v07 import branch_compare

TOKENS=('Y_POS','Y_NEG','Z_POS','Z_NEG','X_POS','X_NEG','CHOOSE_AXIS','WAIT_GAP','CONTINUE','GOAL')

def run(output='synthesis_results_v09/stress.json', siblings=40, repeats=3):
    # Unique siblings share GOAL prefix, with increasing trailing rule context.
    child_rules=[]
    for a in TOKENS:
        for b in TOKENS:
            child_rules.append(('GOAL',a,b))
    candidates=[Program(r) for r in child_rules[:siblings]]
    parent=Program(('GOAL',))
    result={}
    canonical=None
    for backend in ('replay','snapshot','undo'):
        trials=[branch_compare(parent,candidates,lambda:example_scene('static'),mode=backend)
                for _ in range(repeats)]
        if canonical is None:canonical=trials[0]['results']
        assert all(x['results']==canonical for x in trials), 'Backend mismatch'
        result[backend]={
            'median_seconds':statistics.median(x['seconds'] for x in trials),
            'min_seconds':min(x['seconds'] for x in trials),
            'instructions':trials[0]['instructions'],
            'physics_steps':trials[0]['steps'],
            'journal_preimages':trials[0]['journal_entries']}
    obj={'siblings':siblings,'repeats':repeats,'scene':'static',
         'disclaimer':'Artificial shared-prefix branching stress test, not curriculum timing',
         'results':result}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj,indent=2))
    return obj

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='synthesis_results_v09/stress.json')
    ap.add_argument('--siblings',type=int,default=40)
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args();run(args.output,args.siblings,args.repeats)
