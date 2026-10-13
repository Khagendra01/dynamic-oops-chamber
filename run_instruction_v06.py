"""Instruction-prefix branching benchmark (single-observation execution).

Compares multiple child rules at the same previously reached END_NO_MATCH.
No physical wall-clock or episode-level improvement is claimed.
"""
import json
from pathlib import Path
from chamber import example_scene, Observation
from oops_instruction_vm import InstructionVM
from oops_language import Program, MotionConfig

SCENES=('static','moving')
CHILDREN=('Y_POS','Y_NEG','X_POS','X_NEG','Z_POS','Z_NEG','CHOOSE_AXIS','WAIT_GAP')

def evaluate():
    rows=[]
    for scene in SCENES:
        env=example_scene(scene)
        cfg=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed)
        obs0=env.observe()
        obs=Observation(obs0.time_s,obs0.step_index,(.30,.5,.5),(.9,.5,.5),obs0.obstacles,obs0.chamber_size)
        # Parent selection is deterministic. Use a blocked GOAL where possible.
        parent=InstructionVM(config=cfg).start(Program(('GOAL',)),obs)
        while not parent.done and parent.frames[0].bytecode[parent.frames[0].pc][0]!='END_NO_MATCH':
            parent.step_instruction()
        if parent.done:continue
        cp=parent.checkpoint(); inherited=parent.instructions
        child_instructions=0; full_instructions=0
        for rule in CHILDREN:
            parent.restore(cp)
            parent.replace_unmatched_tail(Program(('GOAL',rule)))
            got=parent.run();child_instructions+=parent.instructions-inherited
            fresh=InstructionVM(config=cfg).start(Program(('GOAL',rule)),obs)
            expected=fresh.run();full_instructions+=fresh.instructions
            assert got==expected and parent.trace==fresh.trace and parent.memory==fresh.memory
        rows.append(dict(scene=scene,children=len(CHILDREN),
                         full_replay_instructions=full_instructions,
                         instruction_continuation_instructions=inherited+child_instructions,
                         shared_parent_instructions=inherited))
    return rows

if __name__=='__main__':
    rows=evaluate()
    result={'rows':rows,'total_full_instructions':sum(r['full_replay_instructions'] for r in rows),
      'total_continuation_instructions':sum(r['instruction_continuation_instructions'] for r in rows)}
    folder=Path('synthesis_results_v06');folder.mkdir(exist_ok=True)
    (folder/'instruction_branch_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
