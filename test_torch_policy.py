import unittest, math
import torch
from chamber import example_scene
from oops_language import Interpreter, MotionConfig, VMState, Program
from torch_chamber import TorchChamberBatch
from torch_frozen_policy import TorchFrozenPolicy

FROZEN={
'P001':Program(('GOAL',),'P001'),
'P002':Program(('CALL:P001','CHOOSE_AXIS'),'P002'),
'P003':Program(('WAIT_GAP','CALL:P002'),'P003'),
'P004':Program(('CONTINUE','CALL:P003'),'P004'),
}

class FrozenPolicyTests(unittest.TestCase):
  def test_empty_canonical_random_parity(self):
    names=['empty','static','moving','crossing'] + ['random']*32
    envs=[example_scene(n,i) for i,n in enumerate(names)]
    batch=TorchChamberBatch(envs)
    policy=TorchFrozenPolicy(batch,FROZEN['P004'],FROZEN)
    cpus=[Interpreter(FROZEN,MotionConfig(radius=e.radius,dt=e.dt,max_speed=e.max_speed)) for e in envs]
    memory=[VMState() for e in envs]
    for step in range(40):
      batch_vel=policy.act().tolist()
      for j,e in enumerate(envs):
        if e.done: continue
        ref,_=cpus[j].act(FROZEN['P004'],e.observe(),memory[j])
        for a,b in zip(ref,batch_vel[j]):
            self.assertAlmostEqual(a,b,places=8,msg=f'scene={j}, step={step}, rules={policy.rules}')
        self.assertEqual(memory[j].explicit_waits,int(policy.explicit_waits[j]))
      output=batch.step(torch.tensor(batch_vel,dtype=batch.dtype))
      for j,e in enumerate(envs):
        if e.done:continue
        result=e.step(batch_vel[j]);self.assertEqual(result.collision,bool(output.collision[j]))
        self.assertEqual(result.done,bool(output.done[j]))

  def test_flatten_rejects_recursive_calls(self):
    env=TorchChamberBatch([example_scene('empty')])
    f={'P1':Program(('CALL:P2','GOAL'),'P1'),
       'P2':Program(('CALL:P1','X_NEG'),'P2')}
    p=TorchFrozenPolicy(env,f['P1'],f)
    self.assertEqual(p.rules,('X_NEG','GOAL'))

if __name__=='__main__':unittest.main()
