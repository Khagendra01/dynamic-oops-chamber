import unittest
import torch
from chamber import ChamberEnv, Cuboid, example_scene
from torch_chamber import TorchChamberBatch

class TorchChamberTests(unittest.TestCase):
    def test_canonical_batched_parity(self):
        envs=[example_scene(name,seed=i) for i,name in enumerate(['empty','static','moving','crossing'])]
        gpu=TorchChamberBatch(envs)
        for k in range(25):
            velocities=[]
            for env in envs:
                if env.done:
                    velocities.append((0.,0.,0.)); continue
                p=env.position; g=env.goal
                import math
                d=[g[i]-p[i] for i in range(3)]
                n=math.sqrt(sum(a*a for a in d))
                if k%5==3:
                    velocities.append((0.,0.,0.))
                elif k%7==5:
                    velocities.append((.14,-.07,.08))
                elif n>1e-10:
                    v=min(.12,n)/env.dt/n
                    velocities.append(tuple(x*v for x in d))
                else:
                    velocities.append((0.,0.,0.))
            tensor=torch.tensor(velocities,dtype=torch.float64)
            out=gpu.step(tensor)
            for idx, env in enumerate(envs):
                if env.done: continue
                expected=env.step(velocities[idx])
                self.assertEqual(expected.collision,bool(out.collision[idx]))
                self.assertEqual(expected.safety_intervention,bool(out.intervention[idx]))
                self.assertEqual(expected.done,bool(out.done[idx]))
                for a,b in zip(expected.observation.position,out.observation_positions[idx].tolist()):
                    self.assertAlmostEqual(a,b,places=9)

    def test_random_batched_parity(self):
        envs=[example_scene('random',i) for i in range(20)]
        batched=TorchChamberBatch(envs)
        import random
        r=random.Random(511)
        for k in range(22):
            velocities=[]
            for env in envs:
                if env.done:
                    velocities.append((0.,0.,0.)); continue
                import math
                raw=[r.uniform(-1,1) for _ in range(3)]
                n=math.sqrt(sum(x*x for x in raw))
                velocities.append(tuple(.16*x/n for x in raw))
            res=batched.step(torch.tensor(velocities,dtype=torch.float64))
            for idx,env in enumerate(envs):
                if env.done: continue
                expected=env.step(velocities[idx])
                self.assertEqual(expected.collision,bool(res.collision[idx]),msg=f'seed {idx} time {k}')
                self.assertEqual(expected.safety_intervention,bool(res.intervention[idx]),msg=f'seed {idx} time {k}')
                self.assertEqual(expected.done,bool(res.done[idx]),msg=f'seed {idx} time {k}')
                for a,b in zip(expected.observation.position,res.observation_positions[idx].tolist()):
                    self.assertAlmostEqual(a,b,places=8)

    def test_bounce_between_steps(self):
        env=ChamberEnv(start=(.15,.5,.5),goal=(.85,.5,.5),radius=.035,dt=.5,
            obstacles=[Cuboid('bouncer',(.5,.86,.5),(.2,.2,.2),(0.,.4,0.))])
        batched=TorchChamberBatch([env]); velocities=[(.2,0.,0.)]
        for k in range(7):
            if env.done:break
            old=env.step(velocities[0]); new=batched.step(torch.tensor(velocities,dtype=torch.float64))
            self.assertEqual(old.collision,bool(new.collision[0]))
            self.assertEqual(old.safety_intervention,bool(new.intervention[0]))
            self.assertAlmostEqual(old.observation.position[0],new.observation_positions[0,0].item(),places=9)

    def test_reject_multi_bounce_per_step(self):
        env=ChamberEnv(obstacles=[Cuboid('fast',(.5,.5,.5),(.1,.1,.1),(3.,0.,0.))])
        with self.assertRaises(ValueError): TorchChamberBatch([env])

    def test_cpu_gpu_optional(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA GPU unavailable in this runtime')
        envs=[example_scene('random',i) for i in range(16)]
        cpu=TorchChamberBatch(envs,device='cpu',dtype=torch.float32)
        gpu=TorchChamberBatch(envs,device='cuda',dtype=torch.float32)
        for t in range(8):
            vel=cpu.greedy_action()
            a=cpu.step(vel)
            b=gpu.step(vel.to('cuda'))
            torch.testing.assert_close(a.observation_positions,b.observation_positions.cpu(),atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.equal(a.intervention,b.intervention.cpu()))
            self.assertTrue(torch.equal(a.collision,b.collision.cpu()))

if __name__=='__main__':unittest.main()
