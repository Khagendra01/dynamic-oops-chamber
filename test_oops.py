"""DSL interpreter, observed dynamics, search, frozen reuse and replay tests."""
import tempfile
import unittest
from pathlib import Path

from chamber import ChamberEnv, Cuboid, example_scene, add, mul
from oops_language import (Program, VMState, Interpreter, MotionConfig,
                           predicted_safe, compile_program)
from oops_training import IncrementalSynthesizer, load_library, replay, curriculum


class OopsPrototypeTests(unittest.TestCase):
    def test_compilation_contains_explicit_branch_and_store(self):
        ops=[i[0] for i in compile_program(Program(('GOAL','Y_POS')))]
        self.assertEqual(ops.count('IF_UNSAFE_SKIP'),2)
        self.assertEqual(ops.count('STORE_DIRECTION'),2)
        self.assertEqual(ops[-1],'END_NO_MATCH')

    def test_guarded_goal_rejected_and_detour_selected(self):
        env=ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5),radius=.035,
                       obstacles=(Cuboid('wall',(.3,.5,.5),(.15,.2,.2)),))
        obs=env.reset()
        vm=Interpreter(config=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed))
        action,trace=vm.act(Program(('GOAL','Y_POS')),obs,VMState())
        self.assertIn('BLOCKED',trace)
        self.assertEqual(action[0],0.0)
        self.assertGreater(action[1],0.)

    def test_last_register_and_branch(self):
        env=example_scene('empty')
        vm=Interpreter(config=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed))
        memory=VMState()
        first,_=vm.act(Program(('Y_POS',)),env.observe(),memory)
        self.assertGreater(first[1],0)
        second,trace=vm.act(Program(('LAST','GOAL')),env.observe(),memory)
        self.assertEqual(first,second)
        self.assertIn('AIM LAST',trace)
        blank,trace=vm.act(Program(('LAST','GOAL')),env.observe(),VMState())
        self.assertIn('BLOCKED',trace)
        self.assertGreater(blank[0],0)

    def test_predictor_matches_actual_next_step_with_bounce(self):
        env=ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5),radius=.04,
                       obstacles=[Cuboid('bounce',(.72,.5,.5),(.15,.16,.18),(.38,0,0))],
                       dt=.5,max_speed=.7)
        obs=env.observe()
        cfg=MotionConfig(env.radius,env.dt,env.max_speed)
        for v in [(0,0,0),(.24,0,0),(0,.24,0),(0,0,.24),(.1,.1,0)]:
            expect=env.is_motion_safe(obs.position,add(obs.position,mul(v,env.dt)),0,env.dt)
            self.assertEqual(predicted_safe(obs,v,cfg),expect)

    def test_replay_determinism(self):
        prog=Program(('GOAL','Y_POS'))
        a,log1=replay(prog,lambda:example_scene('moving'),with_trace=True)
        b,log2=replay(prog,lambda:example_scene('moving'),with_trace=True)
        self.assertEqual(a,b)
        self.assertEqual(log1,log2)

    def test_no_oracle_future_data_input(self):
        env=example_scene('moving')
        obs=env.observe()
        self.assertFalse(hasattr(obs,'future_obstacles'))
        self.assertFalse(hasattr(obs,'obstacle_center_at'))

    def test_incremental_freezing_and_reuse(self):
        synth=IncrementalSynthesizer(max_rules=2,budget_per_stage=200)
        a=synth.train_stage(*curriculum()[0])
        b=synth.train_stage(*curriculum()[1])
        self.assertEqual(a.program.rules,('GOAL',))
        self.assertIn('CALL:P001',b.program.rules)
        self.assertTrue(any(x in b.program.rules for x in ('Y_POS','CHOOSE_AXIS')))
        self.assertEqual(len(synth.frozen),2)
        c=synth.train_stage(*curriculum()[2])
        self.assertEqual(c.found_lane,'reused')
        self.assertEqual(len(synth.frozen),2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'library.json'
            synth.save(path)
            loaded=load_library(path)
            self.assertEqual(loaded,synth.frozen)
            result,_=replay(loaded['P002'],lambda:example_scene('static'),loaded)
            self.assertTrue(result.success)

    def test_insufficient_search_budget_does_not_freeze_unsolved_task(self):
        synth=IncrementalSynthesizer(max_rules=1,budget_per_stage=4)
        self.assertTrue(synth.train_stage(*curriculum()[0]).program)
        failed=synth.train_stage(*curriculum()[1])
        self.assertIsNone(failed.program)
        self.assertEqual(len(synth.frozen),1)
        self.assertEqual(len(synth.tasks),1)

    def test_no_collision_in_training_curriculum(self):
        synth=IncrementalSynthesizer(max_rules=2,budget_per_stage=200)
        for name,factory in curriculum():
            discovery=synth.train_stage(name,factory)
            self.assertIsNotNone(discovery.program)
            result,_=replay(discovery.program,factory,synth.frozen)
            self.assertTrue(result.success)
            self.assertFalse(result.collision)
            self.assertEqual(result.interventions,0)


if __name__=='__main__':
    unittest.main()
