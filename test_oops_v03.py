"""v0.3 DSL regression tests for dynamic geometry, waiting and persistent state."""
import tempfile
import unittest
from pathlib import Path

from chamber import example_scene
from oops_language import Interpreter, MotionConfig, Program, VMState, compile_program
from oops_scenarios import north_blocked, south_blocked, moving_gate
from oops_training import IncrementalSynthesizer, extended_curriculum, replay, load_library


class DynamicDsl03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learner = IncrementalSynthesizer(max_rules=3, budget_per_stage=1500)
        cls.stages = []
        for name, scene, wait, nofallback, events, work in extended_curriculum():
            discovery = cls.learner.train_stage(name, scene,
                require_explicit_wait=wait, forbid_fallback=nofallback,
                max_trace_events=events, max_work_units=work)
            cls.stages.append(discovery)
        cls.frozen = cls.learner.frozen

    def test_programs_are_found_and_frozen(self):
        self.assertTrue(all(stage.program for stage in self.stages))
        self.assertEqual([x.id for x in self.frozen.values()],
                         ['P001','P002','P003','P004'])
        self.assertIn('CHOOSE_AXIS',self.frozen['P002'].rules)
        self.assertIn('WAIT_GAP',self.frozen['P003'].rules)
        self.assertIn('CONTINUE',self.frozen['P004'].rules)
        self.assertIn('CALL:P003',self.frozen['P004'].rules)

    def test_opposite_detours_selected_from_live_geometry(self):
        actions = []
        for factory in [north_blocked,south_blocked]:
            _, trace = replay(self.frozen['P004'],factory,self.frozen,True)
            selected=[e['velocity'][1] for e in trace
                      if 'SELECTED:CHOOSE_AXIS' in e['vm'] and e['velocity'][1] != 0]
            self.assertTrue(selected)
            actions.append(selected[0])
        self.assertLess(actions[0],0)  # obstacle blocks upper route: choose -Y
        self.assertGreater(actions[1],0) # blocks lower route: choose +Y

    def test_explicit_wait_for_moving_gate(self):
        result, trace = replay(self.frozen['P004'],moving_gate,self.frozen,True)
        self.assertTrue(result.success)
        self.assertEqual(result.fallback_waits,0)
        self.assertGreater(result.explicit_waits,0)
        self.assertIn('EMIT_WAIT_GAP',trace[0]['vm'])
        self.assertEqual(trace[0]['velocity'],[0.0,0.0,0.0])

    def test_wait_is_not_selected_for_static_blocker(self):
        result,trace=replay(self.frozen['P004'],north_blocked,self.frozen,True)
        self.assertTrue(result.success)
        self.assertEqual(result.explicit_waits,0)
        self.assertFalse(any('EMIT_WAIT_GAP' in step['vm'] for step in trace))

    def test_memory_continuation_selected_for_cost_limited_stage(self):
        mem_prog=self.frozen['P004']
        older=self.frozen['P003']
        old,_=replay(older,lambda:example_scene('empty'),self.frozen)
        new,_=replay(mem_prog,lambda:example_scene('empty'),self.frozen)
        self.assertTrue(old.success and new.success)
        self.assertEqual(old.steps,new.steps)
        self.assertGreater(new.continued_actions,0)
        self.assertLess(new.work_units, old.work_units)
        self.assertLessEqual(new.work_units,70)

    def test_memory_reset_on_each_episode(self):
        a,at=replay(self.frozen['P004'],moving_gate,self.frozen,True)
        b,bt=replay(self.frozen['P004'],moving_gate,self.frozen,True)
        self.assertEqual(a,b)
        self.assertEqual(at,bt)
        self.assertIn('AIM CONTINUE',at[0]['vm'])
        self.assertIn('BLOCKED',at[0]['vm'])

    def test_saved_library_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json'
            self.learner.save(path)
            loaded=load_library(path)
            self.assertEqual(loaded,self.frozen)
            scene=lambda: example_scene('random',102)
            self.assertEqual(replay(loaded['P004'],scene,loaded,True),
                             replay(self.frozen['P004'],scene,self.frozen,True))

    def test_no_causal_violation_at_bounces(self):
        env=moving_gate()
        obs=env.observe()
        # VM gets observations, not env or future obstacle states.
        self.assertFalse(hasattr(obs,'future_obstacles'))
        vm=Interpreter(self.frozen,MotionConfig(env.radius,env.dt,env.max_speed))
        action,trace=vm.act(self.frozen['P004'],obs,VMState())
        self.assertEqual(action,(0.,0.,0.))
        self.assertIn('EMIT_WAIT_GAP',trace)

    def test_bytecode_contains_explicit_wait_conditional(self):
        ops=[x[0] for x in compile_program(Program(('WAIT_GAP','GOAL')))]
        self.assertIn('CHECK_WAIT_GAP',ops)
        self.assertIn('IF_UNSAFE_SKIP',ops)
        self.assertIn('EMIT_WAIT',ops)


if __name__=='__main__':
    unittest.main()
