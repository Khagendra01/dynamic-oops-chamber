import unittest
from chamber import example_scene
from oops_continuable import ContinuableEvaluator
from oops_language import Program
from oops_training import replay

class ContinuationTests(unittest.TestCase):
    def test_exact_results_for_prefix_extensions(self):
        for scenario in ('empty','static','moving','crossing'):
            with self.subTest(scene=scenario):
                ev=ContinuableEvaluator()
                for rules in [('GOAL',),('GOAL','Y_POS'),('GOAL','Y_POS','CHOOSE_AXIS'),
                              ('GOAL','Y_POS','CHOOSE_AXIS','WAIT_GAP')]:
                    got=ev.evaluate(rules,scenario,lambda s=scenario:example_scene(s)).outcome
                    expected,_=replay(Program(rules),lambda s=scenario:example_scene(s))
                    self.assertEqual(got,expected)
    def test_sibling_branches_do_not_mutate_parent(self):
        ev=ContinuableEvaluator()
        factory=lambda:example_scene('static')
        ev.evaluate(('GOAL',),'s',factory)
        checkpoint=ev.checkpoints[('s',('GOAL',))]
        saved=(checkpoint.env.position,checkpoint.env.step_index,checkpoint.mem.last_direction)
        for suffix in ('Y_POS','Y_NEG','CHOOSE_AXIS','WAIT_GAP'):
            got=ev.evaluate(('GOAL',suffix),'s',factory).outcome
            expected,_=replay(Program(('GOAL',suffix)),factory)
            self.assertEqual(got,expected)
        self.assertEqual((checkpoint.env.position,checkpoint.env.step_index,checkpoint.mem.last_direction),saved)
    def test_saved_prefix_work_is_skipped(self):
        ev=ContinuableEvaluator()
        ev.evaluate(('GOAL',),'static',lambda:example_scene('static'))
        result=ev.evaluate(('GOAL','CHOOSE_AXIS'),'static',lambda:example_scene('static'))
        self.assertGreater(result.reused_steps,0)
        self.assertLess(result.simulated_steps,example_scene('static').max_steps)

if __name__=='__main__': unittest.main()
