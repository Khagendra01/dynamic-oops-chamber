"""v0.4 regression, budget, replay, and controlled-ablation tests."""
import tempfile, unittest
from pathlib import Path
from chamber import example_scene
from oops_language import Program
from oops_training import load_library, replay
from oops_scenarios import north_blocked, moving_gate
from oops_prefix import PrefixOOPS, Task, standard_tasks, load_prefix_library
from run_ablation_v04 import ablate, replay_stateless

class V04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=load_library('synthesis_results_v03/frozen_library.json')
        cls.ablation_variants={x:ablate(cls.library,x) for x in ['CHOOSE_AXIS','WAIT_GAP','CONTINUE']}
        cls.prefix_learner=PrefixOOPS(max_phase_budget=131072,max_prefixes_per_phase=30000)
        cls.initial=[]
        for task in standard_tasks()[:5]:
            cls.initial.append(cls.prefix_learner.train_task(task))

    def test_remove_feature_at_all_library_levels(self):
        for feature,lib in self.ablation_variants.items():
            self.assertFalse(any(feature in p.rules for p in lib.values()))
            self.assertEqual(len(lib),len(self.library))
            self.assertEqual(list(lib),list(self.library))

    def test_choose_axis_matters_on_targeted_scene(self):
        lib=self.ablation_variants['CHOOSE_AXIS']
        a,_=replay(self.library['P004'],north_blocked,self.library)
        b,_=replay(lib['P004'],north_blocked,lib)
        self.assertTrue(a.success)
        self.assertFalse(b.success)

    def test_explicit_wait_is_not_silent_fallback(self):
        lib=self.ablation_variants['WAIT_GAP']
        a,_=replay(self.library['P004'],moving_gate,self.library)
        b,_=replay(lib['P004'],moving_gate,lib)
        self.assertGreater(a.explicit_waits,0)
        self.assertEqual(b.explicit_waits,0)

    def test_memory_help_work_proxy(self):
        scene=lambda:example_scene('empty')
        a,_=replay(self.library['P004'],scene,self.library)
        b=replay_stateless(self.library['P004'],scene,self.library)
        self.assertTrue(a.success and b['success'])
        self.assertLess(a.work_units,b['work_units'])
        self.assertGreater(a.continued_actions,0)
        self.assertEqual(b['continued_actions'],0)

    def test_search_is_incremental_and_tests_prefixes(self):
        first,second=self.initial[:2]
        self.assertEqual(first.rules,['GOAL'])
        self.assertEqual(second.rules,['GOAL','CHOOSE_AXIS'])
        self.assertEqual(second.selected_lane,'extend')
        self.assertGreater(second.prefixes_tested,2)
        self.assertLess(second.replay_calls,second.prefixes_tested)

    def test_probability_budget_subtrees_do_not_exceed_allocations(self):
        for stage in self.initial:
            for p in stage.phases:
                self.assertLessEqual(p['work_units'],p['lane_budget'])
            phases=sorted({p['phase_budget'] for p in stage.phases})
            self.assertTrue(all(b==2*a for a,b in zip(phases,phases[1:])))
            for phase in phases:
                rows=[p for p in stage.phases if p['phase_budget']==phase]
                if len(rows)==2:
                    self.assertEqual(rows[0]['lane_budget'],rows[1]['lane_budget'])

    def test_frozen_code_can_be_replayed_without_mutation(self):
        lib=self.prefix_learner.frozen
        self.assertEqual(lib['P001'].rules,('GOAL',))
        self.assertEqual(lib['P002'].rules,('GOAL','CHOOSE_AXIS'))
        outcome1,trace1=replay(lib['P002'],north_blocked,lib,True)
        outcome2,trace2=replay(lib['P002'],north_blocked,lib,True)
        self.assertTrue(outcome1.success)
        self.assertEqual((outcome1,trace1),(outcome2,trace2))

    def test_library_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'frozen_prefix.json'
            self.prefix_learner.save(target)
            self.assertEqual(load_prefix_library(target),self.prefix_learner.frozen)

    def test_call_prefix_is_tested_and_can_be_discovered(self):
        # Work through crossing; fresh lane has CALL available.
        outcome=self.initial[-1]
        self.assertTrue(outcome.success)
        self.assertIn('CALL:P002',outcome.rules)
        self.assertGreater(outcome.prefixes_tested,0)
        self.assertEqual(outcome.selected_lane,'fresh')

    def test_fresh_search_has_no_reused_frozen_primitives(self):
        learner=PrefixOOPS(reuse=False,max_phase_budget=16384,max_prefixes_per_phase=30000)
        result=learner.train_task(standard_tasks()[0]);self.assertTrue(result.success)
        result=learner.train_task(standard_tasks()[1]);self.assertTrue(result.success)
        self.assertFalse(any(rule.startswith('CALL:') for x in learner.frozen.values() for rule in x.rules))
        self.assertEqual(result.selected_lane,'fresh')

    def test_prefix_search_keeps_true_failures(self):
        # This restrictive budget cannot discover any program; no false freeze.
        learner=PrefixOOPS(starting_budget=1,max_phase_budget=1)
        result=learner.train_task(Task('empty',lambda:example_scene('empty')))
        self.assertFalse(result.success)
        self.assertEqual(learner.frozen,{})

if __name__=='__main__':unittest.main()
