"""Correctness, bounded storage, and deterministic search parity for v0.9."""
import unittest
from types import SimpleNamespace
from chamber import example_scene
from oops_language import Program
from oops_training import replay
from oops_prefix import standard_tasks
from oops_prefix_tree_v09 import LazyPrefixCurriculum


class LazyPrefixTreeTests(unittest.TestCase):
    def test_all_candidates_match_clean_replay(self):
        variants=[('GOAL',),('GOAL','Y_POS'),('GOAL','Y_NEG'),
                  ('GOAL','CHOOSE_AXIS'),('WAIT_GAP','GOAL'),
                  ('GOAL','Z_POS','Y_NEG'),('GOAL','Y_POS','Z_NEG'),
                  ('CONTINUE','GOAL','CHOOSE_AXIS')]
        for kind in ('empty','static','moving','crossing'):
            task=SimpleNamespace(name=kind,factory=lambda kind=kind:example_scene(kind))
            for mode in ('snapshot','undo'):
                engine=LazyPrefixCurriculum(backend=mode,promote_after=0,max_nodes=3)
                for rules in variants:
                    with self.subTest(kind=kind,mode=mode,rules=rules):
                        expected,_=replay(Program(rules),task.factory)
                        got,_=engine._evaluate(rules,task)
                        self.assertEqual(expected,got)
                        self.assertLessEqual(len(engine.nodes),3)

    def test_rollback_noncontamination_with_nested_calls(self):
        lib={'P001':Program(('GOAL',),'P001')}
        task=SimpleNamespace(name='static',factory=lambda:example_scene('static'))
        rules=[('CALL:P001','Y_POS'),('CALL:P001','Y_NEG'),
               ('CALL:P001','CHOOSE_AXIS'),('CALL:P001','WAIT_GAP')]
        for mode in ('snapshot','undo'):
            model=LazyPrefixCurriculum(backend=mode,promote_after=0)
            model.frozen=dict(lib)
            for candidate in rules*2:
                expected,_=replay(Program(candidate),task.factory,lib)
                result,_=model._evaluate(candidate,task)
                self.assertEqual(result,expected)
            self.assertGreater(model.branch_hits,0)

    def test_lazy_promotion_and_bounded_eviction(self):
        task=SimpleNamespace(name='moving',factory=lambda:example_scene('moving'))
        engine=LazyPrefixCurriculum(backend='undo',promote_after=2,max_nodes=1)
        candidates=[('GOAL',r) for r in ('Y_NEG','Y_POS','Z_NEG','Z_POS','CHOOSE_AXIS')]
        for c in candidates:
            engine._evaluate(c,task)
        self.assertGreater(engine.branch_hits,0)
        self.assertLessEqual(engine.peak_nodes,1)
        self.assertEqual(engine.created_nodes,1)
        self.assertEqual(engine.extra_parent_physics,3)

    def test_search_parity_with_replay_and_determinism(self):
        models=[]
        for backend in ('replay','snapshot','undo'):
            engine=LazyPrefixCurriculum(backend=backend,promote_after=1,max_nodes=8)
            for task in standard_tasks():
                self.assertTrue(engine.train_task(task).success)
            models.append(engine)
        reference=[(p.id,p.rules) for p in models[0].frozen.values()]
        for model in models[1:]:
            self.assertEqual(reference,[(p.id,p.rules) for p in model.frozen.values()])
            self.assertEqual([s.search_work_units for s in model.history],
                             [s.search_work_units for s in models[0].history])
            self.assertLessEqual(model.peak_nodes,8)

    def test_negative_node_cache_if_prefix_never_falls_through(self):
        task=SimpleNamespace(name='empty',factory=lambda:example_scene('empty'))
        model=LazyPrefixCurriculum(backend='snapshot',promote_after=0)
        for token in ('Y_POS','Y_NEG','WAIT_GAP','CHOOSE_AXIS'):
            model._evaluate(('GOAL',token),task)
        self.assertEqual(model.unusable_nodes,1)
        self.assertEqual(model.branch_hits,0)


if __name__=='__main__':
    unittest.main()
