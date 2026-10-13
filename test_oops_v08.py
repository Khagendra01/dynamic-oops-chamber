import unittest
from chamber import example_scene
from oops_language import Program
from oops_training import replay
from oops_curriculum_v08 import TransactionalCurriculum
from oops_prefix import standard_tasks

class IntegratedCurriculumTests(unittest.TestCase):
    def test_candidates_match_replay(self):
        candidates=[('GOAL',),('GOAL','CHOOSE_AXIS'),('GOAL','Y_POS'),('WAIT_GAP','GOAL'),('CONTINUE','GOAL')]
        for scene in ('empty','static','moving','crossing'):
            from types import SimpleNamespace
            task=SimpleNamespace(name=scene,factory=lambda scene=scene:example_scene(scene))
            for backend in ('snapshot','undo'):
                engine=TransactionalCurriculum(backend=backend)
                for rules in candidates:
                    with self.subTest(scene=scene,backend=backend,rules=rules):
                        expected,_=replay(Program(rules),task.factory,{})
                        got,_=engine._evaluate(rules,task)
                        self.assertEqual(got,expected)
    def test_frozen_code_all_lanes_identical(self):
        models=[]
        for backend in ('replay','snapshot','undo'):
            model=TransactionalCurriculum(backend=backend,max_depth=4)
            for task in standard_tasks():
                self.assertTrue(model.train_task(task).success)
            models.append(model)
        self.assertEqual([tuple(p.rules for p in m.frozen.values()) for m in models][0],
                         tuple(p.rules for p in models[1].frozen.values()))
        self.assertEqual(tuple(p.rules for p in models[0].frozen.values()),
                         tuple(p.rules for p in models[2].frozen.values()))
        self.assertGreater(models[1].branch_hits,0)
        self.assertGreater(models[2].undo_rollbacks,0)

if __name__=='__main__':unittest.main()
