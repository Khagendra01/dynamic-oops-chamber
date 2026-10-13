import unittest
from chamber import example_scene
from oops_language import Program
from oops_transactional_v07 import EpisodeMachine,UndoLog,branch_compare

PARENT=Program(('GOAL',))
CHILDREN=[Program(('GOAL',r)) for r in ('Y_POS','Y_NEG','CHOOSE_AXIS','WAIT_GAP','CONTINUE')]

class MultiStepTests(unittest.TestCase):
    def test_all_modes_match(self):
        for scene in ('static','moving','crossing','empty'):
            with self.subTest(scene=scene):
                factory=lambda scene=scene:example_scene(scene)
                results=[branch_compare(PARENT,CHILDREN,factory,mode=mode) for mode in ('replay','snapshot','undo')]
                self.assertEqual(results[0]['results'],results[1]['results'])
                self.assertEqual(results[0]['results'],results[2]['results'])
                self.assertLessEqual(results[1]['steps'],results[0]['steps'])
                self.assertEqual(results[1]['instructions'],results[2]['instructions'])
    def test_undo_restores_instruction_boundary_and_environment(self):
        machine=EpisodeMachine(lambda:example_scene('static'))
        journal=UndoLog()
        machine.advance(PARENT,True,journal)
        mark=journal.mark()
        base=machine.snapshot()
        journal.before(machine)
        machine.vm.replace_unmatched_tail(CHILDREN[2])
        machine.advance(CHILDREN[2],journal=journal)
        journal.rollback(machine,mark)
        self.assertEqual(machine.env.observe(),base[0].observe())
        self.assertEqual(machine.mem,base[1])
        self.assertEqual(machine.stats,base[3])
        self.assertEqual(machine.vm.frames[0].pc,base[2].frames[0].pc)
        self.assertEqual(machine.vm.frames[0].bytecode,base[2].frames[0].bytecode)
        self.assertEqual(machine.vm.trace,base[2].trace)
    def test_nested_calls(self):
        lib={'P001':Program(('GOAL',),'P001')}
        parent=Program(('CALL:P001',))
        children=[Program(('CALL:P001',r)) for r in ('Y_POS','CHOOSE_AXIS','WAIT_GAP')]
        factory=lambda:example_scene('static')
        runs=[branch_compare(parent,children,factory,lib,mode=mode) for mode in ('replay','snapshot','undo')]
        self.assertEqual(runs[0]['results'],runs[1]['results'])
        self.assertEqual(runs[0]['results'],runs[2]['results'])
if __name__=='__main__':unittest.main()
