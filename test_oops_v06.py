import unittest
from copy import deepcopy
from chamber import example_scene
from oops_language import Program, Interpreter, VMState, MotionConfig
from oops_instruction_vm import InstructionVM

class InstructionTests(unittest.TestCase):
    def setup_vm(self,rules,scene='static',library=None,memory=None):
        env=example_scene(scene)
        cfg=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed)
        vm=InstructionVM(library,config=cfg).start(Program(tuple(rules)),env.observe(),memory)
        return vm,cfg,env.observe()

    def test_bytecode_matches_original_interpreter(self):
        lib={'P001':Program(('GOAL',),'P001')}
        for scene in ('empty','static','moving','crossing'):
            for rules in (('GOAL',),('GOAL','CHOOSE_AXIS'),('WAIT_GAP','CALL:P001','CONTINUE','CHOOSE_AXIS'),('CONTINUE','GOAL')):
                with self.subTest(scene=scene,rules=rules):
                    vm,cfg,obs=self.setup_vm(rules,scene,lib)
                    output=vm.run()
                    oldmem=VMState()
                    oldout,oldtrace=Interpreter(lib,cfg).act(Program(rules),obs,oldmem)
                    self.assertEqual(output,oldout)
                    self.assertEqual(vm.trace,oldtrace)
                    self.assertEqual(vm.memory,oldmem)

    def test_resume_mid_rule_and_rollback(self):
        vm,cfg,obs=self.setup_vm(('GOAL','Y_POS'),'static')
        vm.step_instruction()  # AIM
        checkpoint=vm.checkpoint()
        expected=vm.run(); expected_trace=vm.trace[:]
        vm.restore(checkpoint)
        self.assertEqual(vm.run(),expected)
        self.assertEqual(vm.trace,expected_trace)
        vm.restore(checkpoint)
        vm.memory.last_direction=(1.,2.,3.)
        vm.restore(checkpoint)
        self.assertEqual(vm.memory.last_direction,(0.,0.,0.))

    def test_extend_before_end_and_exact_match(self):
        for scene in ('static','crossing'):
            for next_rule in ('Y_POS','Y_NEG','CHOOSE_AXIS','WAIT_GAP'):
                with self.subTest(scene=scene,rule=next_rule):
                    vm,cfg,obs=self.setup_vm(('GOAL',),scene)
                    while not vm.done and vm.frames[0].bytecode[vm.frames[0].pc][0]!='END_NO_MATCH':
                        vm.step_instruction()
                    if vm.done:continue
                    parent=vm.checkpoint()
                    vm.replace_unmatched_tail(Program(('GOAL',next_rule)))
                    output=vm.run()
                    mem=VMState()
                    old,trace=Interpreter(config=cfg).act(Program(('GOAL',next_rule)),obs,mem)
                    self.assertEqual(output,old)
                    self.assertEqual(vm.trace,trace)
                    self.assertEqual(vm.memory,mem)
                    vm.restore(parent)
                    self.assertEqual(vm.frames[0].bytecode[vm.frames[0].pc][0],'END_NO_MATCH')

    def test_nested_call_stack_checkpoint(self):
        lib={'P001':Program(('Y_NEG','CHOOSE_AXIS'),'P001')}
        vm,cfg,obs=self.setup_vm(('CALL:P001','Y_POS'),'static',lib)
        vm.step_instruction()
        self.assertEqual(len(vm.frames),2)
        cp=vm.checkpoint()
        out=vm.run(); trace=list(vm.trace)
        vm.restore(cp)
        self.assertEqual(len(vm.frames),2)
        self.assertEqual(vm.run(),out)
        self.assertEqual(vm.trace,trace)

if __name__=='__main__':unittest.main()

class BranchSearchTests(unittest.TestCase):
    def test_search_shared_instruction_prefix(self):
        from chamber import Observation
        from oops_instruction_vm import InstructionPrefixBranchSearch
        env=example_scene('static'); old=env.observe()
        obs=Observation(old.time_s,old.step_index,(.30,.5,.5),(.9,.5,.5),old.obstacles,old.chamber_size)
        cfg=MotionConfig(radius=env.radius,dt=env.dt,max_speed=env.max_speed)
        parent=Program(('GOAL',))
        children=[Program(('GOAL',r)) for r in ('Y_POS','Y_NEG','CHOOSE_AXIS','WAIT_GAP')]
        search=InstructionPrefixBranchSearch(config=cfg)
        results=search.alternatives(parent,children,obs)
        naive=0
        for child,out,trace,mem in results:
            vm=InstructionVM(config=cfg).start(child,obs)
            self.assertEqual(vm.run(),out)
            self.assertEqual(vm.trace,trace)
            self.assertEqual(vm.memory,mem)
            naive+=vm.instructions
        self.assertLess(search.instructions_executed,naive)
