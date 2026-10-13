# v0.6 Instruction-Level Continuation (Research Prototype)

## What is implemented

`oops_instruction_vm.py` provides an instruction-step interpreter for the existing DSL. Its checkpoint contains:
- Current compiled bytecode, program counter, candidate velocity, safety register and selected action for every call-stack frame
- Full nested frozen-program call stack
- Persistent VM memory (`last_direction`, emissions, last mode, position, explicit waits)
- Current observation, execution trace, instruction counter, remaining budget, completion and output state

The VM can stop after **any** bytecode instruction, snapshot, and resume or restore from the snapshot. Snapshot restoration is copy-isolated. At a top-level `END_NO_MATCH` boundary, a parent program can be extended with new rules without repeating any executed prefix instruction. `InstructionPrefixBranchSearch` evaluates multiple children from one shared snapshot.

## Limits relative to original OOPS Try

- Full episode/curriculum search in `oops_prefix.py` still uses v0.5 rule-level continuation; v0.6 instruction-level branching currently operates **within a decision observation**, not across an entire sequence of environment steps.
- Checkpoints are full object deep copies, not the more memory-efficient change journal/backtracking stack of the original Try algorithm.
- The search is over a finite domain-specific rule language, not a universal programming language; OOPS bias-optimality results do not apply.
- The branch benchmark counts executed bytecode instructions, NOT wall-clock time, GPU time, or control latency.
- This is a deterministic sphere-versus-axis-aligned-cuboid simulator and is NOT safe for deployment on a physical robot.

## Verification

Run `python -m unittest -q test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py test_oops_v05.py test_oops_v06.py`.

Run `python run_instruction_v06.py` to produce `synthesis_results_v06/instruction_branch_comparison.json`.

The example benchmark shares three parent instructions over eight alternatives for each blocked-observation test. The result is reproducible but not a generalized speed claim.

## Next milestone

Integrate instruction checkpoints across simulator time steps into the full OOPS curriculum search, then replace full-snapshot cloning with an undo log, and compare instruction counts, memory and wall-clock runtime separately.
