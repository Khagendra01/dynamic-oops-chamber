# v0.9: Lazy bounded prefix tree, capped probes, and offline GPU handoff

## Scope

`oops_prefix_tree_v09.py` retains the v0.8 `PrefixOOPS` ordered curriculum, probability-weighted prefix visitation, two extension/fresh search lanes, budget doubling, and frozen-program semantics. It changes only candidate evaluation, **not** the searched program language or task acceptance function. It is a bounded OOPS-inspired macro-DSL implementation—not universal OOPS `Try`, not a guaranteed real-time planner, and not a robot-arm controller.

## Lazy prefix tree

A cached node is indexed by (scene name, factory identity, tuple of DSL rules). A shared node contains a VM stopped at the first top-level `END_NO_MATCH`, plus either a deep snapshot or an instruction/physics write-ahead undo journal. When a new sibling program is evaluated, the node is restored, its `END_NO_MATCH` is replaced with a longer program, and its entire **remaining multi-step** episode is executed. Each child is checked against the same deterministic replay oracle used in earlier versions. To avoid initializing expensive parents for one-off children, a branch is *promoted* only after the configurable number of previously evaluated distinct children; default `promote_after=2` (third child). A `max_probe_steps=4` threshold aborts fruitless parent checkpoints early. This probe work is included in physics/instruction counts, even when no reusable checkpoint is found.

`max_nodes=96` bounds retained entries (including negative/no-fallthrough entries). The cache and child counts are cleared at each new training task; a previously frozen library is not shared as mutable execution state. The VM and simulator use purely deterministic internal state, so restoring or rolling back local Python data does not have external side effects. This is not sufficient for rollback of a real physical robot.

### Limitations and methodological caveats

- The node cache is an LRU-truncated **logical prefix trie**, not an always-live universal search stack. Some prefixes are re-executed to create their shared checkpoint. An instruction-level state record is reused **after** promotion, not at every instruction in the universal language.
- Undo log records Python mutable-state preimages, not Schmidhuber's original `Try` state-change stack semantics or proof assumptions. Snapshots may be faster in this small Python environment.
- The dynamic-obstacle experiments use perfect current coordinates and velocities, axis-aligned cuboids, a spherical agent, deterministic bounce dynamics, and a hand-designed safety shield. Algorithms cannot be directly deployed to a physical robot.
- The collision checker is not a complete rigid-body articulated-arm checker, and the hand-designed `CHOOSE_AXIS` and `WAIT_GAP` primitives encapsulate considerable domain logic. Search gains do not mean the full path planner was learned from raw geometry.
- Synthetic probability budgets differ from true preemptible runtime allocation and do **not** establish bias-optimality or real-time guarantees. Replay results are cached across doubling phases in all backends.

## Results and reproducibility

Run `python run_prefix_tree_v09.py --repeats 3 --holdout 100 --max-probe-steps 4` and compare `replay`, `snapshot`, and `undo` on identical frozen libraries. The time trials rotate backend order. The `run_branch_stress_v09.py --siblings 40 --repeats 3` benchmark deliberately evaluates 40 different candidate programs sharing one blocked prefix; it is a *separate workload*, not representative of the eight-stage curriculum. Use the paired results, never the synthetic search work alone, to judge backtracking overhead. Traces and JSON results are written to `synthesis_results_v09/`.

## CPU-to-GPU milestone

The current engine requires Python standard library only; **do not install Isaac Sim/PyTorch just to run these tests**. On your workstation, first run the CPU test suite and three-run benchmark to obtain hardware baselines. GPU/Isaac Lab support requires an explicit scene adapter exposing `reset`, `observe`, `step`, serialized state, and collision checking. Keep the VLA out of Paper 1 and preserve the CPU oracle until the adapter produces matching deterministic results on the same initial scenes.
