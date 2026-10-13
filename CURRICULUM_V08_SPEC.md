# v0.8: Instruction-Level Transactional Curriculum Integration

## What is integrated

`TransactionalCurriculum` subclasses the bounded `PrefixOOPS` search implementation and retains its existing ordered eight-task curriculum, 50/50 extension/fresh search lanes after the first frozen program, conditional instruction probabilities, phase-budget doubling, task acceptance criteria, and program freezing.

Each candidate receives an **executable program-prefix evaluation**. The `snapshot` and `undo` backends attempt to reuse a saved instruction-boundary state of its immediate parent (the candidate without its final DSL rule). The parent is run to its first top-level `END_NO_MATCH` instruction. A child is patched into that point and completes the **remaining multi-step** episode. The checkpoint includes the interpreter's program counter, nested call frames, registers, persistent motion memory and simulated chamber state. The `undo` backend selectively rolls back changed VM/chamber fields; `snapshot` uses full copying. If a parent finishes without such a branch point, the child runs from the start.

To ensure reproducibility, candidate outcomes and cost accounting are compared with the original full-replay curriculum. Search-bias work units are intentionally unchanged; optimization of simulator execution does not change the candidate's assigned search probability or its acceptance score.

## Commands

```bash
python -m unittest -q test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py test_oops_v05.py test_oops_v06.py test_oops_v07.py test_oops_v08.py
python run_curriculum_v08.py
```

## Measured run (one execution, Python environment)

| Backend | Stages | VM instructions | Physics steps | Checkpoint hits | Seconds | Held-out success (20) |
|---|---:|---:|---:|---:|---:|---:|
| Replay | 8 | 117961 | 18775 | 0 | 2.0320 | 19 |
| Full snapshot | 8 | 122791 | 19616 | 251 | 2.2659 | 19 |
| Undo log | 8 | 122791 | 19616 | 251 | 2.5997 | 19 |

**Interpretation:** In this implementation, eager building of single-parent branch checkpoints costs more simulated work than is saved by reusing them. This is a valid negative systems result, not a claimed speedup. The benchmark is one run; timing is hardware and workload dependent. The 20-scene holdout is small. Neither the current backend nor original OOPS has been established as a real-time safe robot planner.

## Limitations

- Bounded, macro-rule DSL and synthetic probability-weighted budget accounting are inherited from v0.4. It is not a universal OOPS implementation or the original precise `Try` allocation proof.
- Parent checkpoints are eagerly constructed, not shared among all descendants of a trie node, so overlapping prefixes may be evaluated more than once. Parent work is included in accounting.
- Only `END_NO_MATCH` branch points are eligible; the program does not suspend at arbitrary VM instructions for synthesis.
- Program discovery costs should not be conflated with runtime of the frozen planner.
- Full state/undo correctness is verified on deterministic simulated states, not physical robot controllers or external side effects.

## Suggested next optimization

Use a persistent DFS prefix trie: construct a checkpoint only when at least two children need it and retain it across recursive visits, rather than separately initializing an immediate parent for individual candidates. Track actual instruction and physics work including checkpoint construction, peak memory, and repeated time measurements.
