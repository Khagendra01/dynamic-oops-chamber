# v0.7: Multi-step instruction checkpointing and undo-log rollback

## Scope

Extends the v0.6 instruction VM into **complete future simulated episode execution after a candidate branch**. A shared parent rule sequence executes through potentially multiple observations until the first top-level `END_NO_MATCH`. Branch continuations replace that instruction and execute the remaining episode with newly observed moving cuboids at every step.

Backends:

1. `replay`: fresh chamber and VM for each candidate.
2. `snapshot`: `deepcopy` a complete chamber + VM + persistent memory + episode counters at the unmatched instruction.
3. `undo`: record field preimages before each instruction or physics transition and roll them backward to the branch point. The journal captures full mutable field values, frame references, and trace lengths, not whole simulator copies. Immutable compiled bytecode and observations are shared.

## Run

```bash
python -m unittest -q test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py test_oops_v05.py test_oops_v06.py test_oops_v07.py
python run_transaction_v07.py --repeats 5
```

Outputs: `synthesis_results_v07/transaction_comparison.csv` and `.json`.

## Validity

`test_oops_v07.py` verifies identical complete episode outcomes among all three methods for four named scenes, nested frozen-program calls, and rollback of environment state, persistent memory, bytecode, program counter, VM trace and counters after a branched episode.

## Important limitations

- This is **multi-step executable prefix branching**, not yet wired as the backend of the full ordered `PrefixOOPS` curriculum in `oops_prefix.py`.
- Branching is at the first top-level unmatched `END_NO_MATCH` only; multiple arbitrary nested branch points and self-delimiting OOPS programs are not implemented.
- Undo journal is in Python and logs mutable field preimages per transition; it does not implement the full original OOPS `Try` algorithm, asymptotic guarantees, or a universal undo model for arbitrary simulation side effects.
- Running all remaining candidate episodes still includes most later instructions. No claim of proportional savings in all environments.
- Runtime medians from five repetitions are exploratory, not stable microbenchmark measurements. No real-robot safety claim.
- Deterministic point/sphere-vs-moving-axis-aligned-cuboid environment and its known dynamics are the only supported world model.
