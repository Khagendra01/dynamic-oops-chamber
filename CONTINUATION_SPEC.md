# v0.5 — continuable execution and branch restoration

## What was implemented

`oops_continuable.py` adds a transactional checkpoint evaluator for the chamber DSL. Each candidate program is a sequence of fall-through motion rules. When an executed prefix has no safe action, the interpreter falls through to its implicit wait. The evaluator snapshots the complete deterministic `ChamberEnv` and `VMState` *before* this first fallback, including counters used for reward and work accounting. Descendant candidates (programs with the same existing rule prefix and appended rules) can resume their simulation episode from this snapshot rather than from the initial state.

Checkpoint objects are copied before child execution. Sibling program candidates cannot mutate their ancestor's stored state. When a program never falls through, descendants produce the same episode outcome and are returned from the cached result without simulating further steps.

`PrefixOOPS(continuable=True)` activates the continuation evaluator while retaining the v0.4 probability-weighted prefix search, two-lane reuse/fresh search and budget-accounting experiment. The default `False` preserves the v0.4 replay implementation as a control.

## Exactness and limitations

- For the current deterministic rule language and given scene factories, episode outcomes exactly match the original `oops_training.replay` function on the tested scenarios.
- This is **episode-level checkpoint-and-resume at the first unmatched rule**, not instruction-by-instruction `Try` continuation. Within the resumed timestep, the current prefix is interpreted again. It does not implement instruction-level undo logs or writable search bias.
- The procedure deep-copies chamber and VM state; memory and checkpoint-copy overhead are not reported by the simulated-step metric.
- The observed-step count omits any work spent copying checkpoints and compiling instructions, so it is **not a wall-clock speedup**.
- Program search remains bounded by DSL vocabulary, depth, node count and search-budget proxies; OOPS's original universal-machine bias-optimality theorem does not apply.
- There is no real-robot safety guarantee.

## Reproduction

```
python -m unittest -q test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py test_oops_v05.py
python run_continuation_v05.py
```

Saved experiment output: `synthesis_results_v05/continuation_comparison.json`.

## Next algorithmic improvement

Replace the first-fallback episode checkpoint with an instruction-step VM continuation record containing the instruction pointer, call stack, candidate registers, undo-log offsets, observation, and chamber state. This is necessary to restore arbitrary instruction-level branches and make a substantively closer reproduction of the original OOPS `Try` algorithm.
