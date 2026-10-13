# Program-prefix search specification (v0.4)

## Why v0.4 exists

v0.3 searched **fully constructed, ordered rule tuples** ranked by heuristic token scores. v0.4 searches **partial executable rule prefixes**, extending them one instruction at a time in probability-weighted depth-first order. This is a concrete step toward Method 3.1 and the `Try` algorithm in Jürgen Schmidhuber, *Optimal Ordered Problem Solver* (2002): https://arxiv.org/abs/cs/0207097 .

It is NOT a full reimplementation of original universal OOPS; see differences below.

## Domain specific language

The interpreter is unchanged from v0.3 and is defined in `oops_language.py`:

- `GOAL`: try direct progress to the goal.
- `CHOOSE_AXIS`: evaluate six axis-aligned motions with a **hand-designed** lookahead heuristic.
- `WAIT_GAP`: explicitly wait if a moving blocker may open the direct path soon.
- `CONTINUE`: reuse a prior motion direction stored in the VM state, if still useful and safe.
- `X_POS`, `X_NEG`, `Y_POS`, `Y_NEG`, `Z_POS`, `Z_NEG`, `LAST`: other deterministic motion candidates.
- `CALL:Pnnn`: invoke a frozen program. Recursion is prevented and calls are bounded.

Each rule compiles to explicit `AIM`, `CHECK_SAFE`, conditional skip, `STORE_DIRECTION` and `EMIT` instructions, or a waiting test. The first applicable rule emits a velocity, and a failed program implicitly waits.

## Prefix search algorithm

- A search *node* is a prefix of 0..`max_depth` DSL rules. A prefix is interpreted and tested as soon as it is created, even if it would later be extended.
- DFS considers suffix tokens in descending normalized prior probability. Each child receives its `P(token | prefix)` fraction of the parent's remaining work budget. This prevents exhaustive exploration of a high-priority subtree from starving other branches.
- Each successful candidate is checked on the newest task **and all retained earlier tasks**. The system freezes a program only if those tests pass fully within its work allocation.
- The global phase budget doubles: 256, 512, 1024, etc. In later stages, exactly half is assigned to **literal frozen-prefix extension**, and half to **fresh DSL rule search**. In the fresh lane, `CALL:Pnnn` is a legal instruction, so previous code can still be reused even if a new rule must run first.
- For deterministic task replay we use a cache keyed by frozen library + candidate rule prefix + task name. Search work is still charged as if the prefix replay occurred at each phase; *actual* replay calls are reported separately.
- Prefix costs are based on an interpreter trace-work proxy, not measured instruction wall-clock time. When a candidate replay costs more than its allocation, only its allocation is charged. **Important limitation:** the current prototype still evaluates that candidate fully before rejecting it, unlike a preemptible original `Try` implementation.
- A hard maximum program depth and maximum prefixes per phase keep execution finite. Those limits, VM macros, and inability to self-modify search bias mean no original OOPS bias-optimality guarantees apply.

## Fresh-versus-reuse experiment

`run_prefix_v04.py` trains two conditions on the same *ordered, deterministic* 8-stage curriculum (`extended_curriculum()`):

1. `incremental_reuse`: frozen code may be called or literally extended; a solved new task can reuse code immediately.
2. `fresh_search`: no frozen code or `CALL` tokens are visible; all cumulative tasks are re-solved from scratch at every stage. Its discovered program from a prior stage is saved for reporting only, not reused during synthesis.

Both train through `efficient_empty`, which imposes a VM-work threshold; this tests whether prefix search can learn `CONTINUE`, not merely success at navigation. We report total budget-accounting units, real replay calls, candidate prefix evaluations, programs, final held-out success/collisions.

**Interpretation:** This is one deterministic curriculum and prior, not evidence of a statistically general speedup. Comparisons should be repeated on varied curricula, search priors, seed splits, computation limits, and domain abstractions.

## Commands

```bash
python run_prefix_v04.py
python run_ablation_v04.py --count 100
python -m unittest -v test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py
```

Output files (in `synthesis_results_v04/`):

- `incremental_reuse_prefix_trace.json`, `fresh_search_prefix_trace.json`: phase-by-phase search, prefix counts and frozen code.
- `prefix_comparison.csv`, `prefix_summary.json`, `prefix_holdout.csv`: paired search comparison.
- `ablation_results.csv`, `ablation_summary.json`: feature-isolation and memory-reset comparison.

## Next steps for scientific fidelity

1. Replace full-prefix episode replay with a continuable program machine whose partial execution requests the next instruction and whose state changes can be undone exactly, like `Try`.
2. Enforce phase budgets using measured execution steps or preemptible VM instruction costs, rather than retrospective proxy estimates.
3. Permit code-dependent updates to suffix instruction probabilities and evaluate the effect with proper priors and complexity accounting.
4. Test multiple curriculum orders and larger dynamic worlds. Report uncertainty intervals and matched-wall-clock comparisons.
5. Distinguish a real-time motion planner (with an independently tested safety controller) from offline search over reactive programs.
