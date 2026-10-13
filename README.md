# Dynamic OOPS — bounded incremental program synthesis for reactive planning in dynamic 3D chambers

**Research checkpoint v1.2 (experimental validation complete).**

This repository is a runnable, auditable research prototype: a deterministic
3D sphere-in-chamber simulator, a small motion DSL, a *bounded,
OOPS-inspired* incremental program synthesizer (frozen-code reuse,
cumulative curriculum validation, persistent runtime memory — each
independently switchable), three conventional planner baselines, and the
full v1.2 experiment suite (ablations + 2×2×2 factorial, generalization,
planner comparison, observation-degradation sweep, throughput profile).

> **Start here: [`EXPERIMENT_V12_REPORT.md`](EXPERIMENT_V12_REPORT.md)**
> — the complete v1.2 report (methodology, fairness rules, measured results
> with confidence intervals, negative results, limitations, and
> reproducibility commands). Machine-readable aggregates:
> [`experiments_v12/results/analysis.json`](experiments_v12/results/analysis.json);
> derived tables: [`experiments_v12/results/analysis_tables.md`](experiments_v12/results/analysis_tables.md).

## Headline results (all measured; see the report for CIs and design)

| Hypothesis | Result |
|---|---|
| H1 generalization (train L1–2 → test unseen L3–5) | 98.0 % on unseen velocity-event scenes; 97.3 % feasible on unseen observation-corruption; **27.2 % on dense unseen L5 (documented limitation)**; every feasible L1 static scene solved |
| H2 frozen-code reuse | **5.4× less synthesis work** (544k→101k units), 7.5× less wall, **+2.9 pp** deployment success |
| H3 reactive, causal-only | Graceful degradation under 13 observation degradations (worst −18.0 pp, still 0.743); 0 deadline misses in 20 500 C5 episodes |
| H4 vs conventional planners | Wins/ties all 15 paired per-level cells; **+7.5…+20.0 pp vs all three planners on dense L5**; 0.1–0.3 ms decisions vs 3–109 ms |

Documented negative results: cumulative curriculum validation adds **0.0 pp**
deployment at 3.1–15.9× extra synthesis work; persistent memory is −0.2 pp
overall but **+11.7 pp** in the no-reuse stateless factorial corner; GPU
batched search is **overhead-bound (0.17×)** at this curriculum's size
(exact parity maintained).

## Quick start (v1.2)

```bash
python3.11 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m unittest discover -p "test_*.py" -q          # 122 tests
.venv/bin/python -m experiments_v12.run_ablations --seeds 5 --levels 1,2,3,4,5
.venv/bin/python -m experiments_v12.analyze_results              # regenerate all tables
```

GPU profiling (Modal, L40S): `modal run modal_oops_v12.py` (writes
`experiments_v12/results/gpu_profile/l40s_v12_profile.json`).

## Repository map

| Path | Contents |
|---|---|
| `EXPERIMENT_V12_REPORT.md` | **Main v1.2 report** (start here) |
| `experiments_v12/` | v1.2 package: scenarios, causal observer, shared prediction, planners, ablation matrix, synthesis, evaluation, statistics, 5 experiment runners, 57 tests |
| `experiments_v12/results/` | All raw episode rows (JSON/CSV), per-seed synthesis traces, summaries, derived analysis |
| `chamber.py`, `oops_language.py`, `oops_prefix.py`, `oops_training.py` | Core simulator + DSL + prefix synthesizer (unchanged by v1.2) |
| `planners.py` | Privileged oracle (future-reading) — **reference only, never a fair baseline** |
| `oops_batched_search_v11.py`, `modal_oops_v11.py`, `modal_oops_v12.py` | GPU-batched evaluator (v1.1 machinery) + Modal apps |
| `tests/`, `test_*.py` | Legacy v0.x–v1.1 tests + v1.2 tests (122 total) |
| `*_SPEC.md`, `README_V10.md`, `BATCHED_SYNTH_V11.md` | Historical design docs (v0.3–v1.1) |

## Scope and honesty notes

- The synthesizer is **bounded, OOPS-inspired** incremental program search
  (frozen library + extend lane + `CALL`), **not** Schmidhuber's full OOPS
  `Try` algorithm; no bias-optimal timing guarantee.
- The geometry inside primitives (`GOAL`, `CHOOSE_AXIS`, `WAIT_GAP`,
  `CONTINUE`) is **hand-engineered**; the synthesizer discovers *which
  primitives to compose and reuse*. Program-language attribution is an
  open audit item (see report §11).
- No collision-free guarantees beyond recorded episodes; zero deadline
  misses does not imply collision-freedom. Point-sphere agent, axis-aligned
  wall-reflecting cuboids, dt = 0.5 s — not a physical robot.
- All planner comparisons share the same causal observation, the same
  swept-collision rule, the same safety shield, and the same 0.12 m
  per-step command cap. The privileged oracle is labelled and excluded from
  fair tables.

---

## Historical documentation (v0.3–v0.9)

> **v1.0 GPU handoff:** see [README_V10.md](README_V10.md) and [GPU_SETUP_V10.md](GPU_SETUP_V10.md) for PyTorch batched execution, parity tests, and GPU installation. Historical v0.3–v0.9 documentation follows.

# Dynamic 3D Chamber — v0.3 Reactive Program Synthesis

**Runnable, Python-standard-library-only, headless research prototype.**

This release extends the v0.2 moving-cuboid chamber and **bounded OOPS-inspired program search** with three capabilities:

- Geometry-conditioned direction selection (`CHOOSE_AXIS`): at each observation, rank six candidate axis movements using a deterministic one-step lookahead and stored direction preference.
- Deliberate waiting (`WAIT_GAP`): only wait when a direct step is blocked, waiting itself is safe for the next control period, some obstacles are moving, and the known-motion model predicts an opening within four control periods.
- Memory-based continuation (`CONTINUE`): use the previously emitted movement direction if the next step is safe and doesn't move substantially farther from the goal.

**Important distinction:** the geometry routines and the evaluation scores inside these primitives are **manually programmed**. The bounded synthesizer discovers **which primitives to compose and reuse**. This is **not** a faithful implementation of Schmidhuber's full OOPS algorithm, and nothing here guarantees real-time or physical-robot safety.

## Quick start

```bash
cd dynamic_oops_chamber
python -m unittest -v test_chamber.py test_oops.py test_oops_v03.py
python run_extended.py
python run_v03_holdout.py --count 100
```

No GPU or third-party Python packages needed. Tested with Python 3.13; requires Python 3.10+.

## Example frozen programs from the extended curriculum

```text
P001: [GOAL]
P002: [CALL:P001, CHOOSE_AXIS]
P003: [WAIT_GAP, CALL:P002]
P004: [CONTINUE, CALL:P003]
```

A program is a sequence of guarded rules, executed against current observations on every simulated control step. A successful rule emits a velocity. If no rule succeeds, the interpreter falls back to a stationary command, which may itself be unsafe for a moving obstacle; the simulation detects that as a collision.

**Offline:** the synthesizer replays candidate programs on the cumulative deterministic curriculum, allocating *candidate counts* between frozen-code extensions and fresh programs. Candidates satisfying scene and stage criteria become frozen code.

**Online:** the interpreter executes the frozen program with *new current observations*, so a moving cuboid may cause a different action. It does not run full program synthesis every frame.

### Extended curriculum

| Stage | Training constraint | Learned behavior |
|---|---|---|
| `empty` | Solve an empty chamber | `GOAL` |
| `static` | Solve around a cuboid and previous tasks | `CHOOSE_AXIS` |
| `north_blocked`, `south_blocked` | Different obstructed detours | Prior learned code reused |
| `crossing` | Require deliberate wait and zero implicit fallback waits | `WAIT_GAP` |
| `moving_gate` | Enclosed corridor: wait for moving gate | Prior learned code reused |
| `moving` | Static and moving obstacles | Prior learned code reused |
| `efficient_empty` | Lower **proxy** computational work than prior program | `CONTINUE` |

The `crossing` stage intentionally requires explicit waiting, and the final stage uses a designed work-unit threshold. These are *training objectives*, not spontaneous discovery of the objectives themselves.

## Paired 100-scene held-out test

Fixed random seeds **100–199** with exactly the same simulated scenes for both policies:

| Metric | Frozen v0.2 `P002` | Frozen v0.3 `P004` |
|---|---:|---:|
| Successful episodes | 76 / 100 | 97 / 100 |
| Collisions | 24 | 3 |
| Simulator shield interventions | 24 | 3 |
| Mean steps (successful episodes only) | 12.13 | 13.04 |
| Mean weighted **proxy** work units per episode | 88.02 | 77.81 |

All 76 scenes solved by v0.2 were solved by v0.3; v0.3 solved 21 others. Three scenes remain unsolved with detected collisions. The v0.2 rule list is replayed in the v0.3 interpreter (same basic legacy rule semantics) for a paired comparison. These are small demonstration results and **not** a research-quality estimate of real-world generalization.

**Work units are not CPU time.** The deterministic proxy assigns extra weight to more expensive geometric operations (`GOAL`, `CHOOSE_AXIS`, `WAIT_GAP`). The measured program-step counts also have selection bias because failed episodes are excluded.

## Files and outputs

| File | Role |
|---|---|
| `chamber.py` | Deterministic 3D sphere-and-cuboids environment |
| `oops_language.py` | DSL bytecode compiler, interpreter, live observation-based geometry primitives |
| `oops_training.py` | Bounded incremental synthesis, curriculum, freezing, replay |
| `oops_scenarios.py` | Constructed north/south detour and moving-gate scenes |
| `run_extended.py` | Train/freeze v0.3 and save instruction traces |
| `run_v03_holdout.py` | Paired 100-scene evaluation against v0.2 |
| `test_chamber.py`, `test_oops.py`, `test_oops_v03.py` | 27 automated tests |
| `baselines/v02_frozen_library.json` | Original frozen v0.2 rule list, retained for comparison |
| `synthesis_results_v03/frozen_library.json` | Discovered v0.3 programs |
| `synthesis_results_v03/heldout_comparison.csv` | One row per seed per policy |
| `synthesis_results_v03/heldout_summary.json` | Aggregated metrics and caveats |
| `DSL_SPEC.md` | Input contract, bytecode, exact primitive semantics, limitations |

## Design and scope limits

- Point/sphere agent, **not an articulated robotic arm**. Axis-aligned, wall-reflecting cuboids; no general mesh geometries, torque constraints, or inverse kinematics.
- Exact current obstacle positions and velocities; known deterministic movement and reflection law. Future obstacle trajectories are extrapolated from observations, not directly obtained from the simulator.
- Continuous swept collision checking uses conservative expanded AABBs; a simulator shield also validates the chosen step. No claim of real hardware safety.
- The new `CHOOSE_AXIS` scoring and `WAIT_GAP` prediction logic are hand-engineered primitives, not programs synthesized from arithmetic instructions.
- The synthesizer explores short complete rule tuples ranked by hand-set priors and calls frozen code. It does not implement the original OOPS `Try` prefix machinery or bias-optimal timing guarantee.
- Offline curriculum stages are small and deliberately selected. No rigorous confidence intervals or comparisons with strong conventional dynamic planners yet.
- The explicit wait rule may fail with unexpected obstacle accelerations, occlusion, or wrong velocity measurements. Waiting can be unsafe.

The next scientific milestone is to measure wall-clock latency and program-search effort against ablated versions (`no CHOOSE_AXIS`, `no WAIT_GAP`, `no CONTINUE`, and `no frozen calls`) and a non-oracle reactive planner under matched per-decision computation limits.

---

## v0.4: Controlled ablations + executable program-prefix search

This version retains v0.1–v0.3 functionality and introduces:

- `run_ablation_v04.py`: removes one synthesized capability from all frozen programs and compares each against the exact same held-out scenes. Also compares persistent VM memory against resetting memory every observation.
- `oops_prefix.py`: probability-budgeted depth-first search **over executable program prefixes** (instead of completed rule tuples). Each branch gets a conditional-probability share of the remaining budget; search phases double; frozen-prefix extension and fresh search receive separate 50/50 work budgets; passing programs are frozen.
- `run_prefix_v04.py`: paired incremental-reuse versus fresh-from-scratch synthesis across the 8-stage curriculum, with matched held-out evaluation.
- `test_oops_v04.py`: budget, determinism, prefix-search, library roundtrip and ablation tests.
- `PREFIX_SEARCH_SPEC.md` and `ABLATION_REPORT.md`: methods, measured outcomes and important scientific limitations.

Run from the project directory:

```bash
python -m unittest -v test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py
python run_ablation_v04.py --count 100
python run_prefix_v04.py
```

Outputs are saved under `synthesis_results_v04/` by default.

**This v0.4 engine is closer to OOPS but is NOT the full original universal, bias-optimal `Try` algorithm.** It tests executable partial DSL macro-prefixes using reset/replay, not a continuable universal VM with fine-grained state backtracking. Measured *search-work* values are synthetic allocations; measured CPU replay-call counts are reported separately. Do not interpret these results as real-time safety or general robot performance.

## v0.5: Continuable program-prefix execution

Run `python run_continuation_v05.py` to compare complete episode replay against transactional continuation at the first unmatched rule. The implementation is in `oops_continuable.py` and tested by `test_oops_v05.py`. See [`CONTINUATION_SPEC.md`](CONTINUATION_SPEC.md) for exact scope and limitations. This is not the complete original OOPS `Try` algorithm.

## v0.6: Instruction-level resumable VM

See [INSTRUCTION_VM_SPEC.md](INSTRUCTION_VM_SPEC.md). The new `oops_instruction_vm.py` supports exact instruction-boundary snapshots of program counter, nested call stack, registers, trace, observation, and persistent VM state. `InstructionPrefixBranchSearch` reuses a single executed prefix across candidate child programs. This does not yet replace the full curriculum search evaluator or implement the original OOPS `Try` undo-log mechanism.

```bash
python -m unittest -q test_chamber.py test_oops.py test_oops_v03.py test_oops_v04.py test_oops_v05.py test_oops_v06.py
python run_instruction_v06.py
```

## v0.7 multi-step transactional branch evaluator

See [TRANSACTION_V07_SPEC.md](TRANSACTION_V07_SPEC.md). Compare full episode replay,
complete-state snapshots and selective reversible write logs, with seven alternative
rules branching at the first unmatched instruction and continuing for the rest of the
moving-obstacle episode. Run `python run_transaction_v07.py --repeats 5`.

**Scope:** instruction-level multi-step evaluation experiment only. The main ordered
prefix curriculum in `oops_prefix.py` still uses its earlier evaluator.

## v0.8 integrated transactional curriculum

`oops_curriculum_v08.py` wires v0.7 snapshot and undo-log branch execution into the entire ordered, probability-budgeted curriculum from v0.4. `run_curriculum_v08.py` compares full replay, full-state checkpointing, and selective undo-log backtracking using identical training tasks and held-out scenarios. `test_oops_v08.py` validates candidate equivalence and frozen-program parity. See [CURRICULUM_V08_SPEC.md](CURRICULUM_V08_SPEC.md) for details and honest performance results.

```bash
python run_curriculum_v08.py
python -m unittest -q test_oops_v08.py
```

## v0.9 lazy bounded prefix tree (CPU handoff)

The new `oops_prefix_tree_v09.py` implements a capped LRU prefix-state cache with **lazy promotion** after repeated sibling evaluations, instruction-boundary continuation across multi-step episodes, capped prefix-preparation probes, and backends for replay, full snapshots, and reversible write logs. It inherits the v0.8 ordered curriculum, probability budgets, and frozen-code semantics. Its unit tests verify equivalence and rollback isolation. Full replay may still be fastest on small workloads; use the separate branch-stress benchmark to understand crossover behavior.

```bash
python -m unittest discover -p 'test_*.py' -q
python run_prefix_tree_v09.py --repeats 3 --holdout 100
python run_branch_stress_v09.py --siblings 40 --repeats 3
```

See `PREFIX_TREE_V09_SPEC.md` for the algorithm and honest limitations, and `GPU_HANDOFF.md` before setting up Isaac Lab or transferring to a GPU/physical robot. The project is **CPU-only** at this point.
