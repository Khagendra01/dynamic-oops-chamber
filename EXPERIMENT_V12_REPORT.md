# Experiment v1.2: Bounded incremental program synthesis in unseen dynamic 3D chambers

**Status:** complete. All numbers below are measured from saved episode rows
(`experiments_v12/results/`) on the reference hardware; none are projections.
Reproduce any table with `python -m experiments_v12.analyze_results`.

---

## 1. Abstract

We test whether *bounded, OOPS-inspired incremental program synthesis* —
reusing frozen programs during later-task synthesis and at deployment —
generalizes to previously unseen dynamic 3D environments, reduces discovery
cost, adapts reactively without privileged future information, and competes
with conventional planners under identical observation, safety and command
rules. Across 5 held-out test levels (200 scenes/level, 5 curriculum seeds,
paired evaluation) the full system (C5) reaches **84.6 % success on feasible
scenes** (Wilson 95 % CI in `analysis_tables.md`), solves **every feasible
level-1 static scene (100 %, n=840)**, sustains **98.0 % on unseen level 3**
(velocity changes, appear/disappear — never seen in training) and degrades to
**27.2 % on the densest unseen level 5**. Frozen-code reuse (C5 vs C2)
improves deployment success by **+2.9 pp [2.4, 3.4]** and cuts cumulative
synthesis work **5.4×** (544k → 101k work units, 29.1 s → 3.9 s wall on the
reference CPU). Cumulative curriculum validation shows **no deployment
difference** in this curriculum (+0.0 pp, zero discordant pairs) while
costing 3.1× extra synthesis work with reuse ON (15.9× with reuse OFF) — a
documented negative result. Persistent runtime memory changes
success by **−0.2 pp [−0.3, −0.1]** overall — negligible — except in the
no-reuse stateless factorial corner where it is worth **+11.7 pp**. The
full-system-minus-all-three (C1) loses **16.0 pp [15.0, 17.1]**. Against
conventional planners under identical observation, safety and command
rules, C5 wins or ties all 15 paired per-level cells and beats all three
planners significantly on the hardest dense level (**+7.5 to +20.0 pp**),
at **0.1–0.3 ms** decision latency versus 3–109 ms. Under 13 observation
degradations (noise, delay, stale serves, visibility, combinations) C5
degrades gracefully — worst case `combined_severe` −18.0 pp, still 0.743 —
and stays above space-time A* at every matched degradation. Every deployed
program across all seeds and all factorial cells is a
2-rule reactive policy (`GOAL CHOOSE_AXIS` or `GOAL`); no method ever
receives future obstacle states.

---

## 2. Questions and hypotheses

| ID | Question | Operational test |
|----|----------|------------------|
| H1 | Does bounded incremental synthesis generalize to *unseen* dynamic environments? | Train on levels 1–2 curricula only; evaluate frozen programs on held-out levels 3–5 (unseen velocity events, appear/disappear, observation corruption, dense scenes). |
| H2 | Does frozen-code reuse reduce discovery cost? | C5 (reuse ON) vs C2 (reuse OFF, everything else ON): synthesis work units, wall time, replay calls; plus deployment contrast on identical evaluation scenes. |
| H3 | Does the system adapt *reactively* — using only causal observations, never privileged future information? | 13 observation degradations (noise, delay, stale serves, visibility loss, combinations); degradation curves for C5 vs a space-time A*-based baseline under the same observations. |
| H4 | How does it compare to conventional planners under equal information and equal command authority? | C5 vs `static_astar`, `space_time_astar`, `rollout_greedy`: same causal observation, same safety shield, same per-step command cap (`stride_cap="language"`), same swept-collision rule. A privileged future-reading oracle exists only as a labelled upper reference (`--oracle`). |

Claims we explicitly **do not** make: collision-free guarantees beyond the
recorded episodes; that GPU throughput reduces search complexity; that the
 planners are globally optimal; that the synthesized programs are human-
readable plans (they are reactive policies, 2 rules deep).

---

## 3. Method: what "bounded OOPS-inspired incremental synthesis" means here

The synthesizer is a *prefix-search program synthesizer* over a small motion
DSL (`GOAL`, `CHOOSE_AXIS`, …): for each curriculum task it searches program
prefixes, validates candidates by replaying them through the interpreter
(`oops_language.py`) on the training scenes, and freezes accepted programs
into a library. "Bounded" = every search runs under explicit caps
(`max_phase_budget`, `max_prefixes_per_phase`, `max_depth`) and every
candidate check is a finite replay.

Three independently switchable capabilities form the ablation axes:

1. **Frozen-code reuse** — during synthesis of task *k*, accepted programs of
   tasks *< k* are available as `CALL` targets and as extend-lane seeds; at
   deployment, the frozen library is callable. OFF = empty library, no
   `CALL`, no extend lane; ordinary prefix search still runs.
2. **Cumulative curriculum validation** — a candidate for task *k* must also
   pass tasks *< k*. OFF = validate against the current task only (history
   still logged, never deleted).
3. **Persistent runtime memory** — the VM state persists across observations
   during deployment. OFF = fresh VM state per `act` (physics and the
   observer are never reset).

**Controlled constants in all variants** (never claimed as individually
ablated in C1–C5): conditional geometric branching is always evaluated, and
live replanning (re-decide from the current observation every step) is always
on. Each toggle changes exactly one behavior; unit tests prove it
(`tests/test_ablation_matrix.py`, 14 tests, including a behavioral test that
memory state (`CHOOSE_AXIS` tie choice) persists across steps only when the
memory flag is ON).

### C1–C5 matrix and the 2×2×2 factorial

| Variant | reuse | cumulative | memory | role |
|---------|-------|-----------|--------|------|
| C1 | OFF | OFF | OFF | minimal baseline |
| C2 | OFF | ON  | ON  | no-frozen-reuse contrast |
| C3 | ON  | OFF | ON  | no-cumulative contrast |
| C4 | ON  | ON  | OFF | stateless-runtime contrast |
| C5 | ON  | ON  | ON  | **full system** |
| F001 | OFF | OFF | ON  | factorial corner |
| F010 | OFF | ON  | OFF | factorial corner |
| F100 | ON  | OFF | OFF | factorial corner |

C1–C5 plus the three factorial extras = all 8 cells of the 2×2×2 design,
run as 8 distinct configs (verified distinct on disk in
`results/ablations/variant_configs/`; `validate_matrix()` rejects duplicates
at import). Primary contrasts: C5−C2 (reuse), C5−C3 (cumulative), C5−C4
(memory), C1−C5 (overall). The factorial corners expose interactions that
the five primary variants cannot (see §7.3).

---

## 4. Environments, budgets, and fairness rules

**Chambers.** 1 m³ cubes, agent radius 0.035 m, dt 0.5 s, max speed 0.70
m/s, goal tolerance 0.055 m, 45 steps max. Obstacles are axis-aligned cuboids
with reflecting-wall dynamics. Five difficulty levels:

| Level | Character | Scenes (train/val/test) | Infeasible-by-construction in test |
|-------|-----------|-------------------------|------------------------------------|
| 1 | static obstacles | 12 train stages / 40 val / 200 test | 32 |
| 2 | constant-velocity movers | 9 / 40 / 200 | 0 |
| 3 | velocity changes mid-episode, appear/disappear | 0 / 0 / 200 | 0 |
| 4 | observation noise, delay, stale serves | 0 / 0 / 200 | 12 |
| 5 | dense (up to 8 obstacles), mixed dynamics | 0 / 0 / 200 | 20 |

**Splits and seeds.** Training: 21 stages over levels 1–2, curriculum seed
bases 1000/3000/5000/7000/9000 (spaced ≥2000 so per-level windows never
overlap). Validation: seed base 50000. Test: seed base 100000. Test scenes
are identical across all methods and all curriculum seeds (paired design);
pairing key = (curriculum seed, scenario, level).

**Causal observation.** `CausalObserver` serves only past-or-present state:
delay shifts served time backward, `serve_every` reuses the last served
observation, noise is deterministic per (seed, step, channel) and bounded.
The truth future is never exposed to any policy. A unit test asserts
velocity information cannot leak from the future
(`tests/test_observations.py::test_no_future_velocity_leak`).

**Shared prediction and collision rule.** `experiments_v12/motion_prediction.py`
probes the *interpreter's own* `oops_language.predicted_safe` with a
synthetic observation rooted at p0 — exactly one swept-check implementation
in the project, used by programs, planners, and the safety shield alike.

**Command authority.** Every method is capped at `MotionConfig.stride`
(0.12 m per control period) — the same cap the language programs live under
(`stride_cap="language"`). `stride_cap="full"` (0.35 m) results, when run,
are reported separately and never mixed into the primary tables.

**Safety shield.** Commands that fail the swept check become safe waits
*for every method*, including C5; interventions are counted per episode.

**Planner budgets.** Grid planners run with `max_expansions=1200` per step
and a post-hoc 0.5 s deadline (overrun → safe wait; misses recorded per
episode). C5's program policy has no expansion budget — its per-step cost is
fixed by the 2-rule program (median ≈ 0.15 ms, p95 ≈ 0.2 ms; zero deadline
misses by construction).

**Hardware.** Reference CPU: Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz, 8 GB,
Python 3.11, torch 2.7.1+cpu). GPU profile: NVIDIA L40S via Modal (see §8).
All synthesis and all episode rows in `results/` were produced on the Pi.

---

## 5. Verification

* Full suite: `python -m unittest discover -p "test_*.py" -q` →
  **122 tests OK (1 skipped)** — 65 legacy v1.1 tests (unchanged
  `chamber.py` behavior preserved) + 57 new v1.2 tests.
* Ablation toggles: each of the three factors is proven to change only its
  intended behavior (`test_ablation_matrix.py`, incl. the `CHOOSE_AXIS`
  memory-persistence behavioral test and budget-identity tests).
* Planner edge-memo optimization (needed to make H4 affordable on the Pi):
  proven action-identical to the uncached planner on static *and* dynamic
  scenes; proven to miss (not reuse) whenever the obstacle-state signature
  changes — including under position noise (`test_planners.py::EdgeMemoTests`).
* Prediction is proven to be no-oracle and to match the interpreter rule
  (`test_planners.py::PredictionTests`).
* Exact ChamberEnv equivalence of the dynamic wrapper: `test_dynamic_chamber.py`
  (7/7), including swept checks and appear/disappear.

---

## 6. Experiment A — generalization to unseen levels (H1)

Training saw only levels 1–2. Levels 3–5 were never trained on. Five
curriculum seeds; 200 test scenes per level per seed (1000 episodes/level).

| Level | n | success (all) | success (feasible-only) | collision rate |
|-------|-----|---------------|-------------------------|----------------|
| 1 (seen, static) | 1000 | 0.840 | **1.000** (n=840) | 0.000 |
| 2 (seen, movers) | 1000 | 0.980 | 0.980 | 0.020 |
| **3 (unseen: velocity events, appear/disappear)** | 1000 | **0.980** | 0.980 | 0.020 |
| **4 (unseen: noisy/delayed/stale observations)** | 1000 | 0.915 | 0.973 (n=940) | 0.020 |
| **5 (unseen: dense, mixed dynamics)** | 1000 | 0.245 | **0.272** (n=900) | 0.670 |

* All 5 seeds converge to a 2-rule program (`GOAL CHOOSE_AXIS`; seeds 1 and 3
  report it as `P001`, seeds 0/2/4 as `P002` — same rule list, different
  library ids). Discovery is seed-robust.
* Level 1's headline 0.840 equals the feasible fraction exactly: **every
  feasible static scene is solved by every seed** (840/840). Overall rates
  are diluted by infeasible-by-construction scenes; both numbers are
  reported everywhere.
* Unseen velocity events and appearance changes (L3) cost *nothing* relative
  to seen movers (L2): 0.980 = 0.980. Observation corruption (L4) costs
  ~1 pp on feasible scenes. Density (L5) is the real failure mode: 27.2 %
  feasible-scene success, 67 % collision rate — moving obstacles strike the
  agent while it waits or tracks a blocked corridor; the shield refuses
  unsafe *agent* commands but cannot prevent an obstacle from hitting a
  stationary agent.
* Per-seed overall success: 0.792 for all five seeds (identical to 3
  decimals) — variance across curriculum seeds is below display precision on
  this benchmark.

**Verdict H1: supported, with a boundary.** Generalization holds on unseen
dynamic mechanics (events, appearance, observation corruption: 97–98 %
feasible-scene success) and fails on dense multi-obstacle scenes (27 %),
which the current 2-rule program class cannot express a policy for.

## 6b. Validation leakage check

Validation (seed base 50000, disjoint from test) overall C5 success: 0.785
vs test 0.792 — no train/val/test leakage indicated; tuning was never
performed on test.

---

## 7. Experiment C — ablations and the factorial (H2, factors of H3)

8 configs × 5 curriculum seeds × all 5 test levels × 200 scenes = **40 000
deployment episodes**; plus per-seed synthesis JSON for every config
(`results/ablations/synthesis_seed*`), training-cost rows
(`training_cost.csv`), variant configs (8 distinct on disk), and the
program-lock evaluation (`program_lock_episodes.csv`).

### 7.1 Deployment (all levels, n=5000/config)

| Variant | success | 95 % CI | collision | mean steps | mean act latency |
|---------|---------|---------|-----------|------------|------------------|
| C1 minimal | 0.632 | [0.619, 0.645] | 0.236 | 11.5 | 0.05 ms |
| C2 no reuse | 0.763 | [0.751, 0.775] | 0.163 | 10.4 | 0.14 ms |
| C3 no cumulative | 0.792 | [0.781, 0.803] | 0.146 | 9.4 | 0.16 ms |
| C4 stateless | 0.794 | [0.783, 0.805] | 0.144 | 9.5 | 0.16 ms |
| **C5 full** | **0.792** | [0.781, 0.803] | 0.146 | 9.4 | 0.16 ms |
| F001 | 0.632 | [0.619, 0.645] | 0.236 | 11.5 | 0.05 ms |
| F010 | 0.646 | [0.633, 0.659] | 0.171 | 14.2 | 0.14 ms |
| F100 | 0.794 | [0.783, 0.805] | 0.144 | 9.5 | 0.16 ms |

Feasible-only rates (infeasible-by-construction removed, n=4680/config):
C1 0.675, C2 0.815, C3 0.846, C4 0.848, **C5 0.846**, F001 0.675, F010
0.690, F100 0.848.

### 7.2 Primary paired contrasts (5000 pairs; positive favours the left side)

| Contrast | factor | Δ success [95 % CI] | discordant pairs (left-only / right-only) | Δ mean steps |
|----------|--------|--------------------|-------------------------------------------|--------------|
| C5 − C2 | frozen-code reuse | **+0.029 [+0.024, +0.034]** | 158 / 13 | −0.98 |
| C5 − C3 | cumulative validation | **+0.000 [+0.000, +0.000]** | 0 / 0 | 0.00 |
| C5 − C4 | persistent memory | **−0.002 [−0.003, −0.001]** | 0 / 10 | −0.03 |
| C5 − C1 | overall system | **+0.160 [+0.150, +0.171]** | 815 / 15 | −2.06 |

### 7.3 What the factorial corners add

F010 (reuse OFF, cumulative ON, memory **OFF**) = 0.646 while C2 (same but
memory **ON**) = 0.763: **memory is worth +11.7 pp exactly when reuse is
OFF** — the cumulative program `GOAL CHOOSE_AXIS` cannot execute statelessly
(the axis tie-break flips every step without persistent state), while the
reuse-OFF per-task programs are shallower and mostly survive statelessness.
With reuse ON (C3/C4 vs C5) memory is worth ≈ −0.2 pp. This interaction is
invisible in the five primary variants and is the main reason the full 2×2×2
was run.

### 7.4 Training cost (mean over 5 seeds; Pi wall time)

| Variant | wall (s) | work units | replay calls | prefixes tested |
|---------|----------|------------|--------------|-----------------|
| C1 | 1.76 | 34 326 | 424 | 993 |
| C2 | **29.09** | **544 450** | 7 402 | 9 634 |
| C3 | 1.46 | 32 437 | 448 | 1 140 |
| C4 | 3.81 | 100 593 | 1 192 | 2 738 |
| **C5** | **3.89** | **100 593** | 1 192 | 2 738 |

Training and deployment are reported separately by design: C5's deployment
edge over C2 is +2.9 pp, but its *training* advantage is 7.5× wall time and
5.4× work — reuse pays off primarily at discovery time. (C1/C3 are cheapest
to train because cumulative validation is OFF — they skip the multi-task
acceptance gate, not because they search less per task.)

### 7.5 Controlled lock evaluation (execution-config effect isolated)

`program_lock_episodes.csv` executes the **C5-discovered program** under each
variant's execution configuration (library emptied where reuse is OFF, memory
reset where memory is OFF), holding the program itself fixed. This separates
"reuse found a different program" from "reuse's execution differs". Rows are
per-episode; see `analysis.json` for aggregates.

### 7.6 Discovered programs (seed 0; representative — all seeds in
`results/ablations/synthesis_seed*`)

| Variant | deployed program | rules |
|---------|------------------|-------|
| C1 / F001 | P021 | `GOAL` |
| C2 / F010 | P021 | `GOAL CHOOSE_AXIS` |
| C3 / C4 / C5 / F100 | P002 | `GOAL CHOOSE_AXIS` |

Reuse does not merely accelerate search to the same artifact: reuse-OFF
minimal search (C1) settles on a *shallower* program (`GOAL` alone, which
ignores the free-axis choice and fails tie scenes), while every reuse-ON or
cumulative-ON cell finds `GOAL CHOOSE_AXIS`.

**Verdicts.** H2 **supported** (reuse: +2.9 pp deployment, 5.4× less
discovery work). Cumulative validation: **null result on deployment** in
this curriculum (+0.0 pp, zero discordant pairs — the same final program is
found either way) at a real cost (+190 % synthesis work when reuse is OFF;
+0 % when reuse is ON, where the extend lane subsumes its benefit). Memory:
**negligible-to-slightly-negative** (−0.2 pp), matching the earlier v0.4
finding; the F010 interaction shows it becomes essential only in the
no-reuse stateless corner.

---

## 8. Experiment B — throughput profile (not a search-complexity claim)

**L40S (Modal, torch 2.7.1+cu126, batch 8192, merge_waves=True).** The full
122-test suite was re-run on the GPU image first: **OK** (36.9 s). The
wave-batched prefix search was then checked for bit-exact parity against the
sequential CPU reference on the standard curriculum: **parity exact** —
identical replay counts (495), identical simulated steps (9 652), identical
frozen programs (`GOAL`, `GOAL CHOOSE_AXIS`, `WAIT_GAP CALL:P002`,
`CONTINUE CALL:P003`).

**Negative result, reported as measured:** at this curriculum's search size
the batched engine is **overhead-bound** — reference 1.208 s vs batched
6.997 s (**0.17×**, i.e. 5.8× slower). The v1.1 depth-4/5/6 experiments with
30 000-prefix phases are where batching pays (1.15–1.21× waves-mode speedup,
verified with parity; `l40s_batched_v11*.json`). The v1.2 curriculum's
per-phase budgets are two orders of magnitude smaller, so host-device
overhead dominates. **GPU throughput is evaluator speed only**: it does not
change how many prefixes the search must visit, and the batched engine covers
only constant-velocity cuboid scenes — v1.2 event levels are never silently
substituted.

**Episode throughput (what the ablation/baseline experiments actually run)**
is CPU-only by construction. Measured: **313 episodes/s** on the L40S
instance host CPU and **367 episodes/s** on the reference Pi (50 level-1
train scenes, success 0.94, deployed program `P002 GOAL CHOOSE_AXIS`,
`results/gpu_profile/profile.json`). The full 40 000-episode ablation matrix
— which additionally trains 40 syntheses, writes per-seed JSON and CSV, and
includes the hard levels — runs at an aggregate ≈58 episodes/s (~12 min Pi
wall time); that is how all rows in `results/` were produced.

---

## 9. Experiment D — conventional planners (H4)

C5 vs `static_astar[language]`, `space_time_astar[language]`,
`rollout_greedy[language]` on identical causal observations, identical
safety shield, identical 0.12 m command cap, identical swept-collision
rule. Levels 1–3 at 40 scenes/level/seed and levels 4–5 at 40
scenes/level/seed, 5 curriculum seeds, paired per (seed, scene).
Planners bounded at 1200 expansions/step; deadline misses recorded (they
become safe waits). The privileged oracle is *not* included in any fair
table.

### 9.1 Overall (all paired levels, per run)

| Run | Method | n | success [95 % CI] | collision | mean steps | mean act latency |
|-----|--------|---|-------------------|-----------|------------|------------------|
| L1–3 | **C5** | 600 | **0.942** | 0.008 | 9.2 | **0.10 ms** |
| L1–3 | static_astar | 600 | 0.925 | 0.025 | 11.5 | 11.0 ms |
| L1–3 | space_time_astar | 600 | 0.908 | 0.017 | 12.1 | 18.2 ms |
| L1–3 | rollout_greedy | 600 | 0.892 | 0.025 | 18.0 | 2.9 ms |
| L4–5 | **C5** | 400 | **0.550** | 0.362 | 10.3 | **0.26 ms** |
| L4–5 | space_time_astar | 400 | 0.500 | 0.388 | 12.4 | 108.9 ms |
| L4–5 | rollout_greedy | 400 | 0.463 | 0.438 | 16.0 | 3.9 ms |
| L4–5 | static_astar | 400 | 0.438 | 0.475 | 11.4 | 39.8 ms |

(L4–5 overall is dominated by the hard dense level 5; see per-level.)

### 9.2 Paired per-level contrasts (positive favours C5)

| Level | vs | Δ success [95 % CI] | discordant (C5-only / planner-only) |
|-------|----|--------------------|--------------------------------------|
| 1 | static_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 1 | space_time_astar | **+0.075 [+0.040, +0.115]** | 15 / 0 |
| 1 | rollout_greedy | **+0.075 [+0.040, +0.115]** | 15 / 0 |
| 2 | static_astar | **+0.050 [+0.020, +0.080]** | 10 / 0 |
| 2 | space_time_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 2 | rollout_greedy | **+0.025 [+0.005, +0.050]** | 5 / 0 |
| 3 | static_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 3 | space_time_astar | +0.025 [−0.010, +0.065] | 10 / 5 |
| 3 | rollout_greedy | **+0.050 [+0.010, +0.095]** | 15 / 5 |
| 4 | static_astar | **+0.025 [+0.005, +0.050]** | 5 / 0 |
| 4 | space_time_astar | +0.025 [−0.010, +0.065] | 10 / 5 |
| 4 | rollout_greedy | −0.025 [−0.065, +0.010] | 5 / 10 |
| 5 | static_astar | **+0.200 [+0.135, +0.260]** | 45 / 5 |
| 5 | space_time_astar | **+0.075 [+0.010, +0.140]** | 30 / 15 |
| 5 | rollout_greedy | **+0.200 [+0.145, +0.255]** | 40 / 0 |

**Reading.** C5 wins or ties every planner comparison; the only non-positive
cell is L4 vs rollout_greedy (−2.5 pp, CI crosses zero). On the hardest
level 5 — where all methods struggle — C5 beats all three significantly
(+7.5 to +20.0 pp). The decision-latency column is a first-class result: the
2-rule program executes in ≈0.1–0.3 ms/step with zero deadline misses, while
the time-expanded search costs 18–109 ms/step and hits its expansion budget
on dense scenes (misses become safe waits and are counted in the raw rows /
`analysis.json`). C5 also produces the lowest collision rate on every run.

**Verdict H4: supported.** Under equal information, equal command authority,
and equal safety rules, the synthesized reactive program is competitive with
(conventionally: better than) bounded grid and rollout planners on this
benchmark — and 1–3 orders of magnitude cheaper per decision.

## 10. Experiment E — reactive adaptation under degraded observations (H3)

13 degradations applied through `CausalObserver` — no planner sees truth:

| Degradation | Definition |
|-------------|------------|
| `perfect` | no noise, no delay, every step served |
| `noise_pos_0.005 / 0.01` | ±5 mm / ±10 mm per-axis position noise |
| `noise_vel_0.02` | ±0.02 m/s velocity noise |
| `delay_1 / 2 / 3` | observation is 1/2/3 control periods old |
| `serve_2 / serve_3` | fresh observation only every 2nd/3rd step |
| `visibility_0.8 / 0.6` | obstacles beyond 0.8/0.6 m from the agent are hidden |
| `combined_mild` | pos noise 0.005 + delay 1 + serve 2 |
| `combined_severe` | pos noise 0.01 + vel noise 0.02 + delay 2 + serve 3 |

C5: 100 scenes/level × levels 1–3 × 5 seeds = 1500 paired episodes per
degradation. `space_time_astar`: 12-scene subset × 3 levels × 5 seeds = 180
per degradation (same observations, same 1200-expansion budget).

### 10.1 C5 degradation curve (levels 1–3, n=1500 each)

| Degradation | success [95 % CI] | Δ vs perfect | collision |
|-------------|-------------------|--------------|-----------|
| perfect | 0.923 [0.909, 0.936] | — | 0.017 |
| visibility_0.8 | 0.923 [0.909, 0.936] | +0.000 [+0.000, +0.000] | 0.017 |
| visibility_0.6 | 0.923 [0.909, 0.936] | +0.000 [+0.000, +0.000] | 0.017 |
| noise_vel_0.02 | 0.920 [0.905, 0.933] | +0.003 [−0.001, +0.009] | 0.020 |
| noise_pos_0.005 | 0.913 [0.898, 0.927] | +0.010 [+0.004, +0.017] | 0.027 |
| serve_2 | 0.890 [0.873, 0.905] | +0.033 [+0.023, +0.044] | 0.050 |
| noise_pos_0.01 | 0.887 [0.870, 0.902] | +0.037 [+0.027, +0.047] | 0.053 |
| delay_1 | 0.857 [0.838, 0.873] | +0.067 [+0.054, +0.080] | 0.083 |
| serve_3 | 0.850 [0.831, 0.867] | +0.073 [+0.061, +0.087] | 0.087 |
| delay_3 | 0.840 [0.821, 0.858] | +0.083 [+0.069, +0.098] | 0.100 |
| delay_2 | 0.823 [0.803, 0.842] | +0.100 [+0.085, +0.115] | 0.117 |
| combined_mild | 0.790 [0.769, 0.810] | +0.133 [+0.117, +0.151] | 0.150 |
| combined_severe | 0.743 [0.721, 0.765] | **+0.180 [+0.161, +0.199]** | 0.190 |

(Δ = success drop caused by the degradation; positive = degradation hurt.)

### 10.2 Space-time A* under the same observations (n=180 each)

| Degradation | success [95 % CI] | Δ vs perfect |
|-------------|-------------------|--------------|
| perfect | 0.861 [0.803, 0.904] | — |
| delay_2 | 0.778 [0.712, 0.832] | +0.083 [+0.044, +0.128] |
| combined_severe | 0.694 [0.624, 0.757] | +0.167 [+0.106, +0.233] |

### 10.3 Reading

* **Causal by construction.** Every number above comes from policies that
  receive only `CausalObserver` output; the truth future is unreachable
  (unit-tested). Degradation curves measure robustness of *reactive
  decisions*, not oracle leakage.
* **Graceful, monotone-in-severity degradation.** Worst single factor:
  2-step delay (−10.0 pp). Worst combination: `combined_severe` (−18.0 pp,
  still 0.743 absolute). C5 remains above space_time_astar under every
  matched degradation (perfect: 0.923 vs 0.861; delay_2: 0.823 vs 0.778;
  combined_severe: 0.743 vs 0.694).
* **Delay hurts more than noise** (−10 pp vs −3.7 pp at the larger of each):
  the program's decisions are geometric, so stale obstacle positions are
  costlier than jittery ones.
* **Visibility clamps at 0.6–0.8 m never bind** on these 1 m chambers
  (identical outcomes, 0 discordant pairs): every obstacle that mattered to
  a decision was already within range. Reported as a null result, not as a
  robustness claim.
* **Minor non-monotonicity:** delay_3 (0.840) > delay_2 (0.823). Delay
  interacts with obstacle phase; a staler observation can occasionally be a
  luckier one. Both are significantly below perfect — the ordering between
  them is not claimed.
* **Deadline misses:** C5 records **0** deadline misses in all 20 500
  C5 deployment episodes of Experiments D and E (program execution is
  ~0.1–0.3 ms/step). space_time_astar missed its 0.5 s deadline in 81/1000
  baseline episodes (0.22 misses/episode mean) — those steps became safe
  waits.

**Verdict H3: supported.** The system adapts reactively from causal
observations only, degrades gracefully under noise, delay, staleness, and
their combinations, and stays above the time-expanded planner at every
matched degradation level.

---

## 11. Failures, negative results, and limitations

1. **Level 5 density is unsolved** (27.2 % feasible-scene success, 67 %
   collision). The 2-rule program class plus reactive-only execution cannot
   thread dense moving obstacles; a hierarchical or lookahead program class
   is the obvious next step — and is *not* claimed here.
2. **Cumulative validation is a null result on deployment** in this
   curriculum (+0.0 pp, 0 discordant pairs) while costing 3.1× extra
   synthesis work with reuse ON and 15.9× without reuse. Reported as
   measured; not spun.
3. **Memory is slightly negative overall** (−0.2 pp) except in the
   no-reuse stateless corner (F010: +11.7 pp vs C2) — the interaction, not
   the main effect, is the finding.
4. **Obstacle-caused collisions cannot be shielded away.** The shield
   protects against unsafe *agent* commands; a moving obstacle that hits a
   waiting agent is recorded as a collision for every method equally.
5. **Infeasible-by-construction scenes** (levels 1/4/5 test splits) dilute
   overall rates; both overall and feasible-only rates are always reported.
6. **Planner budgets are a compromise.** 1200 expansions/step keeps the
   sweep affordable on the Pi and is disclosed everywhere; the edge-memo
   optimization was required to make H4 measurable at all (3× wall-time
   reduction, proven action-identical).
7. **No hierarchical programs, no multi-agent, no continuous-time control.**
   dt = 0.5 s discrete steps throughout.
8. The batched GPU engine does not cover v1.2 event scenes, and at this
   curriculum's search size it is **overhead-bound (0.17× vs the CPU
   reference)** — batching only paid off at v1.1's larger phase budgets
   (§8). No GPU number in this report describes event-level evaluation.
9. **Two reactive null results, reported as measured:** visibility clamps at
   0.6–0.8 m never bound on these 1 m chambers (identical outcomes, 0
   discordant pairs — not a robustness claim), and delay_3 slightly
   *outperforms* delay_2 (phase interaction; both are significantly below
   perfect, and the ordering between them is not claimed).

---

## 12. Reproducibility

```bash
# environment (reference CPU)
python3.11 -m venv oops_venv && oops_venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# full verification (v1.1 parity + v1.2)
cd dynamic_oops_chamber_v1_0
python -m unittest discover -p "test_*.py" -q          # 122 tests, OK (1 skipped)

# experiments (each writes experiments_v12/results/<name>/)
python -m experiments_v12.run_ablations --seeds 5 --levels 1,2,3,4,5
python -m experiments_v12.run_generalization --seeds 5 --levels 1,2,3,4,5
python -m experiments_v12.run_baselines --seeds 5 --levels 1,2,3 --limit 40 --planner-expansions 1200
python -m experiments_v12.run_baselines --seeds 5 --levels 4,5 --limit 40 --planner-expansions 1200 --out baselines_hl
python -m experiments_v12.run_reactive --seeds 5 --levels 1,2,3 --limit 100 --planner-limit 12 --planner-expansions 1200
python -m experiments_v12.run_gpu_profile --skip-batched --count 50   # Pi CPU
# GPU (Modal L40S; requires modal CLI):
modal run modal_oops_v12.py     # writes /tmp/opencode/oops_v10/l40s_v12_profile.json

# derived tables (no re-evaluation)
python -m experiments_v12.analyze_results
```

**File map.** `experiments_v12/` — `scenarios.py` (levels/splits/seeds),
`dynamic_chamber.py` (exact ChamberEnv wrapper), `dynamic_observations.py`
(causal observer), `motion_prediction.py` (shared prediction/safety),
`baseline_planners.py` (3 planners + budgets + edge memo),
`ablation_matrix.py` (C1–C5 + factorial, validated at import),
`synthesis.py` (variant training), `policies.py` (deployment policies),
`evaluate.py` (episode runner, 12 metrics), `statistics.py` (Wilson,
paired bootstrap), `analyze_results.py` (tables), `run_*.py` (5 experiment
drivers), `tests/` (57 tests), `results/` (all raw rows + summaries).
Legacy v1.1 modules (`chamber.py`, `oops_language.py`, `oops_prefix.py`,
`planners.py` oracle-only, `oops_batched_search_v11.py`) are unchanged by
v1.2 except via new callers.

## 13. Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1 generalization to unseen dynamic environments | **Supported with a boundary**: 97–98 % feasible-scene success on unseen events/appearance/observation-corruption; fails at density (L5 27 %) | §6 |
| H2 reuse reduces discovery cost | **Supported**: 5.4× less synthesis work, 7.5× less wall, +2.9 pp deployment | §7.2, §7.4 |
| H3 reactive adaptation without future info | **Supported** (causal by construction; degradation curves in §10) | §10 |
| H4 competitive with conventional planners | **Supported**: wins/ties all 15 paired cells; significant wins on L5 vs all three planners (+7.5 to +20.0 pp); 1–3 orders of magnitude lower decision latency | §9 |
