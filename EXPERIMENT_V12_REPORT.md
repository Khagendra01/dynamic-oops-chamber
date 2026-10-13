# Experiment v1.2.1: Bounded incremental program synthesis in unseen dynamic 3D chambers

**Status:** v1.2.1 — corrected release. Three methodological defects found in
post-hoc audit were fixed and every affected experiment rerun (see §14):
delayed observations no longer mix current proprioception with old
exteroception; the execution shield is causal (served observations only,
never ground-truth futures); all paired CIs use unique-scene clusters, not
correlated rows; a hand-composed `GOAL CHOOSE_AXIS` attribution baseline was
added. The original v1.2 rows are preserved at git commit `8315872`. All
numbers below are measured from saved episode rows
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
paired evaluation) the full system (C5) reaches **84.0 % success on feasible
scenes** (Wilson 95 % CI in `analysis_tables.md`), solves **every feasible
level-1 static scene (100 %, n=840)**, sustains **97.0 % on unseen level 3**
(velocity changes, appear/disappear — never seen in training) and degrades to
**25.6 % on the densest unseen level 5**. Frozen-code reuse (C5 vs C2)
improves pooled deployment success by **+2.9 pp** — but the gain is driven by
one training seed (seed-level t-CI [−5.1, +10.9] crosses zero; §7.2) — and
cuts cumulative synthesis work **5.4×** (544k → 101k work units, 35.4 s →
5.1 s wall on the reference CPU, on every seed). Cumulative curriculum
validation shows **no deployment difference**
in this curriculum (+0.0 pp, zero discordant pairs) while
costing 3.1× extra synthesis work with reuse ON (15.9× with reuse OFF) — a
documented negative result. Persistent runtime memory changes
success by **−0.2 pp** overall — negligible — except in the
no-reuse stateless factorial corner where it is worth **+11.6 pp**. The
full-system-minus-all-three (C1) loses **15.8 pp [13.5, 18.2]** — robust on
every seed. Against
conventional planners under identical observation, safety and command
rules, C5 shows **no significant losses in any of the 15 paired per-level
cells** (one non-significant negative cell: L4 vs rollout_greedy −2.5 pp,
CI crosses zero) and significant wins on the hardest
dense level versus static_astar and rollout_greedy (**+20.0 to +22.5 pp**),
at **0.1–0.3 ms** decision latency versus 3–112 ms. Under 13 observation
degradations (noise, delay, stale serves, visibility, combinations) C5
degrades gracefully — worst case `combined_severe` −32.7 pp, still 0.593 —
and stays above space-time A* at every matched degradation. Every deployed
program across all seeds and all factorial cells is a
2-rule reactive policy (`GOAL CHOOSE_AXIS` or `GOAL`); no method ever
receives future obstacle states. A hand-composed `GOAL CHOOSE_AXIS`
baseline matches the discovered program exactly (Δ = 0.0 pp on every scene,
clustered CI collapsed to zero) — the deployment-time contribution of
synthesis is *finding* that composition cheaply (5.4× less training work
with reuse), not producing a policy the primitives could not express by
hand (§14.4).

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

**Causal observation (v1.2.1 semantics).** `CausalObserver` serves only
past-or-present state, with an explicit channel split: **proprioception**
(agent pose, goal) is always current; **exteroception** (obstacle states,
observation timestamp) is delayed by `delay_steps`, `serve_every` reuses the
last served observation, and noise is deterministic per (seed, step, channel)
and bounded. The truth future is never exposed to any policy. Unit tests
assert that velocity information cannot leak from the future and that a
delayed observation pairs current pose with last-seen obstacles without
mutating simulator state
(`tests/test_observations.py::test_no_future_velocity_leak`,
`::test_v121_proprioception_current_exteroception_delayed`).

**Shared prediction and collision rule (v1.2.1).**
`oops_language.predicted_safe` — extrapolating each observed cuboid by its
observed velocity with known wall reflections — is the single swept-check
implementation, used by programs, planners, **and the execution shield**
alike. Ground truth (`is_motion_safe`, which knows exact future trajectories
including scheduled appearances) is used *only after execution*, to detect
whether the executed motion collided — a metric, never an executor filter.

**Command authority.** Every method is capped at `MotionConfig.stride`
(0.12 m per control period) — the same cap the language programs live under
(`stride_cap="language"`). `stride_cap="full"` (0.35 m) results, when run,
are reported separately and never mixed into the primary tables.

**Safety shield (v1.2.1 causal).** Commands that fail the *causal* swept
check on the served observation become safe waits *for every method*,
including C5; interventions are counted per episode. The v1.2 release
instead gated execution on the simulator's exact futures (an oracle filter);
that defect and its fix are documented in §14.2, and all rows here are
post-fix.

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
  **126 tests OK (1 skipped)** — 65 legacy v1.1 tests (unchanged
  `chamber.py` behavior preserved) + 61 v1.2/v1.2.1 tests.
* Ablation toggles: each of the three factors is proven to change only its
  intended behavior (`test_ablation_matrix.py`, incl. the `CHOOSE_AXIS`
  memory-persistence behavioral test and budget-identity tests).
* Planner edge-memo optimization (needed to make H4 affordable on the Pi):
  proven action-identical to the uncached planner on static *and* dynamic
  scenes; proven to miss (not reuse) whenever the obstacle-state signature
  changes — including under position noise (`test_planners.py::EdgeMemoTests`).
* Prediction is proven to be no-oracle and to match the interpreter rule
  (`test_planners.py::PredictionTests`).
* Exact ChamberEnv equivalence of the dynamic wrapper:
  `test_dynamic_chamber.py` (10/10), including swept checks,
  appear/disappear, and — new in v1.2.1 — the causal-shield semantics:
  a future-appearance hazard is *not* blocked pre-appearance, and the
  post-hoc detector records the collision (`TestCausalShield`).

---

## 6. Experiment A — generalization to unseen levels (H1)

Training saw only levels 1–2. Levels 3–5 were never trained on. Five
curriculum seeds; 200 test scenes per level per seed (1000 episodes/level).

| Level | n | success (all) | success (feasible-only) | collision rate |
|-------|-----|---------------|-------------------------|----------------|
| 1 (seen, static) | 1000 | 0.840 | **1.000** (n=840) | 0.000 |
| 2 (seen, movers) | 1000 | 0.980 | 0.980 | 0.020 |
| **3 (unseen: velocity events, appear/disappear)** | 1000 | **0.970** | 0.970 | 0.030 |
| **4 (unseen: noisy/delayed/stale observations)** | 1000 | 0.910 | 0.968 (n=940) | 0.025 |
| **5 (unseen: dense, mixed dynamics)** | 1000 | 0.230 | **0.256** (n=900) | 0.685 |

* All 5 seeds converge to a 2-rule program (`GOAL CHOOSE_AXIS`; seeds 1 and 3
  report it as `P001`, seeds 0/2/4 as `P002` — same rule list, different
  library ids). Discovery is seed-robust.
* Level 1's headline 0.840 equals the feasible fraction exactly: **every
  feasible static scene is solved by every seed** (840/840). Overall rates
  are diluted by infeasible-by-construction scenes; both numbers are
  reported everywhere.
* Unseen velocity events and appearance changes (L3) cost ~1 pp relative
  to seen movers (L2): 0.970 vs 0.980. Observation corruption (L4) costs
  ~1.2 pp on feasible scenes. Density (L5) is the real failure mode: 25.6 %
  feasible-scene success, 68.5 % collision rate — moving obstacles strike
  the agent while it waits or tracks a blocked corridor; the causal shield
  refuses unsafe *agent* commands but cannot prevent an obstacle from
  hitting a stationary agent, and (unlike v1.2's oracle shield) cannot
  block a command into a hazard it cannot yet see (§14.2).
* Per-seed overall success: 0.786 for all five seeds (identical to 3
  decimals) — variance across curriculum seeds is below display precision on
  this benchmark.

**Verdict H1: supported, with a boundary.** Generalization holds on unseen
dynamic mechanics (events, appearance, observation corruption: 96.8–97.0 %
feasible-scene success) and fails on dense multi-obstacle scenes (25.6 %),
which the current 2-rule program class cannot express a policy for.

## 6b. Validation leakage check

Validation (seed base 50000, disjoint from test) overall C5 success: 0.780
vs test 0.786 — no train/val/test leakage indicated; tuning was never
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
| C1 minimal | 0.628 | [0.615, 0.641] | 0.240 | 11.4 | 0.05 ms |
| C2 no reuse | 0.757 | [0.745, 0.769] | 0.169 | 10.3 | 0.13 ms |
| C3 no cumulative | 0.786 | [0.774, 0.797] | 0.152 | 9.4 | 0.15 ms |
| C4 stateless | 0.788 | [0.776, 0.799] | 0.150 | 9.4 | 0.15 ms |
| **C5 full** | **0.786** | [0.774, 0.797] | 0.152 | 9.4 | 0.15 ms |
| F001 | 0.628 | [0.615, 0.641] | 0.240 | 11.4 | 0.05 ms |
| F010 | 0.641 | [0.628, 0.654] | 0.177 | 14.2 | 0.13 ms |
| F100 | 0.788 | [0.776, 0.799] | 0.150 | 9.4 | 0.15 ms |

Feasible-only rates (infeasible-by-construction removed, n=4680/config):
C1 0.671, C2 0.809, C3 0.840, C4 0.842, **C5 0.840**, F001 0.671, F010
0.685, F100 0.842.

### 7.2 Primary paired contrasts (positive favours the left side; v1.2.1
scene-clustered CI over 1000 unique scenes, plus seed-level t-CI over the 5
training seeds — the two answer different questions: scene uncertainty
*conditional on these programs*, and uncertainty over which program a random
training seed would produce)

| Contrast | factor | Δ success [scene-clustered 95 % CI] | seed-level t-CI | per-seed means | discordant (L/R) |
|----------|--------|------------------------------------|-----------------|----------------|------------------|
| C5 − C2 | frozen-code reuse | **+0.029 [+0.024, +0.034]** | **+0.029 [−0.051, +0.109]** | s0 0.000; s1 0.000; **s2 +0.144**; s3 0.000; s4 0.000 | 160 / 16 |
| C5 − C3 | cumulative validation | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | all 0.000 | 0 / 0 |
| C5 − C4 | persistent memory | −0.002 [−0.005, +0.000] | −0.002 [−0.002, −0.002] | all −0.002 | 0 / 10 |
| C5 − C1 | overall system | **+0.158 [+0.135, +0.182]** | **+0.158 [+0.158, +0.158]** | all +0.158 | 810 / 20 |

**Reading (v1.2.1 corrected).** The overall-system effect (C5 − C1, +15.8 pp)
is robust: identical across seeds, tight under both intervals. The reuse
effect (C5 − C2) is **+2.9 pp pooled but driven entirely by training seed 2**
(+14.4 pp; the other four seeds are exactly 0.0) — the seed-level t-CI
crosses zero, so with n=5 training seeds the deployment benefit of reuse is
**not established across the seed distribution**; what *is* established is
the discovery-cost result (§7.4, a 5.4× work reduction that holds on every
seed). Memory is a consistent −0.2 pp (10 discordant pairs, all favouring
stateless). Cumulative validation remains a null result.

### 7.3 What the factorial corners add

F010 (reuse OFF, cumulative ON, memory **OFF**) = 0.641 while C2 (same but
memory **ON**) = 0.757: **memory is worth +11.6 pp exactly when reuse is
OFF** — the cumulative program `GOAL CHOOSE_AXIS` cannot execute statelessly
(the axis tie-break flips every step without persistent state), while the
reuse-OFF per-task programs are shallower and mostly survive statelessness.
With reuse ON (C3/C4 vs C5) memory is worth ≈ −0.2 pp. This interaction is
invisible in the five primary variants and is the main reason the full 2×2×2
was run.

### 7.4 Training cost (mean over 5 seeds; Pi wall time)

| Variant | wall (s) | work units | replay calls | prefixes tested |
|---------|----------|------------|--------------|-----------------|
| C1 | 2.17 | 34 326 | 424 | 993 |
| C2 | **35.44** | **544 450** | 7 402 | 9 634 |
| C3 | 1.81 | 32 437 | 448 | 1 140 |
| C4 | 4.92 | 100 593 | 1 192 | 2 738 |
| **C5** | **5.06** | **100 593** | 1 192 | 2 738 |

Training and deployment are reported separately by design: C5's *pooled*
deployment edge over C2 is +2.9 pp (but seed-heterogeneous, §7.2), while its
*training* advantage — 7.0× wall time and 5.4× work — holds on every seed.
Reuse pays off primarily at discovery time. (C1/C3 are cheapest
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

**Verdicts.** H2 **supported on its cost claim** (reuse: 5.4× less discovery
work, 7.0× less wall, on every seed) with an honest caveat on the deployment
claim: the +2.9 pp pooled deployment gain is driven by a single training
seed and its seed-level t-CI crosses zero (§7.2) — deployment parity, not
deployment superiority, is what five seeds establish. Cumulative validation:
**null result on deployment** in
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
instance host CPU and **300 episodes/s** on the reference Pi (50 level-1
train scenes, success 0.94, deployed program `P002 GOAL CHOOSE_AXIS`,
`results/gpu_profile/profile.json`). The full 40 000-episode ablation matrix
— which additionally trains 40 syntheses, writes per-seed JSON and CSV, and
includes the hard levels — runs in ≈17 min Pi wall time (≈65 episodes/s
over the whole pipeline); the complete v1.2.1 chain (ablations +
generalization + baselines + reactive + attribution, ≈76 000 episodes plus
training) takes ≈55 min. That is how all rows in `results/` were produced.

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
| L1–3 | static_astar | 600 | 0.925 | 0.025 | 11.5 | 10.9 ms |
| L1–3 | space_time_astar | 600 | 0.908 | 0.017 | 12.1 | 18.3 ms |
| L1–3 | rollout_greedy | 600 | 0.892 | 0.025 | 18.0 | 2.9 ms |
| L4–5 | **C5** | 400 | **0.550** | 0.363 | 10.2 | **0.27 ms** |
| L4–5 | space_time_astar | 400 | 0.500 | 0.388 | 12.3 | 111.6 ms |
| L4–5 | rollout_greedy | 400 | 0.450 | 0.450 | 15.7 | 4.0 ms |
| L4–5 | static_astar | 400 | 0.438 | 0.475 | 11.4 | 40.4 ms |

(L4–5 overall is dominated by the hard dense level 5; see per-level.)

### 9.2 Paired per-level contrasts (positive favours C5; v1.2.1
scene-clustered CI, 40 unique scenes per level)

| Level | vs | Δ success [scene-clustered 95 % CI] | discordant (C5-only / planner-only, 5-seed rows) |
|-------|----|--------------------|--------------------------------------|
| 1 | static_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 1 | space_time_astar | +0.075 [+0.000, +0.175] | 15 / 0 |
| 1 | rollout_greedy | +0.075 [+0.000, +0.175] | 15 / 0 |
| 2 | static_astar | +0.050 [+0.000, +0.125] | 10 / 0 |
| 2 | space_time_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 2 | rollout_greedy | +0.025 [+0.000, +0.075] | 5 / 0 |
| 3 | static_astar | +0.000 [+0.000, +0.000] | 0 / 0 |
| 3 | space_time_astar | +0.025 [−0.050, +0.100] | 10 / 5 |
| 3 | rollout_greedy | +0.050 [−0.050, +0.150] | 15 / 5 |
| 4 | static_astar | +0.025 [+0.000, +0.075] | 5 / 0 |
| 4 | space_time_astar | +0.025 [−0.050, +0.125] | 10 / 5 |
| 4 | rollout_greedy | −0.025 [−0.100, +0.050] | 5 / 10 |
| 5 | static_astar | **+0.200 [+0.050, +0.350]** | 45 / 5 |
| 5 | space_time_astar | +0.075 [−0.075, +0.225] | 30 / 15 |
| 5 | rollout_greedy | **+0.225 [+0.100, +0.350]** | 45 / 0 |

**Reading (v1.2.1 corrected).** Under honest scene-clustered uncertainty,
C5 shows **no significant loss in any of the 15 cells**; the only
non-positive point estimate is L4 vs rollout_greedy (−2.5 pp, CI crosses
zero). Significant wins remain only on the hardest level 5, versus
static_astar (+20.0 pp [5.0, 35.0]) and rollout_greedy (+22.5 pp [10.0,
35.0]); the L5 vs space_time cell (+7.5 pp) has a CI that crosses zero once
40 unique scenes — not 200 seed-rows — are the resampling unit (v1.2's
row-level CI claimed significance here; §14.3). Pooled over all levels
(200 unique scenes, §9.1 data), C5 beats all three planners with clustered
CIs at or above zero (+4.0 to +7.0 pp). The decision-latency column is a first-class result: the
2-rule program executes in ≈0.1–0.3 ms/step with zero deadline misses, while
the time-expanded search costs 18–112 ms/step and hits its expansion budget
on dense scenes: space_time_astar recorded **220 deadline misses in 80 of
1000 episodes** (misses become safe waits and are counted in the raw rows /
`analysis.json`); every other method recorded zero. C5 also produces the
lowest collision rate on every run.

**Verdict H4: supported, with one qualification.** Under equal information,
equal command authority, and equal safety rules, the synthesized reactive
program has no significant losses against bounded grid and rollout planners
on any level, significant wins on the hardest level versus two of the three,
and is 1–3 orders of magnitude cheaper per decision. The qualification:
with only 40 unique scenes per level, the per-level clustered CIs are wide —
several positive cells (including L5 vs space_time) no longer exclude zero
under honest resampling (§14.3).

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

### 10.1 C5 degradation curve (levels 1–3, n=1500 each; Δ vs perfect is
scene-clustered)

| Degradation | success [95 % CI] | Δ vs perfect [clustered 95 % CI] | collision |
|-------------|-------------------|----------------------------------|-----------|
| perfect | 0.920 [0.905, 0.933] | — | 0.020 |
| visibility_0.8 | 0.920 [0.905, 0.933] | +0.000 [+0.000, +0.000] | 0.020 |
| visibility_0.6 | 0.920 [0.905, 0.933] | +0.000 [+0.000, +0.000] | 0.020 |
| noise_pos_0.005 | 0.883 [0.866, 0.899] | +0.037 [+0.017, +0.060] | 0.070 |
| noise_vel_0.02 | 0.873 [0.856, 0.889] | +0.047 [+0.023, +0.073] | 0.103 |
| noise_pos_0.01 | 0.847 [0.828, 0.864] | +0.073 [+0.043, +0.103] | 0.133 |
| delay_1 | 0.830 [0.810, 0.848] | +0.090 [+0.060, +0.123] | 0.110 |
| delay_2 | 0.787 [0.765, 0.807] | +0.133 [+0.097, +0.173] | 0.153 |
| delay_3 | 0.783 [0.762, 0.803] | +0.137 [+0.100, +0.177] | 0.157 |
| serve_2 | 0.777 [0.755, 0.797] | +0.143 [+0.103, +0.183] | 0.223 |
| serve_3 | 0.720 [0.697, 0.742] | +0.200 [+0.157, +0.247] | 0.280 |
| combined_mild | 0.680 [0.656, 0.703] | +0.240 [+0.190, +0.290] | 0.320 |
| combined_severe | 0.593 [0.568, 0.618] | **+0.327 [+0.273, +0.380]** | 0.407 |

(Δ = success drop caused by the degradation; positive = degradation hurt.)

### 10.2 Space-time A* under the same observations (n=180 each)

| Degradation | success [95 % CI] | Δ vs perfect [clustered 95 % CI] |
|-------------|-------------------|----------------------------------|
| perfect | 0.861 [0.803, 0.904] | — |
| delay_2 | 0.694 [0.624, 0.757] | +0.167 [+0.056, +0.306] |
| combined_severe | 0.194 [0.143, 0.258] | +0.667 [+0.500, +0.806] |

### 10.3 Reading (v1.2.1)

* **Causal by construction — observer *and* executor.** Every number above
  comes from policies that receive only `CausalObserver` output, and the
  execution shield now filters commands through that same served observation
  (§14.2); the truth future is unreachable by either path (unit-tested).
  Degradation curves measure robustness of *reactive decisions*, not oracle
  protection.
* **Graceful, monotone-in-severity degradation.** Worst single factors:
  staleness (`serve_3`, −20.0 pp) and 2-step delay (−13.3 pp). Worst
  combination: `combined_severe` (−32.7 pp, 0.593 absolute) — harsher than
  v1.2's oracle-shielded −18.0 pp, as expected once the oracle filter is
  removed (§14.2). C5 remains above space_time_astar under every matched
  degradation, by larger margins than v1.2 reported (perfect: +5.9 pp;
  delay_2: +9.3 pp; combined_severe: **+39.9 pp**) — the causal shield hurts
  the planner more than the program, because the program re-decides from
  each degraded observation while the time-expanded plan compounds
  prediction errors.
* **Delay hurts more than noise** (−13.7 pp vs −7.3 pp at the larger of
  each): the program's decisions are geometric, so stale obstacle positions
  are costlier than jittery ones.
* **Visibility clamps at 0.6–0.8 m never bind** on these 1 m chambers
  (identical outcomes, 0 discordant pairs): every obstacle that mattered to
  a decision was already within range. Reported as a null result, not as a
  robustness claim.
* **Delay monotonicity now holds:** delay_3 (0.783) ≤ delay_2 (0.787) ≤
  delay_1 (0.830). The v1.2 non-monotonicity (delay_3 > delay_2) was an
  artifact of the mixed-time observation defect (§14.1); under the corrected
  exteroception-only delay the curve is monotone.
* **Deadline misses:** C5 records **0** deadline misses in all C5
  deployment episodes of Experiments D and E (program execution is
  ~0.1–0.3 ms/step). space_time_astar missed its 0.5 s deadline in 80/1000
  baseline episodes (220 misses, 0.22/episode mean) — those steps became
  safe waits.

**Verdict H3: supported.** The system adapts reactively from causal
observations — with a causal executor, not just a causal sensor — degrades
gracefully under noise, delay, staleness, and their combinations, and stays
above the time-expanded planner at every matched degradation level.

---

## 11. Failures, negative results, and limitations

1. **Level 5 density is unsolved** (25.6 % feasible-scene success, 68.5 %
   collision). The 2-rule program class plus reactive-only execution cannot
   thread dense moving obstacles; a hierarchical or lookahead program class
   is the obvious next step — and is *not* claimed here.
2. **Cumulative validation is a null result on deployment** in this
   curriculum (+0.0 pp, 0 discordant pairs) while costing 3.1× extra
   synthesis work with reuse ON and 15.9× without reuse. Reported as
   measured; not spun.
3. **Memory is slightly negative overall** (−0.2 pp) except in the
   no-reuse stateless corner (F010: +11.6 pp vs C2) — the interaction, not
   the main effect, is the finding.
4. **Obstacle-caused collisions cannot be shielded away.** The causal shield
   protects against unsafe *agent* commands as far as the served observation
   permits; a moving obstacle that hits a waiting agent is recorded as a
   collision for every method equally, and (since v1.2.1) a command into a
   hazard the observer cannot yet see is executed and *detected* rather than
   silently blocked (§14.2).
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
9. **Reactive null result, reported as measured:** visibility clamps at
   0.6–0.8 m never bind on these 1 m chambers (identical outcomes, 0
   discordant pairs — not a robustness claim). The v1.2 delay_3 >
   delay_2 non-monotonicity was an artifact of the mixed-time observation
   defect and disappears under v1.2.1 semantics (§14.1).
10. **Five training seeds limit seed-level inference.** Scene-clustered CIs
    are honest about scene sampling but cannot express "which program would
    a new training seed find"; the seed-level t-CIs (§7.2) are wide, and the
    reuse deployment contrast in particular is not established across the
    seed distribution.

---

## 12. Reproducibility

```bash
# environment (reference CPU)
python3.11 -m venv oops_venv && oops_venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

# full verification (v1.1 parity + v1.2.1)
cd dynamic_oops_chamber_v1_0
python -m unittest discover -p "test_*.py" -q          # 126 tests, OK (1 skipped)

# experiments (each writes experiments_v12/results/<name>/)
python -m experiments_v12.run_ablations --seeds 5 --levels 1,2,3,4,5
python -m experiments_v12.run_generalization --seeds 5 --levels 1,2,3,4,5
python -m experiments_v12.run_baselines --seeds 5 --levels 1,2,3 --limit 40 --planner-expansions 1200
python -m experiments_v12.run_baselines --seeds 5 --levels 4,5 --limit 40 --planner-expansions 1200 --out baselines_hl
python -m experiments_v12.run_reactive --seeds 5 --levels 1,2,3 --limit 100 --planner-limit 12 --planner-expansions 1200
python -m experiments_v12.run_attribution --seeds 5 --levels 1,2,3,4,5    # v1.2.1 Experiment F
python -m experiments_v12.run_gpu_profile --skip-batched --count 50   # Pi CPU
# GPU (Modal L40S; requires modal CLI; v1.2 rows, pre-v1.2.1 shield — overhead
# result unaffected by the v1.2.1 semantics changes, not rerun):
modal run modal_oops_v12.py     # writes /tmp/opencode/oops_v10/l40s_v12_profile.json

# derived tables (no re-evaluation)
python -m experiments_v12.analyze_results
```

**File map.** `experiments_v12/` — `scenarios.py` (levels/splits/seeds),
`dynamic_chamber.py` (exact ChamberEnv wrapper + causal execution shield),
`dynamic_observations.py` (causal observer, v1.2.1 proprio/extero split),
`motion_prediction.py` (shared prediction/safety),
`baseline_planners.py` (3 planners + budgets + edge memo),
`ablation_matrix.py` (C1–C5 + factorial, validated at import),
`synthesis.py` (variant training), `policies.py` (deployment policies),
`evaluate.py` (episode runner, 12 metrics), `statistics.py` (Wilson,
paired bootstrap, scene-clustered contrasts), `analyze_results.py` (tables),
`run_*.py` (6 experiment drivers, incl. `run_attribution.py`), `tests/` (61
tests), `results/` (all raw rows + summaries). Legacy v1.1 modules
(`chamber.py`, `oops_language.py`, `oops_prefix.py`, `planners.py`
oracle-only, `oops_batched_search_v11.py`) are unchanged by v1.2 except via
new callers.

## 13. Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1 generalization to unseen dynamic environments | **Supported with a boundary**: 96.8–97.0 % feasible-scene success on unseen events/appearance/observation-corruption; fails at density (L5 25.6 %) | §6 |
| H2 reuse reduces discovery cost | **Supported on cost** (5.4× less synthesis work, 7.0× less wall, every seed); deployment gain +2.9 pp pooled but seed-heterogeneous — seed-level t-CI crosses zero | §7.2, §7.4 |
| H3 reactive adaptation without future info | **Supported** (causal by construction after the v1.2.1 shield fix; degradation curves in §10) | §10, §14.2 |
| H4 competitive with conventional planners | **Supported with a qualification**: no significant losses in any of 15 paired cells (one non-significant negative: L4 vs rollout −2.5 pp); significant wins on L5 vs static and rollout (+20.0/+22.5 pp); per-level clustered CIs are wide (40 scenes/level); 1–3 orders of magnitude lower decision latency | §9, §14.3 |
| *(new, F)* synthesis vs hand composition | **Attribution:** deployment policy quality is fully explained by the hand-composed `GOAL CHOOSE_AXIS` (Δ = 0.0 on every scene); synthesis's measured contribution is discovery efficiency, not policy superiority | §14.4 |

---

## 14. v1.2.1 corrections (post-hoc audit)

A scientific audit of the v1.2 release identified three methodological
defects and one missing control. All four are fixed in v1.2.1; every
affected experiment was retrained and rerun. The original v1.2 raw rows
remain available at git commit `8315872` for provenance.

### 14.1 Delayed observations mixed current and past state

**Defect.** `CausalObserver` with `delay_steps > 0` produced observations
that paired the agent's *current* position with *old* obstacle snapshots,
while labelling the entire observation (including `time_s`) as delayed.
The mechanism was a state-mutation hack (`_truth_at` set
`env.step_index` to the past step and read `env.observe()`), which cannot
roll back `env.position`. The resulting observation was causally valid but
semantically inconsistent — no real sensor produces it, and the labelled
delay did not match the content.

**Fix.** `dynamic_observations.py` now separates the two channels
explicitly: **proprioception** (agent pose, goal) is always current;
**exteroception** (obstacle states, observation timestamp used for motion
prediction) is delayed by `delay_steps`. The mutation hack is removed;
delayed obstacle states are computed directly from `ob.center_at(t_old)`.
The contract is pinned by
`tests/test_observations.py::test_v121_proprioception_current_exteroception_delayed`.

**Effect.** Deployment programs observe a *better* pose signal under delay
than v1.2 reported (current, not stale), and a *consistent* obstacle
snapshot. All delay-involving rows (levels 4–5, reactive `delay_*` and
`combined_*` degradations) are rerun.

### 14.2 Execution shield was oracle, not causal

**Defect.** `DynamicChamber.step()` gated every movement command on
`is_motion_safe()` — the simulator's exact future obstacle trajectories,
including scheduled appearances and reflections. The shield therefore
*prevented* collisions that no causal observer could have predicted (an
obstacle scheduled to appear mid-interval is invisible to any sensor until
it appears). H3's "causal by construction" claim held for the *observer*
but not for the *executor*: the system as deployed was protected by an
oracle filter.

**Fix.** Execution is now filtered by `predicted_safe(shield_obs, velocity,
config)` — the same observable-state extrapolation rule the interpreter and
the planners use. `shield_obs` is the served observation: the policy's own
degraded view when a `CausalObserver` is attached
(`run_episode` sets `env.shield_observer`), otherwise the perfect *current*
observation (still causal — no future knowledge). Ground truth via
`is_motion_safe` is used **only after execution**, to detect whether the
executed motion (or a forced wait) collided — a metric, never an executor
filter. Pinned by three tests in
`tests/test_dynamic_chamber.py::TestCausalShield`, including the exact audit
case: a hazard that appears at t=0.6 s is *not* blocked by the causal shield
at t=0.5 s (v1.2's oracle blocked it), and the post-hoc detector records
the collision.

**Effect.** C5 may now collide where v1.2 silently prevented it — honest
numbers for the causal system that was claimed. All synthesis (training
replay also flows through `step()`) and all deployment episodes are rerun.

### 14.3 Row-level CIs treated correlated rows as independent

**Defect.** Paired bootstrap CIs resampled *rows*. Each scene appears once
per training seed (5 correlated rows), so the effective sample size was
inflated 5×. The reported C5-vs-C2 reuse contrast (+2.9 pp [+2.4, +3.4])
implied a precision that five training seeds cannot support; the reported
L4–5 C5-vs-space-time contrast (+1.0 to +14.0 pp) excluded zero only
because of the inflation (unique-scene bootstrap: −7.5 to +22.5 pp,
crosses zero).

**Fix.** `statistics.py` gains `clustered_paired_rate` (differences averaged
within each unique scene; bootstrap resamples scenes) and
`per_seed_contrasts` (per-training-seed means and seed spread).
`analyze_results.py` emits both for every contrast; scene-clustered CIs are
the authoritative interval, row-level CIs are retained only for provenance.
The per-seed table makes seed heterogeneity visible: on v1.2 rows the
+2.9 pp reuse gain is **+14.5 pp on training seed 2 and 0.0 pp on seeds
0, 1, 3, 4** — a single-seed effect that the pooled row-level CI hid.

**Effect.** Every contrast table in §5–§10 is regenerated from the rerun
with scene-clustered intervals and per-seed breakdowns. Verdicts that lose
significance under honest uncertainty are restated.

### 14.4 Missing control: hand-composed attribution baseline

**Defect.** The discovered programs are short sequences of hand-designed
primitives (`GOAL`, `CHOOSE_AXIS`, …). No baseline established whether
synthesis contributes anything at *deployment* beyond what the primitives
already provide when composed by hand.

**Fix.** New Experiment F (`run_attribution.py`, results in
`results/attribution/`) evaluates three manually composed programs on the
same held-out test splits and scenes as the C5 programs: `manual_GOAL`,
`manual_GOAL_CHOOSE_AXIS` (the README's documented P004-equivalent
composition), and `manual_full_primitives` (all four motion primitives).
Contrasts are paired per scene against every training seed's discovered
program.

**Result (v1.2.1 measured, `results/attribution/`, 1000 scenes × 5 seeds
C5 rows vs replicated manual rows).** The discovered program *is*
`GOAL CHOOSE_AXIS` in every seed; `manual_GOAL_CHOOSE_AXIS` is
indistinguishable from it (**Δ = 0.0 pp on every one of 1000 scenes —
identical behavior by construction and confirmed empirically; the clustered
CI collapses to zero**). `manual_GOAL` alone is far worse (−15.8 pp
[−18.2, −13.5]). The maximal hand composition `manual_full_primitives`
(CONTINUE, WAIT_GAP, GOAL, CHOOSE_AXIS) is slightly *worse* than the
discovered 2-rule program (+0.8 pp [0.0, +1.6] for C5): the extra hand
primitives occasionally hurt (CONTINUE can commit to a bad direction) and
never help enough to matter. Honest reading: the
deployment-time policy quality is fully explained by the hand-designed
primitives; synthesis's measured contribution is **discovery efficiency**
(the H2 result: 5.4× less search work with reuse, finding the effective
composition automatically across all seeds), not policy superiority. This
does not weaken H2 — it sharpens what H2 means — but it does rule out any
claim that the synthesizer found a policy beyond hand composition on this
curriculum.

### 14.5 Corrected claims

* "C5 wins or ties all 15 paired cells" → **no significant losses in any
  cell; one non-significant negative point estimate** (L4 vs
  rollout_greedy, −2.5 pp, CI crosses zero). Corrected in the abstract and
  §9/§13.
* H3 "causal by construction" now refers to the v1.2.1 executor as well as
  the observer (§14.2).
* All CIs quoted in §5–§10 are scene-clustered (§14.3); where significance
  changes, the corrected interval is the one in the table.
