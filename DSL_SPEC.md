# Reactive Geometric DSL v0.3 — Technical Specification

## Inputs and outputs

```text
Observation {
  time_s, step_index,
  position: (x,y,z), goal: (x,y,z),
  chamber_size: (Lx,Ly,Lz),
  obstacles: [{name,center:(x,y,z),size:(dx,dy,dz),velocity:(vx,vy,vz)}]
}
+ MotionConfig {radius,dt,max_speed,stride}
+ VMState {last_direction,last_position,last_mode,emissions,explicit_waits}
+ FrozenLibrary {ID -> Program}
     --> Interpreter.act(Program,Observation,VMState)
     --> (velocity_xyz, execution_trace)
```

All observations come from the **current** simulated time. The interpreter cannot access `ChamberEnv.is_motion_safe()` or full future simulator trajectories. It extrapolates the observed cuboids under the environment's *known constant-velocity + reflecting-walls model*. This assumes perfect current observations and known dynamics.

## Rule tokens

| Token | Semantics |
|---|---|
| `GOAL` | Propose a max-stride velocity toward current goal, clipped when close |
| `X_POS`, `X_NEG`, `Y_POS`, `Y_NEG`, `Z_POS`, `Z_NEG` | Try the corresponding constant axis velocity |
| `LAST` | Attempt the previously saved direction, regardless of goal-progress check |
| `CONTINUE` | Try the saved direction only when it does not increase goal distance by more than 0.02 m and is predicted safe |
| `CHOOSE_AXIS` | Score all six axis motions and attempt the best currently safe candidate |
| `WAIT_GAP` | Explicitly wait if current direct progress is blocked and short-term obstacle motion predicts a gap reopening; additionally verifies waiting itself is safe |
| `CALL:Pnnn` | Reuse an earlier frozen program; fall through if subprogram cannot emit a candidate |

`CHOOSE_AXIS` computes (i) immediate goal distance improvement, (ii) whether a direct motion is predicted safe **from the candidate's endpoint at the next step**, and (iii) alignment with `last_direction`. Invalid axis moves are rejected by swept collision checking. Its scores and weights are **hand-designed and not automatically learned**. Ties are deterministic.

`WAIT_GAP` tests whether the GOAL velocity is currently unsafe, standing still for this step is safe, at least one obstacle is moving, and the model predicts that GOAL movement becomes safe within up to four future control periods. It does not guarantee that the gap will persist when the robot reaches it. It does not examine arbitrary obstacle trajectories, hidden blockers or policy-optimal waiting time.

`CONTINUE` keeps only the previous direction, **not a remembered full path or subgoal**. The persistent VM register is reset per episode.

## Bytecode compilation

Each move rule (including `CHOOSE_AXIS` and `CONTINUE`) emits:

```text
AIM <candidate>
CHECK_SAFE
IF_UNSAFE_SKIP 3
STORE_DIRECTION
EMIT
```

A gap-wait rule emits:

```text
CHECK_WAIT_GAP
IF_UNSAFE_SKIP 2
EMIT_WAIT
```

Calls compile to `CALL <program_id>`. If no rule returns, `END_NO_MATCH` produces no proposal, and `Interpreter.act` emits a zero-velocity `FALLBACK WAIT`. Such a wait is **not** counted as an explicit learned `WAIT_GAP` behavior and may be unsafe.

The VM records `SELECTED:<rule>` immediately before successful emissions; it separately traces candidate `AIM` operations that fail. This prevents overcounting attempted-but-rejected memory or directional choices. Every step has an instruction budget and a recursive call-depth guard.

## Offline incremental synthesis

A `Program` is a *tuple of guarded rules*, not arbitrary Python. `IncrementalSynthesizer` enumerates distinct rule tuples up to `max_rules=3`, prioritizes them by products of manually selected token probabilities, and tests them in a deterministic simulator. The newest task is replayed first; a program must still solve all previously accepted tasks. A program can call frozen code through `CALL`.

Candidate counts are split between the extension and fresh candidate lanes. This is **OOPS-inspired**, **not original OOPS**: there is no prefix-level `Try` machine, no actual original 50/50 *time allocation*, and no bias-optimality guarantee.

The extended curriculum has extra, intentionally specified requirements:
- On `crossing`, the candidate must emit at least one `WAIT_GAP` and avoid implicit fallback waiting. This trains explicit waiting rather than merely inheriting the default zero command.
- On `efficient_empty`, the candidate must keep deterministic `work_units` <=70, with goal reaching and all prior constraints retained. `work_units` is a proxy for relative geometric computation and is **not real measured CPU time**.

The frozen-program output is a JSON library with callable rule lists, compiled bytecode, and curriculum search records. Existing frozen v0.2 programs are readable.

## Baselines and reproducibility

```bash
python -m unittest -v test_chamber.py test_oops.py test_oops_v03.py
python run_extended.py --budget 1500 --max-rules 3
python run_v03_holdout.py --first-seed 100 --count 100
```

The baseline library is saved under `baselines/v02_frozen_library.json`. In the paired evaluation, both versions use the same v0.3 interpreter and random scenes; only the frozen rule list differs. Output includes per-episode success, collision, shield intervention, steps, emitted explicit waits, emitted `CONTINUE`, emitted `CHOOSE_AXIS`, trace event counts, and proxy work units.

## Main limitations

- An arm is *not* yet modeled. A sphere's safe velocity is not sufficient for linked-robot collision avoidance.
- Dynamic obstacle extrapolation assumes exact known bounce dynamics and noiseless positions/velocities.
- `CHOOSE_AXIS` is a packaged heuristic; discovering to invoke it is not the same as synthesizing the internal heuristic.
- `WAIT_GAP` is a fixed 4-step short-term predictor. It is not an adaptive safe interval planner.
- Frozen rule programs remain short and can still collide or fail in unseen scenes, even with a simulator safety shield.
- Program probabilities and stage difficulty targets were selected by the implementer; no unbiased claim about automatic discovery of all three behaviors is warranted.
- Actual latency, scalability, ablation, confidence intervals, and matched compute-baseline comparisons are not established.
