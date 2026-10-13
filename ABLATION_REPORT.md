# v0.4 controlled ablation report

Evaluation: 100 paired held-out chamber seeds 100–199 plus four named diagnostic scenes; same v0.3 interpreter, physics, obstacle trajectories, and sensor access for every variant.

An ablation removes the specified rule **from every frozen program** (preserving all `CALL` wiring), rather than just disabling a feature in the top-level program. For memory-reset control, the complete frozen program is restarted with a new `VMState` on every observation. The latter represents **stateless execution**, not fresh OOPS program synthesis.

| Variant | Success / 100 | Collisions | Average work-proxy units / episode | Difference from complete (success) |
|---|---:|---:|---:|---:|
| Complete P004 | 97 | 3 | 77.81 | -- |
| No `CHOOSE_AXIS` | 73 | 27 | 60.48 | −24 |
| No `WAIT_GAP` | 98 | 2 | 74.18 | +1 |
| No `CONTINUE` | 98 | 2 | 181.10 | +1 |
| Reset memory each observation | 98 | 2 | 206.70 | +1 |

Paired outcomes: disabling `CHOOSE_AXIS` loses 24 of the complete model's successes and adds none; the other three ablations each solve one scene that the full model misses. Thus `WAIT_GAP` and `CONTINUE` are **not supported as aggregate success improvements** by these 100 scenes. `CONTINUE` does substantially lower the *designed* computational cost metric, and `WAIT_GAP` produces explicit anticipatory waiting in the moving-gate challenge. `CHOOSE_AXIS` improves aggregate success but requires more interpreted work than its ablated version.

Targeted behavioral checks:

- `moving_gate`: complete system issues one explicit `WAIT_GAP`, no collision; without `WAIT_GAP` it still finishes this particular scene but waits implicitly/uses another route.
- `north_blocked`, `south_blocked`: removing `CHOOSE_AXIS` prevents goal completion; full program succeeds.
- `empty`: `CONTINUE` produces 11 continuation actions and 60 work units versus 168 with `CONTINUE` disabled and 192 with memory reset.

**Limitations:** spherical point-like agent, axis-aligned cuboids, perfect state observations, deterministic bounce dynamics, and simulator shield intervention. Realistic robotic arms require full joint-space collision validation, dynamics, sensor uncertainty, independent safety, and new benchmarks. Cost proxy ≠ milliseconds; these 100 held-out seeds are not 100 independent real-world environments.
