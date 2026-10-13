# Loose working artifacts (v1.1 GPU / benchmark era)

Files that lived in the scratch working directory (`/tmp/opencode/oops_v10/`)
alongside the project but were not part of the tracked tree. Preserved here
for provenance:

| File | What it is |
|------|------------|
| `modal_oops_v10.py` | v1.0-era Modal GPU app (superseded by `modal_oops_v11.py` / `modal_oops_v12.py` at repo root) |
| `l40s_results.json` | first L40S batched-synthesis profile (v1.0) |
| `l40s_batched_v11.json`, `l40s_batched_v11_b16384.json` | v1.1 L40S profiles (canonical copies also in `experiments_v12/results/gpu_profile/`) |
| `l40s_batched_v11_prev.json` | prior-run snapshot of the v1.1 L40S profile |
| `batched_cpu.json`, `batched_cpu_d4.json` | local CPU batched-engine timing checks |
| `bench_step.py`, `probe_search.py`, `ref_depths.py`, `diag_compile_v11.py` | benchmark / diagnostic scripts used while debugging the wave-batched search |
| `reference_queries.json` | 271 KB reference query set used for CPU/GPU parity checking (v1.1) |

Nothing here is required to run or reproduce the v1.2.1 experiments; the
canonical, maintained entry points are `modal_oops_v11.py`,
`modal_oops_v12.py` and `experiments_v12/run_gpu_profile.py`.
