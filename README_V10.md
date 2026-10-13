# Dynamic OOPS Chamber v1.0 — GPU-ready Batched Execution

This build retains the complete v0.9 CPU research project and adds a **PyTorch-batched chamber and frozen program interpreter**, tested on CPU against the existing reference implementation. It can run on CUDA devices, but this workspace had no NVIDIA GPU, so **GPU execution is unverified**.

## New files

| File | What it does |
|---|---|
| `torch_chamber.py` | Batched AABB sphere-agent continuous collision and wall-bounce simulation |
| `torch_frozen_policy.py` | Batched interpreter for existing frozen rule library and movement memory |
| `verify_torch_v10.py` | Paired verification vs original CPU interpreter on 100 fixed seeds |
| `run_torch_v10.py` | CUDA/CPU workload runner with synchronized timing |
| `test_torch_chamber.py`, `test_torch_policy.py` | Unit tests and CPU/GPU parity test (CUDA conditional) |
| `GPU_SETUP_V10.md` | L40S/Isaac Sim deployment setup, acceptance tests and research limitations |
| `results_v10/` | Locally generated CPU parity and benchmark reports |

## Quick start

```bash
# PyTorch is the only external runtime needed for the new files
python -m unittest discover -p 'test_*.py' -q
python verify_torch_v10.py --device cpu --count 100 --out cpu_parity.json
python run_torch_v10.py --device cpu --scenes 100 --out cpu_run.json
```

On a CUDA-capable machine with working PyTorch:

```bash
python verify_torch_v10.py --device cuda --count 100 --out cuda_parity.json
python run_torch_v10.py --device cuda --scenes 512 --batch-size 512 --out cuda_benchmark.json
```

## Measured in current (CPU-only) workspace

- Complete unit suite: **63 tests passed, 1 CUDA test skipped**.
- Paired CPU interpreter/Torch runner: **100/100 scene-outcome agreement**, **97/100 successes**, **3 collisions** (not safety-ready).
- Batched 100-scene CPU Torch rollout, float64: **0.786 s** excluding initialization in one run; **not a GPU benchmark** and not directly comparable with older timing tables.

The original project remains OOPS-inspired bounded program search. Running the frozen DSL on GPU is not a full GPU-accelerated implementation of OOPS synthesis.
