# GPU Handoff — Dynamic OOPS Chamber v1.0

## Target hardware

- **First choice: L40S (48 GB)** for CUDA batched rollouts *and* later Isaac Sim (RT cores).
- **L4 (24 GB)** for smaller batches; **RTX PRO 6000** only with a verified Blackwell-compatible software stack.
- **A100/H100** are appropriate for compute-only tensor kernels, **not** full Isaac Sim: NVIDIA notes that GPUs without RT cores are unsupported for Isaac Sim.
- On a rented instance, confirm the host exposes a compatible NVIDIA driver, graphics features for Isaac, and enough CPU RAM (32 GB minimum, 64+ preferred). GPU availability alone is not sufficient for ray-traced Isaac rendering.

## Two distinct environments

**1. Torch / CUDA 12.6 runtime (this project)**

Use an existing functioning PyTorch/cu126 environment, or create a fresh Python 3.12 venv on a supported x86_64 Linux host:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu126
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA UNAVAILABLE")'
```

The tested workspace used **PyTorch 2.10.0+cpu** and Python 3.13; the CUDA 12.6 environment and CUDA execution have NOT yet been tested. The 2.7.1/cu126 command is published by PyTorch, but on-device verification remains necessary.

**2. Isaac Sim environment (future, separate)**

NVIDIA publishes an Isaac Sim **6.0.0** container at `nvcr.io/nvidia/isaac-sim:6.0.0`. Use their *Container Installation* guide to perform compatibility checks and run on a cloud host with a GPU that supports RTX. Do **not** install Isaac Sim/Isaac Lab into the PyTorch 2.7.1/cu126 venv above: NVIDIA's current Isaac Lab releases have separate Python/PyTorch/CUDA requirements. Select a release-matched Isaac Lab branch and lock the versions after passing the official smoke test.

Source references:
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html
- https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html
- https://pytorch.org/get-started/previous-versions/

## Validation commands (Torch CUDA environment)

From inside `dynamic_oops_chamber_v1_0`:

```bash
nvidia-smi
python -m unittest discover -p 'test_*.py' -q
python verify_torch_v10.py --device cpu --count 100 --out cpu_parity.json
python verify_torch_v10.py --device cuda --count 100 --out cuda_parity.json
python run_torch_v10.py --device cuda --batch-size 512 --scenes 512 --dtype float64 --out cuda_512.json
python run_torch_v10.py --device cuda --batch-size 1024 --scenes 1024 --dtype float32 --out cuda_1024.json
```

**Acceptance gate:** CUDA parity (100 fixed seeds) must report `equivalent: true` with zero mismatched success, collision, and episode-step outcomes. Stop here if a mismatch occurs. Use float64 initially to isolate GPU numerical differences; then profile float32 separately, and document seed-by-seed changes.

## Scope and limitations

- `torch_chamber.py`: tensorized moving-cuboid geometry, reflection at chamber walls, conservative continuous swept collision checks, batched sphere agent and shield.
- `torch_frozen_policy.py`: batched execution of existing frozen v0.3 program semantics (e.g. `CONTINUE → WAIT_GAP → GOAL → CHOOSE_AXIS`), including per-episode movement memory and `CALL` expansion.
- `verify_torch_v10.py`: compare batched program results against the v0.9 CPU interpreter in 100 deterministic scenes.
- `run_torch_v10.py`: benchmark batched program execution, with actual wall-clock time and CUDA synchronization. Elapsed time excludes scene creation and environment initialization; compare equal batches and settings.
- **NOT implemented:** GPU program *synthesis* (the OOPS prefix search is still the CPU implementation), Isaac Sim/Isaac Lab scene integration, GPU rollback of PhysX state, robotics manipulator kinematics, independent real-world safety controller.
- Exact obstacles and initial velocities are still available, without sensor uncertainty. The simulator uses a spherical agent and axis-aligned boxes (not a real robot arm). AABB inflation is conservative, not exact Euclidean sphere-box contact.
- The GPU tensor backend currently permits at most **one wall bounce per obstacle axis per simulator step**. It rejects environments violating that bound instead of silently approximating them.
- GPU/CUDA test could not run here because the workspace has **no NVIDIA GPU**. Verified locally: CPU PyTorch parity, original v0.9 tests, synthetic moving-obstacle scenarios.

## Experiments for the next research milestone

1. CUDA parity with 100 → 1,000 fixed scenes across several obstacle motion regimes.
2. Matched CPU/GPU action-evaluation throughput and memory utilization (report batch creation separately).
3. Modify the OOPS prefix-search evaluator to evaluate many independent candidate programs in batches. Measure total synthesis wall time, including Python search, host/device transfers, and replays.
4. Compare an equivalent deterministic chamber in Isaac Sim; do not assume its PhysX contacts equal the conservative AABB checker.
5. Only after parity and runtime stability: add articulated-arm geometry and true motion-planning baselines.
