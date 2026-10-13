"""L40S wall-clock comparison: GPU-prefetched prefix search vs CPU reference.

One process, one compiled graph: the batched control step is compiled once and
every synthesis run reuses it, so the report separates the one-time compile
cost from the steady-state search time. Every batched run is checked against
the `PrefixOOPS` reference for exact stage-level parity before it is timed.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import modal

PROJECT = "/project"

REPO_ROOT = str(Path(__file__).resolve().parent)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .add_local_dir(REPO_ROOT, PROJECT, copy=True)
)

cache = modal.Volume.from_name("oops-inductor-cache", create_if_missing=True)

app = modal.App("oops-v11-batched-synthesis", image=image)


def sh(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd), "returncode": p.returncode,
           "elapsed_s": round(time.perf_counter() - t0, 3),
           "stdout": p.stdout, "stderr": p.stderr}
    print(f"[{out['elapsed_s']:8.2f}s] rc={p.returncode} {' '.join(cmd)}", flush=True)
    for stream in (p.stdout, p.stderr):
        if stream.strip():
            print(stream.strip()[-3000:], flush=True)
    return out


@app.function(gpu="L40S", timeout=5400, cpu=16, memory=32768,
              volumes={"/root/.cache/torch/inductor": cache})
def run(depths: list[int] | None = None, repeats: int = 3,
        batch_size: int = 8192, dtype: str = "float64",
        program_width: int = 12,
        budgets: list[int] | None = None,
        modes: list[str] | None = None) -> str:
    depths = depths or [4, 5, 6]
    budgets = budgets or [256]
    modes = modes or ["waves", "merged"]
    sys.path.insert(0, PROJECT)
    import torch

    import run_batched_synthesis_v11 as R
    from oops_language import Program
    from run_torch_v10 import library
    from torch_chamber import TorchChamberBatch
    from torch_job_policy import TorchJobPolicy, compile_program

    out: dict = {
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0),
                        "capability": torch.cuda.get_device_capability(0)},
        "parameters": {"depths": depths, "repeats": repeats,
                       "batch_size": batch_size, "dtype": dtype,
                       "program_width": program_width, "modes": modes,
                       "budgets": budgets},
        "tests": sh([sys.executable, "-m", "unittest", "discover", "-p",
                     "test_*.py", "-q"]),
        "depths": {},
    }
    if out["tests"]["returncode"] != 0:
        raise RuntimeError("unit tests failed on the GPU image")

    def base_args(depth: int, merge: bool = False,
                  budget: int = 256) -> argparse.Namespace:
        return argparse.Namespace(
            device="cuda", dtype=dtype, batch_size=batch_size,
            program_width=program_width, max_depth=depth,
            starting_budget=budget, repeats=repeats, verbose=True,
            merge_waves=merge)

    # ---- one compile, shared by every depth and repeat below ---------------
    compiled = torch.compile(lambda policy: policy.step(), mode="default",
                             dynamic=False)
    step_fn = lambda policy: compiled(policy)   # noqa: E731
    warm_args = base_args(depths[0])
    warm_model = R.BatchedPrefixOOPS(
        device="cuda", dtype=getattr(torch, dtype), batch_size=batch_size,
        program_width=program_width, step_fn=None, verbose=False,
        **R.params(warm_args))
    ids = torch.arange(batch_size) % len(warm_model.templates)
    warm_env = TorchChamberBatch.from_templates(
        warm_model.templates, ids, device="cuda",
        dtype=getattr(torch, dtype), base=warm_model.base_batch)
    frozen, program = library()
    programs = [compile_program(Program(tuple(program.rules)), frozen)
                for _ in range(batch_size)]
    policy = TorchJobPolicy(warm_env, programs, program_width=program_width)
    torch.cuda.synchronize()
    started = time.perf_counter()
    step_fn(policy)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    print(f"compile_seconds={compile_seconds:.2f}", flush=True)
    out["compile_seconds"] = round(compile_seconds, 2)

    # ---- per budget/depth: CPU reference, then each batched mode with parity
    for budget in budgets:
        for depth in depths:
            reference, reference_seconds = R.run_reference(
                base_args(depth, budget=budget))
            print(f"[budget {budget} depth {depth}] reference "
                  f"{reference_seconds:.3f}s stages={len(reference.history)}",
                  flush=True)
            for mode in modes:
                args = base_args(depth, merge=mode == "merged", budget=budget)
                batched, times = R.run_batched(args, step_fn)
                problems = R.compare(R.fingerprint(reference),
                                     R.fingerprint(batched))
                if problems:
                    raise RuntimeError(
                        f"budget {budget} depth {depth} mode {mode} parity "
                        f"failed: {problems[:6]}")
                median = statistics.median(times)
                key = f"b{budget}_d{depth}_{mode}"
                out["depths"][key] = {
                    "starting_budget": budget,
                    "reference_seconds": round(reference_seconds, 4),
                    "batched_seconds": [round(t, 4) for t in times],
                    "median_batched_seconds": round(median, 4),
                    "speedup": round(reference_seconds / median, 3),
                    "parity": True,
                    "stats": batched.stats(),
                    "programs": R.fingerprint(batched)["frozen"],
                }
                print(f"[budget {budget} depth {depth} {mode}] median "
                      f"{median:.3f}s speedup {reference_seconds / median:.2f}x "
                      f"stats={out['depths'][key]['stats']}", flush=True)

    cache.commit()
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    # `BATCH_SIZE=16384 modal run modal_oops_v11.py` selects the batch shape.
    batch_size = int(os.environ.get("BATCH_SIZE", "8192"))
    result = json.loads(run.remote(depths=[4, 5, 6], budgets=[256],
                                   repeats=3, batch_size=batch_size))
    suffix = "" if batch_size == 8192 else f"_b{batch_size}"
    path = (Path(__file__).resolve().parent / "experiments_v12" / "results"
            / "gpu_profile" / f"l40s_batched_v11{suffix}.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tests"}, indent=2))
    print("Saved to", path)


if __name__ == "__main__":
    main()
