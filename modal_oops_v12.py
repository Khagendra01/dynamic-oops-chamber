"""v1.2 profile on L40S: full test suite + batched search + episode throughput.

Runs the complete v1.1+v1.2 unit suite on the GPU image first (any parity or
regression failure aborts the profile), then:

* the wave-batched prefix search (v1.1 machinery) against its sequential CPU
  reference, with bit-exact parity required before timing;
* the v1.2 dynamic-scenario episode throughput on the instance host CPU
  (labelled as such -- the batched cuboid engine does not cover event scenes).

GPU throughput is evaluator speed only; it is never presented as a
search-complexity reduction.
"""
import json
import os
import platform
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

app = modal.App("oops-v12-profile", image=image)


def sh(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd), "returncode": p.returncode,
           "elapsed_s": round(time.perf_counter() - t0, 3),
           "stdout": p.stdout, "stderr": p.stderr}
    print(f"[{out['elapsed_s']:8.2f}s] rc={p.returncode} {' '.join(cmd)}",
          flush=True)
    for stream in (p.stdout, p.stderr):
        if stream.strip():
            print(stream.strip()[-3000:], flush=True)
    return out


@app.function(gpu="L40S", timeout=5400, cpu=16, memory=32768,
              volumes={"/root/.cache/torch/inductor": cache})
def run(batch_size: int = 8192, dynamic_count: int = 200) -> str:
    sys.path.insert(0, PROJECT)
    import torch
    import experiments_v12.run_gpu_profile as G

    out: dict = {
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0),
                        "capability": torch.cuda.get_device_capability(0),
                        "cpu_count": os.cpu_count()},
        "parameters": {"batch_size": batch_size,
                       "dynamic_count": dynamic_count},
        "tests": sh([sys.executable, "-m", "unittest", "discover", "-p",
                     "test_*.py", "-q"]),
    }
    if out["tests"]["returncode"] != 0:
        raise RuntimeError("unit tests failed on the GPU image")

    out["batched"] = G.profile_batched(
        device="cuda", batch_size=batch_size, merge_waves=True,
        max_phase_budget=1048576)
    if not out["batched"]["parity"]:
        raise RuntimeError("batched search parity failed on L40S")

    out["dynamic_scenarios_host_cpu"] = G.profile_dynamic_scenarios(
        dynamic_count)
    out["dynamic_scenarios_host_cpu"]["note"] = (
        "measured on the GPU instance host CPU (not the local Pi); the "
        "batched cuboid engine does not cover v1.2 event scenes")
    cache.commit()
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    batch_size = int(os.environ.get("BATCH_SIZE", "8192"))
    count = int(os.environ.get("DYNAMIC_COUNT", "200"))
    result = json.loads(run.remote(batch_size=batch_size, dynamic_count=count))
    path = (Path(__file__).resolve().parent / "experiments_v12" / "results"
            / "gpu_profile" / "l40s_v12_profile.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "batched": {k: v for k, v in result["batched"].items()
                    if k not in ("reference", "batched")},
        "dynamic_host_cpu": result["dynamic_scenarios_host_cpu"],
        "tests_rc": result["tests"]["returncode"],
    }
    print(json.dumps(summary, indent=2))
    print("Saved to", path)


if __name__ == "__main__":
    main()
