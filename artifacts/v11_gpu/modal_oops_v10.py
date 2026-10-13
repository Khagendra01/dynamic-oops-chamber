"""Dynamic OOPS Chamber v1.0 — first L40S acceptance run on Modal.

Acceptance gate: CUDA parity on 100 fixed seeds must match the CPU reference
exactly (success, collision, step count), otherwise the function raises.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

PROJECT = "/project"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .add_local_dir("/tmp/opencode/oops_v10/stage/dynamic_oops_chamber_v1_0", PROJECT, copy=True)
)

app = modal.App("oops-v10-l40s", image=image)


def sh(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd), "returncode": p.returncode,
           "elapsed_s": round(time.perf_counter() - t0, 3),
           "stdout": p.stdout, "stderr": p.stderr}
    print(f"[{out['elapsed_s']:8.2f}s] rc={p.returncode} {' '.join(cmd)}")
    if p.stdout.strip():
        print(p.stdout.strip()[-4000:])
    if p.stderr.strip():
        print(p.stderr.strip()[-4000:])
    return out


@app.function(gpu="L40S", timeout=5400, cpu=16, memory=32768)
def accept(count: int = 100, scenes: int = 512) -> dict:
    import torch

    env = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "nvidia_smi": sh(["nvidia-smi"])["stdout"],
    }
    print(json.dumps({k: v for k, v in env.items() if k != "nvidia_smi"}, indent=2))
    if not env["cuda_available"]:
        raise RuntimeError("CUDA not available in this container")

    report: dict = {"env": env}
    report["tests"] = sh([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py", "-q"])

    # Gate 1: CPU reference reproduces the frozen 100-scene baseline.
    report["cpu_parity"] = sh([sys.executable, "verify_torch_v10.py", "--device", "cpu",
                               "--count", str(count), "--out", "cpu_parity.json"])
    if report["cpu_parity"]["returncode"] != 0:
        raise RuntimeError("CPU parity gate failed")

    # Gate 2: CUDA parity vs CPU reference — the primary acceptance test.
    report["cuda_parity"] = sh([sys.executable, "verify_torch_v10.py", "--device", "cuda",
                                "--count", str(count), "--out", "cuda_parity.json"])
    if report["cuda_parity"]["returncode"] != 0:
        parity = json.loads(Path(PROJECT, "cuda_parity.json").read_text())
        raise RuntimeError(f"CUDA PARITY GATE FAILED: {parity['mismatches'][:10]}")

    # Matched CPU/GPU throughput on the same host (float64).
    report["cpu_run"] = sh([sys.executable, "run_torch_v10.py", "--device", "cpu",
                            "--batch-size", str(scenes), "--scenes", str(scenes),
                            "--dtype", "float64", "--out", "cpu_run.json"])

    # Primary GPU workload: 512 scenes in one batch, float64.
    report["cuda_512_f64"] = sh([sys.executable, "run_torch_v10.py", "--device", "cuda",
                                 "--batch-size", str(scenes), "--scenes", str(scenes),
                                 "--dtype", "float64", "--out", "cuda_512_f64.json"])

    # Larger batch, float32: profile the throughput/precision trade-off.
    report["cuda_1024_f32"] = sh([sys.executable, "run_torch_v10.py", "--device", "cuda",
                                  "--batch-size", "1024", "--scenes", "1024",
                                  "--dtype", "float32", "--out", "cuda_1024_f32.json"])

    # Summaries + full JSON artifacts, returned to the client.
    files = {}
    for name in ["cpu_parity.json", "cuda_parity.json", "cpu_run.json",
                 "cuda_512_f64.json", "cuda_1024_f32.json"]:
        files[name] = json.loads(Path(PROJECT, name).read_text())

    def summary(key: str) -> dict:
        row = report[key]
        return {"returncode": row["returncode"], "elapsed_s": row["elapsed_s"],
                "tail": row["stdout"].strip().splitlines()[-1] if row["stdout"].strip() else ""}

    summary_out = {
        "env": {k: v for k, v in env.items() if k != "nvidia_smi"},
        "tests": {"returncode": report["tests"]["returncode"],
                  "tail": report["tests"]["stdout"].strip().splitlines()[-3:]},
        "cpu_parity": {k: v for k, v in files["cpu_parity.json"].items() if k != "mismatches"},
        "cuda_parity": {k: v for k, v in files["cuda_parity.json"].items() if k != "mismatches"},
        "commands": {k: summary(k) for k in ["cpu_parity", "cuda_parity", "cpu_run",
                                             "cuda_512_f64", "cuda_1024_f32"]},
        "cpu_run": {k: v for k, v in files["cpu_run.json"].items() if k not in ("runs", "notes")},
        "cuda_512_f64": {k: v for k, v in files["cuda_512_f64.json"].items() if k not in ("runs", "notes")},
        "cuda_1024_f32": {k: v for k, v in files["cuda_1024_f32.json"].items() if k not in ("runs", "notes")},
    }
    return json.dumps({"summary": summary_out, "artifacts": files})


@app.local_entrypoint()
def main():
    import json as _json
    result = _json.loads(accept.remote())
    out = Path("/tmp/opencode/oops_v10/l40s_results.json")
    out.write_text(_json.dumps(result, indent=2) + "\n")
    print(_json.dumps(result["summary"], indent=2))
    print("Full artifacts saved to", out)
