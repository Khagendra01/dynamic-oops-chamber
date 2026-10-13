"""Why is torch.compile of the batched job step so expensive?

The L40S step benchmark cost ~10 minutes of inductor compile for ~1.4 ms/step
steady state. This runs the same compile under TORCH_LOGS=graph_breaks,recompiles
and reports dynamo counters plus inductor kernel counts, so the compile budget
can be attributed to graph breaks, recompiles, or plain codegen.
"""
import json
import os
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
    .add_local_dir("/tmp/opencode/oops_v10/stage/dynamic_oops_chamber_v1_0",
                   PROJECT, copy=True)
)

app = modal.App("oops-v11-compile-diag", image=image)


@app.function(gpu="L40S", timeout=5400, cpu=16, memory=32768,
              env={"TORCH_LOGS": "graph_breaks,recompiles"})
def diag(width: int = 12) -> str:
    os.environ.setdefault("TORCH_LOGS", "graph_breaks,recompiles")
    sys.path.insert(0, PROJECT)
    import torch
    import torch._dynamo
    import torch._inductor.metrics
    from torch._dynamo.utils import counters

    from oops_language import Program
    from run_torch_v10 import library
    from test_batched_v11 import TASK_INDEX, TEMPLATES
    from torch_chamber import TorchChamberBatch
    from torch_job_policy import TorchJobPolicy, compile_program

    frozen, program = library()
    print("program", program.rules, flush=True)
    ids = torch.tensor([TASK_INDEX[n] for n in TASK_INDEX], dtype=torch.long)
    ids = ids.repeat(1024 // len(ids))[:1024]

    def make(w):
        env = TorchChamberBatch.from_templates(TEMPLATES, ids,
                                                device="cuda",
                                                dtype=torch.float64)
        progs = [compile_program(program, frozen) for _ in range(env.B)]
        policy = TorchJobPolicy(env, progs, program_width=w)
        policy.run(max_steps=1)  # allocates per-row state
        return env, policy

    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "torcht_logs": os.environ.get("TORCH_LOGS")}

    def snapshot(tag):
        cg = getattr(torch._inductor.metrics, "generated_kernel_count", None)
        return {
            "unique_graphs": counters["stats"].get("unique_graphs"),
            "graph_breaks": {k: v for k, v in
                             sorted(counters.get("graph_break", {}).items())},
            "recompiles": counters.get("stats", {}).get("dynamo_recompile"),
            "inductor_kernels": cg,
            "tag": tag,
        }

    def eager_row(w):
        torch._dynamo.reset()
        counters.clear()
        _env, policy = make(w)
        for _ in range(5):
            policy.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            policy.step()
        torch.cuda.synchronize()
        return {"ms_per_step": round((time.perf_counter() - t0) / 20 * 1e3, 3)}

    def compile_row(w, dynamic, label):
        torch._dynamo.reset()
        counters.clear()
        torch._inductor.metrics.generated_kernel_count = 0
        _env, policy = make(w)
        fn = torch.compile(lambda p: p.step(), mode="default", dynamic=dynamic)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(policy)
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        rec = {"label": label, "width": w, "dynamic": dynamic,
               "compile_s": round(first, 2)}
        rec.update(snapshot("after first call"))
        try:
            t0 = time.perf_counter()
            for _ in range(30):
                fn(policy)
            torch.cuda.synchronize()
            rec["ms_per_step"] = round((time.perf_counter() - t0) / 30 * 1e3, 3)
        except Exception as exc:
            rec["steady_error"] = f"{type(exc).__name__}: {exc}"[:300]
        rec.update(snapshot("after steady"))
        print(json.dumps(rec, indent=2), flush=True)
        return rec

    out["eager_w12"] = eager_row(width)
    print("eager", out["eager_w12"], flush=True)
    out["a_default_w12"] = compile_row(width, False, "A default width12")
    out["b_dynamic_w12"] = compile_row(width, True, "B dynamic width12")
    out["c_default_w6"] = compile_row(6, False, "C default width6")
    path = Path("/root/compile_diag.json")
    path.write_text(json.dumps(out, indent=2))
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(diag.remote())


if __name__ == "__main__":
    main()
