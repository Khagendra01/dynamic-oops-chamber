"""L40S go/no-go: per-step cost of batched chamber+policy, eager vs torch.compile.

The batched synthesis search needs one GPU call per search level, so per-step
launch overhead (not FLOPs) decides whether GPU beats the CPU reference search.
"""
import json, time
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

app = modal.App("oops-stepbench", image=image)


@app.function(gpu="L40S", timeout=3000, cpu=8, memory=16384)
def bench() -> str:
    import sys
    sys.path.insert(0, PROJECT)
    import torch
    from chamber import example_scene
    from oops_scenarios import north_blocked, south_blocked, moving_gate
    from run_torch_v10 import library
    from torch_chamber import TorchChamberBatch
    from torch_frozen_policy import TorchFrozenPolicy, DIRECTIONS

    factories = [lambda: example_scene('empty'), lambda: example_scene('static'),
                 lambda: example_scene('crossing'), lambda: example_scene('moving'),
                 north_blocked, south_blocked, moving_gate,
                 lambda: example_scene('empty')]
    frozen, program = library()
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    print("device", out["device"], flush=True)

    def make(n):
        envs = [factories[i % len(factories)]() for i in range(n)]
        return TorchChamberBatch(envs, device="cuda", dtype=torch.float64)

    def act_sync_free(policy):
        """TorchFrozenPolicy.act without host synchronization."""
        env = policy.env
        unresolved = ~env.done
        velocity = torch.zeros((env.B, 3), dtype=env.dtype, device=env.device)
        goal_direction = env.goal - env.position
        for rule in policy.rules:
            valid = unresolved
            if rule == "WAIT_GAP":
                proposal = torch.zeros_like(velocity)
                valid = unresolved & policy._wait_gap()
                policy.explicit_waits += valid.to(torch.int64)
            elif rule == "GOAL":
                proposal = policy._velocity(goal_direction, cap=True)
                valid = unresolved & policy._safe(proposal)
            elif rule in DIRECTIONS:
                d = torch.tensor(DIRECTIONS[rule], device=env.device,
                                 dtype=env.dtype).expand(env.B, -1)
                proposal = policy._velocity(d)
                valid = unresolved & policy._safe(proposal)
            elif rule in ("LAST", "CONTINUE"):
                prior = torch.linalg.vector_norm(policy.last_direction, dim=-1) > 1e-12
                proposal = policy._velocity(policy.last_direction)
                if rule == "CONTINUE":
                    tgt = torch.linalg.vector_norm(env.goal - env.position, dim=-1)
                    nxt = torch.linalg.vector_norm(
                        env.goal - (env.position + proposal * env.dt), dim=-1)
                    prior = prior & (nxt <= tgt + .02)
                valid = unresolved & prior & policy._safe(proposal)
            elif rule == "CHOOSE_AXIS":
                proposal, found = policy._choose_axis()
                valid = unresolved & found & policy._safe(proposal)
            else:
                raise ValueError(rule)
            velocity = torch.where(valid[:, None], proposal, velocity)
            if rule != "WAIT_GAP":
                policy.last_direction = torch.where(valid[:, None], proposal,
                                                    policy.last_direction)
            unresolved = unresolved & (~valid)
        policy.emissions += (~env.done).to(torch.int64)
        return velocity

    def one_step(policy, sync_free=True):
        v = act_sync_free(policy) if sync_free else policy.act()
        policy.env.step(v, check_speed=False)

    def time_steps(fn, n_steps, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps

    def eager_row(n):
        batch = make(n)
        policy = TorchFrozenPolicy(batch, program, frozen)
        ms = time_steps(lambda: one_step(policy), 40)
        return {"eager_ms_per_step": round(ms * 1e3, 3),
                "eager_steps_per_s": round(1 / ms)}

    def compile_row(n, mode):
        batch = make(n)
        policy = TorchFrozenPolicy(batch, program, frozen)
        fn = torch.compile(lambda: one_step(policy), mode=mode, dynamic=False)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
        except Exception as exc:
            return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"[:300]}
        try:
            warm = time_steps(fn, 40)
        except Exception as exc:
            return {"mode": mode, "compile_s": round(first, 2),
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
        return {"mode": mode, "compile_s": round(first, 2),
                "ms_per_step": round(warm * 1e3, 3),
                "steps_per_s": round(1 / warm)}

    for n in (256, 1024, 4096):
        out[f"n{n}"] = eager_row(n)
        print("eager", n, out[f"n{n}"], flush=True)

    out["compile_1024_default"] = compile_row(1024, "default")
    print("compile default", out["compile_1024_default"], flush=True)
    out["compile_1024_reduce_overhead"] = compile_row(1024, "reduce-overhead")
    print("compile reduce", out["compile_1024_reduce_overhead"], flush=True)
    out["compile_4096_reduce_overhead"] = compile_row(4096, "reduce-overhead")
    print("compile 4096 reduce", out["compile_4096_reduce_overhead"], flush=True)

    path = Path("/root/stepbench.json")
    path.write_text(json.dumps(out, indent=2))
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    print(bench.remote())


if __name__ == "__main__":
    main()
