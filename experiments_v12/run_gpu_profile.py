"""Experiment B: evaluation-throughput profile (CPU reference vs batched device).

What this measures
------------------
1. **Device batching throughput** (v1.1 machinery, constant-velocity cuboid
   scenes): the sequential reference search versus the wave-batched evaluator,
   with bit-exact parity of the search decisions.  Reported as wall time,
   speedup and parity of the discovered programs.
2. **v1.2 dynamic-scenario throughput** (CPU): episodes/second of the shared
   evaluation path on the v1.2 dynamic chambers, which is what the ablation
   and baseline experiments actually run.

What this does NOT claim
------------------------
GPU throughput is not a search-complexity reduction.  A faster evaluator does
not change how many prefixes the search must visit to find a program; it only
lowers the cost of each visit.  The batched engine models constant-velocity
cuboids with wall reflections, so v1.2 levels 3-5 (velocity events,
appear/disappear) are *not* run through it and are never silently substituted.

CUDA is used only when available and requested; otherwise results are CPU.
"""
from __future__ import annotations

import argparse
import time

from experiments_v12._common import curriculum_for, out_dir, save_json
from experiments_v12.ablation_matrix import C5
from experiments_v12.evaluate import method_row
from experiments_v12.policies import make_policy
from experiments_v12.scenarios import generate
from experiments_v12.synthesis import train_variant
from oops_language import MotionConfig, Program
from oops_prefix import PrefixOOPS, standard_tasks


def profile_batched(device: str, batch_size: int, merge_waves: bool,
                    max_phase_budget: int) -> dict:
    """Sequential reference versus wave-batched search on the v1.1 curriculum."""
    import torch
    from oops_batched_search_v11 import BatchedPrefixOOPS

    def run(engine_cls, **kwargs):
        engine = engine_cls(reuse=True, max_depth=4,
                            max_phase_budget=max_phase_budget,
                            max_prefixes_per_phase=30000, **kwargs)
        started = time.perf_counter()
        for task in standard_tasks():
            engine.train_task(task)
        wall = time.perf_counter() - started
        return {
            "wall_time_s": wall,
            "program_ids": [p.id for p in engine.frozen.values()],
            "rules": [list(p.rules) for p in engine.frozen.values()],
            "real_replay_calls": engine.real_replay_calls,
            "simulated_steps": engine.simulated_steps,
        }

    reference = run(PrefixOOPS)
    batched = run(BatchedPrefixOOPS, device=device, batch_size=batch_size,
                  merge_waves=merge_waves)
    parity = reference["rules"] == batched["rules"]
    return {
        "device": device,
        "batch_size": batch_size,
        "merge_waves": merge_waves,
        "max_phase_budget": max_phase_budget,
        "reference": reference,
        "batched": batched,
        "speedup": (reference["wall_time_s"] / batched["wall_time_s"]
                    if batched["wall_time_s"] else None),
        "parity": parity,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def profile_dynamic_scenarios(count: int) -> dict:
    """Episodes/second of the shared v1.2 evaluation path (CPU)."""
    scenarios = [generate(1, 1000 + i, split="train") for i in range(count)]
    outcome = train_variant(C5, curriculum_for(0))
    program = outcome.library[outcome.deployed_program_id]
    library = dict(outcome.library)
    started = time.perf_counter()
    rows = []
    for spec in scenarios:
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        policy = make_policy(C5, program, library, config)
        rows.append(method_row("C5", spec, env, policy))
    wall = time.perf_counter() - started
    successes = sum(int(r["success"]) for r in rows)
    return {
        "episodes": len(rows),
        "wall_time_s": wall,
        "episodes_per_second": len(rows) / wall if wall else 0.0,
        "mean_steps": sum(r["steps"] for r in rows) / len(rows),
        "success_rate": successes / len(rows),
        "program_id": outcome.deployed_program_id,
        "rules": list(outcome.deployed_rules),
        "note": ("CPU only: v1.2 dynamic chambers (events, appear/disappear) "
                 "are not covered by the batched cuboid engine."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--merge-waves", action="store_true")
    parser.add_argument("--max-phase-budget", type=int, default=1048576)
    parser.add_argument("--count", type=int, default=50,
                        help="dynamic scenarios for the CPU throughput probe")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--skip-batched", action="store_true")
    args = parser.parse_args(argv)

    destination = out_dir(args.out or "gpu_profile")
    payload: dict = {"format": "oops-v12-gpu-profile-1.0"}
    if not args.skip_batched:
        try:
            payload["batched"] = profile_batched(
                args.device, args.batch_size, args.merge_waves,
                args.max_phase_budget)
        except Exception as exc:  # pragma: no cover - environment dependent
            payload["batched"] = {"error": repr(exc),
                                  "note": "batched engine unavailable here"}
    payload["dynamic_scenarios"] = profile_dynamic_scenarios(args.count)
    save_json(payload, destination / "profile.json")
    print(destination / "profile.json")
    print(payload.get("batched", {}))
    print(payload["dynamic_scenarios"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
