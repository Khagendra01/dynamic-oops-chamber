"""Run the GPU-prefixed prefix search and verify it against the CPU reference.

Both searches are the reference `PrefixOOPS` decision code; only the
candidate-episode evaluations differ (batched on `device` versus one Python
`replay` at a time). Stage results, discovered programs, lane spend, visit and
test counts and replay-call counts must match exactly.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from oops_batched_search_v11 import BatchedPrefixOOPS
from oops_prefix import PrefixOOPS, standard_tasks


def params(args) -> dict:
    return dict(reuse=True, max_depth=args.max_depth,
                starting_budget=args.starting_budget,
                max_phase_budget=1048576, max_prefixes_per_phase=30000)


def run_reference(args) -> tuple[PrefixOOPS, float]:
    model = PrefixOOPS(**params(args))
    started = time.perf_counter()
    for task in standard_tasks():
        model.train_task(task)
    return model, time.perf_counter() - started


def run_batched(args, step_fn) -> tuple[BatchedPrefixOOPS, list[float]]:
    def build() -> BatchedPrefixOOPS:
        return BatchedPrefixOOPS(
            device=args.device, dtype=getattr(torch, args.dtype),
        batch_size=args.batch_size, program_width=args.program_width,
        merge_waves=getattr(args, "merge_waves", False),
        step_fn=step_fn, verbose=args.verbose, **params(args))

    model = build()
    times: list[float] = []
    for trial in range(max(1, args.repeats)):
        started = time.perf_counter()
        for task in standard_tasks():
            model.train_task(task)
        seconds = time.perf_counter() - started
        times.append(seconds)
        stats = model.stats()
        print(f"batched[{trial}] {seconds:.3f}s "
              f"episodes={stats['gpu_episodes']} "
              f"batches={stats['gpu_batches']} "
              f"prefetch={stats['prefetch_seconds']}s "
              f"walk={stats['walk_seconds']}s "
              f"timers={stats['timers']}", flush=True)
        if trial + 1 < max(1, args.repeats):
            model = build()
    return model, times


def fingerprint(model) -> dict:
    return {
        "frozen": {pid: list(p.rules) for pid, p in model.frozen.items()},
        "stages": [asdict(stage) for stage in model.history],
        "real_replay_calls": model.real_replay_calls,
        "simulated_steps": model.simulated_steps,
    }


def compare(reference: dict, batched: dict) -> list[str]:
    problems = []
    if reference["frozen"] != batched["frozen"]:
        problems.append(f"frozen programs differ: {reference['frozen']} != "
                        f"{batched['frozen']}")
    if len(reference["stages"]) != len(batched["stages"]):
        problems.append("stage count differs")
        return problems
    for index, (want, have) in enumerate(zip(reference["stages"],
                                             batched["stages"])):
        for field in want:
            if want[field] != have.get(field):
                problems.append(f"stage {index} field {field}: "
                                f"reference={want[field]!r} batched={have.get(field)!r}")
    for field in ("real_replay_calls", "simulated_steps"):
        if reference[field] != batched[field]:
            problems.append(f"{field}: reference={reference[field]} "
                            f"batched={batched[field]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--program-width", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--starting-budget", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the batched control step")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--merge-waves", action="store_true",
                        help="evaluate a whole stage's pairs in one wave")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    reference_model = None
    reference_seconds = 0.0
    if not args.skip_reference:
        reference_model, reference_seconds = run_reference(args)
        print(f"reference  {reference_seconds:.3f}s  "
              f"replay_calls={reference_model.real_replay_calls} "
              f"steps={reference_model.simulated_steps}", flush=True)

    step_fn = None
    compile_seconds = 0.0
    if args.compile:
        compiled = torch.compile(lambda policy: policy.step(),
                                  mode=args.compile_mode, dynamic=False)
        step_fn = lambda policy: compiled(policy)   # noqa: E731
        print(f"compiled step_fn mode={args.compile_mode}", flush=True)

    times: list[float] = []
    batched_model, times = run_batched(args, step_fn)

    problems = []
    if reference_model is not None:
        problems = compare(fingerprint(reference_model),
                           fingerprint(batched_model))
        if problems:
            print("PARITY FAILED")
            for line in problems[:40]:
                print("  ", line)
            return 1
        print(f"PARITY OK: {len(reference_model.history)} stages, "
              f"{reference_model.real_replay_calls} replay calls, "
              f"{reference_model.simulated_steps} simulated steps",
              flush=True)

    median = statistics.median(times)
    print(f"median batched {median:.3f}s over {len(times)} trials")
    if reference_model is not None and reference_seconds:
        print(f"speedup {reference_seconds / median:.2f}x "
              f"({reference_seconds:.3f}s -> {median:.3f}s)")

    if args.output:
        payload = {
            "environment": {"python": sys.version.split()[0],
                            "platform": platform.platform(),
                            "torch": torch.__version__},
            "parameters": {k: v for k, v in vars(args).items()},
            "reference_seconds": reference_seconds,
            "batched_seconds": times,
            "median_batched_seconds": median,
            "speedup": (reference_seconds / median if reference_seconds else None),
            "stats": batched_model.stats(),
            "programs": fingerprint(batched_model)["frozen"],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
