"""Exhaustive CPU-parity sweep for the batched job evaluator.

Compares every EpisodeResult field of `oops_training.replay` against the batched
GPU/CPU evaluator over all unique candidate programs recorded by the reference
search, across all eight curriculum tasks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from oops_language import Program
from oops_training import extended_curriculum, replay
from test_batched_v11 import FIELDS, TASK_INDEX, batched_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="/tmp/opencode/oops_v10/reference_queries.json")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--max-programs", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = json.loads(Path(args.queries).read_text())
    programs = sorted({tuple(row[0]) for row in data["queries"]})
    if args.max_programs:
        programs = programs[: args.max_programs]
    tasks = [(name, factory) for name, factory, *_ in extended_curriculum()]

    pairs = [(rules, name) for rules in programs for name, _ in tasks]
    print(f"{len(programs)} programs x {len(tasks)} tasks = {len(pairs)} episodes")

    failures = []
    t0 = time.perf_counter()
    for start in range(0, len(pairs), args.chunk):
        chunk = pairs[start:start + args.chunk]
        got = batched_episodes(chunk, {}, device=args.device)
        for i, (rules, task_name) in enumerate(chunk):
            factory = dict(tasks)[task_name]
            expected, _ = replay(Program(tuple(rules)), factory, {})
            for field in FIELDS:
                want = getattr(expected, field)
                have = got[field][i].item()
                want = int(bool(want)) if isinstance(want, bool) else int(want)
                if want != int(have):
                    failures.append((rules, task_name, field, want, int(have)))
        done = min(start + args.chunk, len(pairs))
        print(f"  {done}/{len(pairs)} episodes, {len(failures)} mismatches "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        if failures:
            break

    if failures:
        print("FAILED")
        for row in failures[:20]:
            print("  ", row)
        return 1
    print(f"OK: {len(pairs)} episodes matched on all {len(FIELDS)} fields "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
