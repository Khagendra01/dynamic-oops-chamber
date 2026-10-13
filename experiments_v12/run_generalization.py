"""Experiment A: generalization of the full system (C5) to unseen dynamics.

Training uses only the level 1-2 curriculum; evaluation covers every level,
including levels 3-5 whose dynamics (velocity changes, appear/disappear
events, denser scenes) never appear in training.  Each curriculum seed trains
its own program, and every program is evaluated on the *same* held-out
validation and test scenarios, so results are paired across seeds.

Hypothesis H1: bounded OOPS-inspired incremental synthesis generalizes to
unseen dynamic environments.  Reported as per-level success rates with Wilson
intervals and paired contrasts between the seen (L1-L2) and unseen (L3-L5)
level groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments_v12._common import (add_common_args, curriculum_for, out_dir,
                                     parse_levels, save_json, test_split,
                                     validation_split)
from experiments_v12.ablation_matrix import C5
from experiments_v12.evaluate import method_row, summarize, write_rows
from experiments_v12.policies import make_policy
from experiments_v12.statistics import wilson_interval
from experiments_v12.synthesis import deployment_library, train_variant
from oops_language import MotionConfig

SEEN_LEVELS = (1, 2)
UNSEEN_LEVELS = (3, 4, 5)


def load_or_train(seed_index: int, destination: Path, *, limit_stages: int | None = None):
    cached = destination / f"synthesis_seed{seed_index}_C5.json"
    if cached.exists():
        payload = json.loads(cached.read_text())
        if payload["variant"]["id"] == "C5":
            return payload
    curriculum = curriculum_for(seed_index)
    if limit_stages:
        curriculum = curriculum[:limit_stages]
    outcome = train_variant(C5, curriculum)
    save_json(outcome.to_dict(), cached)
    return outcome.to_dict()


def policies_from_payload(payload: dict):
    from oops_language import Program
    library = {entry["id"]: Program(tuple(entry["rules"]), entry["id"])
               for entry in payload.get("library", [])}
    program_id = payload["deployed_program_id"]
    if program_id is None:
        return None, {}
    return library[program_id], library


def evaluate_seed(payload: dict, scenarios, seed_index: int,
                  split_name: str) -> list[dict]:
    program, library = policies_from_payload(payload)
    rows = []
    for spec in scenarios:
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        if program is None:
            row = {"method": "C5", "scenario_id": spec.id, "level": spec.level,
                   "label": spec.label, "seed": spec.seed, "success": False,
                   "collision": False, "steps": 0, "time_to_goal_s": 0.0,
                   "path_length_m": 0.0, "interventions": 0,
                   "min_clearance_m": -1.0, "extra_distance_ratio": 0.0,
                   "mean_speed_ratio": 0.0, "waits": 0,
                   "decision_latency_mean_s": 0.0, "decision_latency_p95_s": 0.0,
                   "deadline_misses": 0, "extras": {"note": "no program"}}
        else:
            row = method_row("C5", spec, env,
                             make_policy(C5, program, library, config))
        row["seed_index"] = seed_index
        row["split"] = split_name
        rows.append(row)
    return rows


def level_rates(rows: list[dict]) -> list[dict]:
    out = []
    for level in sorted({r["level"] for r in rows}):
        members = [r for r in rows if r["level"] == level]
        successes = sum(int(m["success"]) for m in members)
        out.append({"level": level, "n": len(members),
                    **wilson_interval(successes, len(members)),
                    "collision_rate": sum(int(m["collision"]) for m in members)
                    / len(members),
                    "mean_steps_solved": (
                        sum(m["steps"] for m in members if m["success"])
                        / max(1, sum(1 for m in members if m["success"])))})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--include-validation", action="store_true",
                        default=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per level (debugging)")
    args = parser.parse_args(argv)
    levels = parse_levels(args.levels)
    limit = args.limit if args.limit else (3 if args.smoke else None)
    destination = out_dir(args.out or "generalization")

    all_rows: list[dict] = []
    programs = []
    for seed_index in range(args.seeds):
        payload = load_or_train(seed_index, destination)
        programs.append({
            "seed_index": seed_index,
            "program_id": payload["deployed_program_id"],
            "rules": payload["deployed_rules"],
        })
        for level in levels:
            test_specs = test_split(level)
            valid_specs = validation_split(level)
            if limit:
                test_specs, valid_specs = test_specs[:limit], valid_specs[:limit]
            all_rows.extend(evaluate_seed(payload, test_specs, seed_index, "test"))
            if args.include_validation:
                all_rows.extend(evaluate_seed(payload, valid_specs, seed_index,
                                              "validation"))
        print(f"[seed {seed_index}] evaluated program "
              f"{payload['deployed_program_id']} {payload['deployed_rules']}",
              flush=True)

    write_rows(all_rows, destination / "episodes.json")
    write_rows(all_rows, destination / "episodes.csv")

    test_rows = [r for r in all_rows if r["split"] == "test"]
    seen = [r for r in test_rows if r["level"] in SEEN_LEVELS]
    unseen = [r for r in test_rows if r["level"] in UNSEEN_LEVELS]
    per_seed = []
    for seed_index in range(args.seeds):
        seed_rows = [r for r in test_rows if r["seed_index"] == seed_index]
        per_seed.append({
            "seed_index": seed_index,
            "program_id": programs[seed_index]["program_id"],
            "success_rate": sum(int(r["success"]) for r in seed_rows)
            / len(seed_rows) if seed_rows else 0.0,
        })
    summary = {
        "format": "oops-v12-generalization-1.0",
        "training_levels": list(SEEN_LEVELS),
        "evaluated_levels": levels,
        "seeds": args.seeds,
        "limit_per_level": limit,
        "programs": programs,
        "test_by_level": level_rates(test_rows),
        "test_overall": summarize(test_rows, ("method",)),
        "validation_overall": summarize(
            [r for r in all_rows if r["split"] == "validation"], ("method",)),
        "seen_vs_unseen": {
            "seen_levels": list(SEEN_LEVELS),
            "unseen_levels": list(UNSEEN_LEVELS),
            # Scenario sets differ in size, so this contrast is reported with
            # independent Wilson intervals rather than a paired difference.
            "seen": wilson_interval(sum(int(r["success"]) for r in seen),
                                    len(seen)),
            "unseen": wilson_interval(sum(int(r["success"]) for r in unseen),
                                      len(unseen)),
            "unseen_minus_seen": (
                sum(int(r["success"]) for r in unseen) / len(unseen)
                - sum(int(r["success"]) for r in seen) / len(seen))
            if seen and unseen else None,
        },
        "per_seed": per_seed,
    }
    save_json(summary, destination / "summary.json")
    print(f"wrote {destination}/summary.json ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
