"""Experiment C: C1-C5 ablations (plus the full 2x2x2 factorial).

Trains every variant of :mod:`experiments_v12.ablation_matrix` on the identical
curriculum, saves each discovered program, evaluates deployment navigation on
the held-out test split of every level, and reports paired contrasts.

Outputs (under ``experiments_v12/results/ablations/``):
``variant_configs/``        per-variant configuration JSON (the matrix)
``synthesis_<seed>_<id>.json``  discovered programs + per-stage search cost
``episodes_<seed>.json|csv``    raw per-episode rows for every variant
``program_lock_<seed>.json``    C5's deployed program forced through every
                                execution configuration (library x memory),
                                separating discovery effects from deployment
``summary.json``                aggregates and paired statistics

Training cost (search work units, replay calls, wall time, prefixes tested) is
reported separately from deployment navigation metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments_v12._common import (add_common_args, curriculum_for, out_dir,
                                     parse_levels, save_json, test_split)
from experiments_v12.ablation_matrix import (CANONICAL, FACTORIAL, FACTORIAL_EXTRA,
                                             PRIMARY_CONTRASTS, validate_matrix,
                                             write_config_files, by_id)
from experiments_v12.evaluate import method_row, summarize, write_rows
from experiments_v12.policies import make_policy
from experiments_v12.statistics import paired_comparison, paired_rate_comparison
from experiments_v12.synthesis import deployment_library, train_variant
from oops_language import MotionConfig, Program


def search_params(smoke: bool) -> dict:
    if smoke:
        return dict(max_depth=3, starting_budget=8, max_phase_budget=256,
                    max_prefixes_per_phase=200)
    return dict(max_depth=4, starting_budget=256, max_phase_budget=1048576,
                max_prefixes_per_phase=30000)


def evaluate_variant(variant, outcome, scenarios, limit: int | None = None) -> list[dict]:
    rows = []
    program = None
    if outcome.deployed_program_id:
        program = outcome.library[outcome.deployed_program_id]
    library = deployment_library(outcome.learner, variant)
    for spec in scenarios[:limit] if limit else scenarios:
        env = spec.build()
        config = MotionConfig(radius=env.radius, dt=env.dt,
                              max_speed=env.max_speed)
        if program is None:
            rows.append({
                "method": variant.id, "scenario_id": spec.id,
                "level": spec.level, "label": spec.label, "seed": spec.seed,
                "success": False, "collision": False, "steps": 0,
                "time_to_goal_s": 0.0, "path_length_m": 0.0,
                "interventions": 0, "min_clearance_m": -1.0,
                "extra_distance_ratio": 0.0, "mean_speed_ratio": 0.0,
                "waits": 0, "decision_latency_mean_s": 0.0,
                "decision_latency_p95_s": 0.0, "deadline_misses": 0,
                "extras": {"note": "no program discovered"},
            })
            continue
        policy = make_policy(variant, program, library, config)
        rows.append(method_row(variant.id, spec, env, policy))
    return rows


def program_lock_evaluation(outcome, scenarios, limit: int | None = None) -> list[dict]:
    """Execute ONE fixed program (C5's) under every execution configuration."""
    if not outcome.deployed_program_id:
        return []
    program = outcome.library[outcome.deployed_program_id]
    library = dict(outcome.library)
    rows = []
    for variant in FACTORIAL:
        for spec in (scenarios[:limit] if limit else scenarios):
            env = spec.build()
            config = MotionConfig(radius=env.radius, dt=env.dt,
                                  max_speed=env.max_speed)
            policy = make_policy(variant, program, library, config)
            row = method_row(f"lock_{variant.id}", spec, env, policy)
            row["locked_program_id"] = program.id
            row["locked_program_rules"] = list(program.rules)
            rows.append(row)
    return rows


def training_rows(seed_index: int, outcomes: dict) -> list[dict]:
    rows = []
    for variant_id, outcome in outcomes.items():
        totals = dict(outcome.totals)
        rows.append({
            "seed_index": seed_index, "variant": variant_id,
            "deployed_program_id": outcome.deployed_program_id,
            "deployed_rules": " ".join(outcome.deployed_rules),
            "stages_completed": totals["stages_completed"],
            "stages_succeeded": totals["stages_succeeded"],
            "wall_time_s": totals["wall_time_s"],
            "search_work_units": totals["search_work_units"],
            "replay_calls": totals["replay_calls"],
            "prefixes_tested": totals["prefixes_tested"],
            "total_prefix_visits": totals["total_prefix_visits"],
            "simulated_steps": totals["simulated_steps"],
            "reused_steps": totals["reused_steps"],
        })
    return rows


def summarize_training(rows: list[dict]) -> list[dict]:
    """Aggregate search cost per variant (separate from navigation metrics)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    out = []
    for variant, members in sorted(groups.items()):
        n = len(members)
        out.append({
            "variant": variant, "n_seeds": n,
            "mean_wall_time_s": sum(m["wall_time_s"] for m in members) / n,
            "mean_search_work_units": sum(m["search_work_units"] for m in members) / n,
            "mean_replay_calls": sum(m["replay_calls"] for m in members) / n,
            "mean_prefixes_tested": sum(m["prefixes_tested"] for m in members) / n,
            "mean_total_prefix_visits": sum(m["total_prefix_visits"] for m in members) / n,
            "mean_simulated_steps": sum(m["simulated_steps"] for m in members) / n,
            "mean_reused_steps": sum(m["reused_steps"] for m in members) / n,
            "stages_succeeded_total": sum(m["stages_succeeded"] for m in members),
            "programs_found": sum(1 for m in members if m["deployed_program_id"]),
        })
    return out


def contrasts_for(rows: list[dict], variants: tuple[str, ...]) -> dict:
    by_variant = {v: [r for r in rows if r["method"] == v] for v in variants}
    results = {}
    for left, right, label in PRIMARY_CONTRASTS:
        a, b = by_variant.get(left, []), by_variant.get(right, [])
        if not a or not b or len(a) != len(b):
            continue
        key = f"{left}_vs_{right}"
        results[key] = {
            "factor": label,
            "success": paired_rate_comparison([int(r["success"]) for r in a],
                                              [int(r["success"]) for r in b]),
            "steps": paired_comparison([r["steps"] for r in a],
                                       [r["steps"] for r in b]),
            "collision": paired_rate_comparison([int(r["collision"]) for r in a],
                                                [int(r["collision"]) for r in b]),
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per level (debugging)")
    parser.add_argument("--no-factorial-extra", action="store_true",
                        help="run only C1-C5, skip the three extra cells")
    args = parser.parse_args(argv)

    validate_matrix()
    levels = parse_levels(args.levels)
    destination = out_dir(args.out or "ablations")
    configs_dir = destination / "variant_configs"
    variants = CANONICAL if args.no_factorial_extra else FACTORIAL
    write_config_files(configs_dir, variants)
    params = search_params(args.smoke)
    limit = args.limit if args.limit else (5 if args.smoke else None)

    all_rows: list[dict] = []
    all_training: list[dict] = []
    all_locks: list[dict] = []
    seed_summaries = []

    for seed_index in range(args.seeds):
        started = time.perf_counter()
        curriculum = curriculum_for(seed_index)
        outcomes = {}
        for variant in variants:
            print(f"[seed {seed_index}] training {variant.id} "
                  f"({variant.label}) ...", flush=True)
            outcome = train_variant(variant, curriculum, **params)
            outcomes[variant.id] = outcome
            save_json(outcome.to_dict(),
                      destination / f"synthesis_seed{seed_index}_{variant.id}.json")
        all_training.extend(training_rows(seed_index, outcomes))

        scenarios = []
        for level in levels:
            scenarios.extend(test_split(level))
        rows = []
        for variant_id, outcome in outcomes.items():
            rows.extend(evaluate_variant(by_id(variant_id), outcome, scenarios,
                                         limit=limit))
        for row in rows:
            row["seed_index"] = seed_index
        all_rows.extend(rows)

        full = outcomes.get("C5")
        if full is not None:
            locks = program_lock_evaluation(full, scenarios, limit=limit)
            for row in locks:
                row["seed_index"] = seed_index
            all_locks.extend(locks)

        seed_summaries.append({
            "seed_index": seed_index,
            "curriculum_stages": len(curriculum),
            "wall_time_s": time.perf_counter() - started,
            "variants": {vid: {
                "deployed_program_id": o.deployed_program_id,
                "deployed_rules": list(o.deployed_rules),
                "totals": o.totals,
            } for vid, o in outcomes.items()},
        })
        print(f"[seed {seed_index}] done in "
              f"{seed_summaries[-1]['wall_time_s']:.1f}s", flush=True)

    write_rows(all_rows, destination / f"episodes_seed{args.seeds}.json")
    write_rows(all_rows, destination / f"episodes_seed{args.seeds}.csv")
    write_rows(all_training, destination / "training_cost.csv")
    write_rows(all_locks, destination / "program_lock_episodes.csv")

    variant_ids = [v.id for v in variants]
    summary = {
        "format": "oops-v12-ablations-1.0",
        "seeds": args.seeds,
        "levels": levels,
        "smoke": bool(args.smoke),
        "search_params": params,
        "scenarios_per_level": limit or "all",
        "training_cost": summarize_training(all_training),
        "deployment": summarize(all_rows, ("method",)),
        "deployment_by_level": summarize(all_rows, ("method", "level")),
        "contrasts": contrasts_for(all_rows, tuple(variant_ids)),
        "seeds_detail": seed_summaries,
    }
    # Explicit per-seed discovered programs (rule text, not just ids).
    summary["discovered_programs"] = [
        {"seed_index": row["seed_index"], "variant": row["variant"],
         "program_id": row["deployed_program_id"],
         "rules": row["deployed_rules"]}
        for row in all_training
    ]
    save_json(summary, destination / "summary.json")
    print(f"wrote {destination}/summary.json "
          f"({len(all_rows)} episode rows, {len(all_training)} training rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
