"""Experiment D: full system (C5) versus conventional planners.

Every method receives the same causal observation of the same scenario and is
subject to the same safety shield and the same per-step command cap
(``stride_cap="language"``), so differences reflect planning quality rather
than step length.  Planner variants with the environment's full speed envelope
(``stride_cap="full"``) are reported separately as an unconstrained reference.

The privileged time-expanded oracle from ``planners.py`` is included only when
``--oracle N`` is given, is labelled ``oracle_privileged``, and must never be
presented as a fair baseline: it reads the simulator's future obstacle states.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments_v12._common import (add_common_args, curriculum_for, out_dir,
                                     parse_levels, save_json, test_split)
from experiments_v12.ablation_matrix import C5
from experiments_v12.baseline_planners import PLANNERS, PlannerStats, make_planner
from experiments_v12.evaluate import PlannerPolicy, method_row, summarize, write_rows
from experiments_v12.policies import make_policy
from experiments_v12.statistics import paired_rate_comparison, paired_comparison
from experiments_v12.synthesis import deployment_library, train_variant
from oops_language import MotionConfig, Program
from chamber import norm, sub

from experiments_v12.run_generalization import load_or_train, policies_from_payload


def oracle_row(spec, env, max_scenes_hint: int = 0) -> dict:
    from planners import OracleTimeExpandedAStar, next_velocity
    planner = OracleTimeExpandedAStar()
    env.reset()
    collisions = False
    interventions = 0
    path = 0.0
    previous = env.position
    plan = planner.plan(env)
    steps = 0
    while not env.done and steps < env.max_steps:
        velocity = next_velocity(env, plan) if plan.found else (0.0, 0.0, 0.0)
        result = env.step(velocity)
        collisions |= result.collision
        interventions += int(result.safety_intervention)
        path += norm(sub(env.position, previous))
        previous = env.position
        steps += 1
        if result.done:
            break
        plan = planner.plan(env)
    reached = (env.done and not collisions
               and norm(sub(env.position, env.goal)) <= env.goal_tolerance + 1e-9)
    return {
        "method": "oracle_privileged", "scenario_id": spec.id,
        "level": spec.level, "label": spec.label, "seed": spec.seed,
        "success": bool(reached), "collision": bool(collisions),
        "steps": steps, "time_to_goal_s": (steps * env.dt) if reached else 0.0,
        "path_length_m": path, "interventions": interventions,
        "min_clearance_m": -1.0, "extra_distance_ratio":
            path / (norm(sub(env.goal, env.start)) or 1e-9),
        "mean_speed_ratio": path / (steps * env.dt * env.max_speed) if steps else 0.0,
        "waits": 0, "decision_latency_mean_s": 0.0,
        "decision_latency_p95_s": 0.0, "deadline_misses": 0,
        "extras": {"expansions": planner.expansions, "reason": plan.reason,
                   "disclaimer": "reads future obstacle states; not a fair baseline"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner-expansions", type=int, default=1200,
                        help="per-step expansion cap for grid planners")
    parser.add_argument("--planner-deadline", type=float, default=0.5,
                        help="per-step planning deadline (seconds)")
    parser.add_argument("--oracle", type=int, default=0,
                        help="number of scenarios for the privileged oracle")
    parser.add_argument("--full-speed", action="store_true",
                        help="also run planners at max_speed*dt command cap")
    args = parser.parse_args(argv)
    levels = parse_levels(args.levels)
    destination = out_dir(args.out or "baselines")
    limit = args.limit if args.limit else (5 if args.smoke else None)

    planner_specs = [(kind, "language") for kind in PLANNERS]
    if args.full_speed:
        planner_specs += [(kind, "full") for kind in PLANNERS]

    all_rows: list[dict] = []
    for seed_index in range(args.seeds):
        payload = load_or_train(seed_index, destination)
        program, library = policies_from_payload(payload)
        for level in levels:
            scenarios = test_split(level)
            if limit:
                scenarios = scenarios[:limit]
            for position, spec in enumerate(scenarios):
                env = spec.build()
                config = MotionConfig(radius=env.radius, dt=env.dt,
                                      max_speed=env.max_speed)
                if program is not None:
                    policy = make_policy(C5, program, library, config)
                    row = method_row("C5", spec, env, policy)
                    row["seed_index"] = seed_index
                    row["stride_cap"] = "language"
                    all_rows.append(row)
                for kind, cap in planner_specs:
                    planner = make_planner(
                        kind, stride_cap=cap,
                        goal_tolerance=env.goal_tolerance,
                        max_expansions=args.planner_expansions,
                        deadline_seconds=args.planner_deadline)
                    row = method_row(f"{kind}[{cap}]", spec, env,
                                     PlannerPolicy(planner, config))
                    row["seed_index"] = seed_index
                    row["stride_cap"] = cap
                    all_rows.append(row)
                if (position + 1) % 20 == 0:
                    print(f"[seed {seed_index}] L{level} {position + 1}/"
                          f"{len(scenarios)} scenarios", flush=True)
            print(f"[seed {seed_index}] level {level} done", flush=True)
        print(f"[seed {seed_index}] planner comparison done", flush=True)

    if args.oracle:
        oracle_scenarios = []
        for level in levels:
            oracle_scenarios.extend(test_split(level)[:args.oracle])
        for spec in oracle_scenarios:
            env = spec.build()
            row = oracle_row(spec, env)
            row["seed_index"] = 0
            row["stride_cap"] = "language"
            all_rows.append(row)

    write_rows(all_rows, destination / "episodes.json")
    write_rows(all_rows, destination / "episodes.csv")

    methods = sorted({r["method"] for r in all_rows})
    contrasts = {}
    for level in levels:
        level_rows = [r for r in all_rows if r["level"] == level]
        base = [r for r in level_rows if r["method"] == "C5"]
        for method in methods:
            if method == "C5":
                continue
            other = [r for r in level_rows if r["method"] == method]
            if not base or not other or len(base) != len(other):
                continue
            contrasts[f"L{level}:C5_vs_{method}"] = {
                "success": paired_rate_comparison(
                    [int(r["success"]) for r in base],
                    [int(r["success"]) for r in other]),
                "steps": paired_comparison([r["steps"] for r in base],
                                           [r["steps"] for r in other]),
            }
    summary = {
        "format": "oops-v12-baselines-1.0",
        "seeds": args.seeds,
        "levels": levels,
        "methods": methods,
        "overall": summarize(all_rows, ("method",)),
        "by_level": summarize(all_rows, ("method", "level")),
        "contrasts": contrasts,
        "notes": [
            "All methods share the swept-collision rule, the safety shield and "
            "the causal observation.",
            "stride_cap=language caps every method at MotionConfig.stride per "
            "control period; stride_cap=full uses max_speed*dt instead.",
            f"Planners run with max_expansions={args.planner_expansions} per "
            f"step and a {args.planner_deadline}s deadline; the deadline is "
            "post-hoc (overrun actions become safe waits) and misses are "
            "reported per episode.",
            "oracle_privileged reads future obstacle states and is reported "
            "only as an upper reference, never as a fair baseline.",
        ],
    }
    save_json(summary, destination / "summary.json")
    print(f"wrote {destination}/summary.json ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
