"""Experiment E: reactive robustness under degraded observations (level 4).

Hypothesis H3: the method adapts reactively using only causal observations.
Every configuration below is causal by construction: delayed observations
report their own (older) timestamp, noise is a pure function of
(scenario seed, step, channel), and no method ever sees a future obstacle
state.  The same degradation is applied to every method, so differences
between degradation levels measure sensitivity, not unfairness.

Degradation configs: perfect, observation delay, position/velocity noise,
reduced serve rate, limited visibility, and two combined settings.
"""
from __future__ import annotations

import argparse

from experiments_v12._common import (add_common_args, out_dir, parse_levels,
                                     save_json, test_split, curriculum_for)
from experiments_v12.ablation_matrix import C5
from experiments_v12.baseline_planners import make_planner
from experiments_v12.dynamic_observations import ObservationConfig
from experiments_v12.evaluate import (PlannerPolicy, method_row, summarize,
                                      write_rows)
from experiments_v12.policies import make_policy
from experiments_v12.statistics import paired_rate_comparison, paired_comparison
from experiments_v12.synthesis import train_variant
from oops_language import MotionConfig

from experiments_v12.run_generalization import load_or_train, policies_from_payload

DEGRADATIONS = {
    "perfect": {},
    "delay_1": {"delay_steps": 1},
    "delay_2": {"delay_steps": 2},
    "delay_3": {"delay_steps": 3},
    "noise_pos_0.005": {"position_noise": 0.005},
    "noise_pos_0.01": {"position_noise": 0.01},
    "noise_vel_0.02": {"velocity_noise": 0.02},
    "serve_2": {"serve_every": 2},
    "serve_3": {"serve_every": 3},
    "visibility_0.8": {"visibility_range": 0.8},
    "visibility_0.6": {"visibility_range": 0.6},
    "combined_mild": {"delay_steps": 1, "position_noise": 0.005,
                      "velocity_noise": 0.02, "serve_every": 2},
    "combined_severe": {"delay_steps": 2, "position_noise": 0.01,
                        "velocity_noise": 0.05, "serve_every": 3,
                        "visibility_range": 0.8},
}

METHODS = ("C5", "space_time_astar")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="scenarios per level for C5 (default 100)")
    parser.add_argument("--planner-limit", type=int, default=15,
                        help="scenarios per level for the planner subset")
    parser.add_argument("--planner-degradations", type=str,
                        default="perfect,delay_2,combined_severe",
                        help="comma separated subset evaluated for planners")
    parser.add_argument("--planner-expansions", type=int, default=1200,
                        help="per-step expansion cap for the planner")
    args = parser.parse_args(argv)
    levels = parse_levels(args.levels)
    destination = out_dir(args.out or "reactive")
    limit = args.limit if args.limit else (2 if args.smoke else 100)
    planner_limit = args.planner_limit if not args.smoke else min(2, args.planner_limit)
    planner_degradations = [name for name in args.planner_degradations.split(",")
                            if name.strip()]

    all_rows: list[dict] = []
    for seed_index in range(args.seeds):
        payload = load_or_train(seed_index, destination)
        program, library = policies_from_payload(payload)
        for level in levels:
            scenarios = test_split(level)
            for position, spec in enumerate(scenarios):
                if position >= max(limit, planner_limit):
                    break
                env = spec.build()
                config = MotionConfig(radius=env.radius, dt=env.dt,
                                      max_speed=env.max_speed)
                if program is not None and position < limit:
                    # The full system gets the complete degradation sweep.
                    for name, overrides in DEGRADATIONS.items():
                        obs_cfg = ObservationConfig(**{**overrides,
                                                        "seed": spec.seed})
                        row = method_row(
                            "C5", spec, env,
                            make_policy(C5, program, library, config),
                            observation=obs_cfg)
                        row["degradation"] = name
                        row["seed_index"] = seed_index
                        all_rows.append(row)
                # Planners are far more expensive under degraded observations
                # (they re-search every step), so they get a representative
                # subset with default planner budgets.
                if position < planner_limit:
                    for name in planner_degradations:
                        overrides = DEGRADATIONS[name]
                        obs_cfg = ObservationConfig(**{**overrides,
                                                        "seed": spec.seed})
                        planner = make_planner(
                            "space_time_astar",
                            goal_tolerance=env.goal_tolerance,
                            max_expansions=args.planner_expansions)
                        row = method_row("space_time_astar", spec, env,
                                         PlannerPolicy(planner, config),
                                         observation=obs_cfg)
                        row["degradation"] = name
                        row["seed_index"] = seed_index
                        all_rows.append(row)
        print(f"[seed {seed_index}] degradation sweep done", flush=True)

    write_rows(all_rows, destination / "episodes.json")
    write_rows(all_rows, destination / "episodes.csv")

    contrasts = {}
    for method in METHODS:
        perfect = [r for r in all_rows if r["method"] == method
                   and r["degradation"] == "perfect"]
        for name in DEGRADATIONS:
            if name == "perfect":
                continue
            other = [r for r in all_rows if r["method"] == method
                     and r["degradation"] == name]
            if not perfect or not other or len(perfect) != len(other):
                continue
            contrasts[f"{method}:perfect_vs_{name}"] = {
                "success": paired_rate_comparison(
                    [int(r["success"]) for r in perfect],
                    [int(r["success"]) for r in other]),
                "steps": paired_comparison([r["steps"] for r in perfect],
                                           [r["steps"] for r in other]),
            }
    summary = {
        "format": "oops-v12-reactive-1.0",
        "seeds": args.seeds,
        "levels": levels,
        "limit_per_level": limit,
        "degradations": DEGRADATIONS,
        "c5_scenarios_per_level": limit,
        "planner_scenarios_per_level": planner_limit,
        "planner_degradations": planner_degradations,
        "planner_max_expansions": args.planner_expansions,
        "overall": summarize(all_rows, ("method", "degradation")),
        "by_level": summarize(all_rows, ("method", "degradation", "level")),
        "contrasts": contrasts,
        "notes": [
            "All observation models are causal; delayed observations carry "
            "their own older timestamp.",
            "Noise streams depend only on (scenario seed, step, channel).",
        ],
    }
    save_json(summary, destination / "summary.json")
    print(f"wrote {destination}/summary.json ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
