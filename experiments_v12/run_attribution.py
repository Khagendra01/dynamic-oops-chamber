"""Experiment F (v1.2.1): program attribution — synthesis vs hand composition.

The C5 program is a short rule sequence over hand-designed primitives.  This
experiment asks what the *synthesis* actually contributed at deployment time by
comparing the discovered program against manually composed programs built from
the same primitives:

* ``manual_GOAL``                 — goal-directed motion only
* ``manual_GOAL_CHOOSE_AXIS``     — goal-directed + axis-choice fallback
  (the hand-composed composition the README documents as P004-equivalent)
* ``manual_full_primitives``      — every hand-designed primitive in the
  documented order (CONTINUE, WAIT_GAP, GOAL, CHOOSE_AXIS)

All methods are evaluated on the same held-out test splits used by the
generalization experiment (levels 1-5, 200 scenes each), with the same causal
execution shield and observation contract.  The discovered C5 programs (all
training seeds, loaded from the generalization cache — no retraining) are
evaluated on the same scenes, so every contrast is paired per scene.

Reading the result: if ``C5`` and ``manual_GOAL_CHOOSE_AXIS`` are
statistically indistinguishable, the deployment-time advantage is fully
explained by the hand-designed primitives, and synthesis's contribution is
*finding* the effective composition cheaply during training (hypothesis H2),
not by producing a superior policy.  If ``manual_full_primitives`` beats C5,
the discovered two-rule program is not even the best hand policy — an honest
limitation to report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments_v12._common import (RESULTS_ROOT, add_common_args, out_dir,
                                     parse_levels, save_json, test_split)
from experiments_v12.ablation_matrix import C5
from experiments_v12.evaluate import method_row, summarize, write_rows
from experiments_v12.policies import make_policy
from experiments_v12.run_generalization import (evaluate_seed, load_or_train,
                                                policies_from_payload)
from experiments_v12.statistics import (clustered_paired_rate,
                                        per_seed_contrasts)
from oops_language import MotionConfig, Program

MANUAL_PROGRAMS: dict[str, tuple[str, ...]] = {
    "manual_GOAL": ("GOAL",),
    "manual_GOAL_CHOOSE_AXIS": ("GOAL", "CHOOSE_AXIS"),
    "manual_full_primitives": ("CONTINUE", "WAIT_GAP", "GOAL", "CHOOSE_AXIS"),
}


def manual_rows(scenarios, seed_indices: tuple[int, ...]) -> list[dict]:
    """Evaluate each manual program once per scene, replicated per seed tag.

    Replication (``seed_index``) makes rows pair against every C5 training
    seed with the standard (seed_index, scenario_id, level) key; the
    evaluation itself runs once because the programs are seed-independent.
    """
    rows: list[dict] = []
    for name, rules in MANUAL_PROGRAMS.items():
        program = Program(rules, name)
        for spec in scenarios:
            env = spec.build()
            config = MotionConfig(radius=env.radius, dt=env.dt,
                                  max_speed=env.max_speed)
            # Memory ON matches the C5 deployment variant; an empty library
            # makes "reuse" a no-op for hand-composed programs.
            row = method_row(name, spec, env,
                             make_policy(C5, program, {}, config))
            for seed_index in seed_indices:
                tagged = dict(row)
                tagged["seed_index"] = seed_index
                tagged["program_rules"] = list(rules)
                rows.append(tagged)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per level (debugging)")
    parser.add_argument("--synthesis-dir", type=str,
                        default=str(RESULTS_ROOT / "generalization"),
                        help="directory holding the cached C5 synthesis payloads")
    args = parser.parse_args(argv)
    levels = parse_levels(args.levels)
    limit = args.limit if args.limit else (3 if args.smoke else None)
    destination = out_dir(args.out or "attribution")
    synthesis_dir = Path(args.synthesis_dir)

    scenarios = []
    for level in levels:
        specs = test_split(level)
        scenarios.extend(specs[:limit] if limit else specs)

    seed_indices = tuple(range(args.seeds))
    programs = []
    c5_rows: list[dict] = []
    for seed_index in seed_indices:
        payload = load_or_train(seed_index, synthesis_dir)
        programs.append({
            "seed_index": seed_index,
            "program_id": payload["deployed_program_id"],
            "rules": payload["deployed_rules"],
        })
        c5_rows.extend(evaluate_seed(payload, scenarios, seed_index, "test"))
        print(f"[seed {seed_index}] C5 program {payload['deployed_program_id']} "
              f"{payload['deployed_rules']} evaluated on {len(scenarios)} scenes",
              flush=True)

    hand_rows = manual_rows(scenarios, seed_indices)
    for name in MANUAL_PROGRAMS:
        n = sum(1 for r in hand_rows if r["method"] == name)
        print(f"[manual] {name} evaluated ({n} tagged rows)", flush=True)

    all_rows = c5_rows + hand_rows
    write_rows(all_rows, destination / "episodes.json")
    write_rows(all_rows, destination / "episodes.csv")

    def subset(method: str) -> list[dict]:
        return [r for r in all_rows if r["method"] == method]

    contrasts: dict[str, dict] = {}
    for name in MANUAL_PROGRAMS:
        clustered = clustered_paired_rate(c5_rows, subset(name))
        per_seed = per_seed_contrasts(c5_rows, subset(name))
        contrasts[f"C5_vs_{name}"] = {
            "success_clustered": clustered,
            "per_seed": per_seed,
            "note": "positive favours C5 (discovered program)",
        }
        print(f"[contrast] C5 vs {name}: "
              f"{clustered['mean']:+.3f} "
              f"[{clustered['ci_low']:+.3f}, {clustered['ci_high']:+.3f}] "
              f"({clustered['n_scenes']} scenes)", flush=True)

    summary = {
        "format": "oops-v12-attribution-1.0",
        "question": ("Does deployment performance come from synthesis or from "
                     "the hand-designed primitives it composes?"),
        "evaluated_levels": levels,
        "limit_per_level": limit,
        "seeds": args.seeds,
        "manual_programs": {name: list(rules)
                            for name, rules in MANUAL_PROGRAMS.items()},
        "c5_programs": programs,
        "overall": summarize(all_rows, ("method",)),
        "by_level": {
            method: summarize([r for r in subset(method)], ("level",))
            for method in ["C5", *MANUAL_PROGRAMS]
        },
        "contrasts": contrasts,
        "n_rows": len(all_rows),
    }
    save_json(summary, destination / "summary.json")
    print(f"wrote {destination}/summary.json ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
