"""CLI demonstration and optional JSONL trace export."""
from __future__ import annotations
import argparse
import json
import pathlib
import time

from chamber import example_scene
from planners import OracleTimeExpandedAStar, next_velocity


def run(scenario="moving", seed=7, trace_path=None, verbose=True):
    env = example_scene(scenario, seed)
    planner = OracleTimeExpandedAStar()
    records = []
    interventions = 0
    no_plan = 0
    expansions = 0
    start_clock = time.perf_counter()
    if verbose:
        print(f"scenario={scenario} seed={seed} agent_radius={env.radius}m dt={env.dt}s")
        print(f"obstacles={len(env.obstacles)}; deterministic moving AABBs; oracle baseline (not OOPS)")
    for _ in range(env.max_steps):
        if env.done:
            break
        plan = planner.plan(env)
        expansions += plan.expansions
        if not plan.found:
            no_plan += 1
        action = next_velocity(env, plan)
        result = env.step(action)
        interventions += int(result.safety_intervention)
        records.append({
            "scenario": scenario, "seed": seed,
            "action_velocity": action, "planner_found": plan.found,
            "planner_reason": plan.reason,
            "planner_expansions": plan.expansions,
            **result.to_dict(),
        })
        if verbose:
            xyz = ','.join(f"{v:.2f}" for v in result.observation.position)
            print(f"t={result.observation.time_s:4.1f}s | pos=({xyz}) | "
                  f"obstacles={len(result.observation.obstacles)} | "
                  f"plan={'yes' if plan.found else 'no'} | "
                  f"shield={'STOP' if result.safety_intervention else 'ok'}")
        if result.done:
            break
    success = records[-1]["reached_goal"] if records else env.done
    collision = records[-1]["collision"] if records else False
    summary = {
        "scenario": scenario, "seed": seed,
        "success": bool(success), "collision": bool(collision),
        "steps": len(records), "simulation_time_s": env.time_s,
        "safety_interventions": interventions,
        "no_plan_steps": no_plan,
        "total_search_expansions": expansions,
        "wall_clock_s": round(time.perf_counter()-start_clock, 5),
    }
    if trace_path:
        path = pathlib.Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf8') as out:
            out.write(json.dumps({"type": "config", "chamber_size": env.chamber_size,
                                  "dt": env.dt, "radius": env.radius,
                                  "obstacles": [vars(o) for o in env.obstacles]}) + '\n')
            for rec in records:
                out.write(json.dumps({"type": "step", **rec})+'\n')
            out.write(json.dumps({"type": "summary", **summary})+'\n')
    if verbose:
        print("SUMMARY", json.dumps(summary, sort_keys=True))
    return summary, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic 3D chamber smoke-test")
    parser.add_argument('--scenario', choices=['empty','static','moving','crossing','random'], default='moving')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--trace', type=str, default=None, help='Optional output .jsonl file')
    args = parser.parse_args()
    run(args.scenario, args.seed, args.trace)
