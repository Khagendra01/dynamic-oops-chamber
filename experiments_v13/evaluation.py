"""Episode evaluation for v1.3.

One runner for every method (synthesized programs, IK, motion planners).  The
runner gives each method the same causal observation, steps the same
simulator with the same safety shield, and records the same metrics.

Per-episode metrics
-------------------
1.  ``success``                 reached goal, no collision, within max_steps
2.  ``collision``               any true collision (post-execution detection)
3.  ``collision_category``      environment | self | boundary | ""
4.  ``steps``                   control periods consumed
5.  ``time_to_goal_s``          ``steps * dt`` (0 when unsuccessful)
6.  ``final_ee_error_m``        EE-to-goal distance at termination
7.  ``min_ee_error_m``          best EE error achieved in the episode
8.  ``joint_travel_rad``        integrated |executed dq| over all joints
9.  ``control_decisions``       policy decisions taken
10. ``waits``                   zero-command periods
11. ``interventions``           causal shield refusals
12. ``limit_clipped_steps``     commands clamped by joint limits/velocity
13. ``min_obstacle_gap_m``      conservative clearance lower bound over time
14. ``min_self_gap_m``          conservative self-clearance lower bound
15. ``min_boundary_margin_m``   worst wall margin over the trajectory
16. ``decision_latency_mean_s`` / ``decision_latency_p95_s``
17. ``deadline_misses``         reported by the policy
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable

import numpy as np

from experiments_v13.robot_chamber import RobotChamber
from experiments_v13.robot_observations import CausalObserver, ObservationConfig


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[index]


@dataclass
class EpisodeSummary:
    method: str = ""
    scenario_id: str = ""
    level: int = 0
    label: str = ""
    seed: int = 0
    success: bool = False
    collision: bool = False
    collision_category: str = ""
    steps: int = 0
    time_to_goal_s: float = 0.0
    final_ee_error_m: float = -1.0
    min_ee_error_m: float = -1.0
    joint_travel_rad: float = 0.0
    control_decisions: int = 0
    waits: int = 0
    interventions: int = 0
    limit_clipped_steps: int = 0
    min_obstacle_gap_m: float = -1.0
    min_self_gap_m: float = -1.0
    min_boundary_margin_m: float = -1.0
    decision_latency_mean_s: float = 0.0
    decision_latency_p95_s: float = 0.0
    deadline_misses: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def policy_extras(policy) -> dict:
    extras = getattr(policy, "extras", None)
    if callable(extras):
        return extras()
    counters = getattr(policy, "counters", None)
    if counters is not None and hasattr(counters, "to_dict"):
        return counters.to_dict()
    return {}


def run_episode(env: RobotChamber, policy, *,
                observer: CausalObserver | None = None) -> EpisodeSummary:
    """Execute one episode; metrics never feed back into the policy."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    dt = env.dt
    latencies: list[float] = []
    waits = 0
    interventions = 0
    limit_steps = 0
    joint_travel = 0.0
    min_ee_error = math.inf
    min_obstacle_gap = math.inf
    min_self_gap = math.inf
    min_boundary = math.inf

    obs = env.reset()
    if observer is not None:
        observer.reset()
        obs = observer.observe(env)
    env.shield_observer = observer

    # initial clearances
    min_obstacle_gap = min(min_obstacle_gap,
                           env.checker.min_obstacle_gap(
                               env.q, env._gt_field.obstacle_at(env.time_s)[0])
                           if env.obstacles else math.inf)
    min_self_gap = min(min_self_gap, env.checker.min_self_gap(env.q))
    min_boundary = min(min_boundary,
                       env.checker.min_boundary_margin(env.q))
    min_ee_error = min(min_ee_error, env.ee_error_at())

    while not env.done and env.step_index < env.max_steps:
        started = time.perf_counter()
        dq = np.asarray(policy.act(obs), dtype=float)
        latencies.append(time.perf_counter() - started)
        if float(np.linalg.norm(dq)) < 1e-12:
            waits += 1
        result = env.step(dq)
        interventions += int(result.intervention)
        limit_steps += int(result.limit_clipped)
        joint_travel += float(np.sum(np.abs(result.executed_dq)))
        min_ee_error = min(min_ee_error, env.ee_error_at())
        boxes = env._gt_field.obstacle_at(env.time_s)[0]
        if boxes:
            min_obstacle_gap = min(min_obstacle_gap,
                                   env.checker.min_obstacle_gap(env.q, boxes))
        min_self_gap = min(min_self_gap, env.checker.min_self_gap(env.q))
        min_boundary = min(min_boundary,
                           env.checker.min_boundary_margin(env.q))
        obs = result.observation
        if observer is not None:
            obs = observer.observe(env)
        if result.done:
            break

    reached = (env.done and not env.collision_flag
               and env.ee_error_at() <= env.goal_tolerance + 1e-9)
    steps = env.step_index
    extras = policy_extras(policy)
    return EpisodeSummary(
        success=bool(reached), collision=bool(env.collision_flag),
        collision_category=env.collision_category if env.collision_flag else "",
        steps=steps,
        time_to_goal_s=(steps * dt) if reached else 0.0,
        final_ee_error_m=env.ee_error_at(),
        min_ee_error_m=(min_ee_error if math.isfinite(min_ee_error) else -1.0),
        joint_travel_rad=joint_travel,
        control_decisions=len(latencies),
        waits=waits, interventions=interventions,
        limit_clipped_steps=limit_steps,
        min_obstacle_gap_m=(min_obstacle_gap if math.isfinite(min_obstacle_gap) else -1.0),
        min_self_gap_m=(min_self_gap if math.isfinite(min_self_gap) else -1.0),
        min_boundary_margin_m=(min_boundary if math.isfinite(min_boundary) else -1.0),
        decision_latency_mean_s=(sum(latencies) / len(latencies) if latencies else 0.0),
        decision_latency_p95_s=_percentile(latencies, 0.95),
        deadline_misses=int(extras.get("deadline_misses", 0)),
        extras=extras)


def method_row(method: str, spec, env: RobotChamber, policy,
               observation: ObservationConfig | dict | None = None) -> dict:
    obs_cfg = observation if isinstance(observation, ObservationConfig) \
        else ObservationConfig.from_dict(observation)
    observer = CausalObserver(obs_cfg, env.spec) if not obs_cfg.perfect else None
    summary = run_episode(env, policy, observer=observer)
    summary.method = method
    summary.scenario_id = spec.id
    summary.level = spec.level
    summary.label = spec.label
    summary.seed = spec.seed
    return summary.to_dict()


def evaluate_method(scenarios, method: str, policy_factory: Callable[[], Any],
                    *, observation: ObservationConfig | dict | None = None
                    ) -> list[dict]:
    rows = []
    for spec in scenarios:
        env = spec.build()
        rows.append(method_row(method, spec, env, policy_factory(),
                               observation=observation))
    return rows


def write_rows(rows: list[dict], path) -> None:
    import csv
    import json
    from pathlib import Path
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps(rows, indent=2) + "\n")
        return
    if not rows:
        out.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                    for k, v in row.items()}
            writer.writerow(flat)


def summarize(rows: list[dict], group_keys: tuple[str, ...] = ("method",)) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in group_keys), []).append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(members)
        successes = sum(int(m["success"]) for m in members)
        collisions = sum(int(m["collision"]) for m in members)
        env_hits = sum(1 for m in members if m.get("collision_category") == "environment")
        self_hits = sum(1 for m in members if m.get("collision_category") == "self")
        bnd_hits = sum(1 for m in members if m.get("collision_category") == "boundary")
        solved_steps = [m["steps"] for m in members if m["success"]]
        out.append({
            **{k: v for k, v in zip(group_keys, key)},
            "n": n,
            "success_rate": successes / n if n else 0.0,
            "collision_rate": collisions / n if n else 0.0,
            "collision_env": env_hits / n if n else 0.0,
            "collision_self": self_hits / n if n else 0.0,
            "collision_boundary": bnd_hits / n if n else 0.0,
            "mean_steps": sum(m["steps"] for m in members) / n if n else 0.0,
            "mean_steps_solved": (sum(solved_steps) / len(solved_steps)
                                  if solved_steps else 0.0),
            "mean_final_ee_error_m": sum(m["final_ee_error_m"] for m in members) / n if n else 0.0,
            "mean_min_ee_error_m": sum(m["min_ee_error_m"] for m in members) / n if n else 0.0,
            "mean_joint_travel_rad": sum(m["joint_travel_rad"] for m in members) / n if n else 0.0,
            "mean_interventions": sum(m["interventions"] for m in members) / n if n else 0.0,
            "mean_waits": sum(m["waits"] for m in members) / n if n else 0.0,
            "mean_decision_latency_s": sum(m["decision_latency_mean_s"] for m in members) / n if n else 0.0,
            "deadline_misses": sum(m.get("deadline_misses", 0) for m in members),
        })
    return out
