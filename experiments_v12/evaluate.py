"""Episode evaluation for experiment series v1.2.

One runner for every method (synthesized programs and conventional planners).
The runner gives each method the *same* causal observation, steps the same
simulator with the same safety shield, and records the same metrics.

Metrics recorded per episode
----------------------------
1.  ``success``               reached the goal, no collision, within max_steps
2.  ``collision``             any true collision (shield or not)
3.  ``steps``                 control periods consumed
4.  ``time_to_goal_s``        ``steps * dt`` (0 when unsuccessful)
5.  ``path_length_m``         integrated agent displacement
6.  ``interventions``         safety-shield refusals
7.  ``min_clearance_m``       minimum agent-to-obstacle surface clearance
8.  ``extra_distance_ratio``  path length / straight-line start-to-goal length
9.  ``mean_speed_ratio``      mean commanded speed / max_speed
10. ``waits``                 zero-velocity commands
11. ``decision_latency_mean_s`` / ``decision_latency_p95_s``  per-decision time
12. ``deadline_misses``       decisions that exceeded the planning deadline

Program policies additionally report ``work_units``, ``fallback_waits``,
``explicit_waits``, ``continued_actions``, ``directional_choices`` and
``trace_events``; planner policies report ``replans``, ``expansions``,
``no_path_steps`` and ``blocked_stops``.

The safety shield is the environment's own: an unsafe commanded velocity is
refused and the agent holds position (counted as an intervention).  Every
method is subject to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Mapping

from chamber import Observation, Vec3, norm, sub
from experiments_v12.baseline_planners import BasePlanner
from experiments_v12.dynamic_chamber import DynamicChamber
from experiments_v12.dynamic_observations import CausalObserver, ObservationConfig
from experiments_v12.policies import ProgramPolicy
from experiments_v12.scenarios import ScenarioSpec
from oops_language import MotionConfig


class PlannerPolicy:
    """Adapter exposing the baseline planner interface as ``act(obs)``."""

    def __init__(self, planner: BasePlanner, config: MotionConfig):
        self.planner = planner
        self.config = config
        self.name = planner.name

    def reset(self) -> None:
        self.planner.reset()

    def act(self, obs: Observation) -> Vec3:
        return self.planner.act(obs, self.config)

    def extras(self) -> dict:
        return self.planner.stats.summary()


def policy_extras(policy) -> dict:
    counters = getattr(policy, "counters", None)
    if counters is not None:
        return counters.to_dict()
    extras = getattr(policy, "extras", None)
    return extras() if callable(extras) else {}


def surface_clearance(env: DynamicChamber, position: Vec3, t: float) -> float:
    """Distance from the agent's surface to the nearest obstacle surface."""
    best = math.inf
    for box in env.obstacles:
        if not box.active_at(t):
            continue
        center = box.center_at(t, env.chamber_size)
        gaps = [max(abs(position[i] - center[i]) - box.size[i] / 2, 0.0)
                for i in range(3)]
        best = min(best, norm(gaps) - env.radius)
    return best


@dataclass
class EpisodeSummary:
    method: str
    scenario_id: str
    level: int
    label: str
    seed: int
    success: bool
    collision: bool
    steps: int
    time_to_goal_s: float
    path_length_m: float
    interventions: int
    min_clearance_m: float
    extra_distance_ratio: float
    mean_speed_ratio: float
    waits: int
    decision_latency_mean_s: float
    decision_latency_p95_s: float
    deadline_misses: int
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        return payload


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[index]


def run_episode(env: DynamicChamber, policy, *,
                observer: CausalObserver | None = None) -> EpisodeSummary:
    """Execute one episode; metrics never feed back into the policy."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    config_dt = env.dt
    straight = norm(sub(env.goal, env.start)) or 1e-9
    path_length = 0.0
    waits = 0
    interventions = 0
    collisions = False
    latencies: list[float] = []
    min_clearance = math.inf

    obs = env.reset()
    if observer is not None:
        obs = observer.observe(env)
    # v1.2.1: the execution shield filters commands through the *same*
    # observation the policy receives (causal), never ground-truth futures.
    env.shield_observer = observer
    min_clearance = surface_clearance(env, env.position, env.time_s)

    while not env.done and env.step_index < env.max_steps:
        previous = env.position
        started = time.perf_counter()
        velocity = policy.act(obs)
        latencies.append(time.perf_counter() - started)
        if velocity[0] == 0.0 and velocity[1] == 0.0 and velocity[2] == 0.0:
            waits += 1
        result = env.step(velocity)
        interventions += int(result.safety_intervention)
        collisions |= result.collision
        # True simulator displacement (a shielded action moves nothing).
        path_length += norm(sub(env.position, previous))
        obs = result.observation
        if observer is not None:
            obs = observer.observe(env)
        min_clearance = min(min_clearance,
                            surface_clearance(env, env.position, env.time_s))
        if result.done:
            break

    reached = (env.done and not collisions
               and norm(sub(env.position, env.goal)) <= env.goal_tolerance + 1e-9)
    steps = env.step_index
    return EpisodeSummary(
        method="", scenario_id="", level=0, label="", seed=0,
        success=bool(reached), collision=bool(collisions), steps=steps,
        time_to_goal_s=(steps * config_dt) if reached else 0.0,
        path_length_m=path_length, interventions=interventions,
        min_clearance_m=(min_clearance if math.isfinite(min_clearance) else -1.0),
        extra_distance_ratio=path_length / straight,
        mean_speed_ratio=(path_length / (steps * config_dt * env.max_speed)
                          if steps else 0.0),
        waits=waits,
        decision_latency_mean_s=(sum(latencies) / len(latencies)
                                 if latencies else 0.0),
        decision_latency_p95_s=_percentile(latencies, 0.95),
        deadline_misses=int(policy_extras(policy).get("deadline_misses", 0)),
        extras=policy_extras(policy))


def evaluate_method(scenarios: list[ScenarioSpec], method: str,
                    policy_factory: Callable[[], Any], *,
                    observation: ObservationConfig | dict | None = None
                    ) -> list[dict]:
    """Run one method over a scenario list; returns plain dict rows."""
    rows = []
    for spec in scenarios:
        env = spec.build()
        rows.append(method_row(method, spec, env, policy_factory(),
                               observation=observation))
    return rows


def method_row(method: str, spec: ScenarioSpec, env: DynamicChamber, policy,
               observation: ObservationConfig | dict | None = None) -> dict:
    obs_cfg = observation if isinstance(observation, ObservationConfig) \
        else ObservationConfig.from_dict(observation)
    observer = CausalObserver(obs_cfg, episode_seed=spec.seed) \
        if not obs_cfg.perfect else None
    summary = run_episode(env, policy, observer=observer)
    summary.method = method
    summary.scenario_id = spec.id
    summary.level = spec.level
    summary.label = spec.label
    summary.seed = spec.seed
    return summary.to_dict()


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
    """Aggregate rates and means for a group of episode rows."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in group_keys), []).append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(members)
        successes = sum(int(m["success"]) for m in members)
        collisions = sum(int(m["collision"]) for m in members)
        steps = [m["steps"] for m in members]
        solved_steps = [m["steps"] for m in members if m["success"]]
        out.append({
            **{k: v for k, v in zip(group_keys, key)},
            "n": n,
            "success_rate": successes / n if n else 0.0,
            "collision_rate": collisions / n if n else 0.0,
            "mean_steps": sum(steps) / n if n else 0.0,
            "mean_steps_solved": (sum(solved_steps) / len(solved_steps)
                                  if solved_steps else 0.0),
            "mean_interventions": sum(m["interventions"] for m in members) / n if n else 0.0,
            "mean_path_length_m": sum(m["path_length_m"] for m in members) / n if n else 0.0,
            "mean_min_clearance_m": sum(m["min_clearance_m"] for m in members) / n if n else 0.0,
            "mean_decision_latency_s": sum(m["decision_latency_mean_s"] for m in members) / n if n else 0.0,
            "deadline_misses": sum(m.get("deadline_misses", 0) for m in members),
        })
    return out
