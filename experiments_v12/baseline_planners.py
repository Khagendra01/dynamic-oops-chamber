"""Conventional planner baselines for experiment series v1.2.

Three independent, deterministic baselines that receive exactly the same
``Observation`` as the synthesized programs and use the shared prediction /
swept-collision rule from ``experiments_v12.motion_prediction``:

``StaticGridAStar``
    Receding-horizon grid A* over the current obstacle snapshot, re-planned
    every ``replan_every`` control periods.  Deliberately simple: a *static*
    planner with a predictive safety shield, not a dynamic obstacle predictor.

``SpaceTimeAStar``
    Time-expanded A* with explicit wait actions over obstacle states
    extrapolated by the shared constant-velocity predictor.

``RolloutGreedy``
    Model-predictive style short rollout: scores a fixed candidate set by
    simulating a few control periods with the shared predictor.

Command interface fairness
--------------------------
Synthesized programs can issue at most ``MotionConfig.stride`` metres per
control period (the language's command vocabulary).  By default every baseline
is capped with the *same* ``command_stride`` so differences reflect decisions
rather than step length (``stride_cap="language"``).  Setting
``stride_cap="full"`` lets a planner use the environment's full
``max_speed * dt`` envelope; both variants are reported, and the comparison
table always states which one it uses.

Safety contract
---------------
* every executed velocity is validated with the shared swept check; if it is
  not provably safe the planner emits a stop;
* if planning exceeds ``deadline_seconds`` the planner stops instead of
  executing an unchecked action;
* if no path exists the planner reports ``no_path`` and stops.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

from chamber import Observation, Vec3, norm, sub, vec
from experiments_v12.motion_prediction import Predictor, segment_safe

_AXIS_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_ALL_STEPS = [d for d in itertools.product((-1, 0, 1), repeat=3)
              if d != (0, 0, 0)]

STRIDE_CAPS = ("language", "full")


@dataclass
class PlannerStats:
    replans: int = 0
    expansions: int = 0
    deadline_misses: int = 0
    no_path_steps: int = 0
    blocked_stops: int = 0
    status: str = "ok"
    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        return {"replans": self.replans, "expansions": self.expansions,
                "deadline_misses": self.deadline_misses,
                "no_path_steps": self.no_path_steps,
                "blocked_stops": self.blocked_stops, "status": self.status,
                "latency_mean_s": (sum(lat) / len(lat)) if lat else 0.0,
                "latency_p95_s": lat[max(0, int(math.ceil(.95 * len(lat))) - 1)]
                if lat else 0.0}


class BasePlanner:
    name = "base"
    moves = _ALL_STEPS

    def __init__(self, goal_tolerance: float = 0.055,
                 deadline_seconds: float = 0.5, replan_every: int = 1,
                 resolution: float = 0.1, max_expansions: int = 4000,
                 horizon: int = 24, stride_cap: str = "language"):
        if stride_cap not in STRIDE_CAPS:
            raise ValueError(f"stride_cap must be one of {STRIDE_CAPS}")
        self.goal_tolerance = float(goal_tolerance)
        self.deadline_seconds = float(deadline_seconds)
        self.replan_every = int(replan_every)
        self.resolution = float(resolution)
        self.max_expansions = int(max_expansions)
        self.horizon = int(horizon)
        self.stride_cap = stride_cap
        self.stats = PlannerStats()
        self.predictor: Predictor | None = None
        self._edge_cache: dict | None = None

    def reset(self) -> None:
        self.stats = PlannerStats()
        self._edge_cache = None

    # --------------------------------------------- edge-check memoization
    def _edge_memo(self) -> dict:
        """Per-episode memo for ``segment_safe`` edge checks.

        Correctness comes from the key: every entry is keyed by a signature of
        the *exact* obstacle state the check was run against (names, centers,
        sizes, velocities), so any noise, motion, or appearance change alters
        the signature and forces a fresh check.  Perfect static scenes reuse
        entries across search depths and replanning steps.
        """
        if self._edge_cache is None:
            self._edge_cache = {}
        return self._edge_cache

    @staticmethod
    def _scene_signature(obs: Observation) -> tuple:
        return tuple(sorted(
            (o.name,
             round(o.center[0], 9), round(o.center[1], 9), round(o.center[2], 9),
             round(o.size[0], 9), round(o.size[1], 9), round(o.size[2], 9),
             round(o.velocity[0], 9), round(o.velocity[1], 9),
             round(o.velocity[2], 9))
            for o in obs.obstacles))

    def _segment_safe(self, cache: dict, obs: Observation, signature: tuple,
                      a_idx: tuple, b_idx: tuple, base: Vec3, target: Vec3,
                      config) -> bool:
        key = (signature, a_idx, b_idx)
        if key not in cache:
            cache[key] = segment_safe(obs, base, target, config)
        return cache[key]

    # ------------------------------------------------------------- utilities
    def _stride(self, config) -> float:
        if self.stride_cap == "language":
            return min(config.stride, config.max_speed * config.dt)
        return config.max_speed * config.dt

    def _axes(self, chamber_size: Vec3, radius: float) -> list[list[float]]:
        axes = []
        for bound in chamber_size:
            values = []
            value = math.ceil(radius / self.resolution) * self.resolution
            while value <= bound - radius + 1e-9:
                values.append(round(value, 9))
                value += self.resolution
            axes.append(values or [round(bound / 2, 9)])
        return axes

    @staticmethod
    def _point(index: tuple, axes: list[list[float]]) -> Vec3:
        return tuple(axes[i][index[i]] for i in range(3))

    @staticmethod
    def _snap(point: Vec3, axes: list[list[float]]) -> tuple:
        return tuple(min(range(len(axes[i])),
                         key=lambda j: abs(axes[i][j] - point[i]))
                     for i in range(3))

    def _goal_velocity(self, obs: Observation, config) -> Vec3 | None:
        """Velocity that finishes the episode from here, if one exists."""
        delta = sub(obs.goal, obs.position)
        if norm(delta) <= self._stride(config) + 1e-9:
            if segment_safe(obs, obs.position, obs.goal, config):
                return vec(tuple(delta[i] / config.dt for i in range(3)))
        return None

    @staticmethod
    def _velocity_to(obs: Observation, target: Vec3, config,
                     stride: float) -> Vec3:
        delta = sub(target, obs.position)
        distance = norm(delta)
        if distance <= 1e-12:
            return (0.0, 0.0, 0.0)
        travelled = min(distance, stride)
        return vec(tuple(delta[i] * travelled / (distance * config.dt)
                         for i in range(3)))

    def act(self, obs: Observation, config) -> Vec3:
        """Return one velocity command; always provably safe or zero."""
        started = time.perf_counter()
        velocity = self._decide(obs, config)
        elapsed = time.perf_counter() - started
        self.stats.latencies.append(elapsed)
        if elapsed > self.deadline_seconds:
            self.stats.deadline_misses += 1
            return (0.0, 0.0, 0.0)
        return velocity

    def _decide(self, obs: Observation, config) -> Vec3:  # pragma: no cover
        raise NotImplementedError

    # ---------------------------------------------------------- shared search
    def _prepare(self, obs: Observation, config):
        axes = self._axes(obs.chamber_size, config.radius)
        start = self._snap(obs.position, axes)
        goal = self._snap(obs.goal, axes)
        stride = self._stride(config)
        if norm(sub(self._point(start, axes), obs.position)) > stride + 1e-9:
            return None, None, axes, stride, "start_off_grid"
        return start, goal, axes, stride, None

    def _reconstruct(self, parent, start_node, node, axes) -> list[Vec3]:
        chain: list[Vec3] = []
        while node != start_node:
            chain.append(self._point(node[:3], axes))
            node = parent[node]
        chain.reverse()
        return chain

    @staticmethod
    def _mark_deadline_status(stats: PlannerStats, expansions: int,
                              limit: int) -> None:
        stats.status = "expansion_limit" if expansions >= limit else "no_path"


class StaticGridAStar(BasePlanner):
    """Receding-horizon grid A* on the current snapshot + predictive shield."""

    name = "static_astar"

    def _decide(self, obs: Observation, config) -> Vec3:
        direct = self._goal_velocity(obs, config)
        if direct is not None:
            self.stats.status = "goal_direct"
            return direct
        stride = self._stride(config)
        path = self._search(obs, config)
        if path is None:
            self.stats.no_path_steps += 1
            return (0.0, 0.0, 0.0)
        velocity = self._velocity_to(obs, path[0], config, stride)
        if not segment_safe(obs, obs.position, path[0], config):
            self.stats.blocked_stops += 1
            return (0.0, 0.0, 0.0)
        return velocity

    def _search(self, obs: Observation, config) -> list[Vec3] | None:
        self.stats.replans += 1
        start, goal, axes, stride, failure = self._prepare(obs, config)
        if failure:
            self.stats.status = failure
            return None
        goal_point = self._point(goal, axes)
        cache = self._edge_memo()
        scene_sig = self._scene_signature(obs)

        def heuristic(node: tuple) -> float:
            return norm(sub(self._point(node, axes), goal_point)) / stride

        tie = itertools.count()
        open_heap = [(heuristic(start), next(tie), start)]
        best = {start: 0.0}
        parent: dict = {}
        expansions = 0
        found = None
        while open_heap and expansions < self.max_expansions:
            _, _, node = heapq.heappop(open_heap)
            expansions += 1
            if node == goal:
                found = node
                break
            base = self._point(node, axes)
            for delta in self.moves:
                idx = tuple(node[i] + delta[i] for i in range(3))
                if any(idx[i] < 0 or idx[i] >= len(axes[i]) for i in range(3)):
                    continue
                target = self._point(idx, axes)
                if norm(sub(target, base)) > stride + 1e-9:
                    continue
                if not self._segment_safe(cache, obs, scene_sig, node, idx,
                                          base, target, config):
                    continue
                tentative = best[node] + norm(sub(target, base)) / stride
                if tentative + 1e-12 < best.get(idx, math.inf):
                    best[idx] = tentative
                    parent[idx] = node
                    heapq.heappush(open_heap,
                                   (tentative + heuristic(idx), next(tie), idx))
            # an off-grid goal may be reachable directly from this cell
            if (norm(sub(obs.goal, base)) <= stride + 1e-9
                    and self._segment_safe(cache, obs, scene_sig, node, "goal",
                                           base, obs.goal, config)):
                found = node
                break
        self.stats.expansions += expansions
        if found is None:
            self._mark_deadline_status(self.stats, expansions, self.max_expansions)
            return None
        chain = self._reconstruct(parent, start, found, axes)
        if not chain:
            return None
        self.stats.status = "ok"
        return chain


class SpaceTimeAStar(BasePlanner):
    """Time-expanded A* with wait actions over predicted obstacle motion."""

    name = "space_time_astar"

    def _decide(self, obs: Observation, config) -> Vec3:
        direct = self._goal_velocity(obs, config)
        if direct is not None:
            self.stats.status = "goal_direct"
            return direct
        stride = self._stride(config)
        path = self._search(obs, config)
        if path is None:
            self.stats.no_path_steps += 1
            return (0.0, 0.0, 0.0)
        velocity = self._velocity_to(obs, path[0], config, stride)
        if not segment_safe(obs, obs.position, path[0], config):
            self.stats.blocked_stops += 1
            return (0.0, 0.0, 0.0)
        return velocity

    def _search(self, obs: Observation, config) -> list[Vec3] | None:
        self.stats.replans += 1
        start, goal, axes, stride, failure = self._prepare(obs, config)
        if failure:
            self.stats.status = failure
            return None
        goal_point = self._point(goal, axes)
        if self.predictor is None:
            self.predictor = Predictor(config.dt)
        shifted_cache: dict[int, Observation] = {0: obs}
        edge_cache = self._edge_memo()
        signature_cache: dict[int, tuple] = {
            0: self._scene_signature(obs)}

        def at(depth: int) -> Observation:
            cached = shifted_cache.get(depth)
            if cached is None:
                cached = self.predictor.at_time(obs, obs.time_s + depth * config.dt)
                shifted_cache[depth] = cached
                signature_cache[depth] = self._scene_signature(cached)
            return cached

        def heuristic(node: tuple) -> float:
            return norm(sub(self._point(node[:3], axes), goal_point)) / stride

        start_node = (start[0], start[1], start[2], 0)
        tie = itertools.count()
        open_heap = [(heuristic(start_node), next(tie), start_node)]
        best = {start_node: 0.0}
        parent: dict = {}
        expansions = 0
        found = None
        while open_heap and expansions < self.max_expansions:
            _, _, node = heapq.heappop(open_heap)
            expansions += 1
            cell, depth = node[:3], node[3]
            if cell == goal:
                found = node
                break
            if depth >= self.horizon:
                continue
            observation = at(depth)
            depth_sig = signature_cache[depth]
            base = self._point(cell, axes)
            for delta in list(self.moves) + [(0, 0, 0)]:
                idx = tuple(cell[i] + delta[i] for i in range(3))
                if any(idx[i] < 0 or idx[i] >= len(axes[i]) for i in range(3)):
                    continue
                target = self._point(idx, axes)
                if norm(sub(target, base)) > stride + 1e-9:
                    continue
                if not self._segment_safe(edge_cache, observation, depth_sig,
                                          cell, idx, base, target, config):
                    continue
                nxt = (idx[0], idx[1], idx[2], depth + 1)
                tentative = best[node] + 1.0
                if tentative + 1e-12 < best.get(nxt, math.inf):
                    best[nxt] = tentative
                    parent[nxt] = node
                    heapq.heappush(open_heap,
                                   (tentative + heuristic(nxt), next(tie), nxt))
            if (norm(sub(obs.goal, base)) <= stride + 1e-9
                    and self._segment_safe(edge_cache, observation, depth_sig,
                                           cell, "goal", base, obs.goal,
                                           config)):
                found = node
                break
        self.stats.expansions += expansions
        if found is None:
            self._mark_deadline_status(self.stats, expansions, self.max_expansions)
            return None
        chain = self._reconstruct(parent, start_node, found, axes)
        if not chain:
            return None
        self.stats.status = "ok"
        return chain


class RolloutGreedy(BasePlanner):
    """Short model-predictive rollout over a fixed candidate set."""

    name = "rollout_greedy"
    moves = _AXIS_STEPS

    def __init__(self, *args, rollout_steps: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_steps = int(rollout_steps)

    def _candidates(self, obs: Observation, config, stride: float) -> list[Vec3]:
        delta = sub(obs.goal, obs.position)
        distance = norm(delta)
        out = []
        if distance > 1e-9:
            out.append(vec(tuple(delta[i] / distance * stride for i in range(3))))
        for direction in self.moves:
            out.append(vec(tuple(direction[i] * stride for i in range(3))))
        out.append((0.0, 0.0, 0.0))
        return out

    def _decide(self, obs: Observation, config) -> Vec3:
        self.stats.replans += 1
        if self.predictor is None:
            self.predictor = Predictor(config.dt)
        stride = self._stride(config)
        best_velocity = (0.0, 0.0, 0.0)
        best_score = None
        for velocity in self._candidates(obs, config, stride):
            score = self._score(obs, velocity, config, stride)
            if best_score is None or score < best_score:
                best_score = score
                best_velocity = velocity
        if not segment_safe(obs, obs.position, self._advance(obs, best_velocity, config),
                            config):
            self.stats.blocked_stops += 1
            return (0.0, 0.0, 0.0)
        return best_velocity

    @staticmethod
    def _advance(obs: Observation, velocity: Vec3, config) -> Vec3:
        return vec(tuple(obs.position[i] + velocity[i] * config.dt
                         for i in range(3)))

    def _score(self, obs: Observation, first: Vec3, config, stride: float) -> tuple:
        position = obs.position
        candidate = first
        risk = 0
        for step in range(self.rollout_steps):
            target = self._advance(_with_position(obs, position), candidate, config)
            shifted = self.predictor.at_time(obs, obs.time_s + step * config.dt)
            if not segment_safe(shifted, position, target, config):
                risk += 1000
                candidate = (0.0, 0.0, 0.0)
                continue
            position = target
            delta = sub(obs.goal, position)
            distance = norm(delta)
            if distance <= stride + 1e-9:
                return (risk, step, distance)
            if segment_safe(self.predictor.at_time(obs, obs.time_s + (step + 1) * config.dt),
                            position,
                            vec(tuple(position[i] + delta[i] / distance * stride
                                      for i in range(3))), config):
                candidate = vec(tuple(delta[i] / distance * stride for i in range(3)))
            elif risk > 0:
                candidate = (0.0, 0.0, 0.0)
        return (risk, self.rollout_steps, norm(sub(obs.goal, position)))


def _with_position(obs: Observation, position: Vec3) -> Observation:
    from chamber import Observation as _Obs
    return _Obs(time_s=obs.time_s, step_index=obs.step_index, position=position,
                goal=obs.goal, obstacles=obs.obstacles,
                chamber_size=obs.chamber_size)


def make_planner(kind: str, **kwargs) -> BasePlanner:
    """Factory used by the runners; unknown kinds fail loudly."""
    registry = {
        "static_astar": StaticGridAStar,
        "space_time_astar": SpaceTimeAStar,
        "rollout_greedy": RolloutGreedy,
    }
    if kind not in registry:
        raise ValueError(f"unknown planner {kind!r}; choose from {sorted(registry)}")
    return registry[kind](**kwargs)


PLANNERS = ("static_astar", "space_time_astar", "rollout_greedy")
