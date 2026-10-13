"""An optional ORACLE baseline to smoke-test the chamber, NOT OOPS.

Time-expanded lattice A*: the baseline is given future positions of the
simulator's deterministic obstacles. Real camera-based systems do not know the
future exactly. This baseline is deliberately stronger than a live-only planner.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import itertools
import math

from chamber import ChamberEnv, Vec3, norm, sub, mul


@dataclass
class PlanResult:
    path: list[Vec3]
    expansions: int
    reason: str

    @property
    def found(self) -> bool:
        return len(self.path) > 0


class OracleTimeExpandedAStar:
    def __init__(self, resolution: float = .2, margin: float = .1,
                 horizon: int = 14, max_expansions: int = 10000):
        self.resolution = resolution
        self.margin = margin
        self.horizon = horizon
        self.max_expansions = max_expansions
        if resolution <= 0 or horizon < 1 or max_expansions < 1:
            raise ValueError("Invalid planning parameters")

    def plan(self, env: ChamberEnv) -> PlanResult:
        axes = []
        for bound in env.chamber_size:
            values = []
            x = max(self.margin, env.radius)
            while x <= bound - max(self.margin, env.radius) + 1e-9:
                values.append(round(x, 9))
                x += self.resolution
            axes.append(values)
        if any(not v for v in axes):
            return PlanResult([], 0, "no_grid")
        def nearest(point: Vec3):
            return tuple(min(range(len(axes[i])), key=lambda j: abs(axes[i][j]-point[i])) for i in range(3))
        def xyz(idx):
            return (axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]])
        start = nearest(env.position)
        goal = nearest(env.goal)
        # Planner only supports a first and final off-grid position if reachable
        # in a single commanded step. The demos use aligned endpoints.
        if norm(sub(env.position, xyz(start))) > 1e-7:
            return PlanResult([], 0, "start_off_grid")
        if norm(sub(env.goal, xyz(goal))) > env.goal_tolerance:
            return PlanResult([], 0, "goal_off_grid")
        startnode = (start[0],start[1],start[2],0)
        distances = {startnode: 0}
        parent = {}
        open_heap = []
        tie = itertools.count()
        def heuristic(node):
            return max(abs(node[i]-goal[i]) for i in range(3))
        heapq.heappush(open_heap, (heuristic(startnode),next(tie),startnode))
        expansions = 0
        increments = list(itertools.product((-1,0,1), repeat=3))
        while open_heap and expansions < self.max_expansions:
            _, _, node = heapq.heappop(open_heap)
            current_time = env.time_s + node[3] * env.dt
            if node[:3] == goal:
                result = []
                while node != startnode:
                    result.append(xyz(node))
                    node = parent[node]
                result.append(env.position)
                result.reverse()
                return PlanResult(result, expansions, "goal_reached")
            if node[3] >= self.horizon:
                continue
            expansions += 1
            p0 = xyz(node)
            for delta in increments:
                idx = tuple(node[i] + delta[i] for i in range(3))
                if any(idx[i] < 0 or idx[i] >= len(axes[i]) for i in range(3)):
                    continue
                p1 = xyz(idx)
                if norm(sub(p1, p0)) > env.max_speed * env.dt + 1e-9:
                    continue
                if not env.is_motion_safe(p0, p1, current_time, current_time+env.dt):
                    continue
                neighbor = (idx[0],idx[1],idx[2],node[3]+1)
                tentative = distances[node] + 1
                if tentative < distances.get(neighbor, math.inf):
                    distances[neighbor] = tentative
                    parent[neighbor] = node
                    heapq.heappush(open_heap,(tentative+heuristic(neighbor),next(tie),neighbor))
        return PlanResult([], expansions, "expansion_limit" if expansions >= self.max_expansions else "horizon_or_no_path")


def next_velocity(env: ChamberEnv, plan: PlanResult) -> Vec3:
    if len(plan.path) < 2:
        return (0.0, 0.0, 0.0)
    return mul(sub(plan.path[1], env.position), 1 / env.dt)
