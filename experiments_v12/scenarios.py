"""Seeded five-level scenario suite for experiment series v1.2.

Level 1  static geometry (including scenes with no path)
Level 2  predictable dynamic obstacles (constant velocity, reflections)
Level 3  unpredictable dynamic obstacles (velocity changes, appearances)
Level 4  imperfect observations on top of a level 1-3 scene
Level 5  distribution shift outside the training parameter ranges

Every scenario is fully described by a JSON-serialisable ``ScenarioSpec``;
``spec.build()`` reconstructs an identical ``DynamicChamber`` from the stored
description alone (no re-rolling), so a run can be replayed exactly.

Determinism: all sampling goes through ``random.Random(seed)``.  Splits use
disjoint seed ranges so training, validation and held-out test scenes never
overlap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path

from experiments_v12.dynamic_chamber import DynamicBox, DynamicChamber, VelocityEvent

Vec = tuple[float, float, float]

DEFAULT_CONFIG = {
    "chamber_size": [1.0, 1.0, 1.0],
    "radius": 0.035,
    "dt": 0.5,
    "max_speed": 0.70,
    "goal_tolerance": 0.055,
    "max_steps": 45,
    "train": {
        "obstacle_count": [1, 4],
        "size_range": [0.10, 0.26],
        "speed_range": [0.05, 0.20],
        "levels": [1, 2],
        "seeds_per_level": 12,
        "seed_base": 1000,
    },
    "validation": {"count_per_level": 40, "seed_base": 50000},
    "test": {"count_per_level": 200, "seed_base": 100000},
    "level1": {"static_count": [1, 5], "size_range": [0.10, 0.30],
               "infeasible_fraction": 0.15},
    "level2": {"count": [1, 4], "size_range": [0.10, 0.26],
               "speed_range": [0.06, 0.22], "gate_fraction": 0.4},
    "level3": {"count": [1, 4], "size_range": [0.10, 0.26],
               "speed_range": [0.06, 0.24], "events": [1, 4],
               "appear_fraction": 0.5},
    "level4": {"base_levels": [1, 2, 3],
               "position_noise": [0.0, 0.02],
               "velocity_noise": [0.0, 0.05],
               "delay_steps": [0, 2],
               "serve_every": [1, 3],
               "visibility_range": [0.35, 1.6]},
    "level5": {"count": [4, 8], "size_range": [0.24, 0.45],
               "speed_range": [0.15, 0.35], "events": [2, 5],
               "infeasible_fraction": 0.10},
}

GRID_POINTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    data = json.loads(Path(path).read_text(encoding="utf8"))
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update(data)
    return merged


@dataclass(frozen=True)
class ScenarioSpec:
    """Serialisable description of one benchmark scenario."""

    id: str
    level: int
    seed: int
    chamber_size: Vec
    start: Vec
    goal: Vec
    radius: float
    dt: float
    max_speed: float
    goal_tolerance: float
    max_steps: int
    boxes: list[dict] = field(default_factory=list)
    observation: dict = field(default_factory=dict)
    label: str = "feasible"
    note: str = ""

    def build(self) -> DynamicChamber:
        obstacles = []
        for entry in self.boxes:
            obstacles.append(DynamicBox(
                name=entry["name"],
                center=tuple(entry["center"]),
                size=tuple(entry["size"]),
                velocity=tuple(entry.get("velocity", (0.0, 0.0, 0.0))),
                events=[VelocityEvent(e["t"], tuple(e["velocity"]))
                        for e in entry.get("events", [])],
                appear_at=entry.get("appear_at", 0.0),
                disappear_at=entry.get("disappear_at")))
        return DynamicChamber(
            chamber_size=self.chamber_size, start=self.start, goal=self.goal,
            obstacles=obstacles, radius=self.radius, dt=self.dt,
            max_speed=self.max_speed, goal_tolerance=self.goal_tolerance,
            max_steps=self.max_steps)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioSpec":
        return cls(**data)


# --------------------------------------------------------------- helpers
def _box_dict(name: str, center: Vec, size: Vec, velocity: Vec = (0., 0., 0.),
              events: list | None = None, appear_at: float = 0.0,
              disappear_at: float | None = None) -> dict:
    return {"name": name, "center": [float(v) for v in center],
            "size": [float(v) for v in size],
            "velocity": [float(v) for v in velocity],
            "events": [{"t": float(e.t), "velocity": [float(v) for v in e.velocity]}
                       for e in (events or [])],
            "appear_at": float(appear_at),
            "disappear_at": None if disappear_at is None else float(disappear_at)}


def _free_point(rng: random.Random, chamber: Vec, radius: float, box_size: Vec,
                blocked: list) -> Vec:
    """Sample a legal box centre: inside the chamber for ``box_size`` and clear
    of every blocked box plus the agent start/goal keep-out."""
    for _ in range(500):
        point = tuple(rng.uniform(box_size[i] / 2 + 0.01,
                                  chamber[i] - box_size[i] / 2 - 0.01)
                      for i in range(3))
        if all(_clear_of_box(point, b, max(radius + 0.04, 0.03)) for b in blocked):
            return tuple(round(v, 6) for v in point)
    raise RuntimeError("failed to sample a free point")


def _clear_of_box(point: Vec, box: dict, margin: float) -> bool:
    return any(abs(point[i] - box["center"][i])
               > box["size"][i] / 2 + margin for i in range(3))


def _sample_endpoints(rng: random.Random, chamber: Vec, radius: float,
                      blocked: list[dict]) -> tuple[Vec, Vec]:
    candidates = [(x, y, z) for x in GRID_POINTS for y in GRID_POINTS
                  for z in GRID_POINTS]
    rng.shuffle(candidates)
    start = goal = None
    for point in candidates:
        if not all(radius <= point[i] <= chamber[i] - radius for i in range(3)):
            continue
        if not all(_clear_of_box(point, b, radius + 0.03) for b in blocked):
            continue
        if start is None:
            start = point
        elif _distance(start, point) > 0.6:
            goal = point
            break
    if start is None or goal is None:
        start = (0.1, 0.1, 0.1)
        goal = (0.9, 0.9, 0.9)
    return start, goal


def _distance(a: Vec, b: Vec) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _grid_reachable(chamber: Vec, radius: float, boxes: list[dict],
                    start: Vec, goal: Vec, resolution: float = 0.1) -> bool:
    """Independent coarse reachability reference (6-connected occupancy BFS).

    Used only to label deliberately infeasible static scenes; it is not part of
    any evaluated planner's data path.
    """
    import math
    from collections import deque

    def cells(point: Vec) -> tuple:
        return tuple(int(round(point[i] / resolution)) for i in range(3))

    def centre(idx: tuple) -> Vec:
        return tuple(i * resolution for i in idx)

    def blocked(idx: tuple) -> bool:
        p = centre(idx)
        if any(p[i] < radius - 1e-9 or p[i] > chamber[i] - radius + 1e-9
               for i in range(3)):
            return True
        for b in boxes:
            if all(abs(p[i] - b["center"][i]) <= b["size"][i] / 2 + radius + 1e-9
                   for i in range(3)):
                return True
        return False

    lo = tuple(int(math.floor(radius / resolution)) for _ in range(3))
    hi = tuple(int(math.ceil((chamber[i] - radius) / resolution)) for i in range(3))
    seen = {cells(start)}
    queue = deque([cells(start)])
    target = cells(goal)
    moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for delta in moves:
            nxt = tuple(node[i] + delta[i] for i in range(3))
            if any(nxt[i] < lo[i] or nxt[i] > hi[i] for i in range(3)):
                continue
            if nxt in seen or blocked(nxt):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


# ------------------------------------------------------------ generators
def _sample_boxes(rng: random.Random, count_range, size_range, chamber: Vec,
                  radius: float, start: Vec, goal: Vec,
                  velocity_range=None) -> list[dict]:
    boxes: list[dict] = []
    lo_n, hi_n = count_range
    for index in range(rng.randint(lo_n, hi_n)):
        size = tuple(rng.uniform(*size_range) for _ in range(3))
        for _ in range(300):
            center = tuple(rng.uniform(size[i] / 2 + 0.02,
                                       chamber[i] - size[i] / 2 - 0.02)
                           for i in range(3))
            candidate = _box_dict(f"b{index}", center, size)
            if not _clear_of_box(start, candidate, radius + 0.05):
                continue
            if not _clear_of_box(goal, candidate, radius + 0.05):
                continue
            if all(_far_enough(candidate, other, 0.03) for other in boxes):
                boxes.append(candidate)
                break
    return boxes


def _far_enough(a: dict, b: dict, margin: float) -> bool:
    return any(abs(a["center"][i] - b["center"][i])
               > (a["size"][i] + b["size"][i]) / 2 + margin for i in range(3))


def _level1(rng: random.Random, cfg: dict, seed: int, split_cfg: dict) -> tuple[list[dict], str]:
    spec_cfg = cfg["level1"]
    chamber = cfg["chamber_size"]
    radius = cfg["radius"]
    boxes = _sample_boxes(rng, spec_cfg["static_count"], spec_cfg["size_range"],
                          chamber, radius, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
    note = ""
    label = "feasible"
    if rng.random() < spec_cfg["infeasible_fraction"]:
        # Deliberately infeasible: the goal lies strictly inside a static box,
        # so no free position within goal_tolerance exists.
        boxes.append(_box_dict("blocker", (0.9, 0.9, 0.9), (0.16, 0.16, 0.16)))
        label = "infeasible_static"
        note = "goal enclosed by a static box (grid reference reports no path)"
    start, goal = (0.1, 0.1, 0.1), (0.9, 0.9, 0.9)
    if label == "feasible":
        start, goal = _sample_endpoints(rng, chamber, radius, boxes)
    return boxes, label, start, goal, note


def _level2(rng: random.Random, cfg: dict, seed: int) -> tuple[list[dict], str, Vec, Vec, str]:
    spec_cfg = cfg["level2"]
    chamber = cfg["chamber_size"]
    radius = cfg["radius"]
    boxes = _sample_boxes(rng, spec_cfg["count"], spec_cfg["size_range"],
                          chamber, radius, (0.1, 0.5, 0.5), (0.9, 0.5, 0.5),
                          velocity_range=spec_cfg["speed_range"])
    for entry in boxes:
        speed = rng.uniform(*spec_cfg["speed_range"])
        axis = rng.randrange(3)
        direction = rng.choice([-1.0, 1.0])
        velocity = [0.0, 0.0, 0.0]
        velocity[axis] = speed * direction
        entry["velocity"] = velocity
    note = "constant-velocity obstacles with exact wall reflections"
    if rng.random() < spec_cfg["gate_fraction"]:
        # A gate sliding across the direct corridor: waiting beats detouring.
        boxes.append(_box_dict(
            "gate", (0.5, 0.5, 0.5), (0.10, 0.34, 0.34),
            velocity=(0.0, 0.0, 0.16)))
        note += "; sliding gate across the start-goal corridor"
    return boxes, "feasible", (0.1, 0.5, 0.5), (0.9, 0.5, 0.5), note


def _level3(rng: random.Random, cfg: dict, seed: int) -> tuple[list[dict], str, Vec, Vec, str]:
    spec_cfg = cfg["level3"]
    chamber = cfg["chamber_size"]
    radius = cfg["radius"]
    start, goal = (0.1, 0.5, 0.5), (0.9, 0.5, 0.5)
    boxes = _sample_boxes(rng, spec_cfg["count"], spec_cfg["size_range"],
                          chamber, radius, start, goal)
    for entry in boxes:
        speed = rng.uniform(*spec_cfg["speed_range"])
        axis = rng.randrange(3)
        velocity = [0.0, 0.0, 0.0]
        velocity[axis] = speed * rng.choice([-1.0, 1.0])
        entry["velocity"] = velocity          # initial velocity at t=0
        current = list(velocity)
        events = []
        t = rng.uniform(0.8, 1.6)
        for _ in range(rng.randint(*spec_cfg["events"])):
            mode = rng.choice(("redirect", "stop", "speed"))
            if mode == "redirect":
                new_axis = rng.randrange(3)
                new_velocity = [0.0, 0.0, 0.0]
                new_velocity[new_axis] = rng.uniform(*spec_cfg["speed_range"]) \
                    * rng.choice([-1.0, 1.0])
            elif mode == "stop":
                new_velocity = [0.0, 0.0, 0.0]
            else:
                new_velocity = [v * rng.uniform(0.4, 2.2) for v in current]
            events.append(VelocityEvent(round(t, 3), tuple(new_velocity)))
            current = list(new_velocity)
            t += rng.uniform(0.7, 1.8)
        entry["events"] = [{"t": e.t, "velocity": list(e.velocity)} for e in events]
    if rng.random() < spec_cfg["appear_fraction"]:
        appear = round(rng.uniform(1.5, 3.0), 3)
        size = tuple(rng.uniform(*spec_cfg["size_range"]) for _ in range(3))
        keepout = boxes + [{"center": list(start), "size": [0.0, 0.0, 0.0]},
                           {"center": list(goal), "size": [0.0, 0.0, 0.0]}]
        center = _free_point(rng, chamber, radius, size, keepout)
        boxes.append(_box_dict("appearing", center, size,
                               velocity=[rng.uniform(-.15, .15) for _ in range(3)],
                               appear_at=appear))
    return boxes, "feasible", start, goal, \
        "unannounced velocity changes and a late-appearing obstacle"


def _level5(rng: random.Random, cfg: dict, seed: int) -> tuple[list[dict], str, Vec, Vec, str]:
    spec_cfg = cfg["level5"]
    chamber = cfg["chamber_size"]
    radius = cfg["radius"]
    start = tuple(rng.choice(GRID_POINTS) for _ in range(3))
    goal = start
    for _ in range(400):
        goal = tuple(rng.choice(GRID_POINTS) for _ in range(3))
        if _distance(start, goal) >= 0.6:
            break
    else:  # pragma: no cover - exhaustive grid guarantees a match
        goal = tuple(0.9 if v <= 0.5 else 0.1 for v in start)
    boxes = _sample_boxes(rng, spec_cfg["count"], spec_cfg["size_range"],
                          chamber, radius, start, goal)
    label = "feasible"
    note = "held-out parameter ranges (count/size/speed/motion patterns)"
    if rng.random() < spec_cfg["infeasible_fraction"] and boxes:
        # Goal sits strictly inside a static blocker: no free position within
        # goal_tolerance exists, so the scene is infeasible by construction.
        half = 0.15
        centre = tuple(min(max(goal[i], half + 0.01), chamber[i] - half - 0.01)
                       for i in range(3))
        boxes.append(_box_dict("blocker", centre, (2 * half, 2 * half, 2 * half)))
        label = "infeasible_static"
        note = "deliberately infeasible (goal inside a static blocker)"
    for entry in boxes:
        if label == "infeasible_static":
            continue
        speed = rng.uniform(*spec_cfg["speed_range"])
        axis = rng.randrange(3)
        velocity = [0.0, 0.0, 0.0]
        velocity[axis] = speed * rng.choice([-1.0, 1.0])
        entry["velocity"] = velocity          # initial velocity at t=0
        if rng.random() < 0.6:
            events = []
            t = rng.uniform(0.6, 1.4)
            for _ in range(rng.randint(*spec_cfg["events"])):
                new_velocity = [rng.uniform(-.35, .35) for _ in range(3)]
                events.append(VelocityEvent(round(t, 3), tuple(new_velocity)))
                t += rng.uniform(0.6, 1.5)
            entry["events"] = [{"t": e.t, "velocity": list(e.velocity)}
                               for e in events]
    if label == "feasible" and not _grid_reachable(chamber, radius, boxes,
                                                   start, goal):
        label = "infeasible_static"
        note = "grid reference reports no static path"
    return boxes, label, start, goal, note


def generate(level: int, seed: int, config: dict | None = None,
             split: str = "test") -> ScenarioSpec:
    """Deterministically build one scenario specification."""
    cfg = config or load_config()
    if level not in (1, 2, 3, 4, 5):
        raise ValueError("level must be 1..5")
    rng = random.Random(seed * 7919 + level * 104729)
    base_level = level
    observation: dict = {}
    if level == 4:
        level_cfg = cfg["level4"]
        base_level = rng.choice(level_cfg["base_levels"])
        observation = {
            "position_noise": round(rng.uniform(*level_cfg["position_noise"]), 6),
            "velocity_noise": round(rng.uniform(*level_cfg["velocity_noise"]), 6),
            "delay_steps": rng.randint(*level_cfg["delay_steps"]),
            "serve_every": rng.randint(*level_cfg["serve_every"]),
            "visibility_range": round(rng.uniform(
                *level_cfg["visibility_range"]), 6),
            "seed": seed,
        }

    if base_level == 1:
        boxes, label, start, goal, note = _level1(rng, cfg, seed, cfg["test"])
    elif base_level == 2:
        boxes, label, start, goal, note = _level2(rng, cfg, seed)
    elif base_level == 3:
        boxes, label, start, goal, note = _level3(rng, cfg, seed)
    else:
        boxes, label, start, goal, note = _level5(rng, cfg, seed)

    if level == 1 and label == "feasible":
        if not _grid_reachable(cfg["chamber_size"], cfg["radius"], boxes,
                               start, goal):
            label = "infeasible_static"
            note = "grid reference reports no static path"
    if level == 4 and base_level == 1:
        note += "; observations degraded"

    spec = ScenarioSpec(
        id=f"L{level}_s{seed}", level=level, seed=seed,
        chamber_size=tuple(cfg["chamber_size"]), start=start, goal=goal,
        radius=cfg["radius"], dt=cfg["dt"], max_speed=cfg["max_speed"],
        goal_tolerance=cfg["goal_tolerance"], max_steps=cfg["max_steps"],
        boxes=boxes, observation=observation, label=label, note=note)
    spec.build()  # fail fast on an invalid description
    return spec


def make_split(level: int, split: str, config: dict | None = None) -> list[ScenarioSpec]:
    """Deterministic, non-overlapping scenario split for one difficulty level."""
    cfg = config or load_config()
    section = cfg[split]
    if split == "train":
        seeds = [section["seed_base"] + level * 1000 + i
                 for i in range(section["seeds_per_level"])]
    else:
        seeds = [section["seed_base"] + level * 10000 + i
                 for i in range(section["count_per_level"])]
    return [generate(level, seed, cfg, split) for seed in seeds]


def training_curriculum(config: dict | None = None) -> list[tuple[str, ScenarioSpec]]:
    """Named (stage name, scenario) pairs used to train the synthesizer."""
    cfg = config or load_config()
    rows = []
    for level in cfg["train"]["levels"]:
        for spec in make_split(level, "train", cfg):
            if spec.label != "feasible":
                continue
            rows.append((f"L{level}_seed{spec.seed}", spec))
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect generated scenarios")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    for spec in make_split(args.level, args.split)[:args.count]:
        env = spec.build()
        print(f"{spec.id} level={spec.level} label={spec.label} "
              f"obstacles={len(env.obstacles)} start={spec.start} "
              f"goal={spec.goal} note={spec.note}")
