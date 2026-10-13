"""Matplotlib 3-D visualization for the v1.3 articulated arm.

Renders the actual simulated state: oriented link cuboids from FK, chamber
walls, static/moving obstacles, joint pivots, end-effector, goal, and the
executed EE trace.  Deterministic episode recordings are saved as GIF
(PillowWriter) or single-frame PNG snapshots.  Visualization is a debugging
and demonstration aid, never the correctness oracle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from experiments_v13.forward_kinematics import forward_pose
from experiments_v13.robot_model import RobotSpec, link_corners_world

_FACES = (
    (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
    (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
)


def _cuboid_faces(corners: np.ndarray) -> list[np.ndarray]:
    return [corners[list(face)] for face in _FACES]


def _box_corners(center, half) -> np.ndarray:
    c = np.asarray(center, dtype=float)
    h = np.asarray(half, dtype=float)
    signs = np.array([[sx, sy, sz] for sx in (-h[0], h[0])
                      for sy in (-h[1], h[1]) for sz in (-h[2], h[2])])
    return signs + c


def draw_scene(ax, spec: RobotSpec, q: Sequence[float], *,
               obstacles: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
               goal: Sequence[float] | None = None,
               trace: Sequence[Sequence[float]] = (),
               time_s: float = 0.0, elev: float = 18, azim: float = 35):
    """Draw robot + chamber + obstacles into a 3-D axes. Returns the axes."""
    pose = forward_pose(spec, q)
    chamber = spec.chamber_size

    # chamber wireframe
    for xs in ((0, 0), (1, 1)):
        for ys in ((0, 0), (1, 1)):
            pass
    corners = _box_corners([v / 2 for v in chamber], [v / 2 for v in chamber])
    for i, j in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
        ax.plot(*zip(corners[i], corners[j]), color="0.75", lw=0.6)

    # links
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(spec.links)))
    for link, color in zip(spec.links, colors):
        frame = pose.frames[link.drive_joint if link.drive_joint >= 0 else 0]
        world = link_corners_world(link, frame)
        coll = Poly3DCollection(_cuboid_faces(world), alpha=0.55,
                                facecolor=color, edgecolor="k", lw=0.35)
        ax.add_collection3d(coll)

    # joint pivots + bone lines through origins
    origins = [frame[:3, 3] for frame in pose.frames]
    ax.plot(*zip(*origins), color="k", lw=1.2, marker="o", ms=3)
    ee = np.asarray(pose.ee_position)
    ax.scatter(*ee, color="red", s=40, depthshade=False)
    ax.plot(*zip(origins[-1], ee), color="red", lw=1.5)

    # obstacles (axis-aligned world boxes: center, size)
    for center, size in obstacles:
        world = _box_corners(center, [s / 2 for s in size])
        coll = Poly3DCollection(_cuboid_faces(world), alpha=0.35,
                                facecolor="orange", edgecolor="darkorange", lw=0.5)
        ax.add_collection3d(coll)

    if trace:
        arr = np.asarray(trace)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="crimson", lw=1.0, alpha=0.8)

    if goal is not None:
        ax.scatter(*goal, color="lime", marker="*", s=120, depthshade=False)

    ax.set_xlim(0, chamber[0])
    ax.set_ylim(0, chamber[1])
    ax.set_zlim(0, chamber[2])
    ax.set_box_aspect(chamber)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"t = {time_s:.2f}s   EE = ({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f})")
    return ax


def render_frame(spec: RobotSpec, q: Sequence[float], path: str | Path, **kwargs) -> Path:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, spec, q, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def render_episode_gif(spec: RobotSpec, frames: Sequence[dict], path: str | Path,
                       *, fps: int = 8, every: int = 1) -> Path:
    """Save an episode recording.

    ``frames`` is a list of per-step dicts with keys ``q`` (required) and
    optional ``obstacles`` (list of (center, size)), ``goal``, ``trace``,
    ``time_s``.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(enumerate(frames))[:: max(1, every)]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    def _update(pair):
        ax.cla()
        idx, frame = pair
        trace = [f["q"] for f in frames[: idx + 1]]
        # EE trace computed by caller may be provided; else empty
        draw_scene(ax, spec, frame["q"],
                   obstacles=frame.get("obstacles", ()),
                   goal=frame.get("goal"),
                   trace=frame.get("ee_trace", ()),
                   time_s=frame.get("time_s", idx * 0.1))
        return []

    anim = FuncAnimation(fig, _update, frames=selected, interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
