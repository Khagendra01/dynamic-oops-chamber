"""Gate 1 demo: sweep every joint, record the articulated robot moving.

Produces:
* ``results/kinematics/joint_sweep.gif`` — one animated sweep of all six joints
* ``results/kinematics/ee_positions.csv`` — EE positions along the sweep
* ``results/kinematics/summary.json`` — ranges, determinism check

Usage:
    python -m experiments_v13.run_kinematics [--out DIR] [--no-gif]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments_v13.forward_kinematics import ee_position, forward_pose
from experiments_v13.joint_dynamics import JointController
from experiments_v13.robot_model import default_robot, link_corners_world


def joint_sweep_frames(spec, controller, steps_per_joint=24):
    """Move each joint +then- from rest through its range; record q, ee."""
    frames = []
    q = np.zeros(6)
    state = controller.start_state(q)
    frames.append({"q": state.q.copy(), "time_s": 0.0,
                   "ee_trace": [list(ee_position(spec, state.q))]})
    t = 0.0
    for joint in range(6):
        hi = spec.joint_hi[joint]
        lo = spec.joint_lo[joint]
        span = min(abs(hi), abs(lo), math.pi * 0.75)  # stay in a visible range
        for direction in (+1.0, -1.0):
            target = direction * span
            for k in range(steps_per_joint):
                frac = (k + 1) / steps_per_joint
                dq_des = target * frac - float(state.q[joint])
                cmd = np.zeros(6)
                cmd[joint] = dq_des
                step = controller.step(state, cmd)
                state = step.state
                t += controller.dt
                frames.append({
                    "q": state.q.copy(), "time_s": round(t, 3),
                    "ee_trace": [list(ee_position(spec, state.q))],
                })
    return frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments_v13/results/kinematics")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = default_robot()
    controller = JointController(spec, dt=0.1)
    frames = joint_sweep_frames(spec, controller)

    # EE path CSV
    with (out / "ee_positions.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "t"] + [f"q{i}" for i in range(6)]
                        + ["ee_x", "ee_y", "ee_z"])
        for i, frame in enumerate(frames):
            ee = ee_position(spec, frame["q"])
            writer.writerow([i, frame["time_s"]] + [round(float(v), 6) for v in frame["q"]]
                            + [round(float(v), 6) for v in ee])

    # determinism: recompute FK on the same qs
    q0 = np.array(frames[len(frames) // 2]["q"])
    a = ee_position(spec, q0)
    b = ee_position(spec, q0.copy())
    assert a == b

    # EE workspace coverage along the sweep
    ees = np.array([ee_position(spec, f["q"]) for f in frames])
    summary = {
        "frames": len(frames),
        "duration_s": frames[-1]["time_s"],
        "ee_min": ees.min(axis=0).round(4).tolist(),
        "ee_max": ees.max(axis=0).round(4).tolist(),
        "deterministic_ee": a == b,
        "max_reach_bound_m": round(spec.max_reach(), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not args.no_gif:
        from experiments_v13.visualization import render_episode_gif
        gif_frames = []
        for i, frame in enumerate(frames):
            gif_frames.append({"q": frame["q"], "time_s": frame["time_s"],
                               "ee_trace": [list(ee) for ee in ees[: i + 1]]})
        render_episode_gif(spec, gif_frames, out / "joint_sweep.gif", fps=10, every=2)
        print(f"gif -> {out / 'joint_sweep.gif'}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
