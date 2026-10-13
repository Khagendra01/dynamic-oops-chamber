"""Shared helpers for the v1.2 experiment runners."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments_v12.scenarios import DEFAULT_CONFIG, make_split, training_curriculum

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Curriculum seed bases are spaced so the per-level seed windows
# (base + level*1000 .. +level*1000 + seeds_per_level) never overlap.
CURRICULUM_SEED_BASES = (1000, 3000, 5000, 7000, 9000)


def curriculum_config(seed_index: int) -> dict:
    """Config whose training split differs per seed index (held-out fixed)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["train"]["seed_base"] = CURRICULUM_SEED_BASES[seed_index]
    return cfg


def curriculum_for(seed_index: int):
    return training_curriculum(curriculum_config(seed_index))


def test_split(level: int):
    return make_split(level, "test")


def validation_split(level: int):
    return make_split(level, "validation")


def out_dir(name: str, root: Path | None = None) -> Path:
    path = (root or RESULTS_ROOT) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of curriculum seeds (1-5)")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny budget, few scenarios; correctness only")
    return parser


def parse_levels(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]
