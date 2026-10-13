"""Frozen joint-program library persistence for v1.3."""
from __future__ import annotations

import json
from pathlib import Path

from experiments_v13.joint_dsl import JointProgram


def save_library(library: dict[str, JointProgram], path: str | Path) -> Path:
    payload = {
        "format": "robot-oops-joint-lib-0.1",
        "programs": [p.to_dict() for p in library.values()],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    return path


def load_library(path: str | Path) -> dict[str, JointProgram]:
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data.get("format") != "robot-oops-joint-lib-0.1":
        raise ValueError("unsupported joint library format")
    library: dict[str, JointProgram] = {}
    for item in data["programs"]:
        program = JointProgram(tuple(item["rules"]), item["id"])
        if program.id in library:
            raise ValueError("duplicate program id")
        for rule in program.rules:
            if rule.startswith("CALL:") and rule[5:] not in library:
                raise ValueError("unknown or forward CALL reference")
        library[program.id] = program
    return library
