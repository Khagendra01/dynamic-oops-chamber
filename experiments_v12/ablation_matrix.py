"""Ablation matrix for experiment series v1.2 (Paper 1, experiment C).

Five primary variants plus the three remaining cells of the full 2x2x2
factorial over the three factors that the C-series manipulates:

``reuse``
    frozen-code reuse during synthesis (extend lane + ``CALL``) **and** at
    deployment (whether previously discovered programs are callable).
``cumulative``
    cumulative task validation: a candidate program must satisfy every task
    seen so far (ON) rather than only the current task (OFF).
``memory``
    persistent interpreter memory (``VMState``) across observations.

Controlled constants (identical for every variant, never ablated here):
conditional geometric branching (``CHECK_SAFE`` / ``IF_UNSAFE_SKIP`` /
``CHECK_WAIT_GAP``) and live replanning (a fresh decision from the current
observation every control period).  Individual ablation of those two belongs to
separate experiments and must not be claimed from the C-series.

Primary causal contrasts (single-factor):

* ``C5`` vs ``C2`` : frozen-code reuse
* ``C5`` vs ``C3`` : cumulative curriculum validation
* ``C5`` vs ``C4`` : persistent runtime memory

``C1`` vs ``C5`` is an overall comparison only (three factors differ).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

FACTORS = ("reuse", "cumulative", "memory")
CONTROLLED_CONSTANTS = ("conditional_geometric_branching", "live_replanning")


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    reuse: bool
    cumulative: bool
    memory: bool
    note: str = ""

    def bits(self) -> tuple[bool, bool, bool]:
        return (self.reuse, self.cumulative, self.memory)

    def bit_vector(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.bits())  # type: ignore[return-value]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["factors"] = {name: getattr(self, name) for name in FACTORS}
        payload["controlled_constants"] = list(CONTROLLED_CONSTANTS)
        return payload


C1 = Variant("C1", "minimal baseline", reuse=False, cumulative=False,
             memory=False,
             note="All three factors off; not a single-factor comparison.")
C2 = Variant("C2", "no frozen reuse", reuse=False, cumulative=True,
             memory=True,
             note="Isolates frozen-code reuse (C5 vs C2).")
C3 = Variant("C3", "no cumulative validation", reuse=True, cumulative=False,
             memory=True,
             note="Isolates cumulative curriculum validation (C5 vs C3).")
C4 = Variant("C4", "stateless runtime", reuse=True, cumulative=True,
             memory=False,
             note="Isolates persistent runtime memory (C5 vs C4).")
C5 = Variant("C5", "full system", reuse=True, cumulative=True, memory=True,
             note="Reference configuration for Paper 1.")

CANONICAL: tuple[Variant, ...] = (C1, C2, C3, C4, C5)

# Remaining cells of the 2x2x2 factorial (bit order: reuse, cumulative, memory).
FACTORIAL_EXTRA: tuple[Variant, ...] = (
    Variant("F001", "memory only", reuse=False, cumulative=False, memory=True,
            note="Factorial cell: memory without reuse or cumulative validation."),
    Variant("F010", "cumulative only", reuse=False, cumulative=True,
            memory=False,
            note="Factorial cell: cumulative validation alone."),
    Variant("F100", "reuse only", reuse=True, cumulative=False, memory=False,
            note="Factorial cell: reuse alone."),
)

FACTORIAL: tuple[Variant, ...] = CANONICAL + FACTORIAL_EXTRA

PRIMARY_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C5", "C2", "frozen-code reuse"),
    ("C5", "C3", "cumulative curriculum validation"),
    ("C5", "C4", "persistent runtime memory"),
    ("C1", "C5", "overall system (three factors; not single-factor)"),
)


def by_id(variant_id: str) -> Variant:
    for variant in FACTORIAL:
        if variant.id == variant_id:
            return variant
    raise KeyError(f"unknown variant {variant_id!r}; known: "
                   f"{[v.id for v in FACTORIAL]}")


def validate_matrix() -> None:
    """Fail loudly if the matrix is not the specified design."""
    seen: dict[tuple, str] = {}
    for variant in FACTORIAL:
        key = variant.bits()
        if key in seen:
            raise AssertionError(
                f"duplicate configuration: {variant.id} aliases {seen[key]} "
                f"(bits {key}); C2 and C5 must not silently alias")
        seen[key] = variant.id
    if len(FACTORIAL) != 8:
        raise AssertionError(f"factorial must have 8 cells, got {len(FACTORIAL)}")
    expected = {
        "C1": (False, False, False),
        "C2": (False, True, True),
        "C3": (True, False, True),
        "C4": (True, True, False),
        "C5": (True, True, True),
    }
    for variant in CANONICAL:
        if variant.bits() != expected[variant.id]:
            raise AssertionError(
                f"{variant.id} is {variant.bits()}, spec requires "
                f"{expected[variant.id]}")
    for a, b, _ in PRIMARY_CONTRASTS:
        va, vb = by_id(a), by_id(b)
        differing = [name for name in FACTORS
                     if getattr(va, name) != getattr(vb, name)]
        if a in ("C5",) and b in ("C2", "C3", "C4") and len(differing) != 1:
            raise AssertionError(
                f"contrast {a} vs {b} must be single-factor, differs in {differing}")


def write_config_files(out_dir: str | Path, variants: tuple[Variant, ...] = FACTORIAL) -> list[Path]:
    """Persist one JSON config file per variant (and the shared constants)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in variants:
        path = out / f"variant_{variant.id}.json"
        path.write_text(json.dumps(variant.to_dict(), indent=2) + "\n")
        written.append(path)
    summary = {
        "format": "oops-v12-ablation-matrix-1.0",
        "factors": list(FACTORS),
        "controlled_constants": list(CONTROLLED_CONSTANTS),
        "variants": [v.to_dict() for v in variants],
        "primary_contrasts": [
            {"left": a, "right": b, "factor": label}
            for a, b, label in PRIMARY_CONTRASTS
        ],
    }
    path = out / "ablation_matrix.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    written.append(path)
    return written


validate_matrix()
