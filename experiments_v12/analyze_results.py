"""Derive report tables from saved v1.2 results (no re-evaluation).

Reads the raw episode rows written by the runners and produces:

* ``results/analysis.json``   machine-readable aggregates + paired contrasts
* ``results/analysis_tables.md``  markdown tables for the report

Two refinements the raw summaries do not include:

* **feasible-only rates**.  Level 1 and level 4/5 test splits contain scenes
  that are infeasible *by construction* (labelled ``infeasible_static``).
  Solving those is impossible for any method, so overall rates understate
  capability; both numbers are reported.
* **paired contrasts recomputed from raw rows**, so every comparison in the
  report is reproducible from the saved episode files.  v1.2.1 adds
  **scene-clustered bootstrap CIs** (rows that share a scene are correlated
  across training seeds; the resampling unit is the unique scene) and
  **per-training-seed contrast means**, so a contrast driven by one seed is
  visible instead of averaged away.  The row-level interval is retained only
  for provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments_v12._common import RESULTS_ROOT, save_json
from experiments_v12.statistics import (clustered_paired_rate,
                                        paired_comparison,
                                        paired_rate_comparison,
                                        per_seed_contrasts,
                                        wilson_interval)

TRUE = {"1", "true", "True", "TRUE"}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        return json.loads(path.read_text())
    rows = []
    with path.open(newline="", encoding="utf8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("success", "collision"):
                if key in row:
                    row[key] = str(row[key]) in TRUE
            for key in ("steps", "level", "seed", "seed_index", "interventions",
                        "waits", "deadline_misses"):
                if key in row and row[key] not in ("", None):
                    row[key] = int(float(row[key]))
            rows.append(row)
    return rows


def rate(rows: list[dict], key: str = "success") -> dict:
    n = len(rows)
    hits = sum(int(bool(r.get(key))) for r in rows)
    summary = wilson_interval(hits, n)
    misses = [int(r.get("deadline_misses") or 0) for r in rows]
    summary["episodes_with_deadline_miss"] = sum(1 for m in misses if m > 0)
    summary["mean_deadline_misses"] = (sum(misses) / n) if n else 0.0
    return summary


def feasible_only(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("label") != "infeasible_static"]


def rate_table(rows: list[dict], group: str, *, feasible: bool = False) -> list[dict]:
    source = feasible_only(rows) if feasible else rows
    groups: dict[str, list[dict]] = {}
    for row in source:
        groups.setdefault(str(row.get(group, "")), []).append(row)
    out = []
    for name, members in sorted(groups.items()):
        summary = rate(members)
        out.append({
            group: name, "n": summary["n"], "success": summary["rate"],
            "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            "collision_rate": sum(int(bool(r.get("collision"))) for r in members)
            / len(members) if members else 0.0,
            "episodes_with_deadline_miss": summary["episodes_with_deadline_miss"],
            "mean_deadline_misses": summary["mean_deadline_misses"],
        })
    return out


def by_level(rows: list[dict], method_key: str = "method") -> list[dict]:
    out = []
    for level in sorted({r.get("level") for r in rows if r.get("level") is not None}):
        members = [r for r in rows if r.get("level") == level]
        feasible = feasible_only(members)
        out.append({
            "level": level, "n_all": len(members),
            "success_all": rate(members)["rate"],
            "n_feasible": len(feasible),
            "success_feasible": rate(feasible)["rate"] if feasible else None,
            "collision_rate": sum(int(bool(r.get("collision"))) for r in members)
            / len(members) if members else 0.0,
        })
    return out


def pair_key(row: dict) -> tuple:
    """Pair across methods on (curriculum seed, scenario, level)."""
    return (row.get("seed_index", 0), row.get("scenario_id"), row.get("level"))


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict | None:
    index_b = {pair_key(r): r for r in rows_b}
    pairs = [(a, index_b[pair_key(a)]) for a in rows_a
             if pair_key(a) in index_b]
    if not pairs:
        return None
    return {
        "n_pairs": len(pairs),
        # v1.2.1: scene-clustered CI is the authoritative interval (rows that
        # share a scene are correlated across training seeds).
        "success_clustered": clustered_paired_rate(
            [a for a, _ in pairs], [b for _, b in pairs]),
        "per_seed": per_seed_contrasts(
            [a for a, _ in pairs], [b for _, b in pairs]),
        # Row-level interval kept for provenance only.
        "success": paired_rate_comparison(
            [int(bool(a["success"])) for a, _ in pairs],
            [int(bool(b["success"])) for _, b in pairs]),
        "steps": paired_comparison([float(a.get("steps", 0)) for a, _ in pairs],
                                   [float(b.get("steps", 0)) for _, b in pairs]),
    }


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_(no rows)_\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_analysis(root: Path) -> dict:
    analysis: dict = {"root": str(root)}

    ablation_rows = load_rows(root / "ablations" / "episodes_seed5.json")
    if ablation_rows:
        analysis["ablations"] = {
            "overall": rate_table(ablation_rows, "method"),
            "overall_feasible": rate_table(ablation_rows, "method", feasible=True),
            "by_level": {
                method: by_level([r for r in ablation_rows if r["method"] == method])
                for method in sorted({r["method"] for r in ablation_rows})
            },
            "contrasts": {
                f"{left}_vs_{right}": paired(
                    [r for r in ablation_rows if r["method"] == left],
                    [r for r in ablation_rows if r["method"] == right])
                for left, right in (("C5", "C2"), ("C5", "C3"), ("C5", "C4"),
                                    ("C5", "C1"))
            },
        }

    gen_rows = load_rows(root / "generalization" / "episodes.json")
    if gen_rows:
        test = [r for r in gen_rows if r.get("split") == "test"]
        analysis["generalization"] = {
            "by_level_all": by_level(test),
            "by_level_feasible": [
                {**row, "success": row["success_feasible"],
                 "n": row["n_feasible"]}
                for row in by_level(test)
            ],
        }

    baseline_rows = load_rows(root / "baselines" / "episodes.json")
    baseline_rows += load_rows(root / "baselines_hl" / "episodes.json")
    if baseline_rows:
        methods = sorted({r["method"] for r in baseline_rows})
        analysis["baselines"] = {
            "overall": rate_table(baseline_rows, "method"),
            "overall_feasible": rate_table(baseline_rows, "method", feasible=True),
            "by_level": {
                method: by_level([r for r in baseline_rows if r["method"] == method])
                for method in methods
            },
            "contrasts": {
                f"C5_vs_{method}": paired(
                    [r for r in baseline_rows if r["method"] == "C5"],
                    [r for r in baseline_rows if r["method"] == method])
                for method in methods if method != "C5"
            },
        }

    reactive_rows = load_rows(root / "reactive" / "episodes.json")
    if reactive_rows:
        analysis["reactive"] = {
            "by_degradation": {
                method: rate_table(
                    [r for r in reactive_rows if r["method"] == method],
                    "degradation")
                for method in sorted({r["method"] for r in reactive_rows})
            },
        }

    profile = root / "gpu_profile" / "profile.json"
    if profile.exists():
        analysis["gpu_profile"] = json.loads(profile.read_text())
    return analysis


def render_markdown(analysis: dict) -> str:
    parts = ["# v1.2 derived tables\n",
             "Generated by `experiments_v12/analyze_results.py` from saved "
             "episode rows; no re-evaluation.\n"]

    if "ablations" in analysis:
        abl = analysis["ablations"]
        parts.append("## Experiment C: ablations (deployment, all levels)\n")
        parts.append(md_table(abl["overall"],
                              ["method", "n", "success", "ci_low", "ci_high",
                               "collision_rate"]))
        parts.append("Feasible-only (infeasible-by-construction scenes removed):\n")
        parts.append(md_table(abl["overall_feasible"],
                              ["method", "n", "success", "ci_low", "ci_high",
                               "collision_rate"]))
        parts.append("Paired contrasts (positive favours the left method; "
                     "scene-clustered CI; seed-level t-CI over training "
                     "seeds):\n")
        contrast_rows = []
        for name, value in abl["contrasts"].items():
            if not value:
                continue
            clustered = value["success_clustered"]
            seeds = value["per_seed"]["per_seed"]
            tci = value["per_seed"].get("seed_t_ci") or {}
            contrast_rows.append({
                "contrast": name,
                "n_scenes": clustered["n_scenes"],
                "d_success": clustered["mean"],
                "ci_low": clustered["ci_low"],
                "ci_high": clustered["ci_high"],
                "seed_t_ci": (f"{tci.get('mean', 0):+.3f} "
                              f"[{tci.get('ci_low', 0):+.3f}, "
                              f"{tci.get('ci_high', 0):+.3f}]"
                              if tci else "n/a"),
                "per_seed_means": "; ".join(
                    f"s{si}: {m['mean']:+.3f}" for si, m in seeds.items()),
                "left_only": value["success"]["left_only"],
                "right_only": value["success"]["right_only"],
            })
        parts.append(md_table(contrast_rows,
                              ["contrast", "n_scenes", "d_success", "ci_low",
                               "ci_high", "seed_t_ci", "per_seed_means",
                               "left_only", "right_only"]))

    if "generalization" in analysis:
        gen = analysis["generalization"]
        parts.append("\n## Experiment A: generalization by level\n")
        parts.append(md_table(gen["by_level_all"],
                              ["level", "n_all", "success_all", "n_feasible",
                               "success_feasible", "collision_rate"]))

    if "baselines" in analysis:
        base = analysis["baselines"]
        parts.append("\n## Experiment D: baselines\n")
        parts.append(md_table(base["overall"],
                              ["method", "n", "success", "ci_low", "ci_high",
                               "collision_rate", "episodes_with_deadline_miss",
                               "mean_deadline_misses"]))
        parts.append("Feasible-only:\n")
        parts.append(md_table(base["overall_feasible"],
                              ["method", "n", "success", "ci_low", "ci_high",
                               "collision_rate"]))
        parts.append("Paired C5-versus-planner contrasts (scene-clustered CI; "
                     "seed-level t-CI):\n")
        rows = []
        for name, value in base["contrasts"].items():
            if not value:
                continue
            clustered = value["success_clustered"]
            tci = value["per_seed"].get("seed_t_ci") or {}
            rows.append({
                "contrast": name,
                "n_scenes": clustered["n_scenes"],
                "d_success": clustered["mean"],
                "ci_low": clustered["ci_low"],
                "ci_high": clustered["ci_high"],
                "seed_t_ci": (f"{tci.get('mean', 0):+.3f} "
                              f"[{tci.get('ci_low', 0):+.3f}, "
                              f"{tci.get('ci_high', 0):+.3f}]"
                              if tci else "n/a"),
                "left_only": value["success"]["left_only"],
                "right_only": value["success"]["right_only"],
            })
        parts.append(md_table(rows, ["contrast", "n_scenes", "d_success",
                                     "ci_low", "ci_high", "seed_t_ci",
                                     "left_only", "right_only"]))

    if "reactive" in analysis:
        parts.append("\n## Experiment E: degraded observations\n")
        for method, table in analysis["reactive"]["by_degradation"].items():
            parts.append(f"### {method}\n")
            parts.append(md_table(table, ["degradation", "n", "success",
                                          "ci_low", "ci_high", "collision_rate",
                                          "episodes_with_deadline_miss"]))
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=str(RESULTS_ROOT))
    args = parser.parse_args(argv)
    root = Path(args.root)
    analysis = build_analysis(root)
    save_json(analysis, root / "analysis.json")
    (root / "analysis_tables.md").write_text(render_markdown(analysis))
    print(f"wrote {root/'analysis.json'} and {root/'analysis_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
