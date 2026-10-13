"""Paired statistics for experiment series v1.2.

All comparisons in v1.2 are *paired*: every method is evaluated on the same
scenario seeds, so uncertainty is estimated on per-scenario differences, not
on independent samples.  Intervals are percentile bootstrap intervals over the
paired differences; proportions additionally get Wilson intervals.

v1.2.1 — independent-scene inference
------------------------------------
Rows that share a ``(scenario_id, level)`` cluster are correlated across
training seeds (the same frozen program is replayed on the same scene with
different observation-noise draws).  Bootstrapping rows therefore
underestimates uncertainty: the reported +2.9 pp reuse contrast shrinks to
[-7.5, +22.5] once scenes — not rows — are the resampling unit
(:func:`clustered_paired_rate`).  Every contrast quoted in the v1.2.1 report
is scene-clustered; row-level intervals are kept only for provenance.
Per-training-seed contrasts (:func:`per_seed_contrasts`) expose seed
heterogeneity that a pooled interval hides.

Nothing here is seeded by wall-clock time, so re-running an analysis reproduces
the same intervals.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20261013


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_mean_ci(values: Sequence[float], *,
                      n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                      seed: int = DEFAULT_BOOTSTRAP_SEED,
                      alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for a mean (values = paired differences)."""
    values = list(values)
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0,
                "n_samples": n_samples, "seed": seed}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[max(0, int(math.floor(alpha / 2 * n_samples)) - 1
                    if alpha / 2 * n_samples >= 1 else 0)]
    high = means[min(n_samples - 1, int(math.ceil((1 - alpha / 2) * n_samples)))]
    return {"mean": _mean(values), "ci_low": low, "ci_high": high, "n": n,
            "n_samples": n_samples, "seed": seed, "alpha": alpha}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
              / denominator)
    return {"rate": p, "ci_low": max(0.0, centre - margin),
            "ci_high": min(1.0, centre + margin), "n": n, "successes": successes}


def paired_differences(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Per-item (left - right) differences; lengths must match."""
    if len(left) != len(right):
        raise ValueError(f"paired comparison needs equal lengths "
                         f"({len(left)} vs {len(right)})")
    return [a - b for a, b in zip(left, right)]


def paired_comparison(left: Sequence[float], right: Sequence[float], *,
                      n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                      seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Mean difference (left - right) with a paired bootstrap CI."""
    differences = paired_differences(left, right)
    interval = bootstrap_mean_ci(differences, n_samples=n_samples, seed=seed)
    wins = sum(1 for d in differences if d > 0)
    losses = sum(1 for d in differences if d < 0)
    ties = sum(1 for d in differences if d == 0)
    interval.update({"left_mean": _mean(list(left)), "right_mean": _mean(list(right)),
                     "wins": wins, "losses": losses, "ties": ties})
    return interval


def paired_rate_comparison(left: Sequence[int], right: Sequence[int], *,
                           n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                           seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Difference of success *rates* with a paired bootstrap CI."""
    differences = paired_differences([float(x) for x in left],
                                     [float(x) for x in right])
    interval = bootstrap_mean_ci(differences, n_samples=n_samples, seed=seed)
    interval.update({
        "left_rate": _mean([float(x) for x in left]),
        "right_rate": _mean([float(x) for x in right]),
        "left_wilson": wilson_interval(int(sum(left)), len(left)),
        "right_wilson": wilson_interval(int(sum(right)), len(right)),
        "left_only": sum(1 for a, b in zip(left, right) if a and not b),
        "right_only": sum(1 for a, b in zip(left, right) if b and not a),
        "both": sum(1 for a, b in zip(left, right) if a and b),
        "neither": sum(1 for a, b in zip(left, right) if not a and not b),
    })
    return interval


def clustered_paired_rate(left_rows: Sequence[dict], right_rows: Sequence[dict], *,
                          left_field: str = "success", right_field: str = "success",
                          n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
                          seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Scene-clustered difference of success rates (left - right).

    Rows are paired on ``(seed_index, scenario_id, level)``; differences are
    averaged *within each unique scene* (across training seeds) and the
    bootstrap resamples scenes, so 5 training-seed rows of one scene count as
    one independent unit.
    """
    def key(r: dict) -> tuple:
        return (int(r.get("seed_index", r.get("seed", 0))),
                str(r.get("scenario_id", "")), int(r.get("level", -1)))

    right_map = {key(r): r for r in right_rows}
    per_scene: dict[tuple, list[float]] = {}
    left_rate_sum = right_rate_sum = 0.0
    paired = 0
    for left in left_rows:
        right = right_map.get(key(left))
        if right is None:
            continue
        d = float(bool(left[left_field])) - float(bool(right[right_field]))
        scene = (str(left.get("scenario_id", "")), int(left.get("level", -1)))
        per_scene.setdefault(scene, []).append(d)
        left_rate_sum += float(bool(left[left_field]))
        right_rate_sum += float(bool(right[right_field]))
        paired += 1
    scene_means = [sum(v) / len(v) for v in per_scene.values()]
    interval = bootstrap_mean_ci(scene_means, n_samples=n_samples, seed=seed)
    interval.update({
        "unit": "scene",
        "n_scenes": len(scene_means),
        "n_paired_rows": paired,
        "left_rate": left_rate_sum / paired if paired else 0.0,
        "right_rate": right_rate_sum / paired if paired else 0.0,
    })
    return interval


# Two-sided 97.5th percentile of Student's t by degrees of freedom (1..30).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
         25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def per_seed_contrasts(left_rows: Sequence[dict], right_rows: Sequence[dict], *,
                       left_field: str = "success",
                       right_field: str = "success") -> dict:
    """Per-training-seed mean differences (left - right), pooled summary.

    Returns ``{seed_index: {mean, n}}``, ``pooled_mean`` / ``seed_spread``,
    and ``seed_t_ci`` — a Student-t interval over the per-seed means, which
    is the honest interval for *seed-level* uncertainty (scene-clustered CIs
    condition on the trained programs and cannot express it).
    """
    def key(r: dict) -> tuple:
        return (int(r.get("seed_index", r.get("seed", 0))),
                str(r.get("scenario_id", "")), int(r.get("level", -1)))

    right_map = {key(r): r for r in right_rows}
    by_seed: dict[int, list[float]] = {}
    for left in left_rows:
        right = right_map.get(key(left))
        if right is None:
            continue
        d = float(bool(left[left_field])) - float(bool(right[right_field]))
        si = int(left.get("seed_index", left.get("seed", 0)))
        by_seed.setdefault(si, []).append(d)
    per_seed = {s: {"mean": sum(v) / len(v), "n": len(v)}
                for s, v in sorted(by_seed.items())}
    all_d = [d for v in by_seed.values() for d in v]
    means = [m["mean"] for m in per_seed.values()]
    pooled = _mean(all_d)
    seed_t_ci = None
    if len(means) >= 2:
        var = sum((m - pooled) ** 2 for m in means) / (len(means) - 1)
        sem = math.sqrt(var / len(means))
        t_crit = _T975.get(len(means) - 1, 1.96)
        seed_t_ci = {"mean": pooled, "ci_low": pooled - t_crit * sem,
                     "ci_high": pooled + t_crit * sem, "df": len(means) - 1,
                     "sem": sem}
    return {
        "per_seed": per_seed,
        "pooled_mean": pooled,
        "seed_spread": (max(means) - min(means)) if means else 0.0,
        "seed_t_ci": seed_t_ci,
        "n": len(all_d),
    }


def order_rows(rows: list[dict], key: str, *, higher_is_better: bool = True
               ) -> list[dict]:
    """Deterministically order rows by a metric for tables."""
    return sorted(rows, key=lambda r: (-r[key] if higher_is_better else r[key],
                                       str(r.get("method", "")),
                                       str(r.get("variant", ""))))


def format_ci(result: dict, digits: int = 3) -> str:
    return (f"{result['mean']:.{digits}f} "
            f"[{result['ci_low']:.{digits}f}, {result['ci_high']:.{digits}f}]")
