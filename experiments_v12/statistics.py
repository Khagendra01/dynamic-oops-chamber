"""Paired statistics for experiment series v1.2.

All comparisons in v1.2 are *paired*: every method is evaluated on the same
scenario seeds, so uncertainty is estimated on per-scenario differences, not
on independent samples.  Intervals are percentile bootstrap intervals over the
paired differences; proportions additionally get Wilson intervals.

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


def order_rows(rows: list[dict], key: str, *, higher_is_better: bool = True
               ) -> list[dict]:
    """Deterministically order rows by a metric for tables."""
    return sorted(rows, key=lambda r: (-r[key] if higher_is_better else r[key],
                                       str(r.get("method", "")),
                                       str(r.get("variant", ""))))


def format_ci(result: dict, digits: int = 3) -> str:
    return (f"{result['mean']:.{digits}f} "
            f"[{result['ci_low']:.{digits}f}, {result['ci_high']:.{digits}f}]")
