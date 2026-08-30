from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


Metric = Callable[[np.ndarray, np.ndarray], float]


def _metric(name: str) -> Metric:
    if name == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    if name == "pr_auc":
        return lambda y, p: float(average_precision_score(y, p))
    raise ValueError(f"Unsupported bootstrap metric: {name}")


def _cluster_indices(cluster_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique = np.unique(cluster_ids.astype(str))
    groups = [np.flatnonzero(cluster_ids.astype(str) == cluster) for cluster in unique]
    return unique, groups


def _sample_indices(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    selected = rng.integers(0, len(groups), size=len(groups))
    # Concatenation deliberately retains duplicate examination draws.
    return np.concatenate([groups[index] for index in selected])


def cluster_bootstrap_distribution(
    targets,
    probabilities,
    examination_ids,
    *,
    metric: str,
    samples: int = 2000,
    seed: int = 2026,
) -> np.ndarray:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    clusters = np.asarray(examination_ids).reshape(-1)
    if not (y.shape == p.shape == clusters.shape):
        raise ValueError("targets, probabilities, and examination_ids must align")
    if np.unique(y).size < 2:
        raise ValueError("Bootstrap ROC/PR metrics require both target classes")
    if samples <= 0:
        raise ValueError("samples must be positive")
    _, groups = _cluster_indices(clusters)
    if not groups:
        raise ValueError("At least one examination is required")
    scorer = _metric(metric)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    maximum_attempts = samples * 20
    while len(values) < samples and attempts < maximum_attempts:
        attempts += 1
        indices = _sample_indices(groups, rng)
        if np.unique(y[indices]).size < 2:
            continue
        values.append(scorer(y[indices], p[indices]))
    if len(values) != samples:
        raise RuntimeError("Too many degenerate cluster-bootstrap resamples")
    return np.asarray(values, dtype=np.float64)


def cluster_bootstrap_ci(
    targets,
    probabilities,
    examination_ids,
    *,
    metric: str,
    samples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    distribution = cluster_bootstrap_distribution(
        targets,
        probabilities,
        examination_ids,
        metric=metric,
        samples=samples,
        seed=seed,
    )
    alpha = (1.0 - confidence) / 2.0
    scorer = _metric(metric)
    estimate = scorer(np.asarray(targets), np.asarray(probabilities))
    low, high = np.quantile(distribution, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def paired_cluster_bootstrap(
    targets,
    proposed_probabilities,
    comparator_probabilities,
    examination_ids,
    *,
    metric: str,
    samples: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    proposed = np.asarray(proposed_probabilities, dtype=np.float64).reshape(-1)
    comparator = np.asarray(comparator_probabilities, dtype=np.float64).reshape(-1)
    clusters = np.asarray(examination_ids).reshape(-1)
    if not (y.shape == proposed.shape == comparator.shape == clusters.shape):
        raise ValueError("All paired-bootstrap arrays must align")
    if np.unique(y).size < 2:
        raise ValueError("Paired bootstrap requires both target classes")
    if samples <= 0:
        raise ValueError("samples must be positive")
    _, groups = _cluster_indices(clusters)
    if not groups:
        raise ValueError("At least one examination is required")
    scorer = _metric(metric)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    attempts = 0
    maximum_attempts = samples * 20
    while len(deltas) < samples and attempts < maximum_attempts:
        attempts += 1
        indices = _sample_indices(groups, rng)
        if np.unique(y[indices]).size < 2:
            continue
        deltas.append(
            scorer(y[indices], proposed[indices])
            - scorer(y[indices], comparator[indices])
        )
    if len(deltas) != samples:
        raise RuntimeError("Too many degenerate paired cluster-bootstrap resamples")
    delta_array = np.asarray(deltas, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(delta_array, [alpha, 1.0 - alpha])
    # The supplement describes an empirical two-sided probability and reports
    # p<0.001 with B=2000, implying no finite-sample +1 correction.
    p_value = min(
        1.0,
        2.0
        * min(
            float(np.mean(delta_array <= 0.0)),
            float(np.mean(delta_array >= 0.0)),
        ),
    )
    return {
        "delta": float(scorer(y, proposed) - scorer(y, comparator)),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""

    raw = np.asarray(p_values, dtype=np.float64)
    if raw.ndim != 1 or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    order = np.argsort(raw)
    adjusted_sorted = np.empty_like(raw)
    running = 0.0
    family_size = raw.size
    for rank, original_index in enumerate(order):
        corrected = (family_size - rank) * raw[original_index]
        running = max(running, corrected)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(raw)
    adjusted[order] = adjusted_sorted
    return adjusted


def adjust_comparison_families(
    comparisons: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Apply Holm separately within each lesion/metric family."""

    result = [dict(row) for row in comparisons]
    families: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(result):
        families.setdefault((str(row["lesion"]), str(row["metric"])), []).append(index)
    for indices in families.values():
        corrected = holm_adjust([float(result[index]["p_value"]) for index in indices])
        for index, value in zip(indices, corrected, strict=True):
            result[index]["p_value_holm"] = float(value)
    return result
