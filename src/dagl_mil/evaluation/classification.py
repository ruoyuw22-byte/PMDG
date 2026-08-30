from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_1d(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Classification arrays must contain only finite values")
    return result


def binary_classification_metrics(targets, probabilities) -> dict[str, float]:
    y_true = _as_1d(targets).astype(np.int64)
    y_prob = _as_1d(probabilities)
    if y_true.shape != y_prob.shape:
        raise ValueError("Targets and probabilities must have the same shape")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if np.unique(y_true).size < 2:
        raise ValueError("ROC-AUC is undefined when only one class is present")
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        # sklearn average_precision_score implements the step-wise PR integral
        # described in Supplement S5, rather than trapezoidal interpolation.
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "positives": int(y_true.sum()),
        "samples": int(y_true.size),
    }


def classification_metrics(
    targets: Mapping[str, np.ndarray] | np.ndarray,
    probabilities: Mapping[str, np.ndarray] | np.ndarray,
    class_names: tuple[str, ...] = ("mass", "calcification"),
) -> dict[str, dict[str, float]]:
    if isinstance(targets, Mapping):
        return {
            name: binary_classification_metrics(targets[name], probabilities[name])
            for name in class_names
        }

    y_true = np.asarray(targets)
    y_prob = np.asarray(probabilities)
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError("Array inputs must both have shape [samples, classes]")
    if y_true.shape[1] != len(class_names):
        raise ValueError("Number of columns does not match class_names")
    return {
        name: binary_classification_metrics(y_true[:, index], y_prob[:, index])
        for index, name in enumerate(class_names)
    }

