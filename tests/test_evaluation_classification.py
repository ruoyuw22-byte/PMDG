import numpy as np
import pytest

from dagl_mil.evaluation.bootstrap import (
    cluster_bootstrap_ci,
    holm_adjust,
    paired_cluster_bootstrap,
)
from dagl_mil.evaluation.classification import binary_classification_metrics


def test_binary_metrics_are_exact_for_perfect_ranking():
    result = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0


def test_cluster_bootstrap_is_deterministic_and_retains_pairs():
    exams = np.repeat(["a", "b", "c", "d"], 2)
    labels = np.tile([0, 1], 4)
    probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6])
    first = cluster_bootstrap_ci(
        labels, probabilities, exams, metric="roc_auc", samples=50, seed=2026
    )
    second = cluster_bootstrap_ci(
        labels, probabilities, exams, metric="roc_auc", samples=50, seed=2026
    )
    assert first == second
    assert first["estimate"] == 1.0


def test_paired_bootstrap_and_holm():
    exams = np.repeat(np.arange(20), 2)
    labels = np.tile([0, 1], 20)
    proposed = np.tile([0.05, 0.95], 20)
    comparator = np.tile([0.45, 0.55], 20)
    result = paired_cluster_bootstrap(
        labels,
        proposed,
        comparator,
        exams,
        metric="pr_auc",
        samples=50,
        seed=9,
    )
    assert result["delta"] >= 0.0
    corrected = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(corrected, [0.03, 0.06, 0.06])


def test_bootstrap_rejects_single_class_without_retry_loop():
    labels = np.zeros(4, dtype=int)
    probabilities = np.linspace(0.1, 0.4, 4)
    exams = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="both target classes"):
        cluster_bootstrap_ci(
            labels, probabilities, exams, metric="roc_auc", samples=10
        )
    with pytest.raises(ValueError, match="both target classes"):
        paired_cluster_bootstrap(
            labels,
            probabilities,
            probabilities,
            exams,
            metric="roc_auc",
            samples=10,
        )
