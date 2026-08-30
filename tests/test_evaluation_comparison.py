import numpy as np

from dagl_mil.evaluation.artifacts import write_prediction_artifact
from dagl_mil.evaluation.comparison import compare_prediction_artifacts


def _write(path, probabilities):
    records = []
    for index in range(20):
        for side, label in (("L", 0), ("R", 1)):
            records.append(
                {
                    "exam_id": f"e{index:02d}",
                    "laterality": side,
                    "labels": [label, label],
                    "probabilities": probabilities(label),
                    "views": {},
                }
            )
    write_prediction_artifact(path, records, metadata={"test": True})


def test_paired_artifact_comparison_and_holm(tmp_path):
    reference = tmp_path / "reference.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(reference, lambda label: [0.01 + 0.98 * label] * 2)
    _write(first, lambda label: [0.2 + 0.6 * label] * 2)
    _write(second, lambda label: [0.4 + 0.2 * label] * 2)
    report = compare_prediction_artifacts(
        reference,
        {"first": first, "second": second},
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    assert report["breasts"] == 40
    assert report["unique_examinations"] == 20
    assert len(report["comparisons"]) == 8
    assert all("p_value_holm" in row for row in report["comparisons"])
    assert all(np.isfinite(row["delta"]) for row in report["comparisons"])
