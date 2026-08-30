from contextlib import contextmanager

import numpy as np
import pytest

import dagl_mil.evaluation.report as report_module
from dagl_mil.evaluation.artifacts import encode_binary_mask, write_prediction_artifact
from dagl_mil.evaluation.report import (
    evaluate_prediction_artifact,
    write_evaluation_report,
)


def _fixture_records():
    mask = encode_binary_mask(np.ones((4, 4), dtype=bool))
    records = []
    for exam in ("a", "b"):
        for side, label in (("L", 0), ("R", 1)):
            records.append(
                {
                    "exam_id": exam,
                    "breast_id": f"{exam}_{side}",
                    "laterality": side,
                    "labels": [label, label],
                    "probabilities": [0.1 + 0.8 * label, 0.2 + 0.7 * label],
                    "views": {
                        "CC": {
                            "image_id": f"{exam}_{side}_cc",
                            "image_shape": [4, 4],
                            "patch_boxes": [[0, 0, 2, 4], [2, 0, 4, 4]],
                            "candidate_probabilities": [
                                [0.1 + 0.8 * label, 0.2 + 0.7 * label],
                                [0.1 + 0.8 * label, 0.2 + 0.7 * label],
                            ],
                            "breast_mask": mask,
                            "ground_truth": {
                                "mass": [[0, 0, 4, 4]] if label else [],
                                "calcification": [[0, 0, 4, 4]] if label else [],
                            },
                        }
                    },
                }
            )
    return records


def test_report_uses_one_fixed_prediction_artifact(tmp_path):
    records = _fixture_records()
    path = tmp_path / "fixed.jsonl"
    artifact = write_prediction_artifact(
        path, records, metadata={"checkpoint_sha256": "x"}
    )
    report = evaluate_prediction_artifact(path, bootstrap_samples=20, bootstrap_seed=3)
    assert report["breasts"] == 4
    assert report["unique_examinations"] == 2
    assert report["classification"]["mass"]["roc_auc"] == 1.0
    assert report["prediction_metadata"]["checkpoint_sha256"] == "x"
    assert report["prediction_sha256"] == artifact["sha256"]


def test_report_streams_records_and_decodes_each_view_mask_once(tmp_path, monkeypatch):
    path = tmp_path / "streamed.jsonl"
    records = _fixture_records()
    write_prediction_artifact(path, records, metadata={"checkpoint_sha256": "x"})
    events = []
    original_open = report_module.open_prediction_artifact
    original_decode = report_module.decode_binary_mask

    @contextmanager
    def traced_open(*args, **kwargs):
        with original_open(*args, **kwargs) as (header, iterator, info):

            def traced_iterator():
                for index, record in enumerate(iterator):
                    events.append(f"yield:{index}")
                    yield record

            yield header, traced_iterator(), info

    def traced_decode(encoded):
        events.append("decode")
        return original_decode(encoded)

    monkeypatch.setattr(report_module, "open_prediction_artifact", traced_open)
    monkeypatch.setattr(report_module, "decode_binary_mask", traced_decode)
    report_module.evaluate_prediction_artifact(
        path,
        bootstrap_samples=10,
        bootstrap_seed=3,
    )

    assert events.count("decode") == len(records)
    assert events[:3] == ["yield:0", "decode", "yield:1"]


def test_report_writer_serializes_undefined_ap_as_null(tmp_path):
    destination = tmp_path / "report.json"
    write_evaluation_report(destination, {"ap": float("nan")})
    assert destination.read_text(encoding="utf-8").strip() == '{\n  "ap": null\n}'


@pytest.mark.parametrize(
    "labels, match",
    [
        ([0], "shape"),
        ([0.0, 1.0], "binary integers"),
        ([0, 2], "binary integers"),
        ([False, 1], "real numeric"),
    ],
)
def test_report_rejects_non_binary_integer_labels(tmp_path, labels, match):
    records = _fixture_records()
    records[0]["labels"] = labels
    path = tmp_path / "invalid-labels.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match=match):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


@pytest.mark.parametrize(
    "probabilities",
    ([0.2, float("nan")], [0.2, -0.01], [0.2, 1.01]),
)
def test_report_rejects_invalid_breast_probabilities(tmp_path, probabilities):
    records = _fixture_records()
    records[0]["probabilities"] = probabilities
    path = tmp_path / "invalid-breast-probabilities.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


def test_report_requires_views_mapping(tmp_path):
    records = _fixture_records()
    records[0]["views"] = []
    path = tmp_path / "invalid-views.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match="views must be a mapping"):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


@pytest.mark.parametrize(
    "probabilities, match",
    [
        ([[0.1], [0.2]], "shape"),
        ([[0.1, 0.2], [0.3, float("inf")]], "finite"),
        ([[0.1, 0.2], [0.3, 1.0001]], r"in \[0, 1\]"),
    ],
)
def test_report_rejects_invalid_candidate_probabilities(tmp_path, probabilities, match):
    records = _fixture_records()
    records[0]["views"]["CC"]["candidate_probabilities"] = probabilities
    path = tmp_path / "invalid-candidate-probabilities.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match=match):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


def test_report_requires_patch_boxes_to_align_with_candidates(tmp_path):
    records = _fixture_records()
    records[0]["views"]["CC"]["patch_boxes"] = [[0, 0, 2, 4]]
    path = tmp_path / "misaligned-patch-boxes.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match="align with candidate probabilities"):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


@pytest.mark.parametrize(
    "box, match",
    [
        (["0", 0, 4, 4], "real numeric"),
        ([0, 0, float("nan"), 4], "finite"),
        ([4, 0, 0, 4], "ordered"),
    ],
)
def test_report_rejects_invalid_ground_truth_boxes(tmp_path, box, match):
    records = _fixture_records()
    records[0]["views"]["CC"]["ground_truth"]["mass"] = [box]
    path = tmp_path / "invalid-ground-truth.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match=match):
        evaluate_prediction_artifact(path, bootstrap_samples=2)


def test_report_rejects_duplicate_generated_view_ids(tmp_path):
    records = _fixture_records()
    records.append(records[0])
    path = tmp_path / "duplicate-view-id.jsonl"
    write_prediction_artifact(path, records, metadata={})

    with pytest.raises(ValueError, match="Duplicate generated view_id"):
        evaluate_prediction_artifact(path, bootstrap_samples=2)
