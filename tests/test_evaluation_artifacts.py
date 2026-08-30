import json

import numpy as np
import pytest

from dagl_mil.evaluation.artifacts import (
    decode_binary_mask,
    encode_binary_mask,
    open_prediction_artifact,
    read_prediction_artifact,
    write_prediction_artifact,
)


def test_binary_mask_run_length_round_trip():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    mask[2, 3] = False
    assert np.array_equal(decode_binary_mask(encode_binary_mask(mask)), mask)


def test_prediction_artifact_round_trip_and_hash_guard(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = {
        "exam_id": "exam-1",
        "laterality": "L",
        "labels": [1, 0],
        "probabilities": [0.8, 0.1],
        "views": {"CC": {"candidate_probabilities": [[0.7, 0.2]]}},
    }
    summary = write_prediction_artifact(
        path, [record], metadata={"checkpoint_sha256": "abc"}
    )
    assert summary["records"] == 1
    header, records = read_prediction_artifact(path)
    assert header["metadata"]["checkpoint_sha256"] == "abc"
    assert list(records) == [record]

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_prediction_artifact(path)


def test_streaming_reader_validates_sidecar_record_count(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = {
        "exam_id": "exam-1",
        "laterality": "L",
        "labels": [1, 0],
        "probabilities": [0.8, 0.1],
        "views": {},
    }
    summary = write_prediction_artifact(path, [record], metadata={"test": True})
    sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["records"] = 2
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with open_prediction_artifact(path) as (header, records, info):
        assert header["metadata"] == {"test": True}
        assert info.sha256 == summary["sha256"]
        with pytest.raises(RuntimeError, match="record count mismatch"):
            list(records)


def test_streaming_reader_closes_handle_after_consumer_error(tmp_path):
    path = tmp_path / "predictions.jsonl"
    moved = tmp_path / "moved.jsonl"
    record = {
        "exam_id": "exam-1",
        "laterality": "L",
        "labels": [1, 0],
        "probabilities": [0.8, 0.1],
        "views": {},
    }
    write_prediction_artifact(path, [record], metadata={"test": True})

    with pytest.raises(RuntimeError, match="consumer failed"):
        with open_prediction_artifact(path) as (_, records, _):
            assert next(records) == record
            raise RuntimeError("consumer failed")

    path.rename(moved)
    assert moved.is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "wrong.jsonl", "sidecar path"),
        ("schema_version", "wrong-schema", "sidecar schema"),
        ("records", -1, "sidecar records"),
    ],
)
def test_streaming_reader_rejects_invalid_sidecar_metadata(
    tmp_path, field, value, message
):
    path = tmp_path / "predictions.jsonl"
    record = {
        "exam_id": "exam-1",
        "laterality": "L",
        "labels": [1, 0],
        "probabilities": [0.8, 0.1],
        "views": {},
    }
    write_prediction_artifact(path, [record], metadata={"test": True})
    sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = value
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        with open_prediction_artifact(path):
            pass
