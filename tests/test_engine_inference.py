from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import dagl_mil.engine.inference as inference


class _EmptyBackbone(torch.nn.Module):
    pass


class _StaticPredictionModel(torch.nn.Module):
    def __init__(self, candidate_records: list[dict]) -> None:
        super().__init__()
        self.candidate_records = candidate_records

    def forward(self, encoded):
        del encoded
        return {
            "breast_probabilities": torch.tensor([[0.25, 0.75]]),
            "candidate_records": self.candidate_records,
        }


def _view(view_name: str, count: int) -> dict:
    return {
        "image_id": f"image-{view_name.lower()}",
        "view_name": view_name,
        "view_id": {"CC": 0, "MLO": 1}[view_name],
        "image_shape": [8, 8],
        "coords": torch.tensor(
            [[float(index), 0.0, float(index + 2), 2.0] for index in range(count)]
        ),
        "mask": torch.ones(8, 8, dtype=torch.bool),
        "evaluation_metadata": {"bboxes": []},
    }


def _dual_view_batch() -> dict:
    # Deliberately use MLO-first order: bag identity, not an assumed CC-first
    # convention, must govern the model/data contract.
    return {
        "exam_id": ["exam-0"],
        "breast_id": ["breast-0"],
        "laterality": ["L"],
        "labels": torch.tensor([[0, 1]]),
        "views": [{"MLO": _view("MLO", 2), "CC": _view("CC", 1)}],
    }


def _candidate_records() -> list[dict]:
    return [
        {
            "sample_index": 0,
            "bag_index": 0,
            "view_index": 1,
            "indices": torch.tensor([0, 1]),
            "probabilities": torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
        },
        {
            "sample_index": 0,
            "bag_index": 1,
            "view_index": 0,
            "indices": torch.tensor([2]),
            "probabilities": torch.tensor([[0.3, 0.7]]),
        },
    ]


def _predict(monkeypatch, records: list[dict], batch: dict | None = None) -> list[dict]:
    monkeypatch.setattr(
        inference,
        "encode_collated_breasts",
        lambda *args, **kwargs: [{"views": {}}],
    )
    config = {
        "data": {"mean": 0.0, "std": 1.0},
        "training": {"mixed_precision": "none", "patch_micro_batch_size": 1},
        "model": {"name": "dagl_mil"},
    }
    return list(
        inference.iter_prediction_records(
            [_dual_view_batch() if batch is None else batch],
            model=_StaticPredictionModel(records),
            backbone=_EmptyBackbone(),
            config=config,
            device="cpu",
        )
    )


def test_inference_explicitly_pairs_a_dual_view_breast(monkeypatch) -> None:
    result = _predict(monkeypatch, _candidate_records())

    assert list(result[0]["views"]) == ["MLO", "CC"]
    assert result[0]["views"]["MLO"]["candidate_probabilities"] == [
        pytest.approx([0.1, 0.9]),
        pytest.approx([0.2, 0.8]),
    ]
    assert result[0]["views"]["CC"]["candidate_probabilities"] == [
        pytest.approx([0.3, 0.7])
    ]


def test_inference_rejects_candidate_record_order_mismatch(monkeypatch) -> None:
    records = list(reversed(_candidate_records()))

    with pytest.raises(inference.InferenceError, match="order/identity mismatch"):
        _predict(monkeypatch, records)


def test_inference_rejects_duplicate_model_view_identity(monkeypatch) -> None:
    records = _candidate_records()
    records[1]["view_index"] = 1

    with pytest.raises(inference.InferenceError, match="duplicate view identity"):
        _predict(monkeypatch, records)


def test_inference_rejects_duplicate_data_view_identity(monkeypatch) -> None:
    batch = _dual_view_batch()
    batch["views"] = [{"first": _view("CC", 2), "second": _view("CC", 1)}]

    with pytest.raises(inference.InferenceError, match="duplicate CC views"):
        _predict(monkeypatch, _candidate_records(), batch)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records[0].update(sample_index=1), "sample_index"),
        (lambda records: records[1].update(bag_index=0), "duplicate bag_index"),
        (
            lambda records: records[0].update(probabilities=torch.tensor([[0.1, 0.9]])),
            "probability count mismatch",
        ),
        (lambda records: records[0].update(indices=torch.tensor([0])), "index count"),
    ],
)
def test_inference_fails_closed_on_candidate_identity_or_count_corruption(
    monkeypatch, mutation, message
) -> None:
    records = deepcopy(_candidate_records())
    mutation(records)

    with pytest.raises(inference.InferenceError, match=message):
        _predict(monkeypatch, records)
