from __future__ import annotations

import pytest
import torch

from dagl_mil.evaluation.artifacts import write_prediction_artifact
from dagl_mil.evaluation.efficiency import (
    forward_standardized_breast,
    measure_breast_pair_efficiency,
    measure_cuda_inference,
    prepare_standardized_breast,
    select_representative_breast,
    trainable_parameter_count,
)


def _view(name: str, count: int) -> dict:
    return {
        "view_name": name,
        "patch_boxes": [[0, 0, 2, 2] for _ in range(count)],
        "candidate_probabilities": [[0.1, 0.2] for _ in range(count)],
    }


def test_representative_is_observed_lower_median_valid_pair(tmp_path) -> None:
    records = []
    for index, total in enumerate((2, 4, 6, 8)):
        records.append(
            {
                "exam_id": f"e{index}",
                "breast_id": f"e{index}_L",
                "laterality": "L",
                "labels": [0, 0],
                "probabilities": [0.1, 0.2],
                "views": {"CC": _view("CC", total // 2), "MLO": _view("MLO", total // 2)},
            }
        )
    records.append(
        {
            "exam_id": "invalid",
            "laterality": "L",
            "labels": [0, 0],
            "probabilities": [0.1, 0.2],
            "views": {"CC": _view("CC", 99)},
        }
    )
    path = tmp_path / "predictions.jsonl"
    write_prediction_artifact(
        path, records, metadata={"checkpoint_sha256": "checkpoint"}
    )
    selected = select_representative_breast(path)
    assert selected["exam_id"] == "e1"
    assert selected["candidate_counts"] == {"CC": 2, "MLO": 2}
    assert selected["total_candidates"] == 4
    assert selected["rank_zero_based"] == 1
    assert selected["eligible_breast_pairs"] == 4
    assert selected["candidate_count_distribution"] == {
        "minimum": 2,
        "lower_median": 4,
        "mathematical_median": 5.0,
        "maximum": 8,
    }
    assert selected["prediction_checkpoint_sha256"] == "checkpoint"
    assert len(selected["prediction_sha256"]) == 64


class _Backbone(torch.nn.Module):
    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        del micro_batch_size
        assert not return_pyramid
        means = patches.mean(dim=(1, 2, 3))
        return torch.stack((means, means, means, means), dim=1)


class _Model(torch.nn.Module):
    def forward(self, samples):
        assert "labels" not in samples[0]
        features = torch.cat(
            [view["features"] for view in samples[0]["views"].values()], dim=0
        )
        probability = torch.sigmoid(features.mean()).expand(1, 2)
        return {"breast_probabilities": probability}


class _PyramidBackbone(torch.nn.Module):
    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        del micro_batch_size
        assert return_pyramid
        values = patches.mean(dim=1, keepdim=True)
        return {
            8: values,
            16: torch.nn.functional.avg_pool2d(values, 2),
            32: torch.nn.functional.avg_pool2d(values, 2),
        }


class _PyramidModel(torch.nn.Module):
    def forward(self, samples):
        for view in samples[0]["views"].values():
            assert set(view["pyramid"]) == {8, 16, 32}
            assert view["pyramid"][8].shape[0] == 1
        return {"breast_probabilities": torch.ones(1, 2)}


def _batch() -> dict:
    grayscale = torch.tensor([[[[0.25, 0.5], [0.75, 1.0]]]])
    patches = grayscale.expand(-1, 3, -1, -1)

    def view(name: str) -> dict:
        return {
            "view_name": name,
            "patches": patches,
            "anatomy_base": torch.zeros(1, 5),
        }

    return {
        "exam_id": ["e"],
        "breast_id": ["e_L"],
        "laterality": ["L"],
        "views": [{"CC": view("CC"), "MLO": view("MLO")}],
    }


def test_standardized_forward_starts_after_normalization_and_omits_labels() -> None:
    prepared = prepare_standardized_breast(
        _batch(), device="cpu", mean=0.5, std=0.25
    )
    assert prepared["total_candidates"] == 2
    torch.testing.assert_close(
        prepared["views"]["CC"]["patches"][0, 0],
        torch.tensor([[-1.0, 0.0], [1.0, 2.0]]),
    )
    probabilities = forward_standardized_breast(
        prepared,
        model=_Model(),
        backbone=_Backbone(),
        micro_batch_size=1,
    )
    assert probabilities.shape == (1, 2)
    pyramid_probabilities = forward_standardized_breast(
        prepared,
        model=_PyramidModel(),
        backbone=_PyramidBackbone(),
        micro_batch_size=1,
        return_pyramid=True,
    )
    assert pyramid_probabilities.shape == (1, 2)


def test_synchronized_cuda_measurement_reports_distribution(monkeypatch) -> None:
    module = torch.nn.Linear(2, 2)
    calls = {"forward": 0, "sync": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: calls.__setitem__("sync", calls["sync"] + 1),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 300)
    clock = iter((0, 1_000_000, 2_000_000, 4_000_000))
    monkeypatch.setattr("dagl_mil.evaluation.efficiency.time.perf_counter_ns", lambda: next(clock))

    def forward():
        calls["forward"] += 1

    result = measure_cuda_inference(
        module, forward, warmup=1, repetitions=2, device="cuda"
    )
    assert calls == {"forward": 3, "sync": 5}
    assert result["latency_mean_ms"] == 1.5
    assert result["latency_median_ms"] == 1.5
    assert result["latency_std_ms"] == 0.5
    assert result["peak_gpu_memory_bytes"] == 300
    assert result["trainable_parameters"] == trainable_parameter_count(module) == 6
    assert result["flops"] is None
    assert result["flops_status"] == "not_measured"


def test_cuda_measurement_rejects_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        measure_cuda_inference(torch.nn.Linear(1, 1), lambda: None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local CUDA smoke only")
def test_tiny_end_to_end_efficiency_forward_runs_on_local_gpu() -> None:
    result = measure_breast_pair_efficiency(
        model=_Model(),
        backbone=_Backbone(),
        batch=_batch(),
        mean=0.5,
        std=0.25,
        mixed_precision="fp16",
        micro_batch_size=1,
        warmup=1,
        repetitions=2,
        device="cuda",
    )
    assert result["batch_size"] == 1
    assert result["total_candidates"] == 2
    assert result["latency_mean_ms"] > 0
    assert result["peak_gpu_memory_bytes"] > 0
    assert "RTX 4090" in result["cuda_device"]
