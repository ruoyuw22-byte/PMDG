from __future__ import annotations

import json
from pathlib import Path

import torch

import dagl_mil.engine.inference as inference
import dagl_mil.experiment as experiment


class _EmptyModule(torch.nn.Module):
    pass


def _inference_batch(index: int) -> dict:
    return {
        "exam_id": [f"exam-{index}"],
        "breast_id": [f"breast-{index}"],
        "laterality": ["L"],
        "labels": torch.tensor([[0, 1]]),
        "views": [
            {
                "CC": {
                    "image_id": f"image-{index}",
                    "view_name": "CC",
                    "image_shape": [2, 2],
                    "coords": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                    "mask": torch.ones(2, 2, dtype=torch.bool),
                    "evaluation_metadata": {"bboxes": []},
                }
            }
        ],
    }


class _PredictionModel(_EmptyModule):
    def forward(self, encoded):
        del encoded
        return {
            "breast_probabilities": torch.tensor([[0.25, 0.75]]),
            "candidate_records": [
                {
                    "sample_index": 0,
                    "view_index": 0,
                    "bag_index": 0,
                    "indices": torch.tensor([0]),
                    "probabilities": torch.tensor([[0.2, 0.8]]),
                }
            ],
        }


def test_inference_emits_json_progress_every_100_breasts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        inference,
        "encode_collated_breasts",
        lambda *args, **kwargs: [{}],
    )
    config = {
        "data": {"mean": 0.0, "std": 1.0},
        "training": {"mixed_precision": "none", "patch_micro_batch_size": 1},
        "model": {"name": "dagl_mil"},
    }
    events: list[dict] = []
    records = list(
        inference.iter_prediction_records(
            (_inference_batch(index) for index in range(101)),
            model=_PredictionModel(),
            backbone=_EmptyModule(),
            config=config,
            device="cpu",
            progress_interval=100,
            progress_callback=events.append,
        )
    )

    assert len(records) == 101
    assert [(event["event"], event["breasts"]) for event in events] == [
        ("inference_progress", 100),
        ("inference_complete", 101),
    ]
    assert all(event["stage"] == "inference" for event in events)
    assert all(event["timestamp_utc"].endswith("Z") for event in events)
    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]
    assert stdout_events == events


def test_run_pipeline_emits_stage_events_and_forwards_inference_progress(
    tmp_path, monkeypatch, capsys
) -> None:
    config = {
        "experiment": {"name": "observed"},
        "paths": {"output_root": str(tmp_path)},
    }
    callbacks: list[dict] = []

    monkeypatch.setattr(
        experiment,
        "train_experiment",
        lambda *args, **kwargs: {"status": "completed"},
    )

    def fake_inference(*args, **kwargs):
        kwargs["progress_callback"](
            {
                "event": "inference_progress",
                "stage": "inference",
                "status": "running",
                "timestamp_utc": "2026-08-27T00:00:00Z",
                "breasts": 100,
            }
        )
        return {"status": "completed", "records": 100}

    monkeypatch.setattr(experiment, "infer_experiment", fake_inference)
    monkeypatch.setattr(
        experiment,
        "evaluate_experiment",
        lambda *args, **kwargs: {
            "status": "completed",
            "report_path": str(tmp_path / "evaluation.json"),
            "efficiency": {"report_path": str(tmp_path / "efficiency.json")},
        },
    )

    result = experiment.run_pipeline(config, progress_callback=callbacks.append)

    assert result["status"] == "completed"
    stage_events = [
        (event["stage"], event["event"])
        for event in callbacks
        if event["event"].startswith("stage_")
    ]
    assert stage_events == [
        ("training", "stage_started"),
        ("training", "stage_completed"),
        ("inference", "stage_started"),
        ("inference", "stage_completed"),
        ("evaluation", "stage_started"),
        ("evaluation", "stage_completed"),
    ]
    assert any(event.get("breasts") == 100 for event in callbacks)
    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]
    assert len(stdout_events) == 6
    assert all(event["timestamp_utc"].endswith("Z") for event in stdout_events)


def test_run_all_persists_current_stage_and_timestamps(
    tmp_path, monkeypatch, capsys
) -> None:
    paths = [tmp_path / f"config-{index}.yaml" for index in range(13)]
    configs_by_path = {}
    for index, path in enumerate(paths):
        configs_by_path[path.resolve()] = {
            "experiment": {"name": f"experiment-{index}"},
            "paths": {"output_root": str(tmp_path / "outputs")},
            "model": {"name": "dagl_mil" if index == 0 else "abmil"},
        }

    monkeypatch.setattr(
        experiment,
        "load_config",
        lambda path, overrides: configs_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(experiment, "require_formal_cuda", lambda config: None)
    monkeypatch.setattr(
        experiment,
        "require_passed_data_gate",
        lambda config: {
            "run_id": "gate-run",
            "manifest_sha256": "manifest",
            "source_reports": {},
        },
    )

    def fake_pipeline(config, *, resume, skip_completed, progress_callback):
        del resume, skip_completed
        name = config["experiment"]["name"]
        for event, stage, status in (
            ("stage_started", "training", "running"),
            ("stage_completed", "training", "completed"),
            ("stage_started", "inference", "running"),
            ("inference_complete", "inference", "completed"),
            ("stage_started", "evaluation", "running"),
            ("stage_completed", "evaluation", "completed"),
        ):
            progress_callback(
                {
                    "event": event,
                    "stage": stage,
                    "status": status,
                    "experiment": name,
                    "timestamp_utc": "2026-08-27T00:00:00Z",
                    "breasts": 100 if stage == "inference" else None,
                }
            )
        return {"status": "completed", "run_dir": str(tmp_path / name)}

    monkeypatch.setattr(experiment, "run_pipeline", fake_pipeline)
    paired_path = tmp_path / "outputs" / "formal_paired_comparisons.json"
    monkeypatch.setattr(
        experiment,
        "compare_formal_predictions",
        lambda configs: {"status": "completed", "report_path": str(paired_path)},
    )

    result = experiment.run_all_experiments(config_paths=paths)

    assert result["status"] == "prediction_evaluation_completed"
    assert result["current_experiment"] is None
    assert result["current_stage"] is None
    assert result["started_at"].endswith("Z")
    assert result["completed_at"].endswith("Z")
    assert result["stages"]["paired_comparisons"]["status"] == "completed"
    for record in result["experiments"].values():
        assert record["status"] == "completed"
        assert record["current_stage"] is None
        assert record["started_at"].endswith("Z")
        assert record["completed_at"].endswith("Z")
        assert record["stages"]["training"]["status"] == "completed"
        assert record["stages"]["inference"]["status"] == "completed"
        assert record["stages"]["inference"]["last_event"]["breasts"] == 100
        assert record["stages"]["evaluation"]["status"] == "completed"
    saved = json.loads(
        (tmp_path / "outputs" / "formal_matrix_status.json").read_text(encoding="utf-8")
    )
    assert saved == result
    stdout_events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line
    ]
    assert [event["event"] for event in stdout_events] == [
        "stage_started",
        "stage_completed",
    ]
