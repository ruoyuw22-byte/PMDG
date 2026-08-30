from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler

import dagl_mil.experiment as experiment
from dagl_mil.config import load_config
from dagl_mil.evaluation.artifacts import write_prediction_artifact
from dagl_mil.reproducibility import EpochSeededRandomSampler


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.flatten_arguments: list[bool] = []

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return index

    def make_collator(self, *, flatten_patches: bool):
        self.flatten_arguments.append(flatten_patches)
        return lambda values: values


def _base_config(tmp_path: Path) -> dict:
    config = load_config(experiment.DEFAULT_CONFIG)
    config["paths"]["output_root"] = str(tmp_path / "outputs")
    config["paths"]["split_manifest"] = str(tmp_path / "manifest.csv")
    config["training"]["num_workers"] = 0
    config["training"]["eval_num_workers"] = 0
    return config


def test_formal_matrix_is_explicit_and_complete() -> None:
    paths = experiment.formal_config_paths()
    assert len(paths) == 13
    configs = [load_config(path) for path in paths]
    names = [config["experiment"]["name"] for config in configs]
    assert len(set(names)) == 13
    assert configs[0]["model"]["name"] == "dagl_mil"
    assert sum("baselines" in path.parts for path in paths) == 5
    assert sum("ablations" in path.parts for path in paths) == 7
    assert experiment.FORMAL_COMPARATOR_EXPERIMENTS == (
        "fpn_set_transformer",
        "patchgcn_mil",
        "cross_view_transformer_mil",
    )


def test_loader_locks_train_batch_and_never_flattens_eval(tmp_path) -> None:
    config = _base_config(tmp_path)
    config["training"]["num_workers"] = 3
    config["training"]["eval_num_workers"] = 1
    train_dataset = _TinyDataset()
    train_loader = experiment.build_loader(train_dataset, config, training=True)
    assert train_loader.batch_size == 4
    assert train_loader.num_workers == 3
    assert train_loader.prefetch_factor == 2
    assert train_loader.persistent_workers is False
    assert isinstance(train_loader.sampler, EpochSeededRandomSampler)
    assert train_dataset.flatten_arguments == [False]

    eval_dataset = _TinyDataset()
    eval_loader = experiment.build_loader(eval_dataset, config, training=False)
    assert eval_loader.batch_size == 1
    assert eval_loader.num_workers == 1
    assert eval_loader.prefetch_factor == 1
    assert eval_loader.persistent_workers is False
    assert isinstance(eval_loader.sampler, SequentialSampler)
    assert eval_dataset.flatten_arguments == [False]


def test_loader_with_zero_workers_does_not_enable_prefetch(tmp_path) -> None:
    config = _base_config(tmp_path)
    loader = experiment.build_loader(_TinyDataset(), config, training=False)

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False


def test_cuda_contract_checks_batch_accumulation_and_device(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert experiment.require_formal_cuda(config) == torch.device("cuda")

    changed = {**config, "training": {**config["training"], "batch_size": 8}}
    try:
        experiment.require_formal_cuda(changed)
    except experiment.ExperimentError as exc:
        assert "batch_size=4" in str(exc)
    else:
        raise AssertionError("batch-size violation was accepted")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        experiment.require_formal_cuda(config)
    except experiment.ExperimentError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("CPU formal run was accepted")


def test_balanced_smoke_manifest_covers_every_joint_label() -> None:
    records = []
    for split in ("train", "val", "test"):
        for index, (mass, calcification) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            records.append(
                {
                    "split": split,
                    "exam_id": f"{split}-{index}",
                    "laterality": "L",
                    "mass": mass,
                    "calcification": calcification,
                }
            )
    selected = experiment._balanced_smoke_manifest(pd.DataFrame(records))
    assert len(selected) == 12
    for _, rows in selected.groupby("split"):
        assert set(zip(rows["mass"], rows["calcification"], strict=True)) == {
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
        }


def test_run_context_writes_config_environment_and_provenance(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    monkeypatch.setattr(
        experiment,
        "build_run_provenance",
        lambda **_: {
            "manifest_sha256": "m",
            "backbone_sha256": "b",
            "manuscript_sha256": {"paper": "p", "supplement": "s"},
            "config_sha256": "c",
        },
    )
    monkeypatch.setattr(
        experiment,
        "environment_record",
        lambda: {"cuda_available": True, "cuda_devices": [{"name": "GPU"}]},
    )
    monkeypatch.setattr(
        experiment,
        "load_manifest",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "exam_id": ["e"],
                "laterality": ["L"],
                "mass": [0],
                "calcification": [0],
                "split": ["train"],
            }
        ),
    )
    monkeypatch.setattr(experiment, "manifest_fingerprint", lambda _: "fingerprint")
    run_dir = tmp_path / "run"
    provenance = experiment.write_run_context(
        config, run_dir=run_dir, manifest_path=tmp_path / "manifest.csv"
    )
    assert provenance["manifest_sha256"] == "m"
    assert (run_dir / "resolved_config.yaml").is_file()
    assert json.loads((run_dir / "environment.json").read_text())["cuda_available"]
    assert json.loads((run_dir / "provenance.json").read_text())["config_sha256"] == "c"
    assert provenance["manifest_fingerprint"] == "fingerprint"


def test_automatic_resume_starts_fresh_then_uses_last_checkpoint(tmp_path) -> None:
    assert experiment._resolve_resume(tmp_path, True) is None
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert experiment._resolve_resume(tmp_path, "auto") == checkpoint.resolve()


def test_inference_requests_test_boxes_and_hierarchical_loader(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unit")
    calls: dict[str, object] = {}

    class _Dataset:
        include_bboxes = True

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        experiment, "require_formal_cuda", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        experiment,
        "_load_protocol_manifest",
        lambda *args, **kwargs: (
            tmp_path / "manifest.csv",
            pd.DataFrame({"split": ["test"]}),
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_current_run_provenance",
        lambda *args, **kwargs: {"p": 1},
    )
    monkeypatch.setattr(
        experiment, "write_run_context", lambda *args, **kwargs: {"p": 1}
    )

    def fake_dataset(config, manifest, *, split, include_bboxes):
        calls["dataset"] = (split, include_bboxes)
        return _Dataset()

    def fake_loader(dataset, config, *, training):
        calls["loader"] = training
        return ["batch"]

    monkeypatch.setattr(experiment, "build_dataset", fake_dataset)
    monkeypatch.setattr(experiment, "build_loader", fake_loader)
    monkeypatch.setattr(experiment, "build_backbone", lambda _: _Module())
    monkeypatch.setattr(experiment, "build_model", lambda _: _Module())
    monkeypatch.setattr(
        experiment, "load_training_checkpoint", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(experiment, "sha256_file", lambda _: "hash")

    def fake_inference(loader, **kwargs):
        calls["inference_loader"] = loader
        calls["metadata"] = kwargs["metadata"]
        return {"path": str(kwargs["output_path"]), "records": 1, "sha256": "x"}

    monkeypatch.setattr(experiment, "run_inference", fake_inference)
    result = experiment.infer_experiment(
        config,
        checkpoint_path=checkpoint,
        output_dir=run_dir,
        expected_counts=False,
    )
    assert result["status"] == "completed"
    assert calls["dataset"] == ("test", True)
    assert calls["loader"] is False
    assert calls["inference_loader"] == ["batch"]
    assert (
        calls["metadata"]["evaluation_annotation_protocol"]
        == experiment.BBOX_BOUNDARY_PROTOCOL
    )


def test_completion_guards_verify_hash_and_exact_prediction_count(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    provenance = {
        "manifest_sha256": "manifest",
        "backbone_sha256": "backbone",
        "manuscript_sha256": {"paper": "paper", "supplement": "supplement"},
        "preprocessing_protocol": "unit-v2",
        "config_sha256": "config",
        "manifest_fingerprint": "fingerprint",
    }
    provenance_sha256 = experiment.sha256_mapping(provenance)
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "completion_status": "completed",
                "provenance_sha256": provenance_sha256,
                "best_mean_validation_auc": 0.5,
                "epochs_completed": 1,
                "best_checkpoint": str(run_dir / "best.pt"),
                "last_checkpoint": str(run_dir / "last.pt"),
            }
        ),
        encoding="utf-8",
    )
    for name in ("best.pt", "last.pt"):
        checkpoint = run_dir / name
        checkpoint.write_bytes(name.encode())
        (run_dir / f"{name}.sha256.json").write_text(
            json.dumps(
                {
                    "path": name,
                    "sha256": experiment.sha256_file(checkpoint),
                    "format": experiment.CHECKPOINT_FORMAT,
                    "epoch": 0,
                    "provenance_sha256": provenance_sha256,
                }
            ),
            encoding="utf-8",
        )
    assert experiment._training_complete(run_dir, expected_provenance=provenance)
    stale = {**provenance, "preprocessing_protocol": "unit-v1"}
    assert not experiment._training_complete(run_dir, expected_provenance=stale)
    (run_dir / "last.pt").write_bytes(b"tampered")
    assert not experiment._training_complete(run_dir)

    prediction = tmp_path / "predictions.jsonl"
    records = [
        {
            "exam_id": f"e{index}",
            "laterality": "L",
            "labels": [index, index],
            "probabilities": [float(index), float(index)],
            "views": {},
        }
        for index in (0, 1)
    ]
    prediction_metadata = {
        "experiment": "unit",
        "split": "test",
        "fixed_prediction_pass": True,
        "checkpoint_sha256": "checkpoint",
        "provenance": provenance,
        "evaluation_annotation_protocol": experiment.BBOX_BOUNDARY_PROTOCOL,
    }
    write_prediction_artifact(prediction, records, metadata=prediction_metadata)
    assert experiment._prediction_complete(
        prediction,
        expected_records=2,
        expected_checkpoint_sha256="checkpoint",
        expected_provenance=provenance,
        expected_experiment="unit",
        expected_annotation_protocol=experiment.BBOX_BOUNDARY_PROTOCOL,
    )
    assert not experiment._prediction_complete(
        prediction,
        expected_records=2,
        expected_checkpoint_sha256="stale-checkpoint",
        expected_provenance=provenance,
        expected_experiment="unit",
        expected_annotation_protocol=experiment.BBOX_BOUNDARY_PROTOCOL,
    )
    assert not experiment._prediction_complete(
        prediction,
        expected_records=2,
        expected_checkpoint_sha256="checkpoint",
        expected_provenance=stale,
        expected_experiment="unit",
        expected_annotation_protocol=experiment.BBOX_BOUNDARY_PROTOCOL,
    )
    assert not experiment._prediction_complete(
        prediction,
        expected_records=2,
        expected_checkpoint_sha256="checkpoint",
        expected_provenance=provenance,
        expected_experiment="unit",
        expected_annotation_protocol="stale-bbox-protocol",
    )
    assert not experiment._prediction_complete(prediction, expected_records=3)


def test_formal_paired_report_selects_supplement_comparators_and_hashes(
    tmp_path,
) -> None:
    configs = []
    records = []
    for exam in ("a", "b"):
        for laterality, label in (("L", 0), ("R", 1)):
            records.append(
                {
                    "exam_id": exam,
                    "laterality": laterality,
                    "labels": [label, label],
                    "probabilities": [0.1 + 0.8 * label, 0.1 + 0.8 * label],
                    "views": {},
                }
            )
    formal_names = (
        "dagl_mil_full",
        "abmil",
        "set_transformer",
        "fpn_set_transformer",
        "patchgcn_mil",
        "cross_view_transformer_mil",
        "dagl_without_anatomy",
        "dagl_without_prototype",
        "dagl_without_graph",
        "dagl_without_prototype_and_graph",
        "dagl_without_cross_view",
        "dagl_without_topk",
        "dagl_without_diversity",
    )
    for index, name in enumerate(formal_names):
        config = _base_config(tmp_path)
        config["experiment"]["name"] = name
        config["evaluation"]["bootstrap_samples"] = 10
        config["model"]["name"] = "dagl_mil" if index == 0 else "abmil"
        run_dir = experiment.experiment_directory(config)
        run_dir.mkdir(parents=True)
        comparator_records = []
        for record in records:
            changed = dict(record)
            if index:
                label = int(record["labels"][0])
                changed["probabilities"] = [
                    0.4 + 0.2 * label + index * 1e-5,
                    0.4 + 0.2 * label + index * 1e-5,
                ]
            comparator_records.append(changed)
        write_prediction_artifact(
            run_dir / "test_predictions.jsonl", comparator_records, metadata={}
        )
        configs.append(config)
    report = experiment.compare_formal_predictions(
        configs, output_path=tmp_path / "paired.json"
    )
    assert report["status"] == "completed"
    assert report["scope"] == experiment.FORMAL_PAIRED_COMPARISON_SCOPE
    assert report["comparator_names"] == list(
        experiment.FORMAL_COMPARATOR_EXPERIMENTS
    )
    assert report["holm_family_size"] == 3
    assert len(report["comparisons"]) == 3 * 2 * 2
    assert set(report["artifacts"]) == {
        "dagl_mil_full",
        *experiment.FORMAL_COMPARATOR_EXPERIMENTS,
    }
    assert all("p_value_holm" in row for row in report["comparisons"])
    families = {}
    for row in report["comparisons"]:
        families.setdefault((row["lesion"], row["metric"]), []).append(row)
    assert len(families) == 4
    assert all(len(rows) == 3 for rows in families.values())
    report_path = tmp_path / "paired.json"
    assert report_path.is_file()
    sidecar = json.loads(
        (tmp_path / "paired.json.sha256.json").read_text(encoding="utf-8")
    )
    assert sidecar["path"] == "paired.json"
    assert sidecar["sha256"] == experiment.sha256_file(report_path)
    assert sidecar["artifact"] == experiment.PAIRED_COMPARISON_ARTIFACT_FORMAT
    assert sidecar["bindings"]["scope"] == experiment.FORMAL_PAIRED_COMPARISON_SCOPE
    assert sidecar["bindings"]["comparator_names"] == list(
        experiment.FORMAL_COMPARATOR_EXPERIMENTS
    )
    assert sidecar["bindings"]["holm_family_size"] == 3


def _formal_configs_without_artifacts(tmp_path: Path) -> list[dict]:
    names = [
        "dagl_mil_full",
        "abmil",
        "set_transformer",
        "fpn_set_transformer",
        "patchgcn_mil",
        "cross_view_transformer_mil",
        "dagl_without_anatomy",
        "dagl_without_prototype",
        "dagl_without_graph",
        "dagl_without_prototype_and_graph",
        "dagl_without_cross_view",
        "dagl_without_topk",
        "dagl_without_diversity",
    ]
    configs = []
    for index, name in enumerate(names):
        config = _base_config(tmp_path)
        config["experiment"]["name"] = name
        config["model"]["name"] = "dagl_mil" if index == 0 else "abmil"
        configs.append(config)
    return configs


def test_formal_paired_report_fails_when_manuscript_comparator_is_missing(
    tmp_path,
) -> None:
    configs = _formal_configs_without_artifacts(tmp_path)
    configs[3]["experiment"]["name"] = "unexpected_comparator"
    with pytest.raises(experiment.ExperimentError, match="missing manuscript"):
        experiment.compare_formal_predictions(configs)


def test_formal_paired_report_fails_on_duplicate_experiment_name(tmp_path) -> None:
    configs = _formal_configs_without_artifacts(tmp_path)
    configs[-1]["experiment"]["name"] = configs[-2]["experiment"]["name"]
    with pytest.raises(experiment.ExperimentError, match="Duplicate"):
        experiment.compare_formal_predictions(configs)


def test_formal_paired_report_fails_on_bootstrap_protocol_drift(tmp_path) -> None:
    configs = _formal_configs_without_artifacts(tmp_path)
    configs[-1]["evaluation"]["bootstrap_seed"] += 1
    with pytest.raises(experiment.ExperimentError, match="bootstrap samples/seed"):
        experiment.compare_formal_predictions(configs)


def test_efficiency_experiment_records_protocol_without_copying_flops(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = pd.DataFrame(
        {
            "exam_id": ["e"],
            "laterality": ["L"],
            "split": ["test"],
            "mass": [0],
            "calcification": [0],
        }
    )

    class _Dataset:
        def __getitem__(self, index):
            assert index == 0
            patches = torch.zeros(2, 3, 2, 2)

            def view(name: str) -> dict:
                return {
                    "view_name": name,
                    "patches": patches,
                    "anatomy_base": torch.zeros(2, 5),
                }

            return {
                "exam_id": "e",
                "breast_id": "e_L",
                "laterality": "L",
                "labels": torch.zeros(2),
                "views": {"CC": view("CC"), "MLO": view("MLO")},
                "cross_view_valid": True,
            }

        def make_collator(self, *, flatten_patches):
            assert not flatten_patches
            return lambda samples: {
                "exam_id": ["e"],
                "breast_id": ["e_L"],
                "laterality": ["L"],
                "views": [samples[0]["views"]],
            }

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        experiment, "require_formal_cuda", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 4090"
    )
    monkeypatch.setattr(
        experiment,
        "_load_protocol_manifest",
        lambda *args, **kwargs: (tmp_path / "manifest.csv", manifest),
    )
    monkeypatch.setattr(
        experiment,
        "select_representative_breast",
        lambda _: {
            "exam_id": "e",
            "laterality": "L",
            "candidate_counts": {"CC": 2, "MLO": 2},
            "prediction_sha256": "prediction",
            "prediction_checkpoint_sha256": "checkpoint",
        },
    )
    monkeypatch.setattr(experiment, "sha256_file", lambda _: "checkpoint")
    monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(
        experiment,
        "build_run_provenance",
        lambda **kwargs: {
            "manifest_sha256": "manifest",
            "backbone_sha256": "backbone",
            "manuscript_sha256": {"paper": "p", "supplement": "s"},
        },
    )
    monkeypatch.setattr(experiment, "manifest_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(experiment, "build_backbone", lambda _: _Module())
    monkeypatch.setattr(experiment, "build_model", lambda _: _Module())
    monkeypatch.setattr(
        experiment, "load_training_checkpoint", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(experiment, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment,
        "measure_breast_pair_efficiency",
        lambda **kwargs: {
            "trainable_parameters": 1,
            "peak_gpu_memory_bytes": 456,
            "latency_ms": 7.0,
            "flops": None,
            "flops_reason": "unsupported operators",
        },
    )
    monkeypatch.setattr(
        experiment,
        "environment_record",
        lambda: {"cuda_available": True, "cuda_devices": [{"name": "RTX 4090"}]},
    )
    report = experiment.measure_efficiency_experiment(
        config, output_dir=run_dir, checkpoint_path=checkpoint, expected_counts=False
    )
    assert report["status"] == "completed"
    assert report["protocol"]["timing_warmup"] == 20
    assert report["protocol"]["timing_repetitions"] == 100
    assert report["measurement"]["trainable_parameters"] == 1
    parameter_report = report["trainable_parameter_report"]
    assert parameter_report["total_parameters"] == 1
    assert parameter_report["total_millions"] == 1e-6
    assert parameter_report["by_top_level_module"] == {
        "weight": {"parameters": 1, "millions": 1e-6}
    }
    assert parameter_report["frozen_backbone_excluded"] is True
    assert parameter_report["measurement_matches_total"] is True
    assert report["flops_disclosure"]["value"] is None
    assert not report["flops_disclosure"]["manuscript_values_copied"]
    saved = json.loads((run_dir / "efficiency.json").read_text(encoding="utf-8"))
    assert saved["flops_disclosure"]["status"] == "not_measured"
    sidecar = json.loads(
        (run_dir / "efficiency.json.sha256.json").read_text(encoding="utf-8")
    )
    assert sidecar["sha256"] == experiment.sha256_file(run_dir / "efficiency.json")


def test_evaluation_embeds_independent_efficiency_report(tmp_path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "evaluation-run"
    base_report = {
        "prediction_sha256": "prediction",
        "breasts": 2000,
        "unique_examinations": 1000,
        "protocol": {
            "bootstrap_samples": 2000,
            "bootstrap_seed": 2026,
            "heatmap_quantile": 0.95,
            "heatmap_fixed_threshold": 0.5,
            "iou_threshold": 0.25,
            "size_thresholds": [16384, 65536],
            "connected_components": 8,
            "minimum_box_area": None,
            "nms": False,
            "maximum_boxes": None,
        },
    }
    monkeypatch.setattr(
        experiment,
        "evaluate_prediction_artifact",
        lambda *args, **kwargs: dict(base_report),
    )
    monkeypatch.setattr(
        experiment,
        "measure_efficiency_experiment",
        lambda *args, **kwargs: {
            "status": "completed",
            "report_path": str(run_dir / "efficiency.json"),
            "report_sha256": "e" * 64,
            "checkpoint_sha256": "checkpoint",
            "protocol": experiment._efficiency_protocol(config),
            "representative_bag": {
                "exam_id": "e",
                "laterality": "L",
                "prediction_sha256": "prediction",
                "prediction_checkpoint_sha256": "checkpoint",
            },
            "measurement": {"latency_ms": 10.0, "flops": None},
        },
    )
    report = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=tmp_path / "fixed.jsonl",
    )
    assert report["efficiency"]["status"] == "completed"
    saved = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["efficiency"]["measurement"]["latency_ms"] == 10.0
    assert saved["efficiency"]["measurement"]["flops"] is None
    sidecar = json.loads(
        (run_dir / "evaluation.json.sha256.json").read_text(encoding="utf-8")
    )
    assert sidecar["artifact"] == experiment.EVALUATION_ARTIFACT_FORMAT
    assert sidecar["sha256"] == experiment.sha256_file(run_dir / "evaluation.json")
    assert sidecar["bindings"]["prediction_sha256"] == "prediction"


def test_evaluation_requires_formal_counts_but_allows_smoke_counts(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "evaluation-count-gate"
    prediction = tmp_path / "smoke-predictions.jsonl"
    prediction.write_bytes(b"smoke-predictions")

    monkeypatch.setattr(
        experiment,
        "evaluate_prediction_artifact",
        lambda path, **kwargs: {
            "prediction_sha256": experiment.sha256_file(path),
            "breasts": 4,
            "unique_examinations": 2,
            "protocol": experiment._evaluation_protocol(config),
        },
    )
    smoke = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
        expected_counts=False,
    )
    assert smoke["status"] == "completed"
    assert smoke["breasts"] == 4

    with pytest.raises(experiment.ExperimentError, match="count mismatch"):
        experiment.evaluate_experiment(
            config,
            output_dir=run_dir,
            prediction_path=prediction,
            include_efficiency=False,
            expected_counts=True,
            skip_completed=True,
        )


def test_evaluation_skip_requires_valid_hash_prediction_and_protocol_bindings(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "evaluation-bindings"
    prediction = tmp_path / "fixed.jsonl"
    prediction.write_bytes(b"fixed-predictions-v1")
    calls = 0

    def fake_evaluate(path, **kwargs):
        nonlocal calls
        calls += 1
        assert Path(path) == prediction
        return {
            "prediction_sha256": experiment.sha256_file(prediction),
            "breasts": 2000,
            "unique_examinations": 1000,
            "protocol": experiment._evaluation_protocol(config),
        }

    monkeypatch.setattr(experiment, "evaluate_prediction_artifact", fake_evaluate)
    first = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
    )
    assert first["status"] == "completed"
    assert calls == 1

    skipped = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
        skip_completed=True,
    )
    assert skipped["status"] == "skipped_completed"
    assert calls == 1

    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(
        evaluation_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
        skip_completed=True,
    )
    assert calls == 2

    prediction.write_bytes(b"fixed-predictions-v2")
    experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
        skip_completed=True,
    )
    assert calls == 3

    config["experiment"]["protocol_context"] = "test"
    config["evaluation"]["bootstrap_samples"] = 20
    experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        include_efficiency=False,
        skip_completed=True,
    )
    assert calls == 4


def test_evaluation_skip_requires_bound_efficiency_artifact(
    tmp_path, monkeypatch
) -> None:
    config = _base_config(tmp_path)
    run_dir = tmp_path / "evaluation-efficiency-binding"
    prediction = tmp_path / "fixed.jsonl"
    checkpoint = run_dir / "best.pt"
    prediction.write_bytes(b"fixed-predictions")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metric_calls = 0
    efficiency_calls = 0

    def fake_evaluate(path, **kwargs):
        nonlocal metric_calls
        metric_calls += 1
        return {
            "prediction_sha256": experiment.sha256_file(path),
            "breasts": 2000,
            "unique_examinations": 1000,
            "protocol": experiment._evaluation_protocol(config),
        }

    def write_efficiency_report(*args, **kwargs):
        nonlocal efficiency_calls
        efficiency_calls += 1
        efficiency_path = run_dir / "efficiency.json"
        prediction_sha256 = experiment.sha256_file(prediction)
        checkpoint_sha256 = experiment.sha256_file(checkpoint)
        payload = {
            "status": "completed",
            "checkpoint_sha256": checkpoint_sha256,
            "protocol": experiment._efficiency_protocol(config),
            "representative_bag": {
                "exam_id": "exam",
                "laterality": "L",
                "prediction_sha256": prediction_sha256,
                "prediction_checkpoint_sha256": checkpoint_sha256,
            },
            "measurement": {"latency_ms": 1.0},
        }
        experiment.write_evaluation_report(efficiency_path, payload)
        report_sha256 = experiment.sha256_file(efficiency_path)
        experiment._atomic_json(
            efficiency_path.with_suffix(".json.sha256.json"),
            {
                "path": efficiency_path.name,
                "sha256": report_sha256,
                "artifact": experiment.EFFICIENCY_ARTIFACT_FORMAT,
            },
        )
        return {
            **payload,
            "report_path": str(efficiency_path),
            "report_sha256": report_sha256,
        }

    monkeypatch.setattr(experiment, "evaluate_prediction_artifact", fake_evaluate)
    monkeypatch.setattr(
        experiment, "measure_efficiency_experiment", write_efficiency_report
    )
    first = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        checkpoint_path=checkpoint,
    )
    assert first["status"] == "completed"
    assert (metric_calls, efficiency_calls) == (1, 1)

    skipped = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        checkpoint_path=checkpoint,
        skip_completed=True,
    )
    assert skipped["status"] == "skipped_completed"
    assert (metric_calls, efficiency_calls) == (1, 1)

    efficiency_path = run_dir / "efficiency.json"
    stale = json.loads(efficiency_path.read_text(encoding="utf-8"))
    stale["representative_bag"]["prediction_sha256"] = "stale-prediction"
    experiment.write_evaluation_report(efficiency_path, stale)
    experiment._atomic_json(
        efficiency_path.with_suffix(".json.sha256.json"),
        {
            "path": efficiency_path.name,
            "sha256": experiment.sha256_file(efficiency_path),
            "artifact": experiment.EFFICIENCY_ARTIFACT_FORMAT,
        },
    )
    recomputed = experiment.evaluate_experiment(
        config,
        output_dir=run_dir,
        prediction_path=prediction,
        checkpoint_path=checkpoint,
        skip_completed=True,
    )
    assert recomputed["status"] == "completed"
    assert (metric_calls, efficiency_calls) == (2, 2)
