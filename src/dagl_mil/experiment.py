"""Restart-safe orchestration for the manuscript experiment protocol.

This module deliberately contains orchestration only.  The scientific work is
delegated to the data, model, engine, and evaluation packages so the command
line cannot silently replace a method or inject manuscript result numbers.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dagl_mil.config import load_config, save_resolved_config, validate_config
from dagl_mil.data import (
    DICOMPreprocessor,
    IntensityAugment,
    VinDrBreastDataset,
    build_vindr_manifest,
    load_manifest,
    manifest_fingerprint,
    split_counts,
    validate_split_counts,
)
from dagl_mil.data_gates import BBOX_BOUNDARY_PROTOCOL, require_passed_data_gate
from dagl_mil.engine.checkpoint import CHECKPOINT_FORMAT, load_training_checkpoint
from dagl_mil.engine.inference import run_inference
from dagl_mil.engine.trainer import Trainer
from dagl_mil.evaluation import (
    compare_prediction_artifacts,
    evaluate_prediction_artifact,
    measure_breast_pair_efficiency,
    select_representative_breast,
    write_evaluation_report,
)
from dagl_mil.evaluation.artifacts import (
    SCHEMA_VERSION,
    read_prediction_artifact,
    sha256_file,
)
from dagl_mil.models import build_backbone, build_model
from dagl_mil.reproducibility import (
    EpochSeededRandomSampler,
    build_run_provenance,
    environment_record,
    make_generator,
    seed_everything,
    seed_worker,
    sha256_mapping,
)


class ExperimentError(RuntimeError):
    """Raised when a requested run would violate the locked protocol."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dagl.yaml"

# This is an explicit protocol inventory.  A directory glob could silently add
# an exploratory configuration to the formal comparison table.
FORMAL_CONFIG_RELATIVE_PATHS = (
    "configs/dagl.yaml",
    "configs/baselines/abmil.yaml",
    "configs/baselines/set_transformer.yaml",
    "configs/baselines/fpn_set_transformer.yaml",
    "configs/baselines/patchgcn_mil.yaml",
    "configs/baselines/cross_view_transformer_mil.yaml",
    "configs/ablations/without_anatomy.yaml",
    "configs/ablations/without_prototype.yaml",
    "configs/ablations/without_graph.yaml",
    "configs/ablations/without_prototype_and_graph.yaml",
    "configs/ablations/without_cross_view.yaml",
    "configs/ablations/without_topk.yaml",
    "configs/ablations/without_diversity.yaml",
)
FORMAL_COMPARATOR_EXPERIMENTS = (
    "fpn_set_transformer",
    "patchgcn_mil",
    "cross_view_transformer_mil",
)
EVALUATION_ARTIFACT_FORMAT = "dagl-mil-evaluation-v1"
EFFICIENCY_ARTIFACT_FORMAT = "dagl-mil-efficiency-v1"
PAIRED_COMPARISON_ARTIFACT_FORMAT = "dagl-mil-paired-comparison-v1"
FORMAL_PAIRED_COMPARISON_SCOPE = "supplement-table-s2-three-comparators-v1"
EXPANDED_PAIRED_COMPARISON_SCOPE = "expanded-all-twelve-comparators-v1"
FORMAL_EVALUATION_COUNTS = {"breasts": 2000, "unique_examinations": 1000}


def formal_config_paths(project_root: str | Path = PROJECT_ROOT) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    return tuple(root / relative for relative in FORMAL_CONFIG_RELATIVE_PATHS)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _emit_stage_progress(
    stage: str,
    status: str,
    *,
    experiment: str | None = None,
    callback: Callable[[Mapping[str, Any]], None] | None = None,
    event: str = "stage_progress",
    **details: Any,
) -> dict[str, Any]:
    """Write one JSON-lines progress record and optionally persist it upstream."""

    payload: dict[str, Any] = {
        "event": event,
        "stage": stage,
        "status": status,
        "timestamp_utc": _utc_now(),
    }
    if experiment is not None:
        payload["experiment"] = experiment
    payload.update(details)
    print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)
    if callback is not None:
        callback(payload)
    return payload


def _run_observed_stage(
    stage: str,
    operation: Callable[[], dict[str, Any]],
    *,
    experiment: str,
    callback: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _emit_stage_progress(
        stage,
        "running",
        experiment=experiment,
        callback=callback,
        event="stage_started",
    )
    try:
        result = operation()
    except Exception as exc:
        _emit_stage_progress(
            stage,
            "failed",
            experiment=experiment,
            callback=callback,
            event="stage_failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _emit_stage_progress(
        stage,
        "completed",
        experiment=experiment,
        callback=callback,
        event="stage_completed",
        elapsed_seconds=time.perf_counter() - started,
        result_status=str(result.get("status", "completed")),
    )
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentError(f"Expected a JSON object: {path}")
    return value


def _experiment_name(config: Mapping[str, Any]) -> str:
    name = str(config["experiment"]["name"]).strip()
    if not name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in name
    ):
        raise ExperimentError(f"Unsafe experiment name: {name!r}")
    return name


def experiment_directory(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return (Path(config["paths"]["output_root"]) / _experiment_name(config)).resolve()


def require_formal_cuda(config: Mapping[str, Any]) -> torch.device:
    """Fail before formal computation if the local CUDA contract is unmet."""

    training = config["training"]
    if int(training["batch_size"]) != 4:
        raise ExperimentError("Formal training requires DataLoader batch_size=4")
    if int(training["gradient_accumulation"]) != 2:
        raise ExperimentError("Formal training requires gradient_accumulation=2")
    runtime = config.get("runtime", {})
    if str(runtime.get("device", "cuda")).lower() != "cuda":
        raise ExperimentError("Formal experiments must run on the local CUDA device")
    if not bool(runtime.get("require_cuda", True)):
        raise ExperimentError("Formal configs must set runtime.require_cuda=true")
    if not torch.cuda.is_available():
        raise ExperimentError(
            "A local CUDA GPU is required, but torch.cuda.is_available() is false"
        )
    return torch.device("cuda")


def _preprocessor(config: Mapping[str, Any], *, read_only: bool) -> DICOMPreprocessor:
    data = config["data"]
    percentiles = tuple(float(value) for value in data["intensity_percentiles"])
    if len(percentiles) != 2:
        raise ExperimentError("data.intensity_percentiles must contain two values")
    cache_dir = None
    if bool(data.get("cache_preprocessed_images", False)):
        cache_dir = config["paths"].get("preprocessing_cache")
    # Cached arrays precede stochastic augmentation, so all splits may safely
    # populate an atomic cache on a miss.  Read-only evaluation would otherwise
    # decode the same multi-hundred-MB DICOM again on every validation epoch.
    cache_read_only = bool(data.get("cache_read_only", False)) and read_only
    return DICOMPreprocessor(
        lower_percentile=percentiles[0],
        upper_percentile=percentiles[1],
        cache_dir=cache_dir,
        cache_read_only=cache_read_only,
    )


def _augmentation(config: Mapping[str, Any]) -> IntensityAugment:
    data = config["data"]
    return IntensityAugment(
        gamma_probability=float(data["gamma_probability"]),
        gamma_range=tuple(float(value) for value in data["gamma_range"]),
        contrast_probability=float(data["contrast_probability"]),
        contrast_range=tuple(float(value) for value in data["contrast_range"]),
    )


def build_dataset(
    config: Mapping[str, Any],
    manifest: str | Path | pd.DataFrame,
    *,
    split: str,
    include_bboxes: bool = False,
) -> VinDrBreastDataset:
    """Build one breast-level dataset with mode-safe access to bounding boxes."""

    training = split == "train"
    if training and include_bboxes:
        raise ExperimentError("Training must never receive localization boxes")
    data = config["data"]
    return VinDrBreastDataset(
        manifest,
        split=split,
        training=training,
        preprocessor=_preprocessor(config, read_only=not training),
        augmentation=_augmentation(config),
        patch_size=int(data["patch_size"]),
        train_stride=int(data["train_stride"]),
        eval_stride=int(data["eval_stride"]),
        minimum_breast_coverage=float(data["minimum_breast_coverage"]),
        augmentation_seed=int(config["training"]["seed"]),
        strict_two_view_pair=bool(data.get("strict_two_view_pair", False)),
        include_bboxes=include_bboxes,
    )


def build_loader(
    dataset: VinDrBreastDataset,
    config: Mapping[str, Any],
    *,
    training: bool,
) -> DataLoader:
    """Build the formal batch-4 train loader or memory-safe evaluation loader."""

    worker_key = "num_workers" if training else "eval_num_workers"
    workers = int(
        config["training"].get(worker_key, config["training"].get("num_workers", 0))
    )
    if workers < 0:
        raise ExperimentError(f"training.{worker_key} cannot be negative")
    batch_size = int(config["training"]["batch_size"]) if training else 1
    if training and batch_size != 4:
        raise ExperimentError("Formal training DataLoader must use batch_size=4")
    # flatten=False is intentional for both modes.  It avoids materialising a
    # giant RGB concatenation; the online backbone still micro-batches patches.
    collator = dataset.make_collator(flatten_patches=False)
    sampler = (
        EpochSeededRandomSampler(
            dataset,
            seed=int(config["training"]["seed"]),
        )
        if training
        else None
    )
    # The formal train and validation loaders coexist for the whole fit.  Do
    # not keep an idle worker pool alive across phases, and limit the active
    # pool to one prefetched batch per worker.  Sampling and augmentation are
    # epoch/index seeded, so these memory controls do not change data order or
    # tensors.
    worker_memory_options: dict[str, Any] = {}
    if workers > 0:
        # Training bags use stride 512 and need two queued batches to keep the
        # GPU fed.  Evaluation bags use stride 128 and can be much larger, so
        # one queued breast per worker is the safer bound there.
        worker_memory_options["prefetch_factor"] = 2 if training else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=make_generator(int(config["training"]["seed"])),
        collate_fn=collator,
        drop_last=False,
        **worker_memory_options,
    )


def _manifest_path(
    config: Mapping[str, Any], override: str | Path | None = None
) -> Path:
    return Path(override or config["paths"]["split_manifest"]).resolve()


def _load_protocol_manifest(
    config: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    expected_counts: bool = True,
) -> tuple[Path, pd.DataFrame]:
    path = _manifest_path(config, manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Run `dagl-mil prepare` first."
        )
    manifest = load_manifest(path, validate=True)
    if expected_counts:
        validate_split_counts(manifest)
    return path, manifest


def _current_run_provenance(
    config: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    provenance = build_run_provenance(
        manifest_path=manifest_path,
        backbone_path=config["paths"]["checkpoint"],
        config=config,
    )
    provenance["manifest_fingerprint"] = manifest_fingerprint(manifest)
    return provenance


def write_run_context(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the exact config, machine environment, and immutable hashes."""

    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if provenance is None:
        current = _current_run_provenance(
            config,
            manifest_path=manifest_path,
            manifest=load_manifest(manifest_path, validate=True),
        )
    else:
        current = dict(provenance)
    save_resolved_config(dict(config), destination / "resolved_config.yaml")
    _atomic_json(destination / "environment.json", environment_record())
    _atomic_json(destination / "provenance.json", current)
    return current


def prepare_experiment(
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create and lock the official seed-2026 breast manifest."""

    validate_config(dict(config))
    manifest_path = _manifest_path(config)
    anomalies_path = manifest_path.with_name(f"{manifest_path.stem}_anomalies.csv")
    if manifest_path.exists() and not overwrite:
        manifest = load_manifest(manifest_path, validate=True)
        validate_split_counts(manifest)
        status = "reused"
        anomalies = None
    else:
        built = build_vindr_manifest(
            config["paths"]["vindr_root"],
            seed=int(config["training"]["seed"]),
            assert_expected_counts=True,
            manifest_path=manifest_path,
            anomalies_path=anomalies_path,
        )
        manifest = built.manifest
        anomalies = int(len(built.anomalies))
        status = "created"
    provenance = write_run_context(
        config,
        run_dir=manifest_path.parent / "protocol",
        manifest_path=manifest_path,
    )
    summary = {
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": provenance["manifest_sha256"],
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "counts": split_counts(manifest),
        "anomalies_path": str(anomalies_path),
        "anomalies": anomalies,
    }
    _atomic_json(manifest_path.parent / "manifest_summary.json", summary)
    return summary


def _verified_sha_sidecar(path: Path) -> bool:
    sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = _read_json(sidecar_path)
        return (
            sidecar.get("path") == path.name
            and isinstance(sidecar.get("sha256"), str)
            and sha256_file(path) == sidecar["sha256"]
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False


def _evaluation_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "bootstrap_samples": int(evaluation["bootstrap_samples"]),
        "bootstrap_seed": int(evaluation["bootstrap_seed"]),
        "heatmap_quantile": float(evaluation["heatmap_quantile"]),
        "heatmap_fixed_threshold": float(evaluation["heatmap_fixed_threshold"]),
        "iou_threshold": float(evaluation["iou_threshold"]),
        "size_thresholds": [int(value) for value in evaluation["size_thresholds"]],
        "connected_components": 8,
        "minimum_box_area": None,
        "nms": False,
        "maximum_boxes": None,
    }


def _efficiency_completion_binding(
    report: Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
    report_sha256: str | None = None,
) -> dict[str, Any] | None:
    status = report.get("status")
    if status not in {"completed", "skipped_completed"}:
        return None
    protocol = report.get("protocol")
    representative = report.get("representative_bag")
    if not isinstance(protocol, Mapping) or not isinstance(representative, Mapping):
        return None
    resolved_path = report_path or report.get("report_path")
    resolved_sha256 = report_sha256 or report.get("report_sha256")
    if not isinstance(resolved_path, (str, Path)) or not isinstance(
        resolved_sha256, str
    ):
        return None
    return {
        "status": "completed",
        "report_path": str(Path(resolved_path).resolve()),
        "report_sha256": resolved_sha256,
        "protocol_sha256": sha256_mapping(dict(protocol)),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "prediction_sha256": representative.get("prediction_sha256"),
        "prediction_checkpoint_sha256": representative.get(
            "prediction_checkpoint_sha256"
        ),
        "representative_bag_sha256": sha256_mapping(dict(representative)),
    }


def _evaluation_completion_bindings(
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    protocol = report.get("protocol")
    prediction_sha256 = report.get("prediction_sha256")
    efficiency = report.get("efficiency")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(prediction_sha256, str)
        or not isinstance(efficiency, Mapping)
    ):
        return None
    if efficiency.get("status") == "not_requested":
        efficiency_binding: dict[str, Any] | None = {"status": "not_requested"}
    else:
        efficiency_binding = _efficiency_completion_binding(efficiency)
    if efficiency_binding is None:
        return None
    return {
        "prediction_sha256": prediction_sha256,
        "protocol_sha256": sha256_mapping(dict(protocol)),
        "efficiency": efficiency_binding,
    }


def _write_hashed_evaluation_report(path: Path, report: Mapping[str, Any]) -> str:
    bindings = _evaluation_completion_bindings(report)
    if bindings is None:
        raise ExperimentError(
            "Evaluation report lacks complete prediction/protocol/efficiency bindings"
        )
    write_evaluation_report(path, report)
    report_sha256 = sha256_file(path)
    # The sidecar is the completion marker. Both the report and marker are
    # atomically replaced, so interruption can only leave an unverifiable pair.
    _atomic_json(
        path.with_suffix(path.suffix + ".sha256.json"),
        {
            "path": path.name,
            "sha256": report_sha256,
            "artifact": EVALUATION_ARTIFACT_FORMAT,
            "bindings": bindings,
        },
    )
    return report_sha256


def _paired_comparison_bindings(
    report: Mapping[str, Any],
    *,
    scope: str | None = None,
    comparator_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    comparisons = report.get("comparisons")
    artifacts = report.get("artifacts")
    reference = report.get("reference")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        raise ExperimentError("Paired report comparisons must be a sequence")
    if not isinstance(artifacts, Mapping) or not isinstance(reference, str):
        raise ExperimentError("Paired report lacks reference/artifact bindings")

    resolved_scope = scope or report.get("scope")
    if not isinstance(resolved_scope, str) or not resolved_scope:
        raise ExperimentError("Paired report scope is required")
    if comparator_names is None:
        configured = report.get("comparator_names")
        if isinstance(configured, Sequence) and not isinstance(
            configured, (str, bytes)
        ):
            resolved_names = tuple(str(name) for name in configured)
        else:
            resolved_names = tuple(
                dict.fromkeys(str(row["comparator"]) for row in comparisons)
            )
    else:
        resolved_names = tuple(str(name) for name in comparator_names)
    if not resolved_names or len(set(resolved_names)) != len(resolved_names):
        raise ExperimentError("Paired report comparator names must be unique")

    expected_artifacts = {reference, *resolved_names}
    if set(str(name) for name in artifacts) != expected_artifacts:
        raise ExperimentError("Paired report artifact set does not match its scope")
    artifact_hashes: dict[str, str] = {}
    for name in (reference, *resolved_names):
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping) or not isinstance(
            binding.get("sha256"), str
        ):
            raise ExperimentError(f"Paired report artifact lacks SHA-256: {name}")
        artifact_hashes[name] = str(binding["sha256"])

    families: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in comparisons:
        if not isinstance(row, Mapping):
            raise ExperimentError("Paired comparison rows must be mappings")
        comparator = str(row.get("comparator"))
        if comparator not in resolved_names:
            raise ExperimentError(
                f"Paired comparison contains out-of-scope comparator: {comparator}"
            )
        if row.get("reference") != reference:
            raise ExperimentError("Paired comparison reference is inconsistent")
        family = (str(row.get("lesion")), str(row.get("metric")))
        families.setdefault(family, []).append(row)
    expected_families = {
        (lesion, metric)
        for lesion in ("mass", "calcification")
        for metric in ("roc_auc", "pr_auc")
    }
    if set(families) != expected_families or any(
        len(rows) != len(resolved_names) for rows in families.values()
    ):
        raise ExperimentError("Paired report must contain four complete Holm families")
    for rows in families.values():
        if {str(row["comparator"]) for row in rows} != set(resolved_names):
            raise ExperimentError("Paired report Holm family comparator set is incomplete")

    samples = report.get("bootstrap_samples")
    seed = report.get("bootstrap_seed")
    if not isinstance(samples, int) or samples <= 0 or not isinstance(seed, int):
        raise ExperimentError("Paired report bootstrap protocol is invalid")
    return {
        "scope": resolved_scope,
        "reference": reference,
        "comparator_names": list(resolved_names),
        "comparison_count": len(comparisons),
        "holm_family_count": len(families),
        "holm_family_size": len(resolved_names),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "artifact_sha256": artifact_hashes,
        "comparisons_sha256": sha256_mapping({"comparisons": list(comparisons)}),
    }


def _write_paired_comparison_sidecar(
    path: Path,
    report: Mapping[str, Any],
    *,
    scope: str | None = None,
    comparator_names: Sequence[str] | None = None,
) -> str:
    if not path.is_file():
        raise ExperimentError(f"Paired report does not exist: {path}")
    report_sha256 = sha256_file(path)
    _atomic_json(
        path.with_suffix(path.suffix + ".sha256.json"),
        {
            "path": path.name,
            "sha256": report_sha256,
            "artifact": PAIRED_COMPARISON_ARTIFACT_FORMAT,
            "bindings": _paired_comparison_bindings(
                report,
                scope=scope,
                comparator_names=comparator_names,
            ),
        },
    )
    return report_sha256


def _write_hashed_paired_comparison_report(
    path: Path,
    report: Mapping[str, Any],
) -> str:
    write_evaluation_report(path, report)
    return _write_paired_comparison_sidecar(path, report)


def _evaluation_complete(
    path: Path,
    *,
    prediction_path: Path,
    expected_protocol: Mapping[str, Any],
    expected_efficiency_protocol: Mapping[str, Any] | None,
    efficiency_report_path: Path | None,
    checkpoint_path: Path | None,
) -> bool:
    if not _verified_sha_sidecar(path) or not prediction_path.is_file():
        return False
    try:
        sidecar = _read_json(path.with_suffix(path.suffix + ".sha256.json"))
        if sidecar.get("artifact") != EVALUATION_ARTIFACT_FORMAT:
            return False
        report = _read_json(path)
        bindings = _evaluation_completion_bindings(report)
        prediction_sha256 = sha256_file(prediction_path)
        if (
            bindings is None
            or sidecar.get("bindings") != bindings
            or report.get("prediction_sha256") != prediction_sha256
            or report.get("protocol") != dict(expected_protocol)
        ):
            return False

        embedded_efficiency = report.get("efficiency")
        if not isinstance(embedded_efficiency, Mapping):
            return False
        if expected_efficiency_protocol is None:
            return embedded_efficiency.get("status") == "not_requested"
        if efficiency_report_path is None or checkpoint_path is None:
            return False
        if not checkpoint_path.is_file() or not _verified_sha_sidecar(
            efficiency_report_path
        ):
            return False
        efficiency_sidecar = _read_json(
            efficiency_report_path.with_suffix(
                efficiency_report_path.suffix + ".sha256.json"
            )
        )
        if efficiency_sidecar.get("artifact") != EFFICIENCY_ARTIFACT_FORMAT:
            return False
        efficiency_report = _read_json(efficiency_report_path)
        efficiency_sha256 = sha256_file(efficiency_report_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        representative = efficiency_report.get("representative_bag")
        if (
            efficiency_report.get("status") != "completed"
            or efficiency_report.get("protocol") != dict(expected_efficiency_protocol)
            or efficiency_report.get("checkpoint_sha256") != checkpoint_sha256
            or not isinstance(representative, Mapping)
            or representative.get("prediction_sha256") != prediction_sha256
            or representative.get("prediction_checkpoint_sha256") != checkpoint_sha256
        ):
            return False
        expected_efficiency_binding = _efficiency_completion_binding(
            efficiency_report,
            report_path=efficiency_report_path,
            report_sha256=efficiency_sha256,
        )
        return (
            expected_efficiency_binding is not None
            and bindings["efficiency"] == expected_efficiency_binding
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return False


def _training_complete(
    run_dir: Path,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> bool:
    required = (
        run_dir / "training_summary.json",
        run_dir / "best.pt",
        run_dir / "best.pt.sha256.json",
        run_dir / "last.pt",
        run_dir / "last.pt.sha256.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = _read_json(run_dir / "training_summary.json")
        if not {
            "completion_status",
            "provenance_sha256",
            "best_mean_validation_auc",
            "epochs_completed",
            "best_checkpoint",
            "last_checkpoint",
        }.issubset(summary):
            return False
        if summary["completion_status"] != "completed":
            return False
        if (
            Path(summary["best_checkpoint"]).resolve()
            != (run_dir / "best.pt").resolve()
        ):
            return False
        if (
            Path(summary["last_checkpoint"]).resolve()
            != (run_dir / "last.pt").resolve()
        ):
            return False
        persisted_provenance = _read_json(run_dir / "provenance.json")
        persisted_digest = sha256_mapping(persisted_provenance)
        if summary["provenance_sha256"] != persisted_digest:
            return False
        if expected_provenance is not None:
            expected = dict(expected_provenance)
            if persisted_provenance != expected:
                return False
            if summary["provenance_sha256"] != sha256_mapping(expected):
                return False
        best_sidecar = _read_json(run_dir / "best.pt.sha256.json")
        last_sidecar = _read_json(run_dir / "last.pt.sha256.json")
        for name, sidecar in (("best.pt", best_sidecar), ("last.pt", last_sidecar)):
            if sidecar.get("path") != name:
                return False
            if sidecar.get("format") != CHECKPOINT_FORMAT:
                return False
            if sidecar.get("provenance_sha256") != persisted_digest:
                return False
        if int(last_sidecar.get("epoch", -2)) + 1 != int(summary["epochs_completed"]):
            return False
        if int(best_sidecar.get("epoch", -1)) > int(last_sidecar.get("epoch", -1)):
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return _verified_sha_sidecar(run_dir / "best.pt") and _verified_sha_sidecar(
        run_dir / "last.pt"
    )


def _resolve_resume(run_dir: Path, resume: str | Path | bool | None) -> Path | None:
    if resume in (None, False):
        return None
    automatic = resume is True or str(resume) == "auto"
    candidate = run_dir / "last.pt" if automatic else Path(resume)
    if automatic and not candidate.is_file():
        return None
    if not candidate.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate.resolve()


def train_experiment(
    config: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume: str | Path | bool | None = None,
    skip_completed: bool = False,
    expected_counts: bool = True,
) -> dict[str, Any]:
    """Train one configuration with the frozen online Mammo-CLIP backbone."""

    validate_config(dict(config))
    device = require_formal_cuda(config)
    path, manifest = _load_protocol_manifest(
        config, manifest_path=manifest_path, expected_counts=expected_counts
    )
    run_dir = experiment_directory(config, output_dir=output_dir)
    provenance = _current_run_provenance(
        config,
        manifest_path=path,
        manifest=manifest,
    )
    if skip_completed and _training_complete(
        run_dir,
        expected_provenance=provenance,
    ):
        summary = _read_json(run_dir / "training_summary.json")
        return {**summary, "status": "skipped_completed", "run_dir": str(run_dir)}

    seed_everything(
        int(config["training"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", True)),
    )
    write_run_context(
        config,
        run_dir=run_dir,
        manifest_path=path,
        provenance=provenance,
    )
    resume_path = _resolve_resume(run_dir, resume)
    train_dataset = build_dataset(config, manifest, split="train", include_bboxes=False)
    validation_dataset = build_dataset(
        config, manifest, split="val", include_bboxes=False
    )
    train_loader = build_loader(train_dataset, config, training=True)
    validation_loader = build_loader(validation_dataset, config, training=False)
    backbone = build_backbone(config)
    model = build_model(config)
    trainer = Trainer(model=model, backbone=backbone, config=config, device=device)
    summary = trainer.fit(
        train_loader,
        validation_loader,
        output_dir=run_dir,
        provenance=provenance,
        resume=resume_path,
    )
    result = {**summary, "status": "completed", "run_dir": str(run_dir)}
    _atomic_json(run_dir / "train_command_summary.json", result)
    return result


def _prediction_complete(
    path: Path,
    *,
    expected_records: int,
    expected_checkpoint_sha256: str | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    expected_experiment: str | None = None,
    expected_annotation_protocol: str | None = None,
) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        sidecar_payload = _read_json(sidecar)
        if sidecar_payload.get("path") != path.name:
            return False
        if sidecar_payload.get("schema_version") != SCHEMA_VERSION:
            return False
        if int(sidecar_payload.get("records", -1)) != int(expected_records):
            return False
        header, records = read_prediction_artifact(path, verify_hash=True)
        metadata = header.get("metadata")
        if not isinstance(metadata, Mapping):
            return False
        if expected_checkpoint_sha256 is not None and metadata.get(
            "checkpoint_sha256"
        ) != str(expected_checkpoint_sha256):
            return False
        if expected_provenance is not None and metadata.get("provenance") != dict(
            expected_provenance
        ):
            return False
        if expected_experiment is not None and metadata.get("experiment") != str(
            expected_experiment
        ):
            return False
        if expected_annotation_protocol is not None and metadata.get(
            "evaluation_annotation_protocol"
        ) != str(expected_annotation_protocol):
            return False
        if any(
            (
                metadata.get("split") != "test",
                metadata.get("fixed_prediction_pass") is not True,
            )
        ) and any(
            value is not None
            for value in (
                expected_checkpoint_sha256,
                expected_provenance,
                expected_experiment,
                expected_annotation_protocol,
            )
        ):
            return False
        actual_records = sum(1 for _ in records)
        if actual_records != int(expected_records):
            return False
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        StopIteration,
        json.JSONDecodeError,
    ):
        return False
    return True


def infer_experiment(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prediction_path: str | Path | None = None,
    skip_completed: bool = False,
    expected_counts: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed test checkpoint once and persist hash-guarded predictions."""

    validate_config(dict(config))
    device = require_formal_cuda(config)
    path, manifest = _load_protocol_manifest(
        config, manifest_path=manifest_path, expected_counts=expected_counts
    )
    run_dir = experiment_directory(config, output_dir=output_dir)
    prediction = Path(prediction_path or run_dir / "test_predictions.jsonl").resolve()
    checkpoint = Path(checkpoint_path or run_dir / "best.pt").resolve()
    expected_test_records = int((manifest["split"].astype(str) == "test").sum())
    provenance = _current_run_provenance(
        config,
        manifest_path=path,
        manifest=manifest,
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    if skip_completed and _prediction_complete(
        prediction,
        expected_records=expected_test_records,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance=provenance,
        expected_experiment=_experiment_name(config),
        expected_annotation_protocol=BBOX_BOUNDARY_PROTOCOL,
    ):
        sidecar = _read_json(prediction.with_suffix(prediction.suffix + ".sha256.json"))
        return {**sidecar, "status": "skipped_completed", "path": str(prediction)}
    write_run_context(
        config,
        run_dir=run_dir,
        manifest_path=path,
        provenance=provenance,
    )
    seed_everything(
        int(config["training"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", True)),
    )
    test_dataset = build_dataset(config, manifest, split="test", include_bboxes=True)
    if not test_dataset.include_bboxes:
        raise ExperimentError("Test inference must include localization boxes")
    test_loader = build_loader(test_dataset, config, training=False)
    backbone = build_backbone(config).to(device)
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint,
        model=model,
        map_location=device,
        expected_provenance=provenance,
        restore_rng=False,
    )
    metadata = {
        "experiment": _experiment_name(config),
        "split": "test",
        "fixed_prediction_pass": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": provenance,
        "evaluation_annotation_protocol": BBOX_BOUNDARY_PROTOCOL,
    }
    result = run_inference(
        test_loader,
        model=model,
        backbone=backbone,
        config=config,
        output_path=prediction,
        metadata=metadata,
        device=device,
        progress_callback=progress_callback,
    )
    completed = {**result, "status": "completed"}
    _atomic_json(run_dir / "inference_summary.json", completed)
    return completed


def _efficiency_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    warmup = int(evaluation["timing_warmup"])
    repetitions = int(evaluation["timing_repetitions"])
    if warmup != 20 or repetitions != 100:
        raise ExperimentError(
            "Formal efficiency measurement requires 20 warm-ups and 100 repetitions"
        )
    if str(config["training"]["mixed_precision"]).lower() != "fp16":
        raise ExperimentError(
            "Formal efficiency measurement requires fp16 mixed precision"
        )
    return {
        "unit": "one breast-level CC/MLO pair",
        "batch_size": 1,
        "representative_bag": "test lower empirical median total candidate count",
        "timing_warmup": warmup,
        "timing_repetitions": repetitions,
        "synchronize_each_repetition": True,
        "mixed_precision": "fp16",
        "input_start": "standardized 512x512 patch tensors resident on GPU",
        "includes": "frozen EfficientNet-B2 and downstream MIL forward",
        "excludes": (
            "DICOM read, disk I/O, breast mask, patch extraction, normalization, "
            "and host-to-device transfer"
        ),
        "trainable_parameter_scope": "downstream MIL only; frozen B2 excluded",
        "memory_statistic": "torch CUDA peak allocated memory",
    }


def _trainable_parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """Report actual downstream trainable parameters without manuscript targets."""

    by_top_level_module: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name = name.split(".", maxsplit=1)[0]
        by_top_level_module[module_name] = (
            by_top_level_module.get(module_name, 0) + parameter.numel()
        )
    total = int(sum(by_top_level_module.values()))
    return {
        "scope": "downstream MIL model parameters with requires_grad=true",
        "frozen_backbone_excluded": True,
        "total_parameters": total,
        "total_millions": total / 1e6,
        "by_top_level_module": {
            name: {
                "parameters": int(count),
                "millions": int(count) / 1e6,
            }
            for name, count in sorted(by_top_level_module.items())
        },
    }


def measure_efficiency_experiment(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prediction_path: str | Path | None = None,
    report_path: str | Path | None = None,
    skip_completed: bool = False,
    expected_counts: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure one deterministic median-size test breast on the local RTX 4090."""

    validate_config(dict(config))
    experiment_name = _experiment_name(config)
    device = require_formal_cuda(config)
    device_name = torch.cuda.get_device_name(device)
    if "RTX 4090" not in device_name:
        raise ExperimentError(
            "The supplement efficiency protocol requires an RTX 4090; "
            f"current device is {device_name!r}"
        )
    protocol = _efficiency_protocol(config)
    path, manifest = _load_protocol_manifest(
        config, manifest_path=manifest_path, expected_counts=expected_counts
    )
    run_dir = experiment_directory(config, output_dir=output_dir)
    prediction = Path(prediction_path or run_dir / "test_predictions.jsonl").resolve()
    checkpoint = Path(checkpoint_path or run_dir / "best.pt").resolve()
    destination = Path(report_path or run_dir / "efficiency.json").resolve()
    selection_started = time.perf_counter()
    _emit_stage_progress(
        "efficiency.representative_selection",
        "running",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_started",
    )
    representative = select_representative_breast(prediction)
    representative_counts = representative.get("candidate_counts", {})
    representative_total = int(
        representative.get(
            "total_candidates",
            sum(int(value) for value in representative_counts.values()),
        )
    )
    _emit_stage_progress(
        "efficiency.representative_selection",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        elapsed_seconds=time.perf_counter() - selection_started,
        exam_id=representative["exam_id"],
        laterality=representative["laterality"],
        total_candidates=representative_total,
        eligible_breast_pairs=representative.get("eligible_breast_pairs"),
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    prediction_checkpoint = representative.get("prediction_checkpoint_sha256")
    if prediction_checkpoint is not None and prediction_checkpoint != checkpoint_sha256:
        raise ExperimentError(
            "Efficiency checkpoint differs from the checkpoint used for fixed predictions"
        )
    if skip_completed and _verified_sha_sidecar(destination):
        existing = _read_json(destination)
        if (
            existing.get("status") == "completed"
            and existing.get("protocol") == protocol
            and existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("representative_bag", {}).get("prediction_sha256")
            == representative["prediction_sha256"]
            and existing.get("representative_bag", {}).get("exam_id")
            == representative["exam_id"]
            and existing.get("representative_bag", {}).get("laterality")
            == representative["laterality"]
        ):
            _emit_stage_progress(
                "efficiency",
                "completed",
                experiment=experiment_name,
                callback=progress_callback,
                event="stage_completed",
                result_status="skipped_completed",
            )
            return {
                **existing,
                "status": "skipped_completed",
                "report_sha256": sha256_file(destination),
                "report_path": str(destination),
            }

    preparation_started = time.perf_counter()
    _emit_stage_progress(
        "efficiency.data_preparation",
        "running",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_started",
    )
    selected_rows = manifest.loc[
        (manifest["exam_id"].astype(str) == representative["exam_id"])
        & (manifest["laterality"].astype(str) == representative["laterality"])
        & (manifest["split"].astype(str) == "test")
    ].copy()
    if len(selected_rows) != 1:
        raise ExperimentError(
            "Representative prediction does not map to exactly one test manifest row"
        )
    dataset = build_dataset(config, selected_rows, split="test", include_bboxes=False)
    sample = dataset[0]
    actual_counts: dict[str, int] = {}
    for view in sample["views"].values():
        name = str(view["view_name"]).upper()
        if name in actual_counts:
            raise ExperimentError(f"Representative contains duplicate {name} views")
        actual_counts[name] = int(view["patches"].shape[0])
    if actual_counts != representative["candidate_counts"]:
        raise ExperimentError(
            "Regenerated representative candidate counts differ from fixed predictions: "
            f"expected={representative['candidate_counts']}, actual={actual_counts}"
        )
    batch = dataset.make_collator(flatten_patches=False)([sample])
    provenance = build_run_provenance(
        manifest_path=path,
        backbone_path=config["paths"]["checkpoint"],
        config=config,
    )
    provenance["manifest_fingerprint"] = manifest_fingerprint(manifest)
    backbone = build_backbone(config)
    model = build_model(config).to(device)
    load_training_checkpoint(
        checkpoint,
        model=model,
        map_location=device,
        expected_provenance=provenance,
        restore_rng=False,
    )
    parameter_report = _trainable_parameter_report(model)
    seed_everything(
        int(config["training"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", True)),
    )
    _emit_stage_progress(
        "efficiency.data_preparation",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        elapsed_seconds=time.perf_counter() - preparation_started,
        total_candidates=representative_total,
    )
    measurement_started = time.perf_counter()
    _emit_stage_progress(
        "efficiency.measurement",
        "running",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_started",
        timing_warmup=int(config["evaluation"]["timing_warmup"]),
        timing_repetitions=int(config["evaluation"]["timing_repetitions"]),
        total_candidates=representative_total,
    )
    measurement = measure_breast_pair_efficiency(
        model=model,
        backbone=backbone,
        batch=batch,
        mean=float(config["data"]["mean"]),
        std=float(config["data"]["std"]),
        mixed_precision=str(config["training"]["mixed_precision"]),
        micro_batch_size=int(config["training"]["patch_micro_batch_size"]),
        warmup=int(config["evaluation"]["timing_warmup"]),
        repetitions=int(config["evaluation"]["timing_repetitions"]),
        return_pyramid=str(config["model"]["name"]) == "fpn_set_transformer",
        device=device,
    )
    measured_parameters = int(measurement["trainable_parameters"])
    if measured_parameters != parameter_report["total_parameters"]:
        raise ExperimentError(
            "Efficiency parameter count does not match the actual downstream model: "
            f"measurement={measured_parameters}, "
            f"model={parameter_report['total_parameters']}"
        )
    parameter_report["measurement_matches_total"] = True
    _emit_stage_progress(
        "efficiency.measurement",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        elapsed_seconds=time.perf_counter() - measurement_started,
        latency_mean_ms=measurement.get(
            "latency_mean_ms", measurement.get("latency_ms")
        ),
        peak_gpu_memory_bytes=measurement["peak_gpu_memory_bytes"],
        trainable_parameters=measured_parameters,
    )
    report = {
        "status": "completed",
        "experiment": experiment_name,
        "model": str(config["model"]["name"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_path": str(path),
        "manifest_sha256": provenance["manifest_sha256"],
        "protocol": protocol,
        "representative_bag": representative,
        "measurement": measurement,
        "trainable_parameter_report": parameter_report,
        "environment": environment_record(),
        "flops_disclosure": {
            "value": None,
            "status": "not_measured",
            "reason": measurement["flops_reason"],
            "manuscript_values_copied": False,
        },
    }
    write_evaluation_report(destination, report)
    report_sha256 = sha256_file(destination)
    _atomic_json(
        destination.with_suffix(destination.suffix + ".sha256.json"),
        {
            "path": destination.name,
            "sha256": report_sha256,
            "artifact": EFFICIENCY_ARTIFACT_FORMAT,
        },
    )
    report["report_sha256"] = report_sha256
    report["report_path"] = str(destination)
    _emit_stage_progress(
        "efficiency",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        report_path=str(destination),
    )
    return report


def _require_formal_evaluation_counts(report: Mapping[str, Any]) -> None:
    actual = {name: report.get(name) for name in FORMAL_EVALUATION_COUNTS}
    valid = all(
        isinstance(actual[name], int)
        and not isinstance(actual[name], bool)
        and actual[name] == expected
        for name, expected in FORMAL_EVALUATION_COUNTS.items()
    )
    if not valid:
        raise ExperimentError(
            "Formal evaluation count mismatch: "
            f"expected={FORMAL_EVALUATION_COUNTS}, actual={actual}"
        )


def evaluate_experiment(
    config: Mapping[str, Any],
    *,
    prediction_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    skip_completed: bool = False,
    include_efficiency: bool = True,
    checkpoint_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    efficiency_report_path: str | Path | None = None,
    expected_counts: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed predictions and attach a separate efficiency measurement."""

    validate_config(dict(config))
    experiment_name = _experiment_name(config)
    run_dir = experiment_directory(config, output_dir=output_dir)
    prediction = Path(prediction_path or run_dir / "test_predictions.jsonl").resolve()
    destination = Path(report_path or run_dir / "evaluation.json").resolve()
    expected_protocol = _evaluation_protocol(config)
    expected_efficiency_protocol = (
        _efficiency_protocol(config) if include_efficiency else None
    )
    expected_efficiency_path = (
        Path(efficiency_report_path or run_dir / "efficiency.json").resolve()
        if include_efficiency
        else None
    )
    expected_checkpoint_path = (
        Path(checkpoint_path or run_dir / "best.pt").resolve()
        if include_efficiency
        else None
    )
    if skip_completed and _evaluation_complete(
        destination,
        prediction_path=prediction,
        expected_protocol=expected_protocol,
        expected_efficiency_protocol=expected_efficiency_protocol,
        efficiency_report_path=expected_efficiency_path,
        checkpoint_path=expected_checkpoint_path,
    ):
        report = _read_json(destination)
        if expected_counts:
            _require_formal_evaluation_counts(report)
        _emit_stage_progress(
            "evaluation",
            "completed",
            experiment=experiment_name,
            callback=progress_callback,
            event="stage_completed",
            result_status="skipped_completed",
            report_path=str(destination),
        )
        return {
            **report,
            "status": "skipped_completed",
            "report_sha256": sha256_file(destination),
            "report_path": str(destination),
        }
    evaluation = config["evaluation"]
    metrics_started = time.perf_counter()
    _emit_stage_progress(
        "evaluation.metrics_and_localization",
        "running",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_started",
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
    )
    report = evaluate_prediction_artifact(
        prediction,
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        heatmap_quantile=float(evaluation["heatmap_quantile"]),
        heatmap_fixed_threshold=float(evaluation["heatmap_fixed_threshold"]),
        iou_threshold=float(evaluation["iou_threshold"]),
        size_thresholds=tuple(int(value) for value in evaluation["size_thresholds"]),
    )
    if expected_counts:
        _require_formal_evaluation_counts(report)
    _emit_stage_progress(
        "evaluation.metrics_and_localization",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        elapsed_seconds=time.perf_counter() - metrics_started,
        prediction_sha256=report.get("prediction_sha256"),
    )
    if include_efficiency:
        report["efficiency"] = measure_efficiency_experiment(
            config,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            output_dir=run_dir,
            prediction_path=prediction,
            report_path=efficiency_report_path,
            skip_completed=skip_completed,
            expected_counts=expected_counts,
            progress_callback=progress_callback,
        )
    else:
        report["efficiency"] = {
            "status": "not_requested",
            "reason": "This evaluation invocation explicitly excluded GPU efficiency.",
        }
    report_sha256 = _write_hashed_evaluation_report(destination, report)
    completed = {
        **report,
        "status": "completed",
        "report_sha256": report_sha256,
        "report_path": str(destination),
    }
    _emit_stage_progress(
        "evaluation",
        "completed",
        experiment=experiment_name,
        callback=progress_callback,
        event="stage_completed",
        report_path=str(destination),
    )
    return completed


def _balanced_smoke_manifest(
    manifest: pd.DataFrame,
    *,
    rows_per_split: int = 4,
) -> pd.DataFrame:
    """Select real breasts covering all joint labels in every protocol split."""

    if rows_per_split < 4:
        raise ValueError("Smoke runs require at least four breasts per split")
    selected: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        split_rows = manifest.loc[manifest["split"] == split_name].copy()
        if "cross_view_valid" in split_rows:
            valid_pair = (
                split_rows["cross_view_valid"]
                .astype(str)
                .str.lower()
                .isin({"true", "1", "yes"})
            )
            split_rows = split_rows.loc[valid_pair].copy()
        pieces: list[pd.DataFrame] = []
        used: set[int] = set()
        for mass, calcification in ((0, 0), (1, 0), (0, 1), (1, 1)):
            matches = split_rows.loc[
                (split_rows["mass"].astype(int) == mass)
                & (split_rows["calcification"].astype(int) == calcification)
            ]
            if matches.empty:
                raise ExperimentError(
                    f"Smoke split {split_name!r} lacks joint label {mass}{calcification}"
                )
            row = matches.sort_values(["exam_id", "laterality"], kind="stable").iloc[
                [0]
            ]
            pieces.append(row)
            used.add(int(row.index[0]))
        if rows_per_split > 4:
            remaining = split_rows.loc[~split_rows.index.isin(used)].sort_values(
                ["exam_id", "laterality"], kind="stable"
            )
            pieces.append(remaining.iloc[: rows_per_split - 4])
        chosen = pd.concat(pieces, ignore_index=False).iloc[:rows_per_split]
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True).sort_values(
        ["split", "exam_id", "laterality"], kind="stable", ignore_index=True
    )


def smoke_experiment(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    skip_completed: bool = False,
    rows_per_split: int = 4,
) -> dict[str, Any]:
    """Exercise train, inference, and evaluation on a balanced real-data subset."""

    require_passed_data_gate(config)
    require_formal_cuda(config)
    full_path, manifest = _load_protocol_manifest(config, expected_counts=True)
    smoke = _balanced_smoke_manifest(manifest, rows_per_split=rows_per_split)
    smoke_config = copy.deepcopy(dict(config))
    original_name = _experiment_name(smoke_config)
    smoke_config["experiment"]["name"] = f"{original_name}_smoke"
    smoke_config["experiment"]["protocol_context"] = "smoke"
    smoke_config["training"]["epochs"] = 1
    smoke_config["training"]["early_stopping_patience"] = 1
    smoke_root = Path(config["paths"]["output_root"]) / "_smoke" / original_name
    smoke_manifest = smoke_root / "smoke_manifest.csv"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke.to_csv(smoke_manifest, index=False)
    smoke_config["paths"]["split_manifest"] = str(smoke_manifest)
    save_resolved_config(smoke_config, smoke_root / "resolved_config.yaml")
    training = train_experiment(
        smoke_config,
        manifest_path=smoke_manifest,
        output_dir=smoke_root,
        resume=True if resume else None,
        skip_completed=skip_completed,
        expected_counts=False,
    )
    inference = infer_experiment(
        smoke_config,
        checkpoint_path=smoke_root / "best.pt",
        manifest_path=smoke_manifest,
        output_dir=smoke_root,
        skip_completed=skip_completed,
        expected_counts=False,
    )
    evaluation = evaluate_experiment(
        smoke_config,
        output_dir=smoke_root,
        skip_completed=skip_completed,
        include_efficiency=False,
    )
    result = {
        "status": "completed",
        "source_manifest": str(full_path),
        "smoke_manifest": str(smoke_manifest),
        "counts": split_counts(smoke),
        "training": training,
        "inference": inference,
        "evaluation_report": evaluation.get(
            "report_path", str(smoke_root / "evaluation.json")
        ),
    }
    _atomic_json(smoke_root / "smoke_summary.json", result)
    return result


def run_pipeline(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    skip_completed: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run training, the single fixed prediction pass, and post-hoc evaluation."""

    run_dir = experiment_directory(config)
    experiment_name = _experiment_name(config)
    training = _run_observed_stage(
        "training",
        lambda: train_experiment(
            config,
            output_dir=run_dir,
            resume=True if resume else None,
            skip_completed=skip_completed,
        ),
        experiment=experiment_name,
        callback=progress_callback,
    )
    inference = _run_observed_stage(
        "inference",
        lambda: infer_experiment(
            config,
            output_dir=run_dir,
            skip_completed=skip_completed,
            progress_callback=progress_callback,
        ),
        experiment=experiment_name,
        callback=progress_callback,
    )
    evaluation = _run_observed_stage(
        "evaluation",
        lambda: evaluate_experiment(
            config,
            output_dir=run_dir,
            skip_completed=skip_completed,
            progress_callback=progress_callback,
        ),
        experiment=experiment_name,
        callback=progress_callback,
    )
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "training": training,
        "inference": inference,
        "evaluation_report": evaluation.get(
            "report_path", str(run_dir / "evaluation.json")
        ),
        "efficiency_report": evaluation.get("efficiency", {}).get("report_path"),
    }
    _atomic_json(run_dir / "pipeline_summary.json", result)
    return result


def compare_formal_predictions(
    configs: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pair DAGL against the three Supplement Table S2 comparators."""

    if len(configs) != 13:
        raise ExperimentError("Formal paired comparison requires all 13 configs")
    reference_config = configs[0]
    if str(reference_config["model"]["name"]) != "dagl_mil":
        raise ExperimentError("The first formal config must be DAGL-MIL")
    reference_name = _experiment_name(reference_config)
    if reference_name != "dagl_mil_full":
        raise ExperimentError("The formal reference experiment must be dagl_mil_full")
    config_by_name: dict[str, Mapping[str, Any]] = {}
    for config in configs:
        name = _experiment_name(config)
        if name in config_by_name:
            raise ExperimentError(f"Duplicate formal experiment name: {name}")
        config_by_name[name] = config
    missing = [
        name for name in FORMAL_COMPARATOR_EXPERIMENTS if name not in config_by_name
    ]
    if missing:
        raise ExperimentError(
            "Formal paired comparison is missing manuscript comparator(s): "
            + ", ".join(missing)
        )
    reference_path = experiment_directory(reference_config) / "test_predictions.jsonl"
    evaluation = reference_config["evaluation"]
    samples = int(evaluation["bootstrap_samples"])
    seed = int(evaluation["bootstrap_seed"])
    for config in configs:
        candidate_evaluation = config["evaluation"]
        if (
            int(candidate_evaluation["bootstrap_samples"]) != samples
            or int(candidate_evaluation["bootstrap_seed"]) != seed
        ):
            raise ExperimentError(
                "All formal configs must use the reference bootstrap samples/seed"
            )
    comparators: dict[str, Path] = {}
    for comparator_name in FORMAL_COMPARATOR_EXPERIMENTS:
        comparator_config = config_by_name[comparator_name]
        comparators[comparator_name] = (
            experiment_directory(comparator_config) / "test_predictions.jsonl"
        )
    comparison = compare_prediction_artifacts(
        reference_path,
        comparators,
        reference_name=reference_name,
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )
    report: dict[str, Any] = {
        "status": "completed",
        **comparison,
        "scope": FORMAL_PAIRED_COMPARISON_SCOPE,
        "comparator_names": list(FORMAL_COMPARATOR_EXPERIMENTS),
        "holm_family": (
            "three Supplement Table S2 comparators within each lesion/metric pair"
        ),
        "holm_family_size": len(FORMAL_COMPARATOR_EXPERIMENTS),
    }
    if output_path is None:
        output_path = (
            Path(reference_config["paths"]["output_root"])
            / "formal_paired_comparisons.json"
        )
    resolved_output = Path(output_path).resolve()
    report_sha256 = _write_hashed_paired_comparison_report(resolved_output, report)
    report["report_path"] = str(resolved_output)
    report["report_sha256"] = report_sha256
    return report


def run_all_experiments(
    *,
    config_paths: Sequence[str | Path] | None = None,
    overrides: Sequence[str] | None = None,
    resume: bool = False,
    skip_completed: bool = False,
) -> dict[str, Any]:
    """Run DAGL, five baselines, and seven ablations in a locked order."""

    paths = tuple(
        Path(path).resolve() for path in (config_paths or formal_config_paths())
    )
    if len(paths) != 13:
        raise ExperimentError(
            f"The formal matrix must contain 13 configs, got {len(paths)}"
        )
    configs = [load_config(path, list(overrides or ())) for path in paths]
    names = [_experiment_name(config) for config in configs]
    if len(set(names)) != 13:
        raise ExperimentError(f"Formal experiment names must be unique: {names}")
    data_gate = require_passed_data_gate(configs[0])
    # Check once before the first multi-hour run rather than after a partial matrix.
    for config in configs:
        require_formal_cuda(config)

    output_root = Path(configs[0]["paths"]["output_root"]).resolve()
    matrix_path = output_root / "formal_matrix_status.json"
    matrix_started_at = _utc_now()
    matrix: dict[str, Any] = {
        "status": "running",
        "started_at": matrix_started_at,
        "updated_at": matrix_started_at,
        "current_experiment": None,
        "current_stage": None,
        "configs": [str(path) for path in paths],
        "data_gate": {
            "path": str(
                Path(configs[0]["paths"]["output_root"]).resolve()
                / "data_gate_validation.json"
            ),
            "run_id": data_gate.get("run_id"),
            "manifest_sha256": data_gate.get("manifest_sha256"),
            "source_reports": data_gate.get("source_reports"),
        },
        "experiments": {},
        "stages": {},
    }
    _atomic_json(matrix_path, matrix)
    for path, config, name in zip(paths, configs, names, strict=True):
        experiment_started_at = _utc_now()
        experiment_record: dict[str, Any] = {
            "status": "running",
            "config": str(path),
            "started_at": experiment_started_at,
            "updated_at": experiment_started_at,
            "current_stage": None,
            "stages": {},
        }
        matrix["experiments"][name] = experiment_record
        matrix["current_experiment"] = name
        matrix["current_stage"] = None
        matrix["updated_at"] = experiment_started_at
        _atomic_json(matrix_path, matrix)

        def persist_experiment_progress(payload: Mapping[str, Any]) -> None:
            timestamp = str(payload.get("timestamp_utc") or _utc_now())
            stage = str(payload.get("stage") or "unknown")
            status = str(payload.get("status") or "running")
            event = str(payload.get("event") or "stage_progress")
            experiment_record["current_stage"] = stage
            experiment_record["updated_at"] = timestamp
            stage_record = experiment_record["stages"].setdefault(stage, {})
            stage_record["status"] = status
            stage_record["updated_at"] = timestamp
            stage_record["last_event"] = dict(payload)
            if event == "stage_started" and "started_at" not in stage_record:
                stage_record["started_at"] = timestamp
            elif status == "completed":
                stage_record["completed_at"] = timestamp
            elif status == "failed":
                stage_record["failed_at"] = timestamp
            matrix["current_experiment"] = name
            matrix["current_stage"] = stage
            matrix["updated_at"] = timestamp
            _atomic_json(matrix_path, matrix)

        try:
            result = run_pipeline(
                config,
                resume=resume,
                skip_completed=skip_completed,
                progress_callback=persist_experiment_progress,
            )
        except Exception as exc:
            failed_at = _utc_now()
            matrix["status"] = "failed"
            experiment_record["status"] = "failed"
            experiment_record["failed_at"] = failed_at
            experiment_record["updated_at"] = failed_at
            experiment_record["error"] = f"{type(exc).__name__}: {exc}"
            matrix["updated_at"] = failed_at
            matrix["failed_at"] = failed_at
            _atomic_json(matrix_path, matrix)
            raise
        completed_at = _utc_now()
        experiment_record["status"] = "completed"
        experiment_record["completed_at"] = completed_at
        experiment_record["updated_at"] = completed_at
        experiment_record["current_stage"] = None
        experiment_record["result"] = result
        matrix["current_stage"] = None
        matrix["updated_at"] = completed_at
        _atomic_json(matrix_path, matrix)

    comparison_started_at = _utc_now()
    matrix["current_experiment"] = None
    matrix["current_stage"] = "paired_comparisons"
    matrix["updated_at"] = comparison_started_at
    matrix["stages"]["paired_comparisons"] = {
        "status": "running",
        "started_at": comparison_started_at,
        "updated_at": comparison_started_at,
    }
    _emit_stage_progress(
        "paired_comparisons",
        "running",
        event="stage_started",
    )
    _atomic_json(matrix_path, matrix)
    try:
        paired_report = compare_formal_predictions(configs)
    except Exception as exc:
        failed_at = _utc_now()
        matrix["status"] = "failed"
        matrix["paired_comparisons"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        matrix["stages"]["paired_comparisons"].update(
            {
                "status": "failed",
                "failed_at": failed_at,
                "updated_at": failed_at,
            }
        )
        matrix["failed_at"] = failed_at
        matrix["updated_at"] = failed_at
        _emit_stage_progress(
            "paired_comparisons",
            "failed",
            event="stage_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        _atomic_json(matrix_path, matrix)
        raise
    comparison_completed_at = _utc_now()
    matrix["paired_comparisons"] = paired_report["report_path"]
    matrix["stages"]["paired_comparisons"].update(
        {
            "status": "completed",
            "completed_at": comparison_completed_at,
            "updated_at": comparison_completed_at,
            "report_path": paired_report["report_path"],
        }
    )
    _emit_stage_progress(
        "paired_comparisons",
        "completed",
        event="stage_completed",
        report_path=paired_report["report_path"],
    )
    matrix["efficiency"] = {
        "status": "completed_per_experiment",
        "representative_bag": "test lower empirical median total candidate count",
        "flops": "not_measured; see each efficiency.json disclosure",
    }
    matrix["status"] = "prediction_evaluation_completed"
    matrix["completed_at"] = comparison_completed_at
    matrix["updated_at"] = comparison_completed_at
    matrix["current_experiment"] = None
    matrix["current_stage"] = None
    _atomic_json(matrix_path, matrix)
    return matrix


def audit_experiments(
    *,
    config_paths: Sequence[str | Path] | None = None,
    overrides: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit reproducibility inputs and all formal configs without training."""

    paths = tuple(
        Path(path).resolve() for path in (config_paths or formal_config_paths())
    )
    configs: list[dict[str, Any]] = []
    config_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            config = load_config(path, list(overrides or ()))
            configs.append(config)
            config_records.append(
                {
                    "path": str(path),
                    "experiment": _experiment_name(config),
                    "model": str(config["model"]["name"]),
                    "valid": True,
                }
            )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            config_records.append({"path": str(path), "valid": False})
    if len(paths) != 13:
        errors.append(f"Expected 13 formal configs, found {len(paths)}")
    names = [record.get("experiment") for record in config_records if record["valid"]]
    if len(set(names)) != len(names):
        errors.append("Formal experiment names are not unique")

    manifest_record: dict[str, Any] = {}
    image_record: dict[str, Any] = {}
    backbone_record: dict[str, Any] = {}
    data_gate_record: dict[str, Any] = {}
    if configs:
        base = configs[0]
        try:
            data_gate = require_passed_data_gate(base)
            data_gate_record = {
                "status": "passed",
                "path": str(
                    Path(base["paths"]["output_root"]).resolve()
                    / "data_gate_validation.json"
                ),
                "run_id": data_gate.get("run_id"),
                "manifest_sha256": data_gate.get("manifest_sha256"),
                "source_reports": data_gate.get("source_reports"),
            }
        except Exception as exc:
            errors.append(f"Formal data gate: {type(exc).__name__}: {exc}")
            data_gate_record = {"status": "failed"}
        manifest_path = _manifest_path(base)
        try:
            manifest = load_manifest(manifest_path, validate=True)
            validate_split_counts(manifest)
            manifest_record = {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "fingerprint": manifest_fingerprint(manifest),
                "counts": split_counts(manifest),
            }
            image_paths: list[Path] = []
            for raw_views in manifest["views_json"]:
                views = json.loads(str(raw_views))
                if not isinstance(views, list):
                    raise ExperimentError("Manifest views_json must contain a list")
                image_paths.extend(
                    Path(str(view["path"]))
                    for view in views
                    if isinstance(view, Mapping) and str(view.get("path", ""))
                )
            missing_images = [str(path) for path in image_paths if not path.is_file()]
            image_record = {
                "referenced": len(image_paths),
                "present": len(image_paths) - len(missing_images),
                "missing": len(missing_images),
                "missing_examples": missing_images[:20],
            }
            if missing_images:
                errors.append(
                    f"Manifest references {len(missing_images)} missing DICOM files"
                )
        except Exception as exc:
            errors.append(f"Manifest: {type(exc).__name__}: {exc}")
            manifest_record = {"path": str(manifest_path), "valid": False}
        backbone_path = Path(base["paths"]["checkpoint"]).resolve()
        try:
            backbone_record = {
                "path": str(backbone_path),
                "sha256": sha256_file(backbone_path),
                "bytes": backbone_path.stat().st_size,
            }
            strict_backbone = build_backbone(base)
            if strict_backbone.training or any(
                parameter.requires_grad for parameter in strict_backbone.parameters()
            ):
                raise ExperimentError("Strict backbone is not frozen in eval mode")
            backbone_record["strict_load"] = "passed"
            backbone_record["parameters"] = int(
                sum(parameter.numel() for parameter in strict_backbone.parameters())
            )
            del strict_backbone
        except Exception as exc:
            errors.append(f"Backbone: {type(exc).__name__}: {exc}")
            backbone_record = {"path": str(backbone_path), "valid": False}
        for config in configs:
            try:
                require_formal_cuda(config)
            except Exception as exc:
                errors.append(
                    f"{_experiment_name(config)} runtime: {type(exc).__name__}: {exc}"
                )

    report = {
        "status": "passed" if not errors else "failed",
        "formal_config_count": len(paths),
        "configs": config_records,
        "environment": environment_record(),
        "manifest": manifest_record,
        "dicom_images": image_record,
        "data_gate": data_gate_record,
        "backbone": backbone_record,
        "efficiency": {
            "status": "not_run",
            "reason": "Audit does not define a representative variable-length bag.",
        },
        "errors": errors,
    }
    if output_path is None:
        root = (
            Path(configs[0]["paths"]["output_root"])
            if configs
            else PROJECT_ROOT / "outputs"
        )
        output_path = root / "audit.json"
    _atomic_json(output_path, report)
    report["report_path"] = str(Path(output_path).resolve())
    return report


__all__ = [
    "DEFAULT_CONFIG",
    "ExperimentError",
    "EXPANDED_PAIRED_COMPARISON_SCOPE",
    "FORMAL_COMPARATOR_EXPERIMENTS",
    "FORMAL_CONFIG_RELATIVE_PATHS",
    "FORMAL_PAIRED_COMPARISON_SCOPE",
    "PAIRED_COMPARISON_ARTIFACT_FORMAT",
    "PROJECT_ROOT",
    "audit_experiments",
    "build_dataset",
    "build_loader",
    "compare_formal_predictions",
    "evaluate_experiment",
    "experiment_directory",
    "formal_config_paths",
    "infer_experiment",
    "measure_efficiency_experiment",
    "prepare_experiment",
    "require_formal_cuda",
    "run_all_experiments",
    "run_pipeline",
    "smoke_experiment",
    "train_experiment",
    "write_run_context",
]
