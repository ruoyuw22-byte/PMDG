from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dagl_mil.evaluation.artifacts import (
    encode_binary_mask,
    write_prediction_artifact,
)

from .feature_cache import FrozenGAPFeatureCache
from .features import build_evaluation_feature_cache, encode_collated_breasts


TARGET_CATEGORIES = {
    "mass": "Mass",
    "calcification": "Suspicious Calcification",
}
VIEW_TO_INDEX = {"CC": 0, "MLO": 1}


class InferenceError(RuntimeError):
    pass


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _emit_inference_progress(
    *,
    event: str,
    status: str,
    breasts: int,
    started: float,
    callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "event": event,
        "stage": "inference",
        "phase": "inference",
        "status": status,
        "timestamp_utc": _utc_now(),
        "breasts": int(breasts),
        "breasts_completed": int(breasts),
        "elapsed_seconds": elapsed,
        "breasts_per_second": int(breasts) / max(elapsed, 1e-9),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if callback is not None:
        callback(payload)


def _ground_truth(boxes: Sequence[Mapping[str, Any]]) -> dict[str, list[list[float]]]:
    result = {name: [] for name in TARGET_CATEGORIES}
    for box in boxes:
        categories = {str(value) for value in box.get("categories", [])}
        coordinates = [float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
        for name, official in TARGET_CATEGORIES.items():
            if official in categories:
                result[name].append(coordinates)
    return result


def _amp_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _integer_record_field(
    record: Mapping[str, Any], field: str, *, record_position: int
) -> int:
    if field not in record:
        raise InferenceError(
            f"Candidate record {record_position} is missing required field {field!r}"
        )
    value = record[field]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise InferenceError(
                f"Candidate record {record_position} field {field!r} must be scalar"
            )
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InferenceError(
            f"Candidate record {record_position} field {field!r} must be an integer"
        )
    return int(value)


def _data_view_index(
    view_key: Any,
    view: Mapping[str, Any],
    *,
    sample_index: int,
) -> int:
    declared: list[tuple[str, int]] = []
    if "view_name" in view:
        name = str(view["view_name"]).upper()
        if name not in VIEW_TO_INDEX:
            raise InferenceError(
                f"Data view {sample_index}/{view_key} has invalid view_name {name!r}"
            )
        declared.append(("view_name", VIEW_TO_INDEX[name]))
    if "view_id" in view:
        value = view["view_id"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise InferenceError(
                    f"Data view {sample_index}/{view_key} has non-scalar view_id"
                )
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InferenceError(
                f"Data view {sample_index}/{view_key} has non-integer view_id"
            )
        view_id = int(value)
        if view_id not in (0, 1):
            raise InferenceError(
                f"Data view {sample_index}/{view_key} has invalid view_id {view_id}"
            )
        declared.append(("view_id", view_id))

    key_name = str(view_key).upper()
    if key_name in VIEW_TO_INDEX:
        declared.append(("view key", VIEW_TO_INDEX[key_name]))
    if not declared:
        raise InferenceError(
            f"Data view {sample_index}/{view_key} has no canonical CC/MLO identity"
        )
    identities = {identity for _, identity in declared}
    if len(identities) != 1:
        details = ", ".join(f"{source}={identity}" for source, identity in declared)
        raise InferenceError(
            f"Data view {sample_index}/{view_key} has conflicting identities: {details}"
        )
    return identities.pop()


def _validated_candidate_view_pairs(
    candidate_records: Any,
    batch_views: Any,
) -> list[list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]]:
    """Pair model bags with data views only after validating their identities.

    Every formal model emits candidate records in canonical global bag order
    with ``sample_index``, ``bag_index``, ``view_index``, ``indices`` and
    ``probabilities``. Treating this metadata as a contract prevents a sorted
    ``zip`` from silently assigning one view's probabilities to another view.
    """

    if not isinstance(batch_views, Sequence) or isinstance(batch_views, (str, bytes)):
        raise InferenceError("Inference batch views must be a sequence")
    if not isinstance(candidate_records, Sequence) or isinstance(
        candidate_records, (str, bytes)
    ):
        raise InferenceError("Model candidate_records must be a sequence")

    expected: list[tuple[int, int, int, str, Mapping[str, Any], int, int]] = []
    pairs: list[list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = [
        [] for _ in batch_views
    ]
    next_bag = 0
    next_candidate = 0
    for sample_index, sample_views in enumerate(batch_views):
        if not isinstance(sample_views, Mapping) or not sample_views:
            raise InferenceError(f"Inference sample {sample_index} has no view bags")
        seen_data_views: set[int] = set()
        for view_key, view in sample_views.items():
            if not isinstance(view, Mapping):
                raise InferenceError(
                    f"Data view {sample_index}/{view_key} must be a mapping"
                )
            view_index = _data_view_index(view_key, view, sample_index=sample_index)
            if view_index in seen_data_views:
                raise InferenceError(
                    f"Inference sample {sample_index} contains duplicate "
                    f"{('CC', 'MLO')[view_index]} views"
                )
            seen_data_views.add(view_index)
            coords = view.get("coords")
            if (
                not isinstance(coords, torch.Tensor)
                or coords.ndim != 2
                or coords.shape[1] != 4
            ):
                raise InferenceError(
                    f"Data view {sample_index}/{view_key} coords must have shape [N,4]"
                )
            count = int(coords.shape[0])
            if count <= 0:
                raise InferenceError(
                    f"Data view {sample_index}/{view_key} has no candidates"
                )
            expected.append(
                (
                    sample_index,
                    next_bag,
                    view_index,
                    str(view_key),
                    view,
                    next_candidate,
                    count,
                )
            )
            next_bag += 1
            next_candidate += count

    if len(candidate_records) != len(expected):
        raise InferenceError(
            "Candidate/view bag mismatch: "
            f"model={len(candidate_records)}, data={len(expected)}"
        )

    parsed_identities: list[tuple[int, int, int]] = []
    seen_bags: set[int] = set()
    seen_model_views: set[tuple[int, int]] = set()
    for position, record in enumerate(candidate_records):
        if not isinstance(record, Mapping):
            raise InferenceError(f"Candidate record {position} must be a mapping")
        sample_index = _integer_record_field(
            record, "sample_index", record_position=position
        )
        bag_index = _integer_record_field(record, "bag_index", record_position=position)
        view_index = _integer_record_field(
            record, "view_index", record_position=position
        )
        if sample_index < 0 or sample_index >= len(batch_views):
            raise InferenceError(
                f"Candidate record {position} has out-of-range sample_index "
                f"{sample_index}"
            )
        if bag_index < 0:
            raise InferenceError(
                f"Candidate record {position} has negative bag_index {bag_index}"
            )
        if view_index not in (0, 1):
            raise InferenceError(
                f"Candidate record {position} has invalid view_index {view_index}"
            )
        if bag_index in seen_bags:
            raise InferenceError(
                f"Candidate records contain duplicate bag_index {bag_index}"
            )
        seen_bags.add(bag_index)
        sample_view = (sample_index, view_index)
        if sample_view in seen_model_views:
            raise InferenceError(
                "Candidate records contain duplicate view identity for "
                f"sample {sample_index}: {('CC', 'MLO')[view_index]}"
            )
        seen_model_views.add(sample_view)
        parsed_identities.append((sample_index, bag_index, view_index))

    for position, (record, actual, expected_record) in enumerate(
        zip(candidate_records, parsed_identities, expected, strict=True)
    ):
        (
            sample_index,
            bag_index,
            view_index,
            view_key,
            view,
            candidate_start,
            count,
        ) = expected_record
        expected_identity = (sample_index, bag_index, view_index)
        if actual != expected_identity:
            raise InferenceError(
                f"Candidate record order/identity mismatch at position {position}: "
                f"expected sample={sample_index}, bag={bag_index}, "
                f"view={view_index}; got sample={actual[0]}, bag={actual[1]}, "
                f"view={actual[2]}"
            )

        probabilities = record.get("probabilities")
        if not isinstance(probabilities, torch.Tensor) or probabilities.shape != (
            count,
            2,
        ):
            shape = (
                tuple(probabilities.shape)
                if isinstance(probabilities, torch.Tensor)
                else None
            )
            raise InferenceError(
                f"Candidate probability count mismatch for sample {sample_index}/"
                f"{view_key}: expected ({count}, 2), got {shape}"
            )
        indices = record.get("indices")
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.dtype != torch.long
            or indices.shape[0] != count
        ):
            raise InferenceError(
                f"Candidate index count mismatch for sample {sample_index}/{view_key}"
            )
        expected_indices = torch.arange(
            candidate_start,
            candidate_start + count,
            dtype=torch.long,
        )
        if not torch.equal(indices.detach().cpu(), expected_indices):
            raise InferenceError(
                f"Candidate indices disagree with bag {bag_index} for sample "
                f"{sample_index}/{view_key}"
            )
        pairs[sample_index].append((view_key, view, record))
    return pairs


def iter_prediction_records(
    loader: Iterable[Mapping[str, Any]],
    *,
    model: nn.Module,
    backbone: nn.Module,
    config: Mapping[str, Any],
    device: str | torch.device = "cuda",
    feature_cache: FrozenGAPFeatureCache | None = None,
    progress_interval: int = 100,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA inference was requested but CUDA is unavailable")
    model.to(resolved_device).eval()
    backbone.to(resolved_device).eval()
    backbone.requires_grad_(False)
    data_config = config["data"]
    training_config = config["training"]
    return_pyramid = str(config["model"]["name"]) == "fpn_set_transformer"
    breasts_completed = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            with _amp_context(
                resolved_device, str(training_config.get("mixed_precision", "fp16"))
            ):
                encoded = encode_collated_breasts(
                    batch,
                    backbone,
                    device=resolved_device,
                    micro_batch_size=int(training_config["patch_micro_batch_size"]),
                    mean=float(data_config["mean"]),
                    std=float(data_config["std"]),
                    return_pyramid=return_pyramid,
                    feature_cache=feature_cache,
                    deterministic_eval=True,
                )
                output = model(encoded)

            probabilities = output["breast_probabilities"].float().cpu().numpy()
            labels = torch.as_tensor(batch["labels"]).cpu().numpy()
            candidate_view_pairs = _validated_candidate_view_pairs(
                output.get("candidate_records"), batch.get("views")
            )
            for sample_index, sample_pairs in enumerate(candidate_view_pairs):
                serialized_views: dict[str, Any] = {}
                for view_key, view, candidate in sample_pairs:
                    candidate_probabilities = (
                        candidate["probabilities"].detach().float().cpu().numpy()
                    )
                    coords = torch.as_tensor(view["coords"]).cpu().numpy()
                    metadata = view.get("evaluation_metadata", {})
                    serialized_views[str(view_key)] = {
                        "image_id": str(view.get("image_id", "")),
                        "view_name": str(view.get("view_name", view_key)),
                        "mask_strategy": str(view.get("mask_strategy", "otsu_lcc")),
                        "image_shape": [int(value) for value in view["image_shape"]],
                        "patch_boxes": coords.tolist(),
                        "candidate_probabilities": candidate_probabilities.tolist(),
                        "breast_mask": encode_binary_mask(
                            torch.as_tensor(view["mask"]).cpu().numpy()
                        ),
                        "ground_truth": _ground_truth(metadata.get("bboxes", [])),
                    }
                yield {
                    "exam_id": str(batch["exam_id"][sample_index]),
                    "breast_id": str(batch["breast_id"][sample_index]),
                    "laterality": str(batch["laterality"][sample_index]),
                    "labels": labels[sample_index].astype(int).tolist(),
                    "probabilities": probabilities[sample_index].tolist(),
                    "views": serialized_views,
                }
                breasts_completed += 1
                if breasts_completed % progress_interval == 0:
                    _emit_inference_progress(
                        event="inference_progress",
                        status="running",
                        breasts=breasts_completed,
                        started=started,
                        callback=progress_callback,
                    )
    _emit_inference_progress(
        event="inference_complete",
        status="completed",
        breasts=breasts_completed,
        started=started,
        callback=progress_callback,
    )


def run_inference(
    loader: Iterable[Mapping[str, Any]],
    *,
    model: nn.Module,
    backbone: nn.Module,
    config: Mapping[str, Any],
    output_path: str | Path,
    metadata: Mapping[str, Any],
    device: str | torch.device = "cuda",
    progress_interval: int = 100,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed test checkpoint once and write one hash-guarded artifact."""

    provenance = metadata.get("provenance", {})
    backbone_sha256 = (
        provenance.get("backbone_sha256") if isinstance(provenance, Mapping) else None
    )
    manifest_sha256 = (
        provenance.get("manifest_sha256") if isinstance(provenance, Mapping) else None
    )
    feature_cache = (
        build_evaluation_feature_cache(
            config,
            backbone_sha256=str(backbone_sha256),
            manifest_sha256=(
                str(manifest_sha256) if isinstance(manifest_sha256, str) else None
            ),
            device=device,
        )
        if isinstance(backbone_sha256, str)
        else None
    )
    result = write_prediction_artifact(
        output_path,
        iter_prediction_records(
            loader,
            model=model,
            backbone=backbone,
            config=config,
            device=device,
            feature_cache=feature_cache,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        ),
        metadata=metadata,
    )
    if feature_cache is not None:
        result["evaluation_feature_cache"] = feature_cache.report()
    return result
