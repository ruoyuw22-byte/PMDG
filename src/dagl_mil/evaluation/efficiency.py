"""Reproducible local-GPU efficiency measurements for a breast pair.

The supplement fixes the hardware/timing scope but does not identify a
representative variable-size bag. We therefore choose the lower empirical
median of total candidate count among valid test CC/MLO pairs. The chosen
breast and prediction-artifact hash are persisted with every measurement.

DICOM decoding, masking, patch extraction, normalization, and host-to-device
copy occur before timing. Timed forwards start from standardized GPU patch
tensors and include the frozen EfficientNet-B2 plus the downstream MIL model.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .artifacts import read_prediction_artifact, sha256_file


class EfficiencyMeasurementError(RuntimeError):
    """Raised when an efficiency result would not follow the locked scope."""


def trainable_parameter_count(module: nn.Module) -> int:
    """Count downstream parameters updated by the optimizer."""

    return int(
        sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    )


def _valid_pair_counts(record: Mapping[str, Any]) -> dict[str, int] | None:
    views = record.get("views")
    if not isinstance(views, Mapping):
        return None
    by_name: dict[str, list[int]] = {"CC": [], "MLO": []}
    for view in views.values():
        if not isinstance(view, Mapping):
            return None
        view_name = str(view.get("view_name", "")).upper()
        if view_name not in by_name:
            continue
        boxes = view.get("patch_boxes")
        probabilities = view.get("candidate_probabilities")
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            return None
        if not isinstance(probabilities, Sequence) or isinstance(
            probabilities, (str, bytes)
        ):
            return None
        if len(boxes) != len(probabilities) or not boxes:
            raise EfficiencyMeasurementError(
                f"Candidate arrays do not align for {record.get('exam_id')} "
                f"{record.get('laterality')} {view_name}"
            )
        by_name[view_name].append(len(boxes))
    if len(by_name["CC"]) != 1 or len(by_name["MLO"]) != 1:
        return None
    return {"CC": by_name["CC"][0], "MLO": by_name["MLO"][0]}


def select_representative_breast(
    prediction_path: str | Path,
) -> dict[str, Any]:
    """Select the lower-median test CC/MLO pair by total candidate count."""

    header, iterator = read_prediction_artifact(prediction_path, verify_hash=True)
    candidates: list[dict[str, Any]] = []
    for record in iterator:
        counts = _valid_pair_counts(record)
        if counts is None:
            continue
        candidates.append(
            {
                "exam_id": str(record["exam_id"]),
                "breast_id": str(
                    record.get(
                        "breast_id",
                        f"{record['exam_id']}_{record['laterality']}",
                    )
                ),
                "laterality": str(record["laterality"]),
                "candidate_counts": counts,
                "total_candidates": int(counts["CC"] + counts["MLO"]),
            }
        )
    if not candidates:
        raise EfficiencyMeasurementError(
            "Prediction artifact contains no valid single-CC/single-MLO breast pair"
        )
    candidates.sort(
        key=lambda row: (
            int(row["total_candidates"]),
            str(row["exam_id"]),
            str(row["laterality"]),
        )
    )
    rank_zero_based = (len(candidates) - 1) // 2
    selected = dict(candidates[rank_zero_based])
    totals = [int(row["total_candidates"]) for row in candidates]
    prediction_metadata = header.get("metadata", {})
    selected.update(
        {
            "selection_rule": (
                "lower empirical median after sorting valid test CC/MLO pairs "
                "by total candidate count, then exam_id and laterality"
            ),
            "rank_zero_based": rank_zero_based,
            "eligible_breast_pairs": len(candidates),
            "candidate_count_distribution": {
                "minimum": min(totals),
                "lower_median": totals[rank_zero_based],
                "mathematical_median": float(statistics.median(totals)),
                "maximum": max(totals),
            },
            "prediction_path": str(Path(prediction_path).resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_checkpoint_sha256": (
                prediction_metadata.get("checkpoint_sha256")
                if isinstance(prediction_metadata, Mapping)
                else None
            ),
        }
    )
    return selected


def _mixed_precision_context(device: torch.device, mixed_precision: str):
    precision = mixed_precision.lower()
    if precision == "none":
        return nullcontext()
    if device.type != "cuda":
        raise EfficiencyMeasurementError("Mixed-precision efficiency requires CUDA")
    if precision not in {"fp16", "bf16"}:
        raise EfficiencyMeasurementError(
            f"Unsupported mixed precision mode: {mixed_precision!r}"
        )
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prepare_standardized_breast(
    batch: Mapping[str, Any],
    *,
    device: str | torch.device,
    mean: float = 0.3089,
    std: float = 0.2505,
) -> dict[str, Any]:
    """Move one pair to the GPU and standardize it outside the timed region."""

    if std <= 0:
        raise ValueError("Normalization std must be positive")
    resolved_device = torch.device(device)
    samples = batch.get("views")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise EfficiencyMeasurementError("Batch must contain hierarchical view bags")
    if len(samples) != 1 or not isinstance(samples[0], Mapping):
        raise EfficiencyMeasurementError("Efficiency batch size must equal one breast")
    sample_views = samples[0]
    true_names = [
        str(view.get("view_name", key)).upper()
        for key, view in sample_views.items()
    ]
    if true_names.count("CC") != 1 or true_names.count("MLO") != 1:
        raise EfficiencyMeasurementError(
            "Efficiency representative must have exactly one CC and one MLO view"
        )

    prepared_views: dict[str, dict[str, Any]] = {}
    for view_key, view in sample_views.items():
        patches = view.get("patches")
        anatomy = view.get("anatomy_base", view.get("anatomy"))
        if not isinstance(patches, Tensor) or patches.ndim != 4:
            raise EfficiencyMeasurementError(
                f"{view_key} patches must have shape [N,C,H,W]"
            )
        if patches.shape[0] <= 0 or patches.shape[1] not in (1, 3):
            raise EfficiencyMeasurementError(f"{view_key} contains invalid patches")
        if not isinstance(anatomy, Tensor) or anatomy.shape != (patches.shape[0], 5):
            raise EfficiencyMeasurementError(
                f"{view_key} anatomy_base must have shape [N,5]"
            )
        # Expanded grayscale storage is materialized once on the measurement
        # GPU. In-place standardization avoids a second full-bag allocation.
        standardized = patches.to(
            device=resolved_device, dtype=torch.float32, non_blocking=True
        ).contiguous()
        standardized.sub_(float(mean)).div_(float(std))
        prepared_views[str(view_key)] = {
            "view_name": str(view.get("view_name", view_key)).upper(),
            "patches": standardized,
            "anatomy_base": anatomy[:, :5].to(
                device=resolved_device, dtype=torch.float32, non_blocking=True
            ),
            "cross_view_valid": True,
        }
    return {
        "exam_id": str(batch.get("exam_id", [""])[0]),
        "breast_id": str(batch.get("breast_id", [""])[0]),
        "laterality": str(batch.get("laterality", [""])[0]),
        "views": prepared_views,
        "cross_view_valid": True,
        "total_candidates": int(
            sum(view["patches"].shape[0] for view in prepared_views.values())
        ),
    }


def forward_standardized_breast(
    prepared: Mapping[str, Any],
    *,
    model: nn.Module,
    backbone: nn.Module,
    micro_batch_size: int,
    return_pyramid: bool = False,
) -> Tensor:
    """Forward from standardized GPU patches through B2 and the MIL head."""

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    encoded_views: dict[str, dict[str, Any]] = {}
    for view_key, view in prepared["views"].items():
        feature_parts: list[Tensor] = []
        pyramid_parts: dict[int, list[Tensor]] = {8: [], 16: [], 32: []}
        patches = view["patches"]
        for chunk in patches.split(micro_batch_size, dim=0):
            if return_pyramid:
                maps = backbone.encode_patches(
                    chunk,
                    micro_batch_size=micro_batch_size,
                    return_pyramid=True,
                )
                if set(maps) != {8, 16, 32}:
                    raise EfficiencyMeasurementError(
                        "Backbone pyramid must expose OS8/16/32"
                    )
                for stride in pyramid_parts:
                    pyramid_parts[stride].append(maps[stride])
                feature_parts.append(F.adaptive_avg_pool2d(maps[32], 1).flatten(1))
            else:
                features = backbone.encode_patches(
                    chunk, micro_batch_size=micro_batch_size
                )
                if not isinstance(features, Tensor) or features.ndim != 2:
                    raise EfficiencyMeasurementError(
                        "Backbone must return [N,D] feature tensors"
                    )
                feature_parts.append(features)
        combined = torch.cat(feature_parts, dim=0)
        encoded: dict[str, Any] = {
            "view_name": view["view_name"],
            "features": combined,
            "anatomy_base": view["anatomy_base"].to(dtype=combined.dtype),
            "cross_view_valid": True,
        }
        if return_pyramid:
            encoded["pyramid"] = {
                stride: torch.cat(parts, dim=0)
                for stride, parts in pyramid_parts.items()
            }
        encoded_views[str(view_key)] = encoded
    # Labels are intentionally excluded: efficiency is an inference forward,
    # not loss evaluation. Candidate outputs remain part of every model call.
    output = model(
        [
            {
                "exam_id": prepared["exam_id"],
                "breast_id": prepared["breast_id"],
                "laterality": prepared["laterality"],
                "views": encoded_views,
                "cross_view_valid": True,
            }
        ]
    )
    probabilities = output.get("breast_probabilities")
    if not isinstance(probabilities, Tensor) or probabilities.shape != (1, 2):
        raise EfficiencyMeasurementError("MIL model must return probabilities [1,2]")
    return probabilities


def measure_cuda_inference(
    module: nn.Module,
    forward: Callable[[], object],
    *,
    warmup: int = 20,
    repetitions: int = 100,
    device: str | torch.device = "cuda",
) -> dict[str, float | int | str | None]:
    """Measure synchronized per-forward latency and allocated CUDA peak memory."""

    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the manuscript efficiency protocol")
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)
        baseline = int(torch.cuda.memory_allocated(resolved_device))
        elapsed_ms: list[float] = []
        for _ in range(repetitions):
            torch.cuda.synchronize(resolved_device)
            start = time.perf_counter_ns()
            forward()
            torch.cuda.synchronize(resolved_device)
            elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
        peak = int(torch.cuda.max_memory_allocated(resolved_device))
    latency_mean = float(statistics.fmean(elapsed_ms))
    latency_median = float(statistics.median(elapsed_ms))
    latency_std = float(statistics.pstdev(elapsed_ms))
    parameters = trainable_parameter_count(module)
    return {
        "trainable_parameters": parameters,
        "trainable_parameters_millions": parameters / 1e6,
        "baseline_gpu_memory_bytes": baseline,
        "peak_gpu_memory_bytes": peak,
        "peak_gpu_memory_gib": peak / float(1024**3),
        "latency_ms": latency_mean,
        "latency_mean_ms": latency_mean,
        "latency_median_ms": latency_median,
        "latency_std_ms": latency_std,
        "latency_min_ms": float(min(elapsed_ms)),
        "latency_max_ms": float(max(elapsed_ms)),
        "warmup": int(warmup),
        "repetitions": int(repetitions),
        "flops": None,
        "flops_status": "not_measured",
        "flops_reason": (
            "No validated operator-complete FLOPs profiler is available for "
            "variable-size sparse graph, top-k, cross-view, and attention operations."
        ),
    }


def measure_breast_pair_efficiency(
    *,
    model: nn.Module,
    backbone: nn.Module,
    batch: Mapping[str, Any],
    mean: float,
    std: float,
    mixed_precision: str,
    micro_batch_size: int,
    warmup: int = 20,
    repetitions: int = 100,
    return_pyramid: bool = False,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Measure the full standardized-patch-to-MIL breast-pair forward."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise EfficiencyMeasurementError("A local CUDA GPU is required")
    model = model.to(resolved_device).eval()
    backbone = backbone.to(resolved_device).eval()
    backbone.requires_grad_(False)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise EfficiencyMeasurementError("EfficientNet-B2 must remain frozen")
    prepared = prepare_standardized_breast(
        batch, device=resolved_device, mean=mean, std=std
    )

    def forward() -> Tensor:
        with _mixed_precision_context(resolved_device, mixed_precision):
            return forward_standardized_breast(
                prepared,
                model=model,
                backbone=backbone,
                micro_batch_size=micro_batch_size,
                return_pyramid=return_pyramid,
            )

    result = measure_cuda_inference(
        model,
        forward,
        warmup=warmup,
        repetitions=repetitions,
        device=resolved_device,
    )
    if not all(
        math.isfinite(float(result[key]))
        for key in ("latency_mean_ms", "latency_median_ms", "latency_std_ms")
    ):
        raise EfficiencyMeasurementError("Non-finite CUDA timing result")
    result.update(
        {
            "batch_size": 1,
            "mixed_precision": mixed_precision,
            "input_scope": "standardized 512x512 patch tensors resident on GPU",
            "included": "frozen EfficientNet-B2 feature extraction and downstream MIL",
            "excluded": (
                "DICOM read, disk I/O, breast mask, patch extraction, normalization, "
                "and host-to-device transfer"
            ),
            "labels_used": False,
            "total_candidates": int(prepared["total_candidates"]),
            "view_candidate_counts": {
                key: int(view["patches"].shape[0])
                for key, view in prepared["views"].items()
            },
            "cuda_device": torch.cuda.get_device_name(resolved_device),
        }
    )
    return result


__all__ = [
    "EfficiencyMeasurementError",
    "forward_standardized_breast",
    "measure_breast_pair_efficiency",
    "measure_cuda_inference",
    "prepare_standardized_breast",
    "select_representative_breast",
    "trainable_parameter_count",
]
