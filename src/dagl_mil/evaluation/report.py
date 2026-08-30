from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import decode_binary_mask, open_prediction_artifact
from .bootstrap import cluster_bootstrap_ci
from .classification import classification_metrics
from .localization import (
    boxes_from_heatmap,
    localization_metrics,
    prepare_heatmap_geometry,
    reconstruct_heatmap_from_geometry,
)


def _real_numeric_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
        object_values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if array.dtype.kind not in "iuf" or any(
        isinstance(item, (bool, np.bool_)) for item in object_values.flat
    ):
        raise ValueError(f"{name} must contain real numeric values")
    return array


def _validate_ground_truth_boxes(boxes: Any, *, name: str) -> None:
    if isinstance(boxes, (str, bytes)) or not isinstance(boxes, Sequence):
        raise ValueError(f"{name} must be a sequence of xyxy boxes")
    for box_index, box in enumerate(boxes):
        box_name = f"{name}[{box_index}]"
        coordinates = _real_numeric_array(box, name=box_name)
        if coordinates.shape != (4,):
            raise ValueError(f"{box_name} must have shape (4,)")
        coordinates = coordinates.astype(np.float64, copy=False)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{box_name} coordinates must be finite")
        x0, y0, x1, y1 = coordinates
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"{box_name} must be ordered as x0 < x1 and y0 < y1")


def evaluate_prediction_artifact(
    prediction_path: str | Path,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
    heatmap_quantile: float = 0.95,
    heatmap_fixed_threshold: float = 0.5,
    iou_threshold: float = 0.25,
    size_thresholds: tuple[int, int] = (128**2, 256**2),
) -> dict[str, Any]:
    class_names = ("mass", "calcification")
    target_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    examination_rows: list[str] = []
    view_records: list[list[dict[str, Any]]] = [[], []]
    generated_view_ids: set[str] = set()

    with open_prediction_artifact(
        prediction_path,
        verify_hash=True,
    ) as (header, iterator, artifact_info):
        for record_index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise ValueError(f"Prediction record {record_index} must be a mapping")
            labels = _real_numeric_array(
                record.get("labels"), name=f"record {record_index} labels"
            )
            if labels.shape != (2,):
                raise ValueError("Prediction labels must have shape (2,)")
            if labels.dtype.kind not in "iu" or not np.isin(labels, (0, 1)).all():
                raise ValueError("Prediction labels must be binary integers")
            labels = labels.astype(np.int64, copy=False)

            breast_probabilities = _real_numeric_array(
                record.get("probabilities"),
                name=f"record {record_index} breast probabilities",
            )
            if breast_probabilities.shape != (2,):
                raise ValueError("Breast probabilities must have shape (2,)")
            breast_probabilities = breast_probabilities.astype(np.float64, copy=False)
            if (
                not np.isfinite(breast_probabilities).all()
                or (breast_probabilities < 0.0).any()
                or (breast_probabilities > 1.0).any()
            ):
                raise ValueError("Breast probabilities must be finite and in [0, 1]")

            views = record.get("views")
            if not isinstance(views, Mapping):
                raise ValueError("Prediction record views must be a mapping")
            target_rows.append(labels)
            probability_rows.append(breast_probabilities)
            examination_rows.append(str(record["exam_id"]))

            for view_key, view in views.items():
                if not isinstance(view, Mapping):
                    raise ValueError(
                        f"Prediction view {record_index}/{view_key} must be a mapping"
                    )
                mask = decode_binary_mask(view["breast_mask"])
                candidate_probabilities = _real_numeric_array(
                    view.get("candidate_probabilities"),
                    name=(
                        f"record {record_index} view {view_key} candidate probabilities"
                    ),
                )
                if (
                    candidate_probabilities.ndim != 2
                    or candidate_probabilities.shape[1] != 2
                ):
                    raise ValueError(
                        "Candidate probabilities must have shape [patches, 2]"
                    )
                if (
                    not np.isfinite(candidate_probabilities).all()
                    or (candidate_probabilities < 0.0).any()
                    or (candidate_probabilities > 1.0).any()
                ):
                    raise ValueError(
                        "Candidate probabilities must be finite and in [0, 1]"
                    )
                candidate_probabilities = candidate_probabilities.astype(
                    np.float32, copy=False
                )
                patch_boxes = _real_numeric_array(
                    view.get("patch_boxes"),
                    name=f"record {record_index} view {view_key} patch boxes",
                )
                if patch_boxes.shape != (candidate_probabilities.shape[0], 4):
                    raise ValueError(
                        "Patch boxes must have shape [patches, 4] and align with "
                        "candidate probabilities"
                    )
                if not np.isfinite(patch_boxes).all():
                    raise ValueError("Patch box coordinates must be finite")
                geometry = prepare_heatmap_geometry(
                    patch_boxes,
                    tuple(int(value) for value in view["image_shape"]),
                    mask,
                )
                view_id = (
                    f"{record['exam_id']}:{record['laterality']}:"
                    f"{view.get('image_id', view_key)}:{view_key}"
                )
                if view_id in generated_view_ids:
                    raise ValueError(f"Duplicate generated view_id: {view_id}")
                generated_view_ids.add(view_id)
                ground_truth = view.get("ground_truth", {})
                if not isinstance(ground_truth, Mapping):
                    raise ValueError("View ground_truth must be a mapping")
                for class_name in class_names:
                    _validate_ground_truth_boxes(
                        ground_truth.get(class_name, []),
                        name=(
                            f"record {record_index} view {view_key} "
                            f"ground_truth[{class_name!r}]"
                        ),
                    )
                for class_index, class_name in enumerate(class_names):
                    heatmap = reconstruct_heatmap_from_geometry(
                        candidate_probabilities[:, class_index],
                        geometry,
                    )
                    boxes, _threshold, binary = boxes_from_heatmap(
                        heatmap,
                        mask,
                        float(breast_probabilities[class_index]),
                        quantile=heatmap_quantile,
                        fixed_threshold=heatmap_fixed_threshold,
                    )
                    view_records[class_index].append(
                        {
                            "view_id": view_id,
                            "predictions": boxes,
                            "ground_truth": ground_truth.get(class_name, []),
                        }
                    )
                    del heatmap, binary

    if not target_rows:
        raise ValueError("Prediction artifact contains no breast records")
    targets = np.stack(target_rows)
    probabilities = np.stack(probability_rows)
    examinations = np.asarray(examination_rows)
    point = classification_metrics(targets, probabilities)
    classification: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        classification[class_name] = dict(point[class_name])
        for metric in ("roc_auc", "pr_auc"):
            interval = cluster_bootstrap_ci(
                targets[:, class_index],
                probabilities[:, class_index],
                examinations,
                metric=metric,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            classification[class_name][f"{metric}_ci"] = interval

    localization: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        localization[class_name] = localization_metrics(
            view_records[class_index],
            iou_threshold=iou_threshold,
            small_threshold=int(size_thresholds[0]),
            large_threshold=int(size_thresholds[1]),
        )

    return {
        "prediction_path": str(Path(prediction_path).resolve()),
        "prediction_sha256": artifact_info.sha256,
        "prediction_metadata": header["metadata"],
        "breasts": len(target_rows),
        "unique_examinations": int(np.unique(examinations).size),
        "classification": classification,
        "localization": localization,
        "protocol": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "heatmap_quantile": heatmap_quantile,
            "heatmap_fixed_threshold": heatmap_fixed_threshold,
            "iou_threshold": iou_threshold,
            "size_thresholds": list(size_thresholds),
            "connected_components": 8,
            "minimum_box_area": None,
            "nms": False,
            "maximum_boxes": None,
        },
    }


def write_evaluation_report(path: str | Path, report: Mapping[str, Any]) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
