from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


EPSILON = 1e-8


@dataclass(frozen=True)
class PredictedBox:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class HeatmapGeometry:
    image_shape: tuple[int, int]
    mask: np.ndarray
    regions: tuple[tuple[int, int, int, int] | None, ...]
    coverage: np.ndarray


def prepare_heatmap_geometry(
    patch_boxes,
    image_shape: tuple[int, int],
    breast_mask,
) -> HeatmapGeometry:
    """Validate and rasterize class-independent patch geometry once."""

    resolved_shape = tuple(int(value) for value in image_shape)
    boxes = np.asarray(patch_boxes, dtype=np.int64).reshape(-1, 4)
    mask = np.asarray(breast_mask, dtype=bool)
    if mask.shape != resolved_shape:
        raise ValueError("breast_mask shape does not match image_shape")

    coverage = np.zeros(resolved_shape, dtype=np.uint16)
    height, width = resolved_shape
    regions: list[tuple[int, int, int, int] | None] = []
    for x0, y0, x1, y1 in boxes:
        clipped_x0, clipped_x1 = sorted((max(0, int(x0)), min(width, int(x1))))
        clipped_y0, clipped_y1 = sorted((max(0, int(y0)), min(height, int(y1))))
        if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
            regions.append(None)
            continue
        region = (clipped_x0, clipped_y0, clipped_x1, clipped_y1)
        regions.append(region)
        coverage[clipped_y0:clipped_y1, clipped_x0:clipped_x1] += 1
    return HeatmapGeometry(
        image_shape=resolved_shape,
        mask=mask,
        regions=tuple(regions),
        coverage=coverage,
    )


def reconstruct_heatmap_from_geometry(
    candidate_probabilities,
    geometry: HeatmapGeometry,
) -> np.ndarray:
    """Reconstruct one class while preserving the formal float32 add order."""

    probabilities = np.asarray(candidate_probabilities, dtype=np.float32).reshape(-1)
    if probabilities.shape[0] != len(geometry.regions):
        raise ValueError("candidate probabilities and boxes must align")

    response_sum = np.zeros(geometry.image_shape, dtype=np.float32)
    for probability, region in zip(probabilities, geometry.regions, strict=True):
        if region is None:
            continue
        x0, y0, x1, y1 = region
        response_sum[y0:y1, x0:x1] += float(probability)

    raw = np.zeros_like(response_sum)
    covered = geometry.coverage > 0
    raw[covered] = response_sum[covered] / geometry.coverage[covered]
    normalized = np.zeros_like(raw)
    valid_values = raw[geometry.mask]
    if valid_values.size:
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if maximum - minimum > EPSILON:
            normalized[geometry.mask] = (valid_values - minimum) / (
                maximum - minimum + EPSILON
            )
    normalized[~geometry.mask] = 0.0
    return normalized


def reconstruct_heatmap(
    candidate_probabilities,
    patch_boxes,
    image_shape: tuple[int, int],
    breast_mask,
) -> tuple[np.ndarray, np.ndarray]:
    """Average overlapping patch responses and normalize within the breast mask.

    Boxes use half-open ``(x0, y0, x1, y1)`` image coordinates. Pixels in the
    breast mask that are not covered by a retained patch are assigned zero; this
    resolves the undefined zero-denominator case in Supplement S5 explicitly.
    """

    geometry = prepare_heatmap_geometry(patch_boxes, image_shape, breast_mask)
    normalized = reconstruct_heatmap_from_geometry(
        candidate_probabilities,
        geometry,
    )
    return normalized, geometry.coverage


def boxes_from_heatmap(
    heatmap,
    breast_mask,
    breast_probability: float,
    *,
    quantile: float = 0.95,
    fixed_threshold: float = 0.5,
) -> tuple[list[PredictedBox], float, np.ndarray]:
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    mask = np.asarray(breast_mask, dtype=bool)
    if heatmap_array.shape != mask.shape:
        raise ValueError("heatmap and breast_mask must have identical shape")
    if not 0.0 <= breast_probability <= 1.0:
        raise ValueError("breast probability must be in [0, 1]")
    values = heatmap_array[mask]
    if values.size == 0:
        return [], float(fixed_threshold), np.zeros_like(mask)
    threshold = max(float(np.quantile(values, quantile)), float(fixed_threshold))
    binary = mask & (heatmap_array >= threshold)
    labels, components = ndimage.label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    predictions: list[PredictedBox] = []
    for component_id in range(1, components + 1):
        ys, xs = np.nonzero(labels == component_id)
        if xs.size == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        maximum = float(heatmap_array[ys, xs].max())
        predictions.append(PredictedBox(box, float(breast_probability * maximum)))
    predictions.sort(key=lambda item: item.confidence, reverse=True)
    return predictions, threshold, binary


def box_iou(first, second) -> float:
    ax0, ay0, ax1, ay1 = map(float, first)
    bx0, by0, bx1, by1 = map(float, second)
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _prediction_box(item) -> tuple[Sequence[float], float]:
    if isinstance(item, PredictedBox):
        return item.box, item.confidence
    if isinstance(item, Mapping):
        return item["box"], float(item["confidence"])
    box, confidence = item
    return box, float(confidence)


def _box_area(box) -> float:
    x0, y0, x1, y1 = map(float, box)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _area_matches(box, area_range: tuple[float, float] | None) -> bool:
    if area_range is None:
        return True
    area = _box_area(box)
    return area_range[0] <= area < area_range[1]


def _precision_envelope_ap(
    tp: np.ndarray, fp: np.ndarray, ground_truth_count: int
) -> float:
    if ground_truth_count == 0:
        return float("nan")
    cumulative_tp = np.cumsum(tp, dtype=np.float64)
    cumulative_fp = np.cumsum(fp, dtype=np.float64)
    recall = cumulative_tp / ground_truth_count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, EPSILON)
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    padded_precision = np.maximum.accumulate(padded_precision[::-1])[::-1]
    changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
    return float(
        np.sum(
            (padded_recall[changes + 1] - padded_recall[changes])
            * padded_precision[changes + 1]
        )
    )


def average_precision_at_iou(
    views: Iterable[Mapping[str, object]],
    *,
    iou_threshold: float = 0.25,
    area_range: tuple[float, float] | None = None,
    ignore_out_of_range_ground_truth: bool = True,
) -> dict[str, float | int]:
    """Compute view-level detection AP over positive and negative views.

    Each view mapping contains a stable ``view_id``, ``predictions`` and
    ``ground_truth``. For scale-specific AP, detections matching an out-of-range
    lesion are ignored (COCO-style) rather than counted as false positives.
    """

    records = list(views)
    ground_truth: dict[str, list[Sequence[float]]] = {}
    ignored_truth: dict[str, list[Sequence[float]]] = {}
    predictions: list[tuple[float, str, Sequence[float]]] = []
    seen_view_ids: set[str] = set()
    for record in records:
        view_id = str(record["view_id"])
        if view_id in seen_view_ids:
            raise ValueError(f"Duplicate view_id: {view_id}")
        seen_view_ids.add(view_id)
        all_truth = [tuple(box) for box in record.get("ground_truth", [])]
        ground_truth[view_id] = [
            box for box in all_truth if _area_matches(box, area_range)
        ]
        ignored_truth[view_id] = [
            box for box in all_truth if not _area_matches(box, area_range)
        ]
        for item in record.get("predictions", []):
            box, confidence = _prediction_box(item)
            if not np.isfinite(confidence):
                raise ValueError(
                    f"Prediction confidence for view {view_id!r} must be finite"
                )
            try:
                coordinates = tuple(box)
                numeric_coordinates = np.asarray(coordinates, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Prediction coordinates for view {view_id!r} must be numeric"
                ) from error
            if numeric_coordinates.shape != (4,):
                raise ValueError(
                    f"Prediction coordinates for view {view_id!r} must have shape (4,)"
                )
            if not np.isfinite(numeric_coordinates).all():
                raise ValueError(
                    f"Prediction coordinates for view {view_id!r} must be finite"
                )
            predictions.append((confidence, view_id, coordinates))

    predictions.sort(key=lambda item: item[0], reverse=True)
    matched: dict[str, set[int]] = defaultdict(set)
    true_positive: list[float] = []
    false_positive: list[float] = []
    ignored_predictions = 0
    for _, view_id, prediction in predictions:
        candidates = ground_truth.get(view_id, [])
        available = [
            index for index in range(len(candidates)) if index not in matched[view_id]
        ]
        if available:
            overlaps = [box_iou(prediction, candidates[index]) for index in available]
            best_offset = int(np.argmax(overlaps))
            if overlaps[best_offset] >= iou_threshold:
                matched[view_id].add(available[best_offset])
                true_positive.append(1.0)
                false_positive.append(0.0)
                continue
        if ignore_out_of_range_ground_truth and any(
            box_iou(prediction, box) >= iou_threshold
            for box in ignored_truth.get(view_id, [])
        ):
            ignored_predictions += 1
            continue
        true_positive.append(0.0)
        false_positive.append(1.0)

    truth_count = sum(len(boxes) for boxes in ground_truth.values())
    tp_array = np.asarray(true_positive, dtype=np.float64)
    fp_array = np.asarray(false_positive, dtype=np.float64)
    return {
        "ap": _precision_envelope_ap(tp_array, fp_array, truth_count),
        "ground_truth": int(truth_count),
        "predictions": int(tp_array.size),
        "true_positives": int(tp_array.sum()),
        "false_positives": int(fp_array.sum()),
        "ignored_predictions": int(ignored_predictions),
    }


def localization_metrics(
    views: Iterable[Mapping[str, object]],
    *,
    iou_threshold: float = 0.25,
    small_threshold: int = 128**2,
    large_threshold: int = 256**2,
) -> dict[str, dict[str, float | int]]:
    records = list(views)
    return {
        "overall": average_precision_at_iou(records, iou_threshold=iou_threshold),
        "small": average_precision_at_iou(
            records,
            iou_threshold=iou_threshold,
            area_range=(0.0, float(small_threshold)),
        ),
        "medium": average_precision_at_iou(
            records,
            iou_threshold=iou_threshold,
            area_range=(float(small_threshold), float(large_threshold)),
        ),
        "large": average_precision_at_iou(
            records,
            iou_threshold=iou_threshold,
            area_range=(float(large_threshold), float("inf")),
        ),
    }
