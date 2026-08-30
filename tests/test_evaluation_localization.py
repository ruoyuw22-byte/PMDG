import numpy as np
import pytest

from dagl_mil.evaluation.artifacts import decode_binary_mask, encode_binary_mask
from dagl_mil.evaluation.localization import (
    average_precision_at_iou,
    boxes_from_heatmap,
    prepare_heatmap_geometry,
    reconstruct_heatmap,
    reconstruct_heatmap_from_geometry,
)


def _legacy_reconstruct(candidate_probabilities, patch_boxes, image_shape, mask):
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32).reshape(-1)
    boxes = np.asarray(patch_boxes, dtype=np.int64).reshape(-1, 4)
    response_sum = np.zeros(image_shape, dtype=np.float32)
    coverage = np.zeros(image_shape, dtype=np.uint16)
    height, width = image_shape
    for probability, (x0, y0, x1, y1) in zip(probabilities, boxes, strict=True):
        x0, x1 = sorted((max(0, int(x0)), min(width, int(x1))))
        y0, y1 = sorted((max(0, int(y0)), min(height, int(y1))))
        if x1 <= x0 or y1 <= y0:
            continue
        response_sum[y0:y1, x0:x1] += float(probability)
        coverage[y0:y1, x0:x1] += 1
    raw = np.zeros_like(response_sum)
    covered = coverage > 0
    raw[covered] = response_sum[covered] / coverage[covered]
    normalized = np.zeros_like(raw)
    valid_values = raw[mask]
    if valid_values.size:
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if maximum - minimum > 1e-8:
            normalized[mask] = (valid_values - minimum) / (maximum - minimum + 1e-8)
    normalized[~mask] = 0.0
    return normalized, coverage


def test_overlap_average_and_component_box():
    mask = np.ones((8, 8), dtype=bool)
    heatmap, coverage = reconstruct_heatmap(
        [0.2, 0.8],
        [[0, 0, 6, 6], [2, 2, 8, 8]],
        (8, 8),
        mask,
    )
    assert coverage[3, 3] == 2
    assert heatmap[7, 7] > heatmap[0, 0]
    predictions, threshold, binary = boxes_from_heatmap(heatmap, mask, 0.75)
    assert threshold >= 0.5
    assert predictions
    assert 0.0 <= predictions[0].confidence <= 0.75
    assert binary.shape == mask.shape


def test_shared_geometry_is_bitwise_identical_to_independent_reconstruction():
    shape = (9, 10)
    mask = np.ones(shape, dtype=bool)
    mask[:2, :3] = False
    boxes = [
        [0, 0, 6, 6],
        [2, 2, 8, 8],
        [-2, 1, 3, 5],
        [7, 7, 12, 11],
        [6, 6, 4, 4],
    ]
    candidates = np.asarray(
        [
            [0.20000002, 0.8],
            [0.7, 0.10000001],
            [0.33333334, 0.6],
            [0.95, 0.4],
            [0.51, 0.49],
        ],
        dtype=np.float32,
    )
    geometry = prepare_heatmap_geometry(boxes, shape, mask)

    for class_index in range(2):
        expected_heatmap, expected_coverage = _legacy_reconstruct(
            candidates[:, class_index], boxes, shape, mask
        )
        actual_heatmap = reconstruct_heatmap_from_geometry(
            candidates[:, class_index], geometry
        )
        np.testing.assert_array_equal(actual_heatmap, expected_heatmap)
        np.testing.assert_array_equal(geometry.coverage, expected_coverage)

        expected_boxes, expected_threshold, expected_binary = boxes_from_heatmap(
            expected_heatmap, mask, 0.73
        )
        actual_boxes, actual_threshold, actual_binary = boxes_from_heatmap(
            actual_heatmap, mask, 0.73
        )
        assert actual_boxes == expected_boxes
        assert actual_threshold == expected_threshold
        np.testing.assert_array_equal(actual_binary, expected_binary)


def test_constant_heatmap_produces_no_false_component():
    mask = np.ones((5, 5), dtype=bool)
    heatmap, _ = reconstruct_heatmap([0.5], [[0, 0, 5, 5]], (5, 5), mask)
    predictions, _, _ = boxes_from_heatmap(heatmap, mask, 0.9)
    assert predictions == []


def test_negative_view_prediction_counts_as_false_positive():
    views = [
        {
            "view_id": "negative",
            "ground_truth": [],
            "predictions": [{"box": [0, 0, 4, 4], "confidence": 0.99}],
        },
        {
            "view_id": "positive",
            "ground_truth": [[0, 0, 4, 4]],
            "predictions": [{"box": [0, 0, 4, 4], "confidence": 0.8}],
        },
    ]
    result = average_precision_at_iou(views, iou_threshold=0.25)
    assert result["false_positives"] == 1
    assert result["true_positives"] == 1
    assert result["ap"] == 0.5


def test_average_precision_rejects_duplicate_view_ids():
    views = [
        {"view_id": "same", "ground_truth": [], "predictions": []},
        {"view_id": "same", "ground_truth": [], "predictions": []},
    ]

    with pytest.raises(ValueError, match="Duplicate view_id"):
        average_precision_at_iou(views)


@pytest.mark.parametrize(
    "prediction, match",
    [
        ({"box": [0, 0, 4, 4], "confidence": float("nan")}, "confidence"),
        ({"box": [0, 0, float("inf"), 4], "confidence": 0.8}, "coordinates"),
    ],
)
def test_average_precision_rejects_nonfinite_predictions(prediction, match):
    views = [
        {
            "view_id": "view",
            "ground_truth": [],
            "predictions": [prediction],
        }
    ]

    with pytest.raises(ValueError, match=match):
        average_precision_at_iou(views)


def test_mask_rle_round_trip():
    mask = np.zeros((11, 7), dtype=bool)
    mask[1:4, 2:6] = True
    mask[8, 1] = True
    np.testing.assert_array_equal(decode_binary_mask(encode_binary_mask(mask)), mask)
