from __future__ import annotations

import numpy as np
import pytest

from dagl_mil.data.patches import (
    PatchExtractionError,
    breast_geometry,
    extract_patch_bag,
    patch_grid_coverage,
    sliding_window_positions,
)


def test_sliding_grid_adds_only_the_needed_final_boundary() -> None:
    assert sliding_window_positions(1024, 512, 512) == [0, 512]
    assert sliding_window_positions(1100, 512, 512) == [0, 512, 588]
    assert sliding_window_positions(700, 512, 128) == [0, 128, 188]
    assert sliding_window_positions(300, 512, 128) == [0]


def test_patch_grid_covers_last_row_and_column() -> None:
    image = np.linspace(0, 1, 110, dtype=np.float32).reshape(10, 11)
    mask = np.ones_like(image, dtype=bool)
    bag = extract_patch_bag(
        image,
        mask,
        view_name="CC",
        patch_size=4,
        stride=4,
        minimum_breast_coverage=0.5,
    )
    assert bag.patches.shape == (9, 4, 4)
    assert (bag.coords == np.array([7, 6, 11, 10])).all(axis=1).any()
    assert np.all(bag.coverage == 1.0)


def test_exactly_fifty_percent_breast_coverage_is_retained() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:2, :4] = True  # exactly 8 / 16 pixels in the first patch
    bag = extract_patch_bag(
        image,
        mask,
        view_name="CC",
        patch_size=4,
        stride=4,
        minimum_breast_coverage=0.5,
    )
    assert bag.patches.shape[0] == 1
    assert bag.coverage[0] == pytest.approx(0.5)
    assert bag.coords[0].tolist() == [0, 0, 4, 4]


def test_shared_grid_matches_brute_force_with_boundary_padding() -> None:
    rng = np.random.default_rng(2026)
    mask = rng.random((7, 11)) > 0.4
    grid = patch_grid_coverage(
        mask,
        patch_size=8,
        stride=3,
        minimum_breast_coverage=0.5,
    )
    padded = np.zeros(grid.padded_shape, dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    expected = np.asarray(
        [padded[y : y + 8, x : x + 8].mean() for x, y in grid.starts_xy.tolist()]
    )
    np.testing.assert_allclose(grid.coverage, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(
        grid.retained,
        expected + np.finfo(np.float32).eps >= 0.5,
    )


def test_anatomy_descriptor_order_and_view_id_are_fixed() -> None:
    image = np.zeros((8, 12), dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :8] = True
    image[mask] = np.linspace(0.1, 1.0, mask.sum(), dtype=np.float32)
    bag = extract_patch_bag(
        image,
        mask,
        view_name="MLO",
        patch_size=4,
        stride=4,
        minimum_breast_coverage=0.5,
    )
    assert bag.anatomy_base.shape[1] == 5
    assert bag.anatomy.shape[1] == 6
    np.testing.assert_allclose(bag.anatomy[:, :5], bag.anatomy_base)
    assert np.all(bag.anatomy[:, 5] == 1.0)
    assert np.all((bag.anatomy_base[:, :2] >= 0) & (bag.anatomy_base[:, :2] <= 1))
    assert np.all((bag.anatomy_base[:, 4] >= 0) & (bag.anatomy_base[:, 4] <= 1))
    assert bag.geometry.chest_wall_x == 0.0
    assert bag.geometry.nipple_xy[0] == 7.0


def test_breast_geometry_uses_farthest_contour_as_nipple() -> None:
    mask = np.zeros((9, 15), dtype=bool)
    mask[2:7, :8] = True
    mask[4, 8:13] = True
    geometry = breast_geometry(mask)
    assert geometry.bbox_xyxy == (0, 2, 13, 7)
    assert geometry.nipple_xy == (12.0, 4.0)
    assert geometry.chest_wall_x == 0.0


def test_no_subthreshold_fallback_patch_is_silently_added() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[0, 0] = True
    with pytest.raises(PatchExtractionError, match="maximum coverage"):
        extract_patch_bag(
            image,
            mask,
            view_name="CC",
            patch_size=4,
            stride=4,
            minimum_breast_coverage=0.5,
        )
