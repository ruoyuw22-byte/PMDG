from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import dagl_mil.data.preprocessing as preprocessing_module
from dagl_mil.data.preprocessing import (
    DICOMPreprocessor,
    IntensityAugment,
    PreprocessedMammogram,
    PreprocessingError,
    apply_grayscale_polarity,
    extract_breast_mask,
    largest_connected_component,
    percentile_normalize,
    preprocess_array,
    preprocess_dicom,
    standardize_laterality,
    transform_bboxes_for_laterality,
)


def test_monochrome1_is_inverted_and_padding_stays_background() -> None:
    pixels = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.float32)
    padding = pixels == 0
    mono1 = apply_grayscale_polarity(pixels, "MONOCHROME1", padding_mask=padding)
    mono2 = apply_grayscale_polarity(pixels, "MONOCHROME2", padding_mask=padding)
    assert mono1[0, 1] > mono1[1, 2]
    assert mono2[0, 1] < mono2[1, 2]
    assert np.all(mono1[padding] == mono1[~padding].min())


def test_padding_mask_includes_large_untagged_border_code() -> None:
    pixels = np.zeros((20, 20), dtype=np.uint16)
    pixels[4:16, 4:16] = 3000
    pixels[:, -2:] = 10000
    dataset = FileDataset("unit.dicom", {}, file_meta=FileMetaDataset())
    dataset.PixelPaddingValue = 10000
    padding = preprocessing_module._padding_mask(dataset, pixels)
    assert bool(padding[pixels == 0].all())
    assert bool(padding[pixels == 10000].all())
    assert not bool(padding[pixels == 3000].any())


def test_largest_component_and_hole_filling_define_breast_mask() -> None:
    image = np.zeros((30, 40), dtype=np.float32)
    image[4:26, 2:25] = 10.0
    image[12:15, 10:13] = 0.0  # internal hole
    image[1:4, 34:38] = 10.0  # disconnected acquisition artifact
    mask = extract_breast_mask(image)
    assert mask[13, 11]  # filled internal hole
    assert not mask[2, 35]  # small disconnected object was discarded
    assert mask.sum() == 22 * 23

    raw = np.zeros((8, 10), dtype=bool)
    raw[1:6, 1:5] = True
    raw[1:3, 8:10] = True
    component = largest_connected_component(raw)
    assert component.sum() == 20


def test_padding_topology_rejects_a_brighter_disconnected_artifact() -> None:
    image = np.zeros((100, 100), dtype=np.float32)
    main_breast = np.zeros_like(image, dtype=bool)
    main_breast[5:75, 5:55] = True
    artifact = np.zeros_like(image, dtype=bool)
    artifact[80:100, 70:100] = True
    image[main_breast] = np.linspace(
        0.0, 20.0, int(main_breast.sum()), dtype=np.float32
    )
    image[artifact] = 100.0
    support = main_breast | artifact

    otsu_only = extract_breast_mask(image)
    guarded = extract_breast_mask(image, foreground_support=support)

    assert bool(otsu_only[artifact].all())
    assert not bool(otsu_only[main_breast].any())
    assert bool(guarded[main_breast].all())
    assert not bool(guarded[artifact].any())

    connected_support = support.copy()
    connected_support[74, 54:71] = True
    connected_support[74:81, 70] = True
    one_component = extract_breast_mask(
        image,
        foreground_support=connected_support,
    )
    assert bool(one_component[artifact].all())
    assert not bool(one_component[main_breast].any())

    result = preprocess_array(
        image,
        laterality="L",
        view_name="CC",
        padding_mask=~support,
    )
    assert result.mask_strategy == "nonpadding_lcc_topology_fallback"
    np.testing.assert_array_equal(result.mask, main_breast)


def test_laterality_standardization_places_right_chest_wall_on_left() -> None:
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    mask = np.ones_like(image, dtype=bool)
    left_image, left_mask, left_flipped = standardize_laterality(image, mask, "L")
    right_image, right_mask, right_flipped = standardize_laterality(image, mask, "R")
    assert not left_flipped
    assert left_image.tolist() == image.tolist()
    assert left_mask.all()
    assert right_flipped
    assert right_image.tolist() == image[:, ::-1].tolist()
    assert right_mask.all()


def test_percentile_normalization_uses_only_breast_pixels() -> None:
    image = np.full((12, 12), 9999.0, dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[1:11, 1:11] = True
    image[mask] = np.arange(100, dtype=np.float32)
    normalized = percentile_normalize(image, mask)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert np.all(normalized[~mask] == 0.0)
    assert normalized[1, 1] == 0.0  # below p1 is clipped
    assert normalized[10, 10] == 1.0  # above p99 is clipped


def test_gamma_contrast_augmentation_is_photometric_only() -> None:
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[False, True], [True, True]])
    augment = IntensityAugment(
        gamma_probability=1.0,
        gamma_range=(2.0, 2.0),
        contrast_probability=0.0,
    )
    result = augment(image, mask, rng=np.random.default_rng(3))
    assert result.shape == image.shape
    assert result[0, 0] == 0.0
    np.testing.assert_allclose(result[mask], np.square(image[mask]))


def test_right_bboxes_are_transformed_only_for_evaluation_metadata() -> None:
    boxes = [
        {
            "categories": ["Mass"],
            "xmin": 10.0,
            "ymin": 20.0,
            "xmax": 30.0,
            "ymax": 40.0,
        }
    ]
    transformed = transform_bboxes_for_laterality(
        boxes, width=100, height=100, laterality="R"
    )
    assert transformed[0]["xmin"] == 70.0
    assert transformed[0]["xmax"] == 90.0
    assert boxes[0]["xmin"] == 10.0  # no in-place mutation
    assert (
        transform_bboxes_for_laterality(boxes, width=100, height=100, laterality="L")
        == boxes
    )


def test_partially_out_of_canvas_bboxes_are_clipped_before_laterality() -> None:
    boxes = [
        {
            "categories": ["Mass"],
            "xmin": -10.0,
            "ymin": -2.0,
            "xmax": 105.0,
            "ymax": 44.0,
        }
    ]
    left = transform_bboxes_for_laterality(boxes, width=100, height=40, laterality="L")[
        0
    ]
    right = transform_bboxes_for_laterality(
        boxes, width=100, height=40, laterality="R"
    )[0]
    assert (left["xmin"], left["ymin"], left["xmax"], left["ymax"]) == (
        0.0,
        0.0,
        100.0,
        40.0,
    )
    assert right == left
    assert boxes[0]["xmin"] == -10.0  # no in-place clipping


def test_right_bbox_is_clipped_before_non_symmetric_mirroring() -> None:
    transformed = transform_bboxes_for_laterality(
        [{"xmin": 90.0, "ymin": 2.0, "xmax": 105.0, "ymax": 8.0}],
        width=100,
        height=40,
        laterality="R",
    )[0]
    assert (transformed["xmin"], transformed["xmax"]) == (0.0, 10.0)


@pytest.mark.parametrize(
    "box",
    [
        {"xmin": -20.0, "ymin": 1.0, "xmax": -1.0, "ymax": 5.0},
        {"xmin": 100.0, "ymin": 1.0, "xmax": 110.0, "ymax": 5.0},
        {"xmin": 1.0, "ymin": 40.0, "xmax": 5.0, "ymax": 45.0},
        {"xmin": 5.0, "ymin": 1.0, "xmax": 5.0, "ymax": 4.0},
        {"xmin": float("nan"), "ymin": 1.0, "xmax": 5.0, "ymax": 4.0},
        {"xmin": 1.0, "ymin": 1.0, "xmax": float("inf"), "ymax": 4.0},
        {"ymin": 1.0, "xmax": 5.0, "ymax": 4.0},
    ],
)
def test_invalid_or_disjoint_bboxes_fail_closed(box: dict[str, float]) -> None:
    with pytest.raises(PreprocessingError):
        transform_bboxes_for_laterality([box], width=100, height=40, laterality="L")


def _write_test_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    photometric_interpretation: str = "MONOCHROME2",
    pixel_padding_value: int = 0,
    laterality: str = "R",
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelPaddingValue = pixel_padding_value
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = 10
    dataset.ImageLaterality = laterality
    dataset.ViewPosition = "CC"
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path)


def test_raw_dicom_pipeline_decodes_masks_flips_and_normalizes(tmp_path: Path) -> None:
    y, x = np.ogrid[:80, :100]
    breast = ((x - 72) / 27) ** 2 + ((y - 40) / 36) ** 2 <= 1
    pixels = np.zeros((80, 100), dtype=np.uint16)
    pixels[breast] = (100 + x.repeat(80, axis=0)[breast]).astype(np.uint16)
    path = tmp_path / "synthetic.dicom"
    _write_test_dicom(path, pixels)

    result = preprocess_dicom(path)
    assert result.image.shape == pixels.shape
    assert result.image.dtype == np.float32
    assert result.mask.dtype == np.bool_
    assert result.horizontally_flipped
    assert result.laterality == "R"
    assert result.view_name == "CC"
    assert result.image.min() == 0.0
    assert result.image.max() == 1.0
    assert result.mask[:, :50].sum() > result.mask[:, 50:].sum()


def test_raw_monochrome1_dicom_excludes_tagged_10000_and_untagged_zero_padding(
    tmp_path: Path,
) -> None:
    height, width = 96, 128
    y, x = np.ogrid[:height, :width]
    breast = ((x - 48) / 42) ** 2 + ((y - 48) / 42) ** 2 <= 1
    pixels = np.zeros((height, width), dtype=np.uint16)
    gradient = 900 + 12 * x.repeat(height, axis=0) + 3 * y.repeat(width, axis=1)
    pixels[breast] = gradient[breast].astype(np.uint16)
    pixels[:, -8:] = 10000
    path = tmp_path / "mono1-dual-padding.dicom"
    _write_test_dicom(
        path,
        pixels,
        photometric_interpretation="MONOCHROME1",
        pixel_padding_value=10000,
        laterality="L",
    )

    result = preprocess_dicom(path)

    assert result.laterality == "L" and not result.horizontally_flipped
    assert not bool(result.mask[pixels == 0].any())
    assert not bool(result.mask[pixels == 10000].any())
    # Ordinary Otsu is intentionally allowed to select the denser portion of
    # a connected breast; this regression targets polarity/padding semantics.
    assert float(result.mask[breast].mean()) > 0.6
    assert bool(np.all(result.image[~result.mask] == 0.0))
    assert result.image.dtype == np.float32


def test_degenerate_image_fails_loudly() -> None:
    with pytest.raises(PreprocessingError, match="constant"):
        extract_breast_mask(np.ones((10, 10), dtype=np.float32))


def test_preprocessor_cache_is_preaugmentation_atomic_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.dicom"
    source.write_bytes(b"stable-source-identity")
    calls: list[Path] = []

    def fake_preprocess(path: str | Path, **_: object) -> PreprocessedMammogram:
        calls.append(Path(path))
        image = np.linspace(0, 1, 36, dtype=np.float32).reshape(6, 6)
        return PreprocessedMammogram(
            image=image,
            mask=np.ones((6, 6), dtype=bool),
            laterality="L",
            view_name="CC",
            original_shape=(6, 6),
            horizontally_flipped=False,
            source_path=str(path),
        )

    monkeypatch.setattr(preprocessing_module, "preprocess_dicom", fake_preprocess)
    cache_dir = tmp_path / "cache"
    processor = DICOMPreprocessor(cache_dir=cache_dir)
    first = processor(source, laterality="L", view_name="CC")
    second = processor(source, laterality="L", view_name="CC")
    assert calls == [source]
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.mask, second.mask)
    assert second.mask_strategy == "otsu_lcc"
    cache_files = list(cache_dir.rglob("*.npz"))
    assert len(cache_files) == 1
    assert not list(cache_dir.rglob("*.tmp"))

    cache_files[0].write_bytes(b"truncated zip")
    recovered = processor(source, laterality="L", view_name="CC")
    assert calls == [source, source]
    np.testing.assert_array_equal(recovered.image, first.image)


def test_read_only_cache_miss_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.dicom"
    source.write_bytes(b"source")

    def fake_preprocess(path: str | Path, **_: object) -> PreprocessedMammogram:
        return PreprocessedMammogram(
            image=np.ones((4, 4), dtype=np.float32),
            mask=np.ones((4, 4), dtype=bool),
            laterality="L",
            view_name="MLO",
            original_shape=(4, 4),
            horizontally_flipped=False,
            source_path=str(path),
        )

    monkeypatch.setattr(preprocessing_module, "preprocess_dicom", fake_preprocess)
    cache_dir = tmp_path / "read-only-cache"
    processor = DICOMPreprocessor(cache_dir=cache_dir, cache_read_only=True)
    result = processor(source, laterality="L", view_name="MLO")
    assert result.view_name == "MLO"
    assert not cache_dir.exists()
