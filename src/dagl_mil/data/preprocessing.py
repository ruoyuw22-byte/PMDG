"""Raw DICOM preprocessing for the manuscript-aligned VinDr pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pydicom
from scipy import ndimage
from skimage.filters import threshold_otsu


class PreprocessingError(ValueError):
    """Raised when a mammogram cannot be safely standardized."""


MASK_STRATEGIES = frozenset({"otsu_lcc", "nonpadding_lcc_topology_fallback"})


@dataclass(frozen=True)
class PreprocessedMammogram:
    """A polarity/orientation-standardized mammogram and its breast mask."""

    image: np.ndarray
    mask: np.ndarray
    laterality: str
    view_name: str
    original_shape: tuple[int, int]
    horizontally_flipped: bool
    source_path: str = ""
    mask_strategy: str = "otsu_lcc"

    def __post_init__(self) -> None:
        if self.image.ndim != 2 or self.mask.ndim != 2:
            raise PreprocessingError("image and mask must both be two-dimensional")
        if self.image.shape != self.mask.shape:
            raise PreprocessingError("image and mask shapes do not match")
        if self.image.dtype != np.float32:
            raise PreprocessingError("preprocessed image must have dtype float32")
        if self.mask.dtype != np.bool_:
            raise PreprocessingError("breast mask must have dtype bool")
        if not np.isfinite(self.image).all():
            raise PreprocessingError("preprocessed image contains non-finite values")
        if not bool(self.mask.any()):
            raise PreprocessingError("breast mask is empty")
        if self.laterality not in {"L", "R"}:
            raise PreprocessingError(f"unsupported laterality: {self.laterality!r}")
        if self.view_name not in {"CC", "MLO"}:
            raise PreprocessingError(f"unsupported view name: {self.view_name!r}")
        if tuple(int(value) for value in self.original_shape) != self.image.shape:
            raise PreprocessingError(
                "original DICOM shape does not match standardized image shape"
            )
        tolerance = float(np.finfo(np.float32).eps)
        if (
            float(self.image.min()) < -tolerance
            or float(self.image.max()) > 1.0 + tolerance
        ):
            raise PreprocessingError("preprocessed image must lie in [0, 1]")
        if bool(np.any(self.image[~self.mask] != 0.0)):
            raise PreprocessingError("pixels outside the breast mask must be zero")
        if self.mask_strategy not in MASK_STRATEGIES:
            raise PreprocessingError(
                f"unsupported breast-mask strategy: {self.mask_strategy!r}"
            )


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest 8-connected foreground component."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise PreprocessingError("breast mask must be two-dimensional")
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(binary)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labels == largest_label


def _extract_breast_mask_with_strategy(
    image: np.ndarray,
    *,
    foreground_support: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Segment breast tissue and retain its maximum connected region.

    Polarity must be standardized before this function is called.  Hole
    filling is applied after selecting the largest component so internal dark
    structures do not fragment the usable breast area.  When DICOM padding
    supplies an independent foreground support, it is used as a conservative
    topology guard: if the globally brightest Otsu component lies entirely in
    a smaller disconnected acquisition artifact, the largest non-padding
    component is the breast mask instead.
    """

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise PreprocessingError("mammogram must be a two-dimensional array")
    finite = np.isfinite(array)
    values = array[finite]
    if values.size == 0:
        raise PreprocessingError("mammogram contains no finite pixels")
    if float(np.ptp(values)) <= np.finfo(np.float32).eps:
        raise PreprocessingError("constant mammogram has no separable breast region")

    threshold = float(threshold_otsu(values))
    candidate = finite & (array > threshold)
    component = largest_connected_component(candidate)
    if not bool(component.any()):
        raise PreprocessingError("Otsu thresholding produced an empty breast mask")

    strategy = "otsu_lcc"
    if foreground_support is not None:
        support = np.asarray(foreground_support, dtype=bool)
        if support.shape != array.shape:
            raise PreprocessingError(
                "foreground support shape does not match mammogram"
            )
        support_component = largest_connected_component(support & finite)
        if not bool(support_component.any()):
            raise PreprocessingError("foreground support is empty")
        if not bool((component & support_component).any()):
            component = support_component
            strategy = "nonpadding_lcc_topology_fallback"
    filled = ndimage.binary_fill_holes(component)
    mask = largest_connected_component(filled).astype(bool, copy=False)
    return mask, strategy


def extract_breast_mask(
    image: np.ndarray,
    *,
    foreground_support: np.ndarray | None = None,
) -> np.ndarray:
    """Return the deterministic breast mask; see the topology guard above."""

    mask, _ = _extract_breast_mask_with_strategy(
        image,
        foreground_support=foreground_support,
    )
    return mask


def apply_grayscale_polarity(
    pixels: np.ndarray,
    photometric_interpretation: str,
    *,
    padding_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert MONOCHROME1/2 pixels to a bright-tissue representation."""

    image = np.asarray(pixels, dtype=np.float32).copy()
    if image.ndim != 2:
        raise PreprocessingError("DICOM pixel array must be two-dimensional")
    finite = np.isfinite(image)
    if padding_mask is not None:
        padding = np.asarray(padding_mask, dtype=bool)
        if padding.shape != image.shape:
            raise PreprocessingError("padding mask shape does not match pixels")
    else:
        padding = np.zeros_like(image, dtype=bool)
    tissue_candidates = finite & ~padding
    if not bool(tissue_candidates.any()):
        raise PreprocessingError("DICOM contains no non-padding finite pixels")

    polarity = str(photometric_interpretation).strip().upper()
    if polarity == "MONOCHROME1":
        values = image[tissue_candidates]
        image = float(values.min() + values.max()) - image
    elif polarity != "MONOCHROME2":
        raise PreprocessingError(
            f"Unsupported PhotometricInterpretation: {photometric_interpretation!r}"
        )

    # Padding is background even after MONOCHROME1 inversion.  Resetting it to
    # the standardized minimum prevents a bright padding frame from becoming
    # the largest connected component.
    standardized_min = float(image[tissue_candidates].min())
    image[padding | ~finite] = standardized_min
    return image.astype(np.float32, copy=False)


def standardize_laterality(
    image: np.ndarray,
    mask: np.ndarray,
    laterality: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Orient VinDr breasts with the chest wall on the left.

    In the raw VinDr DICOMs, L images already have the chest wall on the left,
    while R images have it on the right.  R images are therefore mirrored;
    the actual view name is never changed.
    """

    side = str(laterality).strip().upper()
    if side not in {"L", "R"}:
        raise PreprocessingError(f"Unsupported laterality: {laterality!r}")
    array = np.asarray(image)
    binary = np.asarray(mask, dtype=bool)
    if array.shape != binary.shape:
        raise PreprocessingError("image and mask shapes do not match")
    flipped = side == "R"
    if flipped:
        return (
            np.ascontiguousarray(array[:, ::-1]),
            np.ascontiguousarray(binary[:, ::-1]),
            True,
        )
    return np.ascontiguousarray(array), np.ascontiguousarray(binary), False


def percentile_normalize(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Clip p1-p99 inside the breast and scale it to [0, 1]."""

    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise PreprocessingError("percentiles must satisfy 0 <= lower < upper <= 100")
    array = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    if array.shape != binary.shape or array.ndim != 2:
        raise PreprocessingError("image and mask must be matching 2D arrays")
    values = array[binary & np.isfinite(array)]
    if values.size == 0:
        raise PreprocessingError("cannot normalize an empty breast mask")
    low, high = np.percentile(values, (lower_percentile, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise PreprocessingError(
            "breast intensities have a degenerate percentile range"
        )
    normalized = (np.clip(array, low, high) - low) / (high - low)
    normalized[~binary] = 0.0
    return normalized.astype(np.float32, copy=False)


def _padding_mask(dataset: pydicom.dataset.Dataset, pixels: np.ndarray) -> np.ndarray:
    value = dataset.get("PixelPaddingValue")
    if value is None:
        padding = np.zeros_like(pixels, dtype=bool)
    else:
        try:
            lower = float(value)
        except (TypeError, ValueError):
            padding = np.zeros_like(pixels, dtype=bool)
        else:
            limit = dataset.get("PixelPaddingRangeLimit")
            if limit is None:
                padding = np.asarray(pixels) == lower
            else:
                try:
                    upper = float(limit)
                except (TypeError, ValueError):
                    padding = np.asarray(pixels) == lower
                else:
                    low, high = sorted((lower, upper))
                    padding = (np.asarray(pixels) >= low) & (np.asarray(pixels) <= high)

    # Some VinDr images contain both tagged padding (usually 10000) and a
    # large exact-zero border field.  MONOCHROME1 would invert that untagged
    # field to maximum brightness, causing Otsu to select constant background.
    # Add only the dominant exact border code when it occupies a material
    # image region; do not threshold away low-valued breast tissue.
    array = np.asarray(pixels)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    border_values, border_counts = np.unique(border, return_counts=True)
    if border_values.size:
        dominant = border_values[int(np.argmax(border_counts))]
        dominant_mask = array == dominant
        if float(dominant_mask.mean()) >= 0.01:
            padding = padding | dominant_mask
    return np.asarray(padding, dtype=bool)


def preprocess_array(
    pixels: np.ndarray,
    *,
    laterality: str,
    view_name: str,
    photometric_interpretation: str = "MONOCHROME2",
    padding_mask: np.ndarray | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    source_path: str = "",
) -> PreprocessedMammogram:
    """Preprocess an already decoded array (also useful for deterministic tests)."""

    polarity_image = apply_grayscale_polarity(
        pixels,
        photometric_interpretation,
        padding_mask=padding_mask,
    )
    foreground_support = None
    if padding_mask is not None:
        foreground_support = ~np.asarray(padding_mask, dtype=bool)
    mask, mask_strategy = _extract_breast_mask_with_strategy(
        polarity_image,
        foreground_support=foreground_support,
    )
    oriented_image, oriented_mask, flipped = standardize_laterality(
        polarity_image, mask, laterality
    )
    normalized = percentile_normalize(
        oriented_image,
        oriented_mask,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    return PreprocessedMammogram(
        image=normalized,
        mask=oriented_mask.astype(bool, copy=False),
        laterality=str(laterality).strip().upper(),
        view_name=str(view_name).strip().upper(),
        original_shape=tuple(int(value) for value in np.asarray(pixels).shape),
        horizontally_flipped=flipped,
        source_path=source_path,
        mask_strategy=mask_strategy,
    )


def preprocess_dicom(
    path: str | Path,
    *,
    laterality: str | None = None,
    view_name: str | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> PreprocessedMammogram:
    """Read and preprocess a raw VinDr DICOM without applying a display window."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    dataset = pydicom.dcmread(source)
    raw_pixels = np.asarray(dataset.pixel_array)
    if raw_pixels.ndim != 2:
        raise PreprocessingError(
            f"Expected a single-frame 2D DICOM, got shape {raw_pixels.shape}"
        )
    padding = _padding_mask(dataset, raw_pixels)
    slope = float(dataset.get("RescaleSlope", 1.0))
    intercept = float(dataset.get("RescaleIntercept", 0.0))
    rescaled = raw_pixels.astype(np.float32) * slope + intercept

    resolved_laterality = (
        laterality or dataset.get("ImageLaterality") or dataset.get("Laterality")
    )
    resolved_view = view_name or dataset.get("ViewPosition")
    if resolved_laterality is None:
        raise PreprocessingError(f"DICOM has no laterality: {source}")
    if resolved_view is None:
        raise PreprocessingError(f"DICOM has no ViewPosition: {source}")
    return preprocess_array(
        rescaled,
        laterality=str(resolved_laterality),
        view_name=str(resolved_view),
        photometric_interpretation=str(dataset.get("PhotometricInterpretation", "")),
        padding_mask=padding,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        source_path=str(source),
    )


_CACHE_PROTOCOL_VERSION = "vindr-polarity-padding-topology-mask-laterality-p1p99-v2"


@dataclass(frozen=True)
class DICOMPreprocessor:
    """Configurable DICOM preprocessor with an optional atomic disk cache.

    Cached arrays are the deterministic result before gamma/contrast
    augmentation.  ``cache_read_only`` is useful for validation/inference on
    a cache populated during preparation; a miss is computed normally but is
    not written.
    """

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    cache_dir: str | Path | None = None
    cache_read_only: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise PreprocessingError(
                "percentiles must satisfy 0 <= lower < upper <= 100"
            )

    def _cache_path(
        self,
        source: Path,
        laterality: str | None,
        view_name: str | None,
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        stat = source.stat()
        identity = {
            "protocol": _CACHE_PROTOCOL_VERSION,
            "source": str(source.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "laterality": str(laterality or "").strip().upper(),
            "view_name": str(view_name or "").strip().upper(),
            "lower_percentile": float(self.lower_percentile),
            "upper_percentile": float(self.upper_percentile),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return Path(self.cache_dir) / digest[:2] / f"{digest}.npz"

    @staticmethod
    def _load_cached(path: Path) -> PreprocessedMammogram | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cached:
                protocol = str(cached["protocol"].item())
                if protocol != _CACHE_PROTOCOL_VERSION:
                    return None
                image = np.asarray(cached["image"], dtype=np.float32)
                mask = np.asarray(cached["mask"], dtype=bool)
                laterality = str(cached["laterality"].item())
                view_name = str(cached["view_name"].item())
                original_shape_values = np.asarray(
                    cached["original_shape"], dtype=np.int64
                ).tolist()
                flipped = bool(cached["horizontally_flipped"].item())
                source_path = str(cached["source_path"].item())
                mask_strategy = str(cached["mask_strategy"].item())
            if (
                not isinstance(original_shape_values, list)
                or len(original_shape_values) != 2
            ):
                return None
            return PreprocessedMammogram(
                image=np.ascontiguousarray(image),
                mask=np.ascontiguousarray(mask),
                laterality=laterality,
                view_name=view_name,
                original_shape=(
                    int(original_shape_values[0]),
                    int(original_shape_values[1]),
                ),
                horizontally_flipped=flipped,
                source_path=source_path,
                mask_strategy=mask_strategy,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            EOFError,
            zipfile.BadZipFile,
            PreprocessingError,
        ):
            # An interrupted external copy or stale incompatible file is a
            # cache miss; the raw DICOM remains the source of truth.
            return None

    @staticmethod
    def _write_cached(path: Path, result: PreprocessedMammogram) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{path.stem}.",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                np.savez_compressed(
                    temporary,
                    protocol=np.asarray(_CACHE_PROTOCOL_VERSION),
                    image=result.image,
                    mask=result.mask.astype(np.uint8, copy=False),
                    laterality=np.asarray(result.laterality),
                    view_name=np.asarray(result.view_name),
                    original_shape=np.asarray(result.original_shape, dtype=np.int64),
                    horizontally_flipped=np.asarray(
                        result.horizontally_flipped, dtype=np.bool_
                    ),
                    source_path=np.asarray(result.source_path),
                    mask_strategy=np.asarray(result.mask_strategy),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.replace(temporary_name, path)
                temporary_name = None
            except OSError:
                # Another DataLoader worker may have won the same cache race.
                # Only a fully validated winner is sufficient; mere existence
                # could be a truncated file from another interrupted writer.
                if DICOMPreprocessor._load_cached(path) is None:
                    raise
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass

    def __call__(
        self,
        path: str | Path,
        *,
        laterality: str | None = None,
        view_name: str | None = None,
    ) -> PreprocessedMammogram:
        source = Path(path)
        cache_path = self._cache_path(source, laterality, view_name)
        if cache_path is not None:
            cached = self._load_cached(cache_path)
            if cached is not None:
                return cached
        result = preprocess_dicom(
            source,
            laterality=laterality,
            view_name=view_name,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        if cache_path is not None and not self.cache_read_only:
            self._write_cached(cache_path, result)
        return result


@dataclass(frozen=True)
class IntensityAugment:
    """Mild gamma/contrast augmentation with no geometric transforms."""

    gamma_probability: float = 0.5
    gamma_range: tuple[float, float] = (0.9, 1.1)
    contrast_probability: float = 0.5
    contrast_range: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self) -> None:
        for name, probability in (
            ("gamma_probability", self.gamma_probability),
            ("contrast_probability", self.contrast_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise PreprocessingError(f"{name} must lie in [0, 1]")
        for name, interval in (
            ("gamma_range", self.gamma_range),
            ("contrast_range", self.contrast_range),
        ):
            if len(interval) != 2 or interval[0] <= 0 or interval[1] < interval[0]:
                raise PreprocessingError(f"Invalid {name}: {interval}")

    @staticmethod
    def _random(rng: np.random.Generator | None) -> float:
        return float(rng.random() if rng is not None else np.random.random())

    @staticmethod
    def _uniform(
        interval: tuple[float, float], rng: np.random.Generator | None
    ) -> float:
        if interval[0] == interval[1]:
            return float(interval[0])
        if rng is not None:
            return float(rng.uniform(interval[0], interval[1]))
        return float(np.random.uniform(interval[0], interval[1]))

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32).copy()
        binary = np.asarray(mask, dtype=bool)
        if array.shape != binary.shape:
            raise PreprocessingError("image and mask shapes do not match")

        if self._random(rng) < self.gamma_probability:
            gamma = self._uniform(self.gamma_range, rng)
            array[binary] = np.power(np.clip(array[binary], 0.0, 1.0), gamma)
        if self._random(rng) < self.contrast_probability:
            factor = self._uniform(self.contrast_range, rng)
            mean = float(array[binary].mean())
            array[binary] = (array[binary] - mean) * factor + mean
        np.clip(array, 0.0, 1.0, out=array)
        array[~binary] = 0.0
        return array.astype(np.float32, copy=False)


def transform_bboxes_for_laterality(
    bboxes: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    laterality: str,
) -> list[dict[str, Any]]:
    """Clip raw boxes to the image and map them into evaluation coordinates.

    VinDr contains a small number of annotations whose edges extend a few pixels
    beyond the stored DICOM canvas.  A box is usable when it has a non-empty
    intersection with the image; only that intersection is evaluated.  Malformed,
    degenerate, or wholly disjoint boxes still fail closed.
    """

    side = str(laterality).strip().upper()
    if side not in {"L", "R"}:
        raise PreprocessingError(f"Unsupported laterality: {laterality!r}")
    image_width = float(width)
    image_height = float(height)
    if not np.isfinite(image_width) or image_width <= 0.0:
        raise PreprocessingError(f"invalid image width: {width!r}")
    if not np.isfinite(image_height) or image_height <= 0.0:
        raise PreprocessingError(f"invalid image height: {height!r}")
    transformed: list[dict[str, Any]] = []
    for original in bboxes:
        box = dict(original)
        try:
            xmin, ymin, xmax, ymax = (
                float(original[key]) for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreprocessingError(f"invalid bounding box: {original!r}") from exc
        if not np.isfinite((xmin, ymin, xmax, ymax)).all():
            raise PreprocessingError(f"non-finite bounding box: {original!r}")
        if xmin >= xmax or ymin >= ymax:
            raise PreprocessingError(f"degenerate bounding box: {original!r}")

        xmin = max(0.0, xmin)
        xmax = min(image_width, xmax)
        ymin = max(0.0, ymin)
        ymax = min(image_height, ymax)
        if xmin >= xmax or ymin >= ymax:
            raise PreprocessingError(
                f"bounding box does not intersect the image: {original!r}"
            )

        if side == "R":
            box["xmin"] = image_width - xmax
            box["xmax"] = image_width - xmin
        else:
            box["xmin"] = xmin
            box["xmax"] = xmax
        box["ymin"] = ymin
        box["ymax"] = ymax
        transformed.append(box)
    return transformed
