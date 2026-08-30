"""Boundary-complete mammogram patching and anatomical descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


class PatchExtractionError(ValueError):
    """Raised when a view cannot produce a protocol-valid candidate bag."""


VIEW_TO_ID = {"CC": 0, "MLO": 1}


@dataclass(frozen=True)
class PatchGridCoverage:
    """Canonical row-major patch grid and exact breast coverages."""

    starts_xy: np.ndarray  # [G, 2], x/y starts in row-major grid order.
    coverage: np.ndarray  # [G], float64 fractions from int64 window sums.
    retained: np.ndarray  # [G], the manuscript's inclusive threshold test.
    image_shape: tuple[int, int]
    padded_shape: tuple[int, int]

    @property
    def maximum_coverage(self) -> float:
        return float(self.coverage.max(initial=0.0))

    @property
    def retained_count(self) -> int:
        return int(self.retained.sum())


@dataclass(frozen=True)
class BreastGeometry:
    """References measured in standardized (chest-wall-left) coordinates."""

    bbox_xyxy: tuple[int, int, int, int]
    nipple_xy: tuple[float, float]
    chest_wall_x: float
    diagonal: float


@dataclass(frozen=True)
class PatchBag:
    """A variable-length single-view bag before conversion to torch tensors."""

    patches: np.ndarray  # [N, H, W], float32; Dataset exposes zero-copy 3 channels.
    anatomy_base: np.ndarray  # [N, 5]: x, y, d_nipple, d_chest, rho.
    anatomy: np.ndarray  # [N, 6]: anatomy_base followed by the discrete view id.
    coords: np.ndarray  # [N, 4], standardized xyxy coordinates.
    centers: np.ndarray  # [N, 2], standardized absolute xy centers.
    coverage: np.ndarray  # [N].
    view_name: str
    view_id: int
    image_shape: tuple[int, int]
    geometry: BreastGeometry

    def __post_init__(self) -> None:
        count = int(self.patches.shape[0])
        if count == 0:
            raise PatchExtractionError("a view bag must contain at least one patch")
        expected = {
            "anatomy_base": self.anatomy_base.shape[0],
            "anatomy": self.anatomy.shape[0],
            "coords": self.coords.shape[0],
            "centers": self.centers.shape[0],
            "coverage": self.coverage.shape[0],
        }
        mismatched = {name: size for name, size in expected.items() if size != count}
        if mismatched:
            raise PatchExtractionError(f"patch metadata length mismatch: {mismatched}")


def sliding_window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return grid starts and always include the final valid boundary start."""

    if length <= 0 or patch_size <= 0 or stride <= 0:
        raise PatchExtractionError("length, patch_size, and stride must be positive")
    if length <= patch_size:
        return [0]
    last = length - patch_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def patch_grid_coverage(
    mask: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    minimum_breast_coverage: float,
) -> PatchGridCoverage:
    """Build the single source of truth for candidate coordinates/coverage.

    The integral image uses ``int64`` so a full-resolution mammogram cannot
    overflow.  Padding is bottom/right only and therefore contributes zero
    breast pixels to the fixed ``patch_size ** 2`` denominator.
    """

    if not 0.0 <= minimum_breast_coverage <= 1.0:
        raise PatchExtractionError("minimum breast coverage must lie in [0, 1]")
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not bool(binary.any()):
        raise PatchExtractionError("breast mask must be a non-empty 2D array")
    height, width = (int(value) for value in binary.shape)
    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = binary
    integral = np.pad(
        padded.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64),
        ((1, 0), (1, 0)),
    )
    ys = np.asarray(
        sliding_window_positions(padded_height, patch_size, stride),
        dtype=np.int64,
    )
    xs = np.asarray(
        sliding_window_positions(padded_width, patch_size, stride),
        dtype=np.int64,
    )
    y0, x0 = np.meshgrid(ys, xs, indexing="ij")
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    breast_pixels = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )
    coverage = np.ascontiguousarray(
        breast_pixels.reshape(-1).astype(np.float64) / float(patch_size * patch_size)
    )
    starts_xy = np.ascontiguousarray(
        np.stack((x0.reshape(-1), y0.reshape(-1)), axis=1), dtype=np.int64
    )
    retained = np.ascontiguousarray(
        coverage + np.finfo(np.float32).eps >= minimum_breast_coverage
    )
    return PatchGridCoverage(
        starts_xy=starts_xy,
        coverage=coverage,
        retained=retained,
        image_shape=(height, width),
        padded_shape=(padded_height, padded_width),
    )


def breast_geometry(mask: np.ndarray) -> BreastGeometry:
    """Estimate breast bbox, nipple point, and chest-wall reference.

    The nipple is the standardized breast-contour point farthest from the
    chest-wall boundary.  If several pixels share that maximum x coordinate,
    their median y coordinate gives a stable reference.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not bool(binary.any()):
        raise PatchExtractionError("cannot compute geometry from an empty 2D mask")
    y_indices, x_indices = np.nonzero(binary)
    xmin = int(x_indices.min())
    xmax = int(x_indices.max()) + 1
    ymin = int(y_indices.min())
    ymax = int(y_indices.max()) + 1

    eroded = ndimage.binary_erosion(
        binary, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    contour = binary & ~eroded
    contour_y, contour_x = np.nonzero(contour)
    farthest_x = int(contour_x.max())
    farthest_y = contour_y[contour_x == farthest_x]
    nipple = (float(farthest_x), float(np.median(farthest_y)))
    diagonal = float(np.hypot(max(xmax - xmin, 1), max(ymax - ymin, 1)))
    return BreastGeometry(
        bbox_xyxy=(xmin, ymin, xmax, ymax),
        nipple_xy=nipple,
        chest_wall_x=float(xmin),
        diagonal=max(diagonal, np.finfo(np.float32).eps),
    )


def _canonical_view(view_name: str) -> str:
    view = str(view_name).strip().upper().replace("-", "")
    aliases = {
        "CRANIOCAUDAL": "CC",
        "MEDIOLATERALOBLIQUE": "MLO",
        "MEDIOLATERAL OBLIQUE": "MLO",
    }
    view = aliases.get(view, view)
    if view not in VIEW_TO_ID:
        raise PatchExtractionError(f"Unsupported view name: {view_name!r}")
    return view


def _anatomy_descriptor(
    center_x: float,
    center_y: float,
    rho: float,
    geometry: BreastGeometry,
) -> tuple[float, float, float, float, float]:
    xmin, ymin, xmax, ymax = geometry.bbox_xyxy
    width = max(xmax - xmin, 1)
    height = max(ymax - ymin, 1)
    normalized_x = float(np.clip((center_x - xmin) / width, 0.0, 1.0))
    normalized_y = float(np.clip((center_y - ymin) / height, 0.0, 1.0))
    nipple_x, nipple_y = geometry.nipple_xy
    nipple_distance = float(
        np.hypot(center_x - nipple_x, center_y - nipple_y) / geometry.diagonal
    )
    chest_distance = float(abs(center_x - geometry.chest_wall_x) / geometry.diagonal)
    return (
        normalized_x,
        normalized_y,
        nipple_distance,
        chest_distance,
        float(rho),
    )


def extract_patch_bag(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    view_name: str,
    patch_size: int = 512,
    stride: int = 512,
    minimum_breast_coverage: float = 0.5,
) -> PatchBag:
    """Extract a variable-length bag using the formal patch protocol.

    Border starts are appended when a stride grid does not land on the final
    boundary.  Images smaller than a patch are padded at bottom/right with
    zeros; padding counts as non-breast area in the coverage denominator.
    """

    array = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or binary.ndim != 2 or array.shape != binary.shape:
        raise PatchExtractionError("image and mask must be matching 2D arrays")
    if not bool(binary.any()):
        raise PatchExtractionError("breast mask is empty")
    if not np.isfinite(array).all():
        raise PatchExtractionError("image contains non-finite values")

    view = _canonical_view(view_name)
    view_id = VIEW_TO_ID[view]
    height, width = (int(value) for value in array.shape)
    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)
    if (padded_height, padded_width) == (height, width):
        padded_image = array
        padded_mask = binary
    else:
        padded_image = np.zeros((padded_height, padded_width), dtype=np.float32)
        padded_mask = np.zeros((padded_height, padded_width), dtype=bool)
        padded_image[:height, :width] = array
        padded_mask[:height, :width] = binary

    grid = patch_grid_coverage(
        binary,
        patch_size=patch_size,
        stride=stride,
        minimum_breast_coverage=minimum_breast_coverage,
    )
    geometry = breast_geometry(binary)
    patches: list[np.ndarray] = []
    anatomy_base: list[tuple[float, float, float, float, float]] = []
    coords: list[tuple[int, int, int, int]] = []
    centers: list[tuple[float, float]] = []
    coverages: list[float] = []

    retained_starts = grid.starts_xy[grid.retained]
    retained_coverages = grid.coverage[grid.retained]
    for (x0_raw, y0_raw), coverage_raw in zip(
        retained_starts, retained_coverages, strict=True
    ):
        x0, y0 = int(x0_raw), int(y0_raw)
        y1 = y0 + patch_size
        x1 = x0 + patch_size
        patch_mask = padded_mask[y0:y1, x0:x1]
        patch = padded_image[y0:y1, x0:x1]
        breast_values = patch[patch_mask]
        rho = float(breast_values.mean()) if breast_values.size else 0.0
        center_x = x0 + patch_size / 2.0
        center_y = y0 + patch_size / 2.0
        patches.append(np.ascontiguousarray(patch, dtype=np.float32))
        anatomy_base.append(_anatomy_descriptor(center_x, center_y, rho, geometry))
        coords.append((x0, y0, min(x1, width), min(y1, height)))
        centers.append((center_x, center_y))
        coverages.append(float(coverage_raw))

    if not patches:
        raise PatchExtractionError(
            "No patch meets the breast-coverage threshold "
            f"{minimum_breast_coverage:.3f}; maximum coverage was "
            f"{grid.maximum_coverage:.3f}"
        )

    anatomy_base_array = np.asarray(anatomy_base, dtype=np.float32)
    view_column = np.full((len(patches), 1), float(view_id), dtype=np.float32)
    return PatchBag(
        patches=np.stack(patches).astype(np.float32, copy=False),
        anatomy_base=anatomy_base_array,
        anatomy=np.concatenate((anatomy_base_array, view_column), axis=1),
        coords=np.asarray(coords, dtype=np.int64),
        centers=np.asarray(centers, dtype=np.float32),
        coverage=np.asarray(coverages, dtype=np.float32),
        view_name=view,
        view_id=view_id,
        image_shape=(height, width),
        geometry=geometry,
    )
