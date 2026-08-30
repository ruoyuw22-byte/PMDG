"""Breast-level CC/MLO Dataset and memory-conscious ragged collation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .patches import VIEW_TO_ID, PatchExtractionError, extract_patch_bag
from .preprocessing import (
    DICOMPreprocessor,
    IntensityAugment,
    PreprocessedMammogram,
    PreprocessingError,
    transform_bboxes_for_laterality,
)


class DatasetError(ValueError):
    """Raised when a manifest row cannot form a valid breast sample."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n", ""}:
        return False
    raise DatasetError(f"Cannot parse boolean value: {value!r}")


def _read_manifest(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"exam_id": str, "study_id": str})


def _parse_json_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        parsed = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        parsed = []
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON in {field}: {exc}") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) for item in parsed
    ):
        raise DatasetError(f"{field} must encode a list of objects")
    return [dict(item) for item in parsed]


def _legacy_view_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for view_name in ("CC", "MLO"):
        prefix = view_name.lower()
        image_id = row.get(f"{prefix}_image_id", "")
        path = row.get(f"{prefix}_path", "")
        if image_id is None or not str(image_id).strip():
            continue
        boxes = _parse_json_list(
            row.get(f"{prefix}_bboxes_json", "[]"),
            field=f"{prefix}_bboxes_json",
        )
        records.append(
            {
                "image_id": str(image_id),
                "view_name": view_name,
                "slot": view_name,
                "path": str(path),
                "bboxes": boxes,
            }
        )
    return records


def _view_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "views_json" in row and row.get("views_json") is not None:
        records = _parse_json_list(row.get("views_json"), field="views_json")
    else:
        records = _legacy_view_records(row)
    valid: list[dict[str, Any]] = []
    for record in records:
        view_name = str(record.get("view_name", "")).strip().upper()
        if view_name not in VIEW_TO_ID:
            continue
        record["view_name"] = view_name
        valid.append(record)
    return valid


class VinDrBreastDataset(Dataset[dict[str, Any]]):
    """Create one variable-size bag for each available ipsilateral view.

    The returned ``patches`` tensors have shape ``[N, 1, P, P]``.  Replication
    to the three channels expected by Mammo-CLIP occurs only inside each GPU
    micro-batch; this keeps collation and pinned-memory transfer free of both
    redundant storage and overlapping-stride writes.  Training
    samples cannot expose boxes: setting ``include_bboxes=True`` while
    ``training=True`` is rejected rather than relying on downstream code to
    ignore privileged localization labels.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        split: str | None = None,
        training: bool | None = None,
        preprocessor: Callable[..., PreprocessedMammogram] | None = None,
        augmentation: IntensityAugment | None = None,
        patch_size: int = 512,
        train_stride: int = 512,
        eval_stride: int = 128,
        minimum_breast_coverage: float = 0.5,
        augmentation_seed: int = 2026,
        strict_two_view_pair: bool = False,
        include_bboxes: bool = False,
    ) -> None:
        frame = _read_manifest(manifest)
        required = {"exam_id", "laterality", "mass", "calcification"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DatasetError(f"Manifest is missing required columns: {missing}")
        if split is not None:
            if "split" not in frame.columns:
                raise DatasetError(
                    "A split was requested but manifest has no split column"
                )
            frame = frame.loc[frame["split"].astype(str) == str(split)].copy()
        if frame.empty:
            raise DatasetError(f"No samples remain after split filtering: {split!r}")

        if training is None:
            training = split == "train"
        if training and include_bboxes:
            raise DatasetError(
                "Bounding boxes are post-hoc evaluation metadata and cannot be "
                "exposed by a training Dataset"
            )
        if patch_size <= 0 or train_stride <= 0 or eval_stride <= 0:
            raise DatasetError("patch size and strides must be positive")

        self.manifest = frame.reset_index(drop=True)
        self.training = bool(training)
        self.preprocessor = preprocessor or DICOMPreprocessor()
        self.augmentation = augmentation or IntensityAugment()
        self.patch_size = int(patch_size)
        self.train_stride = int(train_stride)
        self.eval_stride = int(eval_stride)
        self.minimum_breast_coverage = float(minimum_breast_coverage)
        if int(augmentation_seed) < 0:
            raise DatasetError("augmentation seed must be non-negative")
        self.augmentation_seed = int(augmentation_seed)
        self.strict_two_view_pair = bool(strict_two_view_pair)
        self.include_bboxes = bool(include_bboxes)

    def __len__(self) -> int:
        return len(self.manifest)

    def make_collator(
        self, *, flatten_patches: bool | None = None
    ) -> "RaggedBreastCollator":
        """Return a mode-safe collator.

        Training defaults to flat tensors for online backbone micro-batching.
        Evaluation defaults to hierarchical bags so stride-128 expanded RGB
        views are not copied into one potentially multi-gigabyte tensor.
        """

        if flatten_patches is None:
            flatten_patches = self.training
        return RaggedBreastCollator(flatten_patches=bool(flatten_patches))

    def _load_view(
        self,
        record: Mapping[str, Any],
        *,
        laterality: str,
    ) -> PreprocessedMammogram:
        path = str(record.get("path", ""))
        try:
            processed = self.preprocessor(
                path,
                laterality=laterality,
                view_name=str(record["view_name"]),
            )
        except (PreprocessingError, TypeError) as exc:
            raise DatasetError(f"Failed to preprocess DICOM {path}: {exc}") from exc
        if not isinstance(processed, PreprocessedMammogram):
            raise DatasetError(
                "preprocessor must return a PreprocessedMammogram, got "
                f"{type(processed).__name__}"
            )
        if processed.laterality != laterality:
            raise DatasetError(
                f"Laterality mismatch for {path}: manifest={laterality}, "
                f"DICOM={processed.laterality}"
            )
        expected_view = str(record["view_name"]).strip().upper()
        if processed.view_name != expected_view:
            raise DatasetError(
                f"View mismatch for {path}: manifest={expected_view}, "
                f"DICOM={processed.view_name}"
            )
        expected_height = record.get("height")
        expected_width = record.get("width")
        if expected_height is not None and expected_width is not None:
            try:
                expected_shape = (int(expected_height), int(expected_width))
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"Invalid manifest dimensions for {path}: "
                    f"{expected_height!r}x{expected_width!r}"
                ) from exc
            if processed.original_shape != expected_shape:
                raise DatasetError(
                    f"Image shape mismatch for {path}: manifest={expected_shape}, "
                    f"DICOM={processed.original_shape}"
                )
        return processed

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise DatasetError(f"Invalid epoch-tagged sample index: {index!r}")
            epoch, sample_index = (int(value) for value in index)
        else:
            epoch, sample_index = 0, int(index)
        if epoch < 0:
            raise DatasetError("augmentation epoch must be non-negative")
        row = self.manifest.iloc[sample_index].to_dict()
        exam_id = str(row["exam_id"])
        laterality = str(row["laterality"]).strip().upper()
        records = _view_records(row)
        if not records:
            raise DatasetError(f"Breast {exam_id}_{laterality} has no valid views")

        counts = {
            view: sum(record["view_name"] == view for record in records)
            for view in VIEW_TO_ID
        }
        actual_pair_valid = counts["CC"] == 1 and counts["MLO"] == 1
        manifest_pair_valid = _as_bool(row.get("cross_view_valid", actual_pair_valid))
        cross_view_valid = bool(actual_pair_valid and manifest_pair_valid)
        if self.strict_two_view_pair and not cross_view_valid:
            raise DatasetError(
                f"Breast {exam_id}_{laterality} does not have exactly one CC and one MLO"
            )

        stride = self.train_stride if self.training else self.eval_stride
        seen = {view: 0 for view in VIEW_TO_ID}
        views: dict[str, dict[str, Any]] = {}
        for record in records:
            view_name = record["view_name"]
            seen[view_name] += 1
            fallback_slot = (
                view_name if seen[view_name] == 1 else f"{view_name}_{seen[view_name]}"
            )
            view_key = str(record.get("slot", fallback_slot)).strip() or fallback_slot
            if view_key in views:
                raise DatasetError(
                    f"Breast {exam_id}_{laterality} has duplicate view slot {view_key!r}"
                )
            processed = self._load_view(record, laterality=laterality)
            image = processed.image
            if self.training:
                augmentation_seed = np.random.SeedSequence(
                    (
                        self.augmentation_seed,
                        epoch,
                        sample_index,
                        VIEW_TO_ID[view_name],
                        seen[view_name],
                    )
                )
                image = self.augmentation(
                    image,
                    processed.mask,
                    rng=np.random.default_rng(augmentation_seed),
                )
            try:
                bag = extract_patch_bag(
                    image,
                    processed.mask,
                    view_name=view_name,
                    patch_size=self.patch_size,
                    stride=stride,
                    minimum_breast_coverage=self.minimum_breast_coverage,
                )
            except PatchExtractionError as exc:
                raise DatasetError(
                    f"Failed to patch {exam_id}_{laterality}/{view_key}: {exc}"
                ) from exc

            patches = torch.from_numpy(bag.patches).unsqueeze(1)
            view_data: dict[str, Any] = {
                "view_name": view_name,
                "view_id": bag.view_id,
                "cross_view_valid": cross_view_valid,
                "image_id": str(record.get("image_id", "")),
                "source_path": processed.source_path or str(record.get("path", "")),
                "mask_strategy": processed.mask_strategy,
                "patches": patches,
                "anatomy_base": torch.from_numpy(bag.anatomy_base),
                "anatomy": torch.from_numpy(bag.anatomy),
                "coords": torch.from_numpy(bag.coords),
                "coverage": torch.from_numpy(bag.coverage),
                "image_shape": bag.image_shape,
            }
            if self.include_bboxes:
                # Full-resolution masks are needed only by the fixed test-set
                # localization export.  Keeping them out of train/validation
                # batches avoids pinning several megabytes per view.
                view_data["mask"] = torch.from_numpy(
                    np.ascontiguousarray(processed.mask)
                )
                raw_boxes = record.get("bboxes", [])
                if not isinstance(raw_boxes, list):
                    raise DatasetError("view bboxes must be a list")
                view_data["evaluation_metadata"] = {
                    "bboxes": transform_bboxes_for_laterality(
                        raw_boxes,
                        width=processed.original_shape[1],
                        height=processed.original_shape[0],
                        laterality=laterality,
                    )
                }
            views[view_key] = view_data

        labels = torch.tensor(
            [int(row["mass"]), int(row["calcification"])], dtype=torch.float32
        )
        return {
            "exam_id": exam_id,
            "breast_id": str(row.get("breast_id", f"{exam_id}_{laterality}")),
            "laterality": laterality,
            "labels": labels,
            "views": views,
            "cross_view_valid": cross_view_valid,
        }


def collate_breast_bags(
    batch: Sequence[Mapping[str, Any]],
    *,
    flatten_patches: bool = False,
) -> dict[str, Any]:
    """Collate variable CC/MLO bags without padding to the largest bag.

    With ``flatten_patches=True`` all candidates are concatenated and indexed
    by ``sample_index``/``view_index``.  Returned per-sample ``views`` point to
    slices of the same flat storage, so the convenient hierarchical schema
    does not retain a second copy of patch pixels.
    """

    if not batch:
        raise DatasetError("Cannot collate an empty batch")
    labels = torch.stack([sample["labels"] for sample in batch])
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise DatasetError(f"Expected labels [B,2], got {tuple(labels.shape)}")

    patch_chunks: list[torch.Tensor] = []
    anatomy_base_chunks: list[torch.Tensor] = []
    anatomy_chunks: list[torch.Tensor] = []
    coords_chunks: list[torch.Tensor] = []
    coverage_chunks: list[torch.Tensor] = []
    sample_indices: list[torch.Tensor] = []
    view_indices: list[torch.Tensor] = []
    bag_slices: list[dict[str, tuple[int, int]]] = []
    slice_tensor = torch.full((len(batch), 2, 2), -1, dtype=torch.long)
    cursor = 0

    for sample_index, sample in enumerate(batch):
        sample_slices: dict[str, tuple[int, int]] = {}
        views = sample.get("views")
        if not isinstance(views, Mapping) or not views:
            raise DatasetError(f"Sample {sample_index} has no view bags")
        for view_key, view in views.items():
            patches = view["patches"]
            count = int(patches.shape[0])
            if count <= 0:
                raise DatasetError(f"Sample {sample_index}/{view_key} has an empty bag")
            start, end = cursor, cursor + count
            sample_slices[str(view_key)] = (start, end)
            view_id = int(view["view_id"])
            if view_id not in (0, 1):
                raise DatasetError(f"Invalid view id: {view_id}")
            if int(slice_tensor[sample_index, view_id, 0]) < 0:
                slice_tensor[sample_index, view_id] = torch.tensor((start, end))

            patch_chunks.append(patches)
            anatomy_base_chunks.append(view["anatomy_base"])
            anatomy_chunks.append(view["anatomy"])
            coords_chunks.append(view["coords"])
            coverage_chunks.append(view["coverage"])
            sample_indices.append(torch.full((count,), sample_index, dtype=torch.long))
            view_indices.append(torch.full((count,), view_id, dtype=torch.long))
            cursor = end
        bag_slices.append(sample_slices)

    flat: dict[str, torch.Tensor] = {
        "anatomy_base": torch.cat(anatomy_base_chunks, dim=0),
        "anatomy": torch.cat(anatomy_chunks, dim=0),
        "coords": torch.cat(coords_chunks, dim=0),
        "coverage": torch.cat(coverage_chunks, dim=0),
        "sample_index": torch.cat(sample_indices, dim=0),
        "view_index": torch.cat(view_indices, dim=0),
    }
    if flatten_patches:
        flat["patches"] = torch.cat(patch_chunks, dim=0)

    # Rebuild hierarchical view dictionaries.  When patches were flattened,
    # each view references its segment of the flat allocation.
    collated_views: list[dict[str, dict[str, Any]]] = []
    for sample_index, sample in enumerate(batch):
        sample_views: dict[str, dict[str, Any]] = {}
        for view_key, original in sample["views"].items():
            start, end = bag_slices[sample_index][str(view_key)]
            copied = dict(original)
            if flatten_patches:
                copied["patches"] = flat["patches"][start:end]
            copied["anatomy_base"] = flat["anatomy_base"][start:end]
            copied["anatomy"] = flat["anatomy"][start:end]
            copied["coords"] = flat["coords"][start:end]
            copied["coverage"] = flat["coverage"][start:end]
            sample_views[str(view_key)] = copied
        collated_views.append(sample_views)

    result: dict[str, Any] = {
        "exam_id": [str(sample["exam_id"]) for sample in batch],
        "breast_id": [str(sample["breast_id"]) for sample in batch],
        "laterality": [str(sample["laterality"]) for sample in batch],
        "labels": labels,
        "views": collated_views,
        "cross_view_valid": torch.tensor(
            [bool(sample["cross_view_valid"]) for sample in batch], dtype=torch.bool
        ),
        "bag_slices": bag_slices,
        "bag_slice_tensor": slice_tensor,
        **flat,
    }
    return result


@dataclass(frozen=True)
class RaggedBreastCollator:
    """Picklable DataLoader ``collate_fn`` wrapper."""

    flatten_patches: bool = False

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_breast_bags(batch, flatten_patches=self.flatten_patches)
