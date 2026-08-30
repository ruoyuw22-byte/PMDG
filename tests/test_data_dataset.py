from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dagl_mil.data.dataset import (
    DatasetError,
    RaggedBreastCollator,
    VinDrBreastDataset,
    collate_breast_bags,
)
from dagl_mil.data.preprocessing import IntensityAugment, PreprocessedMammogram
from dagl_mil.reproducibility import EpochSeededRandomSampler


def _processed(
    height: int,
    width: int,
    *,
    laterality: str,
    view: str,
    path: str,
) -> PreprocessedMammogram:
    image = np.linspace(0, 1, height * width, dtype=np.float32).reshape(height, width)
    mask = np.ones((height, width), dtype=bool)
    return PreprocessedMammogram(
        image=image,
        mask=mask,
        laterality=laterality,
        view_name=view,
        original_shape=(height, width),
        horizontally_flipped=laterality == "R",
        source_path=path,
    )


class MappingLoader:
    def __init__(self, mapping: dict[str, PreprocessedMammogram]) -> None:
        self.mapping = mapping

    def __call__(self, path: str, **_: object) -> PreprocessedMammogram:
        return self.mapping[path]


def _manifest_row(
    *,
    exam: str = "exam-1",
    laterality: str = "L",
    views: list[dict[str, object]],
    cross_view_valid: bool = True,
    mass: int = 1,
    calcification: int = 0,
) -> dict[str, object]:
    return {
        "exam_id": exam,
        "breast_id": f"{exam}_{laterality}",
        "laterality": laterality,
        "mass": mass,
        "calcification": calcification,
        "cross_view_valid": cross_view_valid,
        "views_json": json.dumps(views),
        "split": "train",
    }


def test_dataset_and_collate_preserve_variable_cc_mlo_bags() -> None:
    cc = _processed(8, 8, laterality="L", view="CC", path="cc")
    mlo = _processed(8, 12, laterality="L", view="MLO", path="mlo")
    views = [
        {"image_id": "cc-id", "view_name": "CC", "path": "cc", "bboxes": []},
        {
            "image_id": "mlo-id",
            "view_name": "MLO",
            "path": "mlo",
            "bboxes": [],
        },
    ]
    dataset = VinDrBreastDataset(
        pd.DataFrame([_manifest_row(views=views)]),
        training=False,
        preprocessor=MappingLoader({"cc": cc, "mlo": mlo}),
        patch_size=4,
        eval_stride=4,
    )
    sample = dataset[0]
    assert sample["exam_id"] == "exam-1"
    assert sample["laterality"] == "L"
    assert sample["labels"].tolist() == [1.0, 0.0]
    assert sample["cross_view_valid"]
    assert list(sample["views"]) == ["CC", "MLO"]
    assert sample["views"]["CC"]["patches"].shape == (4, 1, 4, 4)
    assert sample["views"]["MLO"]["patches"].shape == (6, 1, 4, 4)
    assert sample["views"]["CC"]["anatomy_base"].shape == (4, 5)
    assert sample["views"]["CC"]["view_name"] == "CC"
    assert sample["views"]["CC"]["patches"].is_contiguous()
    assert "mask" not in sample["views"]["CC"]
    assert "centers" not in sample["views"]["CC"]
    assert "breast_bbox" not in sample["views"]["CC"]

    batch = collate_breast_bags([sample, sample], flatten_patches=True)
    assert batch["exam_id"] == ["exam-1", "exam-1"]
    assert batch["laterality"] == ["L", "L"]
    assert batch["labels"].shape == (2, 2)
    assert batch["patches"].shape == (20, 1, 4, 4)
    assert batch["anatomy_base"].shape == (20, 5)
    assert batch["anatomy"].shape == (20, 6)
    assert batch["sample_index"].tolist() == [0] * 10 + [1] * 10
    assert batch["view_index"].tolist() == ([0] * 4 + [1] * 6) * 2
    assert batch["bag_slice_tensor"].tolist() == [
        [[0, 4], [4, 10]],
        [[10, 14], [14, 20]],
    ]
    assert batch["views"][0]["MLO"]["patches"].shape[0] == 6

    eval_batch = dataset.make_collator()([sample])
    assert "patches" not in eval_batch
    assert eval_batch["views"][0]["CC"]["patches"].shape == (4, 1, 4, 4)
    # The dataset keeps one contiguous grayscale channel.  The frozen backbone
    # expands each GPU micro-batch to RGB immediately before feature extraction.
    cc_patches = eval_batch["views"][0]["CC"]["patches"]
    assert cc_patches.shape[1] == 1
    assert cc_patches.is_contiguous()
    assert cc_patches.stride(1) == cc_patches.shape[2] * cc_patches.shape[3]


def test_duplicate_cc_is_kept_in_stable_slots_and_not_faked_as_mlo() -> None:
    first = _processed(8, 8, laterality="L", view="CC", path="cc-1")
    second = _processed(8, 8, laterality="L", view="CC", path="cc-2")
    views = [
        {"image_id": "1", "view_name": "CC", "path": "cc-1", "bboxes": []},
        {"image_id": "2", "view_name": "CC", "path": "cc-2", "bboxes": []},
    ]
    dataset = VinDrBreastDataset(
        pd.DataFrame([_manifest_row(views=views, cross_view_valid=False, mass=0)]),
        training=False,
        preprocessor=MappingLoader({"cc-1": first, "cc-2": second}),
        patch_size=4,
        eval_stride=4,
    )
    sample = dataset[0]
    assert list(sample["views"]) == ["CC", "CC_2"]
    assert {view["view_name"] for view in sample["views"].values()} == {"CC"}
    assert not sample["cross_view_valid"]

    batch = RaggedBreastCollator()([sample])
    assert set(batch["bag_slices"][0]) == {"CC", "CC_2"}
    assert torch.all(batch["view_index"] == 0)
    assert batch["bag_slice_tensor"][0, 1].tolist() == [-1, -1]


def test_dataset_fails_closed_on_manifest_shape_or_view_mismatch() -> None:
    processed = _processed(8, 8, laterality="L", view="CC", path="cc")
    wrong_shape = {
        "image_id": "1",
        "view_name": "CC",
        "path": "cc",
        "height": 7,
        "width": 8,
        "bboxes": [],
    }
    with pytest.raises(DatasetError, match="Image shape mismatch"):
        VinDrBreastDataset(
            pd.DataFrame(
                [_manifest_row(views=[wrong_shape], cross_view_valid=False, mass=0)]
            ),
            training=False,
            preprocessor=MappingLoader({"cc": processed}),
            patch_size=4,
            eval_stride=4,
        )[0]

    wrong_view = _processed(8, 8, laterality="L", view="MLO", path="cc")
    record = {**wrong_shape, "height": 8}
    with pytest.raises(DatasetError, match="View mismatch"):
        VinDrBreastDataset(
            pd.DataFrame(
                [_manifest_row(views=[record], cross_view_valid=False, mass=0)]
            ),
            training=False,
            preprocessor=MappingLoader({"cc": wrong_view}),
            patch_size=4,
            eval_stride=4,
        )[0]


def test_training_stride_and_eval_stride_follow_protocol_modes() -> None:
    processed = _processed(10, 10, laterality="L", view="CC", path="cc")
    views = [{"image_id": "1", "view_name": "CC", "path": "cc", "bboxes": []}]
    frame = pd.DataFrame([_manifest_row(views=views, cross_view_valid=False, mass=0)])
    no_augmentation = IntensityAugment(gamma_probability=0.0, contrast_probability=0.0)
    train = VinDrBreastDataset(
        frame,
        training=True,
        preprocessor=MappingLoader({"cc": processed}),
        augmentation=no_augmentation,
        patch_size=4,
        train_stride=4,
        eval_stride=2,
    )[0]
    evaluate = VinDrBreastDataset(
        frame,
        training=False,
        preprocessor=MappingLoader({"cc": processed}),
        patch_size=4,
        train_stride=4,
        eval_stride=2,
    )[0]
    assert train["views"]["CC"]["patches"].shape[0] == 9
    assert evaluate["views"]["CC"]["patches"].shape[0] == 16


def test_training_augmentation_is_stateless_across_resume_and_workers() -> None:
    processed = _processed(8, 8, laterality="L", view="CC", path="cc")
    views = [{"image_id": "1", "view_name": "CC", "path": "cc", "bboxes": []}]
    frame = pd.DataFrame([_manifest_row(views=views, cross_view_valid=False, mass=0)])
    dataset = VinDrBreastDataset(
        frame,
        training=True,
        preprocessor=MappingLoader({"cc": processed}),
        augmentation=IntensityAugment(
            gamma_probability=1.0,
            gamma_range=(0.7, 1.3),
            contrast_probability=1.0,
            contrast_range=(0.7, 1.3),
        ),
        augmentation_seed=2026,
        patch_size=4,
        train_stride=4,
    )

    epoch_three = dataset[(3, 0)]["views"]["CC"]["patches"]
    np.random.random(1000)
    resumed_epoch_three = dataset[(3, 0)]["views"]["CC"]["patches"]
    epoch_four = dataset[(4, 0)]["views"]["CC"]["patches"]

    torch.testing.assert_close(epoch_three, resumed_epoch_three, rtol=0, atol=0)
    assert not torch.equal(epoch_three, epoch_four)


def test_multiworker_resumed_epoch_matches_uninterrupted_epoch() -> None:
    views_by_path = {}
    rows = []
    for index in range(6):
        path = f"cc-{index}"
        views_by_path[path] = _processed(8, 8, laterality="L", view="CC", path=path)
        rows.append(
            _manifest_row(
                exam=f"exam-{index}",
                views=[
                    {
                        "image_id": f"image-{index}",
                        "view_name": "CC",
                        "path": path,
                        "bboxes": [],
                    }
                ],
                cross_view_valid=False,
                mass=index % 2,
            )
        )

    def build(
        *, persistent_workers: bool = True, prefetch_factor: int = 2
    ) -> tuple[DataLoader, EpochSeededRandomSampler]:
        dataset = VinDrBreastDataset(
            pd.DataFrame(rows),
            training=True,
            preprocessor=MappingLoader(views_by_path),
            augmentation=IntensityAugment(
                gamma_probability=1.0,
                gamma_range=(0.8, 1.2),
                contrast_probability=1.0,
                contrast_range=(0.8, 1.2),
            ),
            augmentation_seed=2026,
            patch_size=4,
            train_stride=4,
        )
        sampler = EpochSeededRandomSampler(dataset, seed=2026)
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            num_workers=2,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
            collate_fn=dataset.make_collator(flatten_patches=False),
        )
        return loader, sampler

    def collect(loader: DataLoader) -> tuple[list[str], list[torch.Tensor]]:
        exams: list[str] = []
        patches: list[torch.Tensor] = []
        for batch in loader:
            exams.extend(batch["exam_id"])
            patches.extend(
                sample_views["CC"]["patches"].clone() for sample_views in batch["views"]
            )
        return exams, patches

    uninterrupted_loader, uninterrupted_sampler = build()
    for epoch in range(3):
        uninterrupted_sampler.set_epoch(epoch)
        uninterrupted = collect(uninterrupted_loader)

    resumed_loader, resumed_sampler = build()
    resumed_sampler.set_epoch(2)
    resumed = collect(resumed_loader)

    memory_safe_loader, memory_safe_sampler = build(
        persistent_workers=False, prefetch_factor=1
    )
    memory_safe_sampler.set_epoch(2)
    memory_safe = collect(memory_safe_loader)

    assert resumed[0] == uninterrupted[0]
    for actual, expected in zip(resumed[1], uninterrupted[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert memory_safe[0] == uninterrupted[0]
    for actual, expected in zip(memory_safe[1], uninterrupted[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_boxes_are_forbidden_in_training_and_do_not_change_eval_inputs() -> None:
    processed = _processed(8, 8, laterality="R", view="CC", path="cc")
    base_view = {
        "image_id": "1",
        "view_name": "CC",
        "path": "cc",
        "bboxes": [
            {
                "categories": ["Mass"],
                "xmin": 1.0,
                "ymin": 2.0,
                "xmax": 3.0,
                "ymax": 5.0,
            }
        ],
    }
    frame = pd.DataFrame(
        [
            _manifest_row(
                laterality="R",
                views=[base_view],
                cross_view_valid=False,
            )
        ]
    )
    with pytest.raises(DatasetError, match="post-hoc evaluation"):
        VinDrBreastDataset(frame, training=True, include_bboxes=True)

    loader = MappingLoader({"cc": processed})
    with_boxes = VinDrBreastDataset(
        frame,
        training=False,
        include_bboxes=True,
        preprocessor=loader,
        patch_size=4,
        eval_stride=4,
    )[0]
    changed_view = dict(base_view)
    changed_view["bboxes"] = [
        {
            "categories": ["Mass"],
            "xmin": 0.0,
            "ymin": 0.0,
            "xmax": 8.0,
            "ymax": 8.0,
        }
    ]
    changed_frame = pd.DataFrame(
        [
            _manifest_row(
                laterality="R",
                views=[changed_view],
                cross_view_valid=False,
            )
        ]
    )
    changed = VinDrBreastDataset(
        changed_frame,
        training=False,
        include_bboxes=True,
        preprocessor=loader,
        patch_size=4,
        eval_stride=4,
    )[0]

    torch.testing.assert_close(
        with_boxes["views"]["CC"]["patches"],
        changed["views"]["CC"]["patches"],
    )
    torch.testing.assert_close(
        with_boxes["views"]["CC"]["anatomy_base"],
        changed["views"]["CC"]["anatomy_base"],
    )
    box = with_boxes["views"]["CC"]["evaluation_metadata"]["bboxes"][0]
    assert (box["xmin"], box["xmax"]) == (5.0, 7.0)
    assert with_boxes["views"]["CC"]["mask"].dtype == torch.bool

    without_boxes = VinDrBreastDataset(
        frame,
        training=False,
        include_bboxes=False,
        preprocessor=loader,
        patch_size=4,
        eval_stride=4,
    )[0]
    assert "evaluation_metadata" not in without_boxes["views"]["CC"]
    assert "mask" not in without_boxes["views"]["CC"]
