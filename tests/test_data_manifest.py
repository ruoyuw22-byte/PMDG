from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dagl_mil.constants import DEFAULT_VINDR_ROOT, EXPECTED_SPLIT_COUNTS
from dagl_mil.data.manifest import (
    ManifestError,
    aggregate_vindr_annotations,
    assign_seed2026_splits,
    build_vindr_manifest,
    manifest_fingerprint,
    parse_finding_categories,
    split_counts,
    validate_no_exam_leakage,
)


def _image_row(
    exam: str,
    side: str,
    view: str,
    image: str,
    split: str = "training",
) -> dict[str, str]:
    return {
        "study_id": exam,
        "series_id": f"series-{exam}",
        "image_id": image,
        "laterality": side,
        "view_position": view,
        "height": "1024",
        "width": "768",
        "breast_birads": "BI-RADS 1",
        "breast_density": "DENSITY B",
        "split": split,
    }


def _finding_row(
    exam: str,
    side: str,
    view: str,
    image: str,
    categories: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    coords = bbox or (None, None, None, None)
    return {
        "study_id": exam,
        "series_id": f"series-{exam}",
        "image_id": image,
        "laterality": side,
        "view_position": view,
        "height": "1024",
        "width": "768",
        "breast_birads": "BI-RADS 4",
        "breast_density": "DENSITY B",
        "finding_categories": categories,
        "finding_birads": "BI-RADS 4",
        "xmin": coords[0],
        "ymin": coords[1],
        "xmax": coords[2],
        "ymax": coords[3],
        "split": "training",
    }


def test_parse_finding_categories_preserves_exact_official_names() -> None:
    assert parse_finding_categories("['Suspicious Calcification', 'Mass']") == (
        "Suspicious Calcification",
        "Mass",
    )
    assert parse_finding_categories("Mass;Suspicious Calcification") == (
        "Mass",
        "Suspicious Calcification",
    )
    assert parse_finding_categories(float("nan")) == ()


def test_aggregate_dual_labels_boxes_and_duplicate_views(tmp_path: Path) -> None:
    breast = pd.DataFrame(
        [
            _image_row("exam-a", "L", "CC", "cc-1"),
            _image_row("exam-a", "L", "CC", "cc-2"),
            _image_row("exam-a", "R", "CC", "r-cc"),
            _image_row("exam-a", "R", "MLO", "r-mlo"),
        ]
    )
    findings = pd.DataFrame(
        [
            _finding_row(
                "exam-a",
                "L",
                "CC",
                "cc-1",
                "['Mass']",
                (10.0, 20.0, 30.0, 40.0),
            ),
            _finding_row(
                "exam-a",
                "L",
                "CC",
                "cc-2",
                "['Suspicious Calcification']",
                (1.0, 2.0, 5.0, 8.0),
            ),
            _finding_row("exam-a", "R", "CC", "r-cc", "['No Finding']"),
            _finding_row("exam-a", "R", "MLO", "r-mlo", "['No Finding']"),
        ]
    )

    result = aggregate_vindr_annotations(
        breast, findings, images_root=tmp_path / "images"
    )
    left = result.manifest.query("laterality == 'L'").iloc[0]
    assert (left["mass"], left["calcification"]) == (1, 1)
    assert not bool(left["cross_view_valid"])
    views = json.loads(left["views_json"])
    assert [view["view_name"] for view in views] == ["CC", "CC"]
    assert [view["image_id"] for view in views] == ["cc-1", "cc-2"]
    assert [view["slot"] for view in views] == ["CC", "CC_2"]
    assert views[0]["bboxes"][0]["categories"] == ["Mass"]
    assert set(result.anomalies["code"]) == {"duplicate_cc", "missing_mlo"}

    right = result.manifest.query("laterality == 'R'").iloc[0]
    assert (right["mass"], right["calcification"]) == (0, 0)
    assert bool(right["cross_view_valid"])


def test_target_finding_with_invalid_bbox_fails_manifest_build() -> None:
    breast = pd.DataFrame(
        [
            _image_row("exam-a", "L", "CC", "cc"),
            _image_row("exam-a", "L", "MLO", "mlo"),
        ]
    )
    findings = pd.DataFrame([_finding_row("exam-a", "L", "CC", "cc", "['Mass']")])
    with pytest.raises(
        ManifestError, match="Target finding has an invalid bounding box"
    ):
        aggregate_vindr_annotations(breast, findings)


def test_seed2026_split_is_exam_grouped_joint_stratified_and_deterministic() -> None:
    rows: list[dict[str, object]] = []
    # Ten examinations per joint exam label provide enough members in each
    # stratum for an exact 80/20 stratified split.
    for mass in (0, 1):
        for calcification in (0, 1):
            for number in range(10):
                exam = f"train-{mass}{calcification}-{number:02d}"
                for side in ("L", "R"):
                    rows.append(
                        {
                            "exam_id": exam,
                            "laterality": side,
                            "official_split": "training",
                            # Put positives on only one side to verify that
                            # stratification first aggregates by examination.
                            "mass": int(mass and side == "L"),
                            "calcification": int(calcification and side == "R"),
                        }
                    )
    for number in range(4):
        for side in ("L", "R"):
            rows.append(
                {
                    "exam_id": f"test-{number:02d}",
                    "laterality": side,
                    "official_split": "test",
                    "mass": 0,
                    "calcification": 0,
                }
            )
    manifest = pd.DataFrame(rows).sample(frac=1.0, random_state=17)

    first = assign_seed2026_splits(manifest, assert_expected_counts=False)
    second = assign_seed2026_splits(manifest, assert_expected_counts=False)
    validate_no_exam_leakage(first)
    assert first[["exam_id", "laterality", "split"]].equals(
        second[["exam_id", "laterality", "split"]]
    )
    assert first.query("split == 'train'")["exam_id"].nunique() == 32
    assert first.query("split == 'val'")["exam_id"].nunique() == 8
    assert first.query("split == 'test'")["exam_id"].nunique() == 4
    assert manifest_fingerprint(first) == manifest_fingerprint(
        first.sample(frac=1.0, random_state=99)
    )


def test_local_raw_vindr_reproduces_locked_manuscript_counts() -> None:
    root = Path(DEFAULT_VINDR_ROOT)
    if not (root / "breast-level_annotations.csv").is_file():
        pytest.skip("local raw VinDr annotations are not available")
    result = build_vindr_manifest(root)
    assert split_counts(result.manifest) == EXPECTED_SPLIT_COUNTS
    assert result.fingerprint == manifest_fingerprint(result.manifest)
    assert (
        result.fingerprint
        == "68e1571a7c5a838c10a6a1eeecb831fbaa15c949c27c6921a865d0e1271d10d2"
    )
    validate_no_exam_leakage(result.manifest)

    # The source release contains one known malformed breast: two CC images
    # and no MLO.  It remains in the manifest, is never relabeled, and is
    # explicitly auditable.
    anomaly = result.anomalies.query(
        "exam_id == 'dbca9d28baa3207b3187c4d07dc81a80' and laterality == 'L'"
    )
    assert set(anomaly["code"]) == {"duplicate_cc", "missing_mlo"}
