"""VinDr-Mammo annotation aggregation and the locked seed-2026 split.

The learning unit used by DAGL-MIL is one breast, not one image and not a
four-view examination.  A breast contains every available ipsilateral view;
normally this is one CC and one MLO image.  The official test examinations are
kept untouched.  Official training examinations are split as a group, with
stratification on examination-level presence of the two target findings.

Bounding boxes are deliberately serialized as opaque evaluation metadata.
They are never consulted while assigning labels, splitting examinations, or
constructing training samples.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dagl_mil.constants import EXPECTED_SPLIT_COUNTS


TARGET_FINDINGS: Mapping[str, str] = {
    "mass": "Mass",
    "calcification": "Suspicious Calcification",
}
VALID_LATERALITIES = ("L", "R")
VALID_VIEWS = ("CC", "MLO")


class ManifestError(ValueError):
    """Raised when annotations cannot form a leakage-free manifest."""


@dataclass(frozen=True)
class ManifestBuildResult:
    """Breast manifest plus a machine-readable examination anomaly audit."""

    manifest: pd.DataFrame
    anomalies: pd.DataFrame

    @property
    def fingerprint(self) -> str:
        """Stable training-protocol digest (paths and boxes are excluded)."""

        return manifest_fingerprint(self.manifest)

    def write(
        self,
        manifest_path: str | Path,
        anomalies_path: str | Path | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.to_csv(manifest_path, index=False)
        if anomalies_path is not None:
            anomalies_path = Path(anomalies_path)
            anomalies_path.parent.mkdir(parents=True, exist_ok=True)
            self.anomalies.to_csv(anomalies_path, index=False)


def parse_finding_categories(value: Any) -> tuple[str, ...]:
    """Parse VinDr's stringified Python lists without accepting arbitrary code."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())

    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        # This fallback makes hand-authored/synthetic CSVs convenient while
        # preserving exact official category names.
        parsed = [part.strip() for part in text.split(";")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple, set)):
        raise ManifestError(f"Invalid finding_categories value: {value!r}")
    return tuple(str(item).strip() for item in parsed if str(item).strip())


def _read_frame(source: str | Path | pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"{name} CSV not found: {path}")
    return pd.read_csv(path, dtype=str)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ManifestError(f"{name} is missing required columns: {missing}")


def _normalise_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_view(value: Any) -> str:
    view = _normalise_token(value).upper().replace("-", "")
    aliases = {
        "CRANIOCAUDAL": "CC",
        "MEDIOLATERALOBLIQUE": "MLO",
        "MEDIOLATERAL OBLIQUE": "MLO",
    }
    return aliases.get(view, view)


def _safe_number(value: Any, *, integer: bool = False) -> int | float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if integer else float(number)


def _bbox_record(row: pd.Series, categories: tuple[str, ...]) -> dict[str, Any] | None:
    coords = [_safe_number(row.get(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
    if any(value is None for value in coords):
        return None
    xmin, ymin, xmax, ymax = (float(value) for value in coords)
    if xmax <= xmin or ymax <= ymin:
        return None
    return {
        "categories": list(categories),
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def aggregate_vindr_annotations(
    breast_annotations: str | Path | pd.DataFrame,
    finding_annotations: str | Path | pd.DataFrame,
    *,
    images_root: str | Path | None = None,
) -> ManifestBuildResult:
    """Aggregate official image rows into dual-label breast samples.

    Duplicate or missing views are retained in ``views_json`` whenever they
    have a recognized view name.  Such examinations are recorded in the
    anomaly table and marked ``cross_view_valid=False``; a second CC is never
    relabeled as MLO merely to manufacture a pair.
    """

    breast = _read_frame(breast_annotations, "breast-level annotations")
    findings = _read_frame(finding_annotations, "finding annotations")
    _require_columns(
        breast,
        (
            "study_id",
            "series_id",
            "image_id",
            "laterality",
            "view_position",
            "height",
            "width",
            "split",
        ),
        "breast-level annotations",
    )
    _require_columns(
        findings,
        (
            "study_id",
            "image_id",
            "laterality",
            "view_position",
            "finding_categories",
        ),
        "finding annotations",
    )

    for column in ("study_id", "series_id", "image_id", "laterality", "split"):
        breast[column] = breast[column].map(_normalise_token)
    breast["laterality"] = breast["laterality"].str.upper()
    breast["view_position"] = breast["view_position"].map(_normalise_view)

    for column in ("study_id", "image_id", "laterality"):
        findings[column] = findings[column].map(_normalise_token)
    findings["laterality"] = findings["laterality"].str.upper()
    findings["view_position"] = findings["view_position"].map(_normalise_view)
    findings["_categories"] = findings["finding_categories"].map(
        parse_finding_categories
    )
    for label, official_name in TARGET_FINDINGS.items():
        findings[f"_{label}"] = findings["_categories"].map(
            lambda categories, target=official_name: int(target in categories)
        )

    anomaly_rows: list[dict[str, str]] = []

    def anomaly(study_id: str, laterality: str, code: str, detail: str) -> None:
        anomaly_rows.append(
            {
                "exam_id": study_id,
                "laterality": laterality,
                "code": code,
                "detail": detail,
            }
        )

    invalid_laterality = breast.loc[~breast["laterality"].isin(VALID_LATERALITIES)]
    for row in invalid_laterality.itertuples(index=False):
        anomaly(row.study_id, row.laterality, "invalid_laterality", row.image_id)

    invalid_view = breast.loc[~breast["view_position"].isin(VALID_VIEWS)]
    for row in invalid_view.itertuples(index=False):
        anomaly(
            row.study_id,
            row.laterality,
            "invalid_view",
            f"{row.image_id}:{row.view_position}",
        )

    duplicate_images = breast.loc[breast["image_id"].duplicated(keep=False)]
    for image_id, rows in duplicate_images.groupby("image_id", sort=True):
        first = rows.iloc[0]
        anomaly(
            first["study_id"],
            first["laterality"],
            "duplicate_image_id",
            f"{image_id} appears {len(rows)} times",
        )

    valid_breast = breast.loc[
        breast["laterality"].isin(VALID_LATERALITIES)
        & breast["view_position"].isin(VALID_VIEWS)
    ].copy()
    if valid_breast.empty:
        raise ManifestError("No valid L/R CC/MLO rows were found")

    # Index finding rows once.  Labels use all annotations from either view;
    # boxes remain attached to their original image solely as metadata.
    finding_groups = {
        key: group.copy()
        for key, group in findings.groupby(["study_id", "laterality"], sort=False)
    }
    finding_image_groups = {
        key: group.copy() for key, group in findings.groupby("image_id", sort=False)
    }
    known_images = set(valid_breast["image_id"])
    for row in findings.loc[~findings["image_id"].isin(known_images)].itertuples(
        index=False
    ):
        anomaly(
            row.study_id,
            row.laterality,
            "finding_without_image",
            row.image_id,
        )

    images_root_path = Path(images_root) if images_root is not None else None
    records: list[dict[str, Any]] = []
    for (study_id, laterality), rows in valid_breast.groupby(
        ["study_id", "laterality"], sort=True
    ):
        official_splits = sorted(value for value in rows["split"].unique() if value)
        if len(official_splits) != 1:
            anomaly(
                study_id,
                laterality,
                "inconsistent_official_split",
                json.dumps(official_splits),
            )
        official_split = official_splits[0] if official_splits else ""

        breast_findings = finding_groups.get((study_id, laterality))
        labels = {
            label: int(
                breast_findings is not None and bool(breast_findings[f"_{label}"].max())
            )
            for label in TARGET_FINDINGS
        }

        view_counts = rows["view_position"].value_counts().to_dict()
        for view_name in VALID_VIEWS:
            count = int(view_counts.get(view_name, 0))
            if count == 0:
                anomaly(study_id, laterality, f"missing_{view_name.lower()}", "")
            elif count > 1:
                anomaly(
                    study_id,
                    laterality,
                    f"duplicate_{view_name.lower()}",
                    f"{count} images",
                )

        view_records: list[dict[str, Any]] = []
        sorted_rows = rows.sort_values(["view_position", "image_id"], kind="stable")
        seen_view_slots = {view_name: 0 for view_name in VALID_VIEWS}
        for row in sorted_rows.to_dict(orient="records"):
            image_id = row["image_id"]
            view_name = row["view_position"]
            seen_view_slots[view_name] += 1
            slot = (
                view_name
                if seen_view_slots[view_name] == 1
                else f"{view_name}_{seen_view_slots[view_name]}"
            )
            image_findings = finding_image_groups.get(image_id)
            boxes: list[dict[str, Any]] = []
            if image_findings is not None:
                for _, finding in image_findings.iterrows():
                    box = _bbox_record(finding, finding["_categories"])
                    if box is not None:
                        boxes.append(box)
                    elif any(
                        target in finding["_categories"]
                        for target in TARGET_FINDINGS.values()
                    ):
                        raise ManifestError(
                            "Target finding has an invalid bounding box: "
                            f"{study_id}/{laterality}/{image_id}"
                        )

            if images_root_path is None:
                image_path = ""
            else:
                image_path = str(images_root_path / study_id / f"{image_id}.dicom")
            view_records.append(
                {
                    "image_id": image_id,
                    "series_id": row["series_id"],
                    "view_name": view_name,
                    "slot": slot,
                    "height": _safe_number(row.get("height"), integer=True),
                    "width": _safe_number(row.get("width"), integer=True),
                    "path": image_path,
                    "bboxes": boxes,
                }
            )

        by_view = {
            view_name: [view for view in view_records if view["view_name"] == view_name]
            for view_name in VALID_VIEWS
        }
        cross_view_valid = all(
            len(by_view[view_name]) == 1 for view_name in VALID_VIEWS
        )
        record: dict[str, Any] = {
            "exam_id": study_id,
            "study_id": study_id,
            "laterality": laterality,
            "breast_id": f"{study_id}_{laterality}",
            "official_split": official_split,
            "mass": labels["mass"],
            "calcification": labels["calcification"],
            "label_stratum": f"{labels['mass']}{labels['calcification']}",
            "cross_view_valid": bool(cross_view_valid),
            "views_json": json.dumps(
                view_records, ensure_ascii=False, separators=(",", ":")
            ),
        }
        # Convenience columns are useful for inspection, but ``views_json`` is
        # authoritative and preserves anomalous duplicate views.
        for view_name in VALID_VIEWS:
            prefix = view_name.lower()
            first = by_view[view_name][0] if by_view[view_name] else None
            record[f"{prefix}_image_id"] = first["image_id"] if first else ""
            record[f"{prefix}_path"] = first["path"] if first else ""
            record[f"{prefix}_bboxes_json"] = json.dumps(
                first["bboxes"] if first else [],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        records.append(record)

    manifest = pd.DataFrame.from_records(records).sort_values(
        ["exam_id", "laterality"], kind="stable", ignore_index=True
    )
    anomalies = pd.DataFrame.from_records(
        anomaly_rows, columns=("exam_id", "laterality", "code", "detail")
    )
    if not anomalies.empty:
        anomalies = anomalies.sort_values(
            ["exam_id", "laterality", "code", "detail"],
            kind="stable",
            ignore_index=True,
        ).drop_duplicates(ignore_index=True)
    return ManifestBuildResult(manifest=manifest, anomalies=anomalies)


def _canonical_official_split(value: Any) -> str:
    token = _normalise_token(value).lower()
    aliases = {
        "train": "training",
        "training": "training",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(token, token)


def assign_seed2026_splits(
    manifest: pd.DataFrame,
    *,
    seed: int = 2026,
    validation_fraction: float = 0.20,
    assert_expected_counts: bool = True,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SPLIT_COUNTS,
) -> pd.DataFrame:
    """Apply the manuscript's examination-grouped, joint-label split.

    Stratification is based on two examination-level indicators: whether
    either breast contains a Mass and whether either contains a Suspicious
    Calcification.  Sorting examination IDs before ``train_test_split`` is a
    required part of the locked seed-2026 protocol.
    """

    _require_columns(
        manifest,
        ("exam_id", "laterality", "official_split", "mass", "calcification"),
        "manifest",
    )
    if seed != 2026:
        raise ManifestError("The formal VinDr split is locked to seed 2026")
    if not np.isclose(validation_fraction, 0.20):
        raise ManifestError("The formal VinDr validation fraction is 0.20")

    result = manifest.copy()
    result["official_split"] = result["official_split"].map(_canonical_official_split)
    unknown = sorted(set(result["official_split"]).difference({"training", "test"}))
    if unknown:
        raise ManifestError(f"Unknown official split values: {unknown}")

    per_exam_splits = result.groupby("exam_id", sort=True)["official_split"].nunique()
    inconsistent = per_exam_splits[per_exam_splits != 1]
    if not inconsistent.empty:
        raise ManifestError(
            "Official split is inconsistent within examinations: "
            + ", ".join(inconsistent.index[:10])
        )

    official_training = result.loc[result["official_split"] == "training"]
    exam_labels = official_training.groupby("exam_id", sort=True).agg(
        mass=("mass", "max"),
        calcification=("calcification", "max"),
    )
    if exam_labels.empty:
        raise ManifestError("Official training split contains no examinations")
    strata = exam_labels["mass"].astype(int).astype(str) + exam_labels[
        "calcification"
    ].astype(int).astype(str)
    try:
        train_exams, val_exams = train_test_split(
            exam_labels.index.to_numpy(),
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=strata.to_numpy(),
        )
    except ValueError as exc:
        raise ManifestError(f"Examination-level stratification failed: {exc}") from exc

    train_set = set(train_exams.tolist())
    val_set = set(val_exams.tolist())
    test_set = set(
        result.loc[result["official_split"] == "test", "exam_id"].unique().tolist()
    )
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ManifestError("Examination leakage detected between data splits")

    result["split"] = ""
    result.loc[result["exam_id"].isin(train_set), "split"] = "train"
    result.loc[result["exam_id"].isin(val_set), "split"] = "val"
    result.loc[result["exam_id"].isin(test_set), "split"] = "test"
    if bool((result["split"] == "").any()):
        missing = result.loc[result["split"] == "", "exam_id"].unique()[:10]
        raise ManifestError(f"Unassigned examinations: {missing.tolist()}")

    result = result.sort_values(
        ["split", "exam_id", "laterality"], kind="stable", ignore_index=True
    )
    validate_no_exam_leakage(result)
    if assert_expected_counts:
        validate_split_counts(result, expected_counts=expected_counts)
    return result


def split_counts(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return the exact breast/label counts used in manuscript assertions."""

    _require_columns(manifest, ("split", "mass", "calcification"), "split manifest")
    counts: dict[str, dict[str, int]] = {}
    for split_name, rows in manifest.groupby("split", sort=False):
        mass = rows["mass"].astype(int)
        calcification = rows["calcification"].astype(int)
        counts[str(split_name)] = {
            "breasts": int(len(rows)),
            "mass": int(mass.sum()),
            "calcification": int(calcification.sum()),
            "both": int(((mass == 1) & (calcification == 1)).sum()),
        }
    return counts


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Return a path- and row-order-independent SHA256 protocol fingerprint.

    The digest covers exam membership, breast labels, assigned/official
    splits, and the identities/order-independent set of real CC/MLO images.
    Absolute filesystem paths and bounding boxes are intentionally excluded:
    paths vary between machines, while boxes are not training inputs.
    """

    _require_columns(
        manifest,
        ("exam_id", "laterality", "mass", "calcification", "split"),
        "split manifest",
    )
    payload: list[dict[str, Any]] = []
    ordered = manifest.sort_values(
        ["exam_id", "laterality"], kind="stable", ignore_index=True
    )
    for row in ordered.to_dict(orient="records"):
        views = _view_identity_records(row)
        payload.append(
            {
                "exam_id": _normalise_token(row["exam_id"]),
                "laterality": _normalise_token(row["laterality"]).upper(),
                "official_split": _canonical_official_split(
                    row.get("official_split", "")
                ),
                "split": _normalise_token(row["split"]).lower(),
                "mass": int(row["mass"]),
                "calcification": int(row["calcification"]),
                "views": views,
            }
        )
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _view_identity_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("views_json")
    if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
        try:
            views = json.loads(str(raw)) if not isinstance(raw, list) else raw
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"Invalid views_json while fingerprinting: {exc}"
            ) from exc
        if not isinstance(views, list):
            raise ManifestError("views_json must encode a list")
    else:
        views = []
        for view_name in VALID_VIEWS:
            prefix = view_name.lower()
            image_id = _normalise_token(row.get(f"{prefix}_image_id", ""))
            if image_id:
                views.append({"view_name": view_name, "image_id": image_id})

    identities: list[dict[str, Any]] = []
    for view in views:
        if not isinstance(view, Mapping):
            raise ManifestError("views_json entries must be objects")
        identities.append(
            {
                "view_name": _normalise_view(view.get("view_name", "")),
                "slot": _normalise_token(view.get("slot", "")),
                "image_id": _normalise_token(view.get("image_id", "")),
                "series_id": _normalise_token(view.get("series_id", "")),
                "height": _safe_number(view.get("height"), integer=True),
                "width": _safe_number(view.get("width"), integer=True),
            }
        )
    return sorted(
        identities,
        key=lambda item: (
            str(item["view_name"]),
            str(item["image_id"]),
            str(item["series_id"]),
        ),
    )


def validate_split_counts(
    manifest: pd.DataFrame,
    *,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SPLIT_COUNTS,
) -> None:
    actual = split_counts(manifest)
    expected = {split: dict(values) for split, values in expected_counts.items()}
    if actual != expected:
        raise ManifestError(
            "Split counts do not match the locked manuscript protocol. "
            f"expected={expected}, actual={actual}"
        )


def validate_no_exam_leakage(manifest: pd.DataFrame) -> None:
    _require_columns(manifest, ("exam_id", "split"), "split manifest")
    split_per_exam = manifest.groupby("exam_id", sort=False)["split"].nunique()
    leaking = split_per_exam[split_per_exam != 1]
    if not leaking.empty:
        raise ManifestError(
            "Examinations occur in more than one split: "
            + ", ".join(leaking.index[:10])
        )


def build_vindr_manifest(
    vindr_root: str | Path,
    *,
    seed: int = 2026,
    assert_expected_counts: bool = True,
    manifest_path: str | Path | None = None,
    anomalies_path: str | Path | None = None,
) -> ManifestBuildResult:
    """Build, validate, and optionally persist the complete VinDr manifest."""

    root = Path(vindr_root)
    aggregated = aggregate_vindr_annotations(
        root / "breast-level_annotations.csv",
        root / "finding_annotations.csv",
        images_root=root / "images",
    )
    split_manifest = assign_seed2026_splits(
        aggregated.manifest,
        seed=seed,
        assert_expected_counts=assert_expected_counts,
    )
    result = ManifestBuildResult(split_manifest, aggregated.anomalies)
    if manifest_path is not None:
        result.write(manifest_path, anomalies_path)
    elif anomalies_path is not None:
        anomalies_path = Path(anomalies_path)
        anomalies_path.parent.mkdir(parents=True, exist_ok=True)
        result.anomalies.to_csv(anomalies_path, index=False)
    return result


def load_manifest(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"exam_id": str, "study_id": str})
    if validate:
        validate_no_exam_leakage(manifest)
    return manifest
