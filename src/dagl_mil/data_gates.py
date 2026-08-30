from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .reproducibility import sha256_file


LOCKED_PREPROCESSING_PROTOCOL = (
    "vindr-polarity-padding-topology-mask-laterality-p1p99-v2"
)
EXPECTED_TOTAL_VIEWS = 20_000
EXPECTED_SPLIT_VIEWS = {"train": 12_800, "val": 3_200, "test": 4_000}
EXPECTED_TOPOLOGY_FALLBACK_IDS = frozenset(
    {
        "fb8fd3863ad17037894b4171da696251",
        "8ebbb6116d0a6187d9dc5eed5836e196",
        "863bde43e71a55208eda16c6baa2a974",
        "7e1c0fa32a1924fc7f80c5efbc3753d8",
    }
)
EXPECTED_TOPOLOGY_FALLBACK_SPLITS = {"train": 3, "val": 1, "test": 0}
ALLOWED_MASK_STRATEGIES = frozenset({"otsu_lcc", "nonpadding_lcc_topology_fallback"})
BBOX_BOUNDARY_PROTOCOL = "clip-partially-intersecting-boxes-to-dicom-canvas-v1"
EXPECTED_BBOX_COUNTS = {
    "total_boxes": 2_254,
    "in_bounds_boxes": 2_212,
    "partially_out_of_canvas_boxes": 42,
    "invalid_or_disjoint_boxes": 0,
}
EXPECTED_CLIPPED_BBOX_SPLITS = {"train": 25, "val": 10, "test": 7}


class DataGateError(RuntimeError):
    """Raised when a formal run lacks a current sealed data gate."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return str(uuid.uuid4())


def atomic_write_json(destination: str | Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON report without exposing a partial document."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def initial_report(
    *,
    report_kind: str,
    config_path: str | Path,
    manifest_path: str | Path,
    preprocessing_protocol: str,
    run_id: str | None = None,
    started_at: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    config = Path(config_path).resolve()
    manifest = Path(manifest_path).resolve()
    return {
        **fields,
        "report_schema": "dagl-data-gate-v1",
        "report_kind": str(report_kind),
        "status": "running",
        "run_id": run_id or new_run_id(),
        "started_at": started_at or utc_timestamp(),
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "preprocessing_protocol": str(preprocessing_protocol),
    }


def finish_report(
    running_report: Mapping[str, Any],
    *,
    status: str,
    **fields: Any,
) -> dict[str, Any]:
    if status not in {"passed", "failed"}:
        raise ValueError("terminal report status must be 'passed' or 'failed'")
    return {
        **dict(running_report),
        **fields,
        "status": status,
        "completed_at": utc_timestamp(),
    }


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_bbox_audit(audit: Any) -> list[str]:
    """Validate the fixed-manifest bounding-box boundary audit."""

    errors: list[str] = []
    if not isinstance(audit, Mapping):
        return ["bounding-box audit is missing or is not an object"]
    _require(
        audit.get("protocol") == BBOX_BOUNDARY_PROTOCOL,
        "bounding-box audit protocol is invalid",
        errors,
    )
    for field, expected in EXPECTED_BBOX_COUNTS.items():
        _require(
            audit.get(field) == expected,
            f"bounding-box audit {field} != {expected}",
            errors,
        )
    _require(
        audit.get("clipped_split_counts") == EXPECTED_CLIPPED_BBOX_SPLITS,
        "bounding-box audit clipped split counts differ from train/val/test=25/10/7",
        errors,
    )
    records = audit.get("clipped_records")
    if not isinstance(records, list):
        errors.append("bounding-box audit clipped_records is not a list")
    else:
        _require(
            len(records) == EXPECTED_BBOX_COUNTS["partially_out_of_canvas_boxes"],
            "bounding-box audit clipped record count != 42",
            errors,
        )
        record_splits = Counter(
            str(record.get("split"))
            for record in records
            if isinstance(record, Mapping)
        )
        _require(
            {split: record_splits.get(split, 0) for split in EXPECTED_SPLIT_VIEWS}
            == EXPECTED_CLIPPED_BBOX_SPLITS,
            "bounding-box audit clipped record split counts are inconsistent",
            errors,
        )
    count_values = [audit.get(field) for field in EXPECTED_BBOX_COUNTS]
    if all(
        isinstance(value, int) and not isinstance(value, bool) for value in count_values
    ):
        _require(
            audit["total_boxes"]
            == audit["in_bounds_boxes"]
            + audit["partially_out_of_canvas_boxes"]
            + audit["invalid_or_disjoint_boxes"],
            "bounding-box audit counts do not sum to total_boxes",
            errors,
        )
    return errors


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _validate_common(
    report: Mapping[str, Any],
    *,
    report_name: str,
    config_sha256: str,
    manifest_sha256: str,
) -> list[str]:
    errors: list[str] = []
    _require(
        report.get("status") == "passed", f"{report_name}: status is not passed", errors
    )
    _require(
        report.get("preprocessing_protocol") == LOCKED_PREPROCESSING_PROTOCOL,
        f"{report_name}: preprocessing protocol is not the locked v2 protocol",
        errors,
    )
    # Reports produced by an already-running pre-upgrade process do not contain
    # these two fields.  When present, they are fail-closed; the final validator
    # always binds the source files and their hashes in its own signed report.
    if "config_sha256" in report:
        _require(
            report.get("config_sha256") == config_sha256,
            f"{report_name}: config SHA-256 mismatch",
            errors,
        )
    if "manifest_sha256" in report:
        _require(
            report.get("manifest_sha256") == manifest_sha256,
            f"{report_name}: manifest SHA-256 mismatch",
            errors,
        )
    return errors


def validate_precache_report(
    report: Mapping[str, Any],
    *,
    expected_ids_from_config: set[str],
    image_split_by_id: Mapping[str, str],
    config_sha256: str,
    manifest_sha256: str,
) -> list[str]:
    errors = _validate_common(
        report,
        report_name="precache",
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
    )
    _require(
        expected_ids_from_config == set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
        "precache: config does not contain the four locked topology fallback IDs",
        errors,
    )
    _require(
        report.get("images") == EXPECTED_TOTAL_VIEWS,
        "precache: images != 20000",
        errors,
    )
    _require(
        report.get("completed") == EXPECTED_TOTAL_VIEWS,
        "precache: completed != 20000",
        errors,
    )
    failures = report.get("failures")
    _require(
        isinstance(failures, list) and not failures,
        "precache: failures is not empty",
        errors,
    )

    strategies = report.get("mask_strategy_counts")
    if not isinstance(strategies, Mapping):
        errors.append("precache: mask_strategy_counts is not an object")
        strategy_counts: dict[str, int] = {}
    else:
        strategy_counts = {}
        for key, value in strategies.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"precache: invalid mask strategy count for {key!r}")
                continue
            strategy_counts[str(key)] = value
        _require(
            set(strategy_counts).issubset(ALLOWED_MASK_STRATEGIES),
            "precache: unknown mask strategy present",
            errors,
        )
    _require(
        sum(strategy_counts.values()) == EXPECTED_TOTAL_VIEWS,
        "precache: mask strategy counts do not sum to 20000",
        errors,
    )
    _require(
        strategy_counts.get("nonpadding_lcc_topology_fallback") == 4,
        "precache: topology fallback strategy count != 4",
        errors,
    )

    fallback_records = _as_list(report.get("topology_fallbacks"))
    _require(
        isinstance(report.get("topology_fallbacks"), list),
        "precache: topology_fallbacks is not a list",
        errors,
    )
    actual_ids = [
        str(record.get("image_id"))
        for record in fallback_records
        if isinstance(record, Mapping) and record.get("image_id") is not None
    ]
    _require(
        len(actual_ids) == 4, "precache: topology fallback record count != 4", errors
    )
    _require(
        len(set(actual_ids)) == len(actual_ids),
        "precache: duplicate topology fallback ID",
        errors,
    )
    _require(
        set(actual_ids) == set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
        "precache: topology fallback IDs differ from the four locked IDs",
        errors,
    )
    reported_expected_ids = report.get("expected_topology_fallback_image_ids")
    if reported_expected_ids is not None:
        _require(
            isinstance(reported_expected_ids, list)
            and set(map(str, reported_expected_ids))
            == set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
            "precache: report's expected topology fallback IDs are not locked",
            errors,
        )
    topology_validation = report.get("topology_fallback_validation")
    if topology_validation is not None:
        _require(
            isinstance(topology_validation, Mapping)
            and topology_validation.get("status") == "passed"
            and topology_validation.get("errors") == [],
            "precache: embedded topology fallback validation did not pass",
            errors,
        )
    fallback_splits: Counter[str] = Counter()
    for image_id in actual_ids:
        split = image_split_by_id.get(image_id)
        if split is None:
            errors.append(f"precache: fallback ID absent from manifest: {image_id}")
        else:
            fallback_splits[str(split)] += 1
    _require(
        {split: fallback_splits.get(split, 0) for split in EXPECTED_SPLIT_VIEWS}
        == EXPECTED_TOPOLOGY_FALLBACK_SPLITS,
        "precache: topology fallback split counts are not train/val/test=3/1/0",
        errors,
    )
    for record in fallback_records:
        if not isinstance(record, Mapping) or "split" not in record:
            continue
        image_id = str(record.get("image_id"))
        _require(
            str(record.get("split")) == image_split_by_id.get(image_id),
            f"precache: reported split mismatch for {image_id}",
            errors,
        )
    return errors


def validate_patch_report(
    report: Mapping[str, Any],
    *,
    data_config: Mapping[str, Any],
    config_sha256: str,
    manifest_sha256: str,
) -> list[str]:
    errors = _validate_common(
        report,
        report_name="patch audit",
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
    )
    _require(
        report.get("views") == EXPECTED_TOTAL_VIEWS,
        "patch audit: views != 20000",
        errors,
    )
    _require(
        report.get("completed") == EXPECTED_TOTAL_VIEWS,
        "patch audit: completed != 20000",
        errors,
    )
    failures = report.get("failures")
    _require(
        isinstance(failures, list) and not failures,
        "patch audit: failures is not empty",
        errors,
    )
    _require(
        report.get("fallback_view_count") == 0,
        "patch audit: fallback_view_count != 0",
        errors,
    )
    _require(
        report.get("fallback_breast_count") == 0,
        "patch audit: fallback_breast_count != 0",
        errors,
    )
    fallback_views = report.get("fallback_views")
    _require(
        isinstance(fallback_views, list) and not fallback_views,
        "patch audit: fallback_views is not empty",
        errors,
    )
    multiple_fallbacks = report.get("breasts_with_multiple_fallback_views")
    if multiple_fallbacks is not None:
        _require(
            isinstance(multiple_fallbacks, list) and not multiple_fallbacks,
            "patch audit: breasts_with_multiple_fallback_views is not empty",
            errors,
        )
    embedded_errors = report.get("validation_errors")
    if embedded_errors is not None:
        _require(
            isinstance(embedded_errors, list) and not embedded_errors,
            "patch audit: embedded validation errors are not empty",
            errors,
        )

    locked_scalars = {
        "patch_size": int(data_config["patch_size"]),
        "train_stride": int(data_config["train_stride"]),
        "eval_stride": int(data_config["eval_stride"]),
        "minimum_breast_coverage": float(data_config["minimum_breast_coverage"]),
    }
    for field, expected in locked_scalars.items():
        _require(
            report.get(field) == expected, f"patch audit: {field} mismatch", errors
        )

    summaries = report.get("split_summaries")
    if not isinstance(summaries, Mapping):
        errors.append("patch audit: split_summaries is not an object")
        return errors
    _require(
        set(summaries) == set(EXPECTED_SPLIT_VIEWS),
        "patch audit: split summary keys differ from train/val/test",
        errors,
    )
    for split, expected_views in EXPECTED_SPLIT_VIEWS.items():
        summary = summaries.get(split)
        if not isinstance(summary, Mapping):
            errors.append(f"patch audit: missing {split} summary")
            continue
        _require(
            summary.get("views") == expected_views,
            f"patch audit: {split} views != {expected_views}",
            errors,
        )
        _require(
            summary.get("zero_candidate_views") == 0,
            f"patch audit: {split} has zero-candidate views",
            errors,
        )
        minimum = summary.get("minimum_candidates")
        _require(
            isinstance(minimum, int) and not isinstance(minimum, bool) and minimum >= 1,
            f"patch audit: {split} minimum_candidates is not positive",
            errors,
        )
    return errors


def require_passed_data_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the current manifest has an immutable passed seal."""

    try:
        output_root = Path(str(config["paths"]["output_root"])).resolve()
        manifest_path = Path(str(config["paths"]["split_manifest"])).resolve()
    except (KeyError, TypeError) as exc:
        raise DataGateError("config does not define formal data-gate paths") from exc
    seal_path = output_root / "data_gate_validation.json"
    if not seal_path.is_file():
        raise DataGateError(f"formal data-gate seal is missing: {seal_path}")
    try:
        report = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataGateError(
            f"formal data-gate seal is unreadable: {seal_path}"
        ) from exc
    if not isinstance(report, dict):
        raise DataGateError("formal data-gate seal root is not an object")

    errors: list[str] = []
    _require(report.get("status") == "passed", "seal status is not passed", errors)
    _require(
        report.get("report_kind") == "formal_data_gate_validation",
        "seal report kind is invalid",
        errors,
    )
    _require(
        report.get("preprocessing_protocol") == LOCKED_PREPROCESSING_PROTOCOL,
        "seal preprocessing protocol is not locked v2",
        errors,
    )
    _require(report.get("errors") == [], "seal contains validation errors", errors)
    _require(
        report.get("checks")
        == {
            "manifest": "passed",
            "precache": "passed",
            "patch_feasibility": "passed",
        },
        "seal checks are incomplete",
        errors,
    )
    errors.extend(validate_bbox_audit(report.get("manifest_bbox_audit")))
    _require(
        str(report.get("manifest_path", "")) == str(manifest_path),
        "seal manifest path differs from the formal config",
        errors,
    )
    if manifest_path.is_file():
        _require(
            report.get("manifest_sha256") == sha256_file(manifest_path),
            "sealed manifest SHA-256 is stale",
            errors,
        )
    else:
        errors.append(f"sealed manifest is missing: {manifest_path}")

    config_path = Path(str(report.get("config_path", "")))
    if config_path.is_file():
        _require(
            report.get("config_sha256") == sha256_file(config_path),
            "sealed data config SHA-256 is stale",
            errors,
        )
    else:
        errors.append(f"sealed data config is missing: {config_path}")

    sources = report.get("source_reports")
    if not isinstance(sources, Mapping):
        errors.append("seal source_reports is not an object")
    else:
        for name in ("precache", "patch_feasibility"):
            metadata = sources.get(name)
            if not isinstance(metadata, Mapping):
                errors.append(f"seal source report metadata is missing: {name}")
                continue
            source_path = Path(str(metadata.get("path", "")))
            if not source_path.is_file():
                errors.append(f"sealed source report is missing: {source_path}")
                continue
            if metadata.get("sha256") != sha256_file(source_path):
                errors.append(f"sealed source report SHA-256 is stale: {name}")

    if errors:
        raise DataGateError("formal data-gate seal rejected: " + "; ".join(errors))
    return report
