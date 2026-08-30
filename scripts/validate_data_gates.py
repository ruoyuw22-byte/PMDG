from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from dagl_mil.config import load_config
from dagl_mil.data import load_manifest
from dagl_mil.data.preprocessing import (
    _CACHE_PROTOCOL_VERSION,
    PreprocessingError,
    transform_bboxes_for_laterality,
)
from dagl_mil.data_gates import (
    BBOX_BOUNDARY_PROTOCOL,
    EXPECTED_SPLIT_VIEWS,
    EXPECTED_TOTAL_VIEWS,
    LOCKED_PREPROCESSING_PROTOCOL,
    atomic_write_json,
    finish_report,
    initial_report,
    validate_bbox_audit,
    validate_patch_report,
    validate_precache_report,
)
from dagl_mil.reproducibility import environment_record, sha256_file


def _read_report(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    encoded = path.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"report root must be a JSON object: {path}")
    metadata = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return payload, metadata


def _manifest_image_splits(
    manifest,
) -> tuple[dict[str, str], Counter[str], list[str], dict[str, Any]]:
    image_split_by_id: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    clipped_split_counts: Counter[str] = Counter()
    total_boxes = 0
    clipped_records: list[dict[str, Any]] = []
    invalid_boxes = 0
    errors: list[str] = []
    for row in manifest.to_dict(orient="records"):
        split = str(row["split"])
        for view in json.loads(str(row["views_json"])):
            image_id = str(view["image_id"])
            if image_id in image_split_by_id:
                errors.append(f"manifest contains duplicate image_id: {image_id}")
            image_split_by_id[image_id] = split
            split_counts[split] += 1
            boxes = view.get("bboxes", [])
            if not isinstance(boxes, list):
                errors.append(f"manifest bboxes are not a list: {image_id}")
                continue
            try:
                width = int(view["width"])
                height = int(view["height"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"manifest image dimensions are invalid: {image_id}: {exc}"
                )
                continue
            for box_index, box in enumerate(boxes):
                total_boxes += 1
                if not isinstance(box, Mapping):
                    invalid_boxes += 1
                    errors.append(
                        f"manifest bounding box is not an object: {image_id}[{box_index}]"
                    )
                    continue
                try:
                    transform_bboxes_for_laterality(
                        [box],
                        width=width,
                        height=height,
                        laterality=str(row["laterality"]),
                    )
                    xmin, ymin, xmax, ymax = (
                        float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax")
                    )
                except (PreprocessingError, KeyError, TypeError, ValueError) as exc:
                    invalid_boxes += 1
                    errors.append(
                        f"manifest bounding box is invalid: {image_id}[{box_index}]: {exc}"
                    )
                    continue
                if xmin < 0.0 or ymin < 0.0 or xmax > width or ymax > height:
                    clipped_split_counts[split] += 1
                    clipped_records.append(
                        {
                            "split": split,
                            "breast_id": str(row["breast_id"]),
                            "image_id": image_id,
                            "view_name": str(view["view_name"]),
                            "box_index": box_index,
                            "image_shape": [height, width],
                            "raw_xyxy": [xmin, ymin, xmax, ymax],
                            "clipped_xyxy": [
                                max(0.0, xmin),
                                max(0.0, ymin),
                                min(float(width), xmax),
                                min(float(height), ymax),
                            ],
                        }
                    )
    actual_splits = {
        split: split_counts.get(split, 0) for split in EXPECTED_SPLIT_VIEWS
    }
    if actual_splits != EXPECTED_SPLIT_VIEWS:
        errors.append(
            "manifest view split counts differ from train/val/test=12800/3200/4000: "
            f"actual={actual_splits}"
        )
    if len(image_split_by_id) != EXPECTED_TOTAL_VIEWS:
        errors.append(
            f"manifest unique image count={len(image_split_by_id)}, "
            f"expected={EXPECTED_TOTAL_VIEWS}"
        )
    bbox_audit = {
        "protocol": BBOX_BOUNDARY_PROTOCOL,
        "total_boxes": total_boxes,
        "in_bounds_boxes": total_boxes - len(clipped_records) - invalid_boxes,
        "partially_out_of_canvas_boxes": len(clipped_records),
        "invalid_or_disjoint_boxes": invalid_boxes,
        "clipped_split_counts": {
            split: clipped_split_counts.get(split, 0) for split in EXPECTED_SPLIT_VIEWS
        },
        "clipped_records": clipped_records,
    }
    return image_split_by_id, split_counts, errors, bbox_audit


def _resolved_report_paths(
    config: Mapping[str, Any],
    precache_report: Path | None,
    patch_report: Path,
) -> tuple[Path, Path]:
    precache = precache_report or (
        Path(str(config["paths"]["preprocessing_cache"])) / "precache_report.json"
    )
    return precache.resolve(), patch_report.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly seal the formal VinDr precache and patch data gates."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dagl.yaml"))
    parser.add_argument("--precache-report", type=Path, default=None)
    parser.add_argument(
        "--patch-report",
        type=Path,
        default=Path("outputs/patch_feasibility_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_gate_validation.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(str(config["paths"]["split_manifest"])).resolve()
    manifest = load_manifest(manifest_path, validate=True)
    precache_path, patch_path = _resolved_report_paths(
        config, args.precache_report, args.patch_report
    )
    output_path = args.output.resolve()
    if output_path in {precache_path, patch_path}:
        raise ValueError("validation output must differ from both source reports")

    running_report = initial_report(
        report_kind="formal_data_gate_validation",
        config_path=args.config,
        manifest_path=manifest_path,
        preprocessing_protocol=_CACHE_PROTOCOL_VERSION,
        precache_report_path=str(precache_path),
        patch_report_path=str(patch_path),
        environment=environment_record(),
        errors=[],
    )
    atomic_write_json(output_path, running_report)
    started = time.perf_counter()
    try:
        errors: list[str] = []
        if _CACHE_PROTOCOL_VERSION != LOCKED_PREPROCESSING_PROTOCOL:
            errors.append(
                "runtime preprocessing protocol is not the locked v2 protocol"
            )
        (
            image_split_by_id,
            split_counts,
            manifest_errors,
            bbox_audit,
        ) = _manifest_image_splits(manifest)
        bbox_errors = validate_bbox_audit(bbox_audit)
        manifest_errors.extend(bbox_errors)
        errors.extend(manifest_errors)
        precache, precache_metadata = _read_report(precache_path)
        patch, patch_metadata = _read_report(patch_path)
        config_sha256 = str(running_report["config_sha256"])
        manifest_sha256 = str(running_report["manifest_sha256"])
        expected_ids = {
            str(value)
            for value in config["data"].get("expected_topology_fallback_image_ids", [])
        }
        precache_errors = validate_precache_report(
            precache,
            expected_ids_from_config=expected_ids,
            image_split_by_id=image_split_by_id,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
        )
        patch_errors = validate_patch_report(
            patch,
            data_config=config["data"],
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
        )
        errors.extend(precache_errors)
        errors.extend(patch_errors)
        if sha256_file(args.config) != running_report["config_sha256"]:
            errors.append("config changed while final validation was running")
        if sha256_file(manifest_path) != running_report["manifest_sha256"]:
            errors.append("manifest changed while final validation was running")
        report = finish_report(
            running_report,
            status="passed" if not errors else "failed",
            errors=errors,
            checks={
                "manifest": "passed" if not manifest_errors else "failed",
                "precache": "passed" if not precache_errors else "failed",
                "patch_feasibility": "passed" if not patch_errors else "failed",
            },
            manifest_split_view_counts={
                split: split_counts.get(split, 0) for split in EXPECTED_SPLIT_VIEWS
            },
            manifest_bbox_audit=bbox_audit,
            source_reports={
                "precache": precache_metadata,
                "patch_feasibility": patch_metadata,
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(output_path, report)
    except Exception as exc:
        failed_report = finish_report(
            running_report,
            status="failed",
            errors=[f"{type(exc).__name__}: {exc}"],
            fatal_error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(output_path, failed_report)
        print(output_path, flush=True)
        return 1

    print(output_path, flush=True)
    if report["status"] != "passed":
        for error in report["errors"]:
            print(f"ERROR: {error}", flush=True)
        return 1
    print("formal data gates passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
