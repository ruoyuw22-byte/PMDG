from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from dagl_mil.config import load_config
from dagl_mil.data import DICOMPreprocessor, load_manifest
from dagl_mil.data_gates import (
    EXPECTED_SPLIT_VIEWS,
    EXPECTED_TOTAL_VIEWS,
    LOCKED_PREPROCESSING_PROTOCOL,
    atomic_write_json,
    finish_report,
    initial_report,
)
from dagl_mil.data.patches import patch_grid_coverage
from dagl_mil.data.preprocessing import _CACHE_PROTOCOL_VERSION
from dagl_mil.reproducibility import environment_record
from dagl_mil.reproducibility import sha256_file


def _jobs(manifest) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        for view in json.loads(str(row["views_json"])):
            jobs.append(
                {
                    "exam_id": str(row["exam_id"]),
                    "breast_id": str(row["breast_id"]),
                    "laterality": str(row["laterality"]),
                    "split": str(row["split"]),
                    "mass": int(row["mass"]),
                    "calcification": int(row["calcification"]),
                    "image_id": str(view["image_id"]),
                    "view_name": str(view["view_name"]),
                    "path": str(view["path"]),
                    "height": int(view["height"]),
                    "width": int(view["width"]),
                    "bboxes": list(view.get("bboxes", [])),
                }
            )
    return jobs


def _coverage_summary(
    mask: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    threshold: float,
) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    grid = patch_grid_coverage(
        binary,
        patch_size=patch_size,
        stride=stride,
        minimum_breast_coverage=threshold,
    )
    best_flat = int(np.argmax(grid.coverage))
    best_x, best_y = (int(value) for value in grid.starts_xy[best_flat])
    return {
        "candidate_count": grid.retained_count,
        "maximum_coverage": grid.maximum_coverage,
        "best_xy": [best_x, best_y],
        "grid_windows": int(grid.coverage.size),
        "mask_pixels": int(binary.sum()),
        "image_shape": [int(value) for value in grid.image_shape],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit every formal VinDr view for at least one >=threshold patch."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dagl.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/patch_feasibility_audit.json")
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    config = load_config(args.config)
    if _CACHE_PROTOCOL_VERSION != LOCKED_PREPROCESSING_PROTOCOL:
        raise RuntimeError(
            "preprocessing protocol drifted from the locked formal v2 protocol"
        )
    manifest_path = Path(config["paths"]["split_manifest"]).resolve()
    manifest = load_manifest(manifest_path, validate=True)
    jobs = _jobs(manifest)
    data = config["data"]
    patch_size = int(data["patch_size"])
    threshold = float(data["minimum_breast_coverage"])
    preprocessor = DICOMPreprocessor(
        cache_dir=config["paths"]["preprocessing_cache"], cache_read_only=True
    )
    running_report = initial_report(
        report_kind="patch_feasibility_audit",
        config_path=args.config,
        manifest_path=manifest_path,
        preprocessing_protocol=_CACHE_PROTOCOL_VERSION,
        views=len(jobs),
        completed=0,
        failures=[],
        fallback_view_count=0,
        environment=environment_record(),
        workers=args.workers,
    )
    atomic_write_json(args.output, running_report)
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    fallback_views: list[dict[str, Any]] = []
    candidate_counts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    completed = 0

    def process(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_path = preprocessor._cache_path(
            Path(job["path"]), job["laterality"], job["view_name"]
        )
        if cache_path is None or not cache_path.is_file():
            raise FileNotFoundError(f"standardized cache miss: {cache_path}")
        processed = DICOMPreprocessor._load_cached(cache_path)
        if processed is None:
            raise ValueError(f"invalid standardized v2 cache entry: {cache_path}")
        expected_shape = (int(job["height"]), int(job["width"]))
        if processed.original_shape != expected_shape:
            raise ValueError(
                "manifest/cache shape mismatch: "
                f"expected={expected_shape}, actual={processed.original_shape}"
            )
        if processed.laterality != job["laterality"]:
            raise ValueError("manifest/cache laterality mismatch")
        if processed.view_name != job["view_name"]:
            raise ValueError("manifest/cache view mismatch")
        stride = (
            int(data["train_stride"])
            if job["split"] == "train"
            else int(data["eval_stride"])
        )
        return job, _coverage_summary(
            processed.mask,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record, summary = future.result()
                    candidate_counts[record["split"]].append(summary["candidate_count"])
                    if summary["candidate_count"] == 0:
                        fallback_views.append({**record, **summary})
                except Exception as exc:
                    failures.append(
                        {
                            **job,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                if completed % 100 == 0 or completed == len(jobs):
                    print(
                        f"patch-audit {completed}/{len(jobs)} "
                        f"fallback={len(fallback_views)} failures={len(failures)} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

        breast_fallback_counts: dict[str, int] = {}
        for record in fallback_views:
            key = str(record["breast_id"])
            breast_fallback_counts[key] = breast_fallback_counts.get(key, 0) + 1
        summaries = {}
        for split, values in candidate_counts.items():
            array = np.asarray(values, dtype=np.int64)
            summaries[split] = {
                "views": int(array.size),
                "minimum_candidates": int(array.min()) if array.size else None,
                "median_candidates": float(np.median(array)) if array.size else None,
                "maximum_candidates": int(array.max()) if array.size else None,
                "zero_candidate_views": int((array == 0).sum()),
            }
        validation_errors: list[str] = []
        if len(jobs) != EXPECTED_TOTAL_VIEWS:
            validation_errors.append(
                f"manifest view count is {len(jobs)}, expected {EXPECTED_TOTAL_VIEWS}"
            )
        if completed != len(jobs):
            validation_errors.append("not every manifest view completed")
        if sha256_file(args.config) != running_report["config_sha256"]:
            validation_errors.append("config changed while patch audit was running")
        if sha256_file(manifest_path) != running_report["manifest_sha256"]:
            validation_errors.append("manifest changed while patch audit was running")
        for split, expected_views in EXPECTED_SPLIT_VIEWS.items():
            summary = summaries.get(split, {})
            if summary.get("views") != expected_views:
                validation_errors.append(
                    f"{split} completed views={summary.get('views')}, "
                    f"expected={expected_views}"
                )
            if summary.get("zero_candidate_views") != 0:
                validation_errors.append(f"{split} contains zero-candidate views")
            minimum = summary.get("minimum_candidates")
            if not isinstance(minimum, int) or minimum < 1:
                validation_errors.append(
                    f"{split} minimum candidate count is not positive"
                )
        report = finish_report(
            running_report,
            status=(
                "passed"
                if not failures and not fallback_views and not validation_errors
                else "failed"
            ),
            patch_size=patch_size,
            train_stride=int(data["train_stride"]),
            eval_stride=int(data["eval_stride"]),
            minimum_breast_coverage=threshold,
            views=len(jobs),
            completed=completed,
            fallback_view_count=len(fallback_views),
            fallback_breast_count=len(breast_fallback_counts),
            breasts_with_multiple_fallback_views=sorted(
                key for key, count in breast_fallback_counts.items() if count > 1
            ),
            split_summaries=summaries,
            fallback_views=sorted(
                fallback_views,
                key=lambda item: (
                    item["split"],
                    item["exam_id"],
                    item["laterality"],
                    item["view_name"],
                ),
            ),
            failures=failures,
            validation_errors=validation_errors,
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(args.output, report)
    except BaseException as exc:
        failed_report = finish_report(
            running_report,
            status="failed",
            views=len(jobs),
            completed=completed,
            fallback_view_count=len(fallback_views),
            fallback_views=fallback_views,
            failures=failures,
            fatal_error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(args.output, failed_report)
        raise
    print(args.output.resolve(), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
