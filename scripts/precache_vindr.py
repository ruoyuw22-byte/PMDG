from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from dagl_mil.config import load_config
from dagl_mil.data import DICOMPreprocessor, load_manifest
from dagl_mil.data_gates import (
    ALLOWED_MASK_STRATEGIES,
    EXPECTED_SPLIT_VIEWS,
    EXPECTED_TOPOLOGY_FALLBACK_IDS,
    EXPECTED_TOPOLOGY_FALLBACK_SPLITS,
    EXPECTED_TOTAL_VIEWS,
    LOCKED_PREPROCESSING_PROTOCOL,
    atomic_write_json,
    finish_report,
    initial_report,
)
from dagl_mil.data.preprocessing import _CACHE_PROTOCOL_VERSION
from dagl_mil.reproducibility import environment_record
from dagl_mil.reproducibility import sha256_file


def _jobs(
    manifest: pd.DataFrame,
) -> list[tuple[str, str, str, str, int, int, str]]:
    jobs: list[tuple[str, str, str, str, int, int, str]] = []
    for row in manifest.to_dict(orient="records"):
        for view in json.loads(str(row["views_json"])):
            jobs.append(
                (
                    str(view["path"]),
                    str(row["laterality"]),
                    str(view["view_name"]),
                    str(view["image_id"]),
                    int(view["height"]),
                    int(view["width"]),
                    str(row["split"]),
                )
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic, pre-augmentation VinDr DICOM cache."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dagl.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    config = load_config(args.config)
    if _CACHE_PROTOCOL_VERSION != LOCKED_PREPROCESSING_PROTOCOL:
        raise RuntimeError(
            "preprocessing protocol drifted from the locked formal v2 protocol"
        )
    data = config["data"]
    percentiles = tuple(float(value) for value in data["intensity_percentiles"])
    preprocessor = DICOMPreprocessor(
        lower_percentile=percentiles[0],
        upper_percentile=percentiles[1],
        cache_dir=config["paths"]["preprocessing_cache"],
        cache_read_only=False,
    )
    manifest = load_manifest(config["paths"]["split_manifest"], validate=True)
    jobs = _jobs(manifest)
    manifest_path = Path(config["paths"]["split_manifest"]).resolve()
    destination = args.report or (
        Path(config["paths"]["preprocessing_cache"]) / "precache_report.json"
    )
    running_report = initial_report(
        report_kind="vindr_precache",
        config_path=args.config,
        manifest_path=manifest_path,
        preprocessing_protocol=_CACHE_PROTOCOL_VERSION,
        images=len(jobs),
        completed=0,
        failures=[],
        cache_dir=str(Path(config["paths"]["preprocessing_cache"]).resolve()),
        environment=environment_record(),
        workers=args.workers,
    )
    atomic_write_json(destination, running_report)
    started = time.perf_counter()
    failures: list[dict[str, str]] = []
    mask_strategies: Counter[str] = Counter()
    topology_fallbacks: list[dict[str, str]] = []
    completed = 0

    def process(
        job: tuple[str, str, str, str, int, int, str],
    ) -> tuple[str, tuple[int, int], str]:
        path, laterality, view_name, image_id, height, width, _ = job
        result = preprocessor(path, laterality=laterality, view_name=view_name)
        expected_shape = (height, width)
        if result.original_shape != expected_shape:
            raise ValueError(
                f"manifest/DICOM shape mismatch for {image_id}: "
                f"expected={expected_shape}, actual={result.original_shape}"
            )
        return image_id, result.image.shape, result.mask_strategy

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    _, _, strategy = future.result()
                    mask_strategies[strategy] += 1
                    if strategy == "nonpadding_lcc_topology_fallback":
                        topology_fallbacks.append(
                            {
                                "path": job[0],
                                "laterality": job[1],
                                "view_name": job[2],
                                "image_id": job[3],
                                "expected_height": job[4],
                                "expected_width": job[5],
                                "split": job[6],
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "path": job[0],
                            "laterality": job[1],
                            "view_name": job[2],
                            "image_id": job[3],
                            "expected_height": str(job[4]),
                            "expected_width": str(job[5]),
                            "split": job[6],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                if completed % 100 == 0 or completed == len(jobs):
                    print(
                        f"precache {completed}/{len(jobs)} failures={len(failures)} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

        expected_fallback_ids = {
            str(value) for value in data.get("expected_topology_fallback_image_ids", [])
        }
        actual_fallback_ids = [record["image_id"] for record in topology_fallbacks]
        fallback_splits = Counter(record["split"] for record in topology_fallbacks)
        validation_errors: list[str] = []
        if len(jobs) != EXPECTED_TOTAL_VIEWS:
            validation_errors.append(
                f"manifest image count is {len(jobs)}, expected {EXPECTED_TOTAL_VIEWS}"
            )
        if completed != len(jobs):
            validation_errors.append("not every manifest image completed")
        if sha256_file(args.config) != running_report["config_sha256"]:
            validation_errors.append("config changed while precache was running")
        if sha256_file(manifest_path) != running_report["manifest_sha256"]:
            validation_errors.append("manifest changed while precache was running")
        if sum(mask_strategies.values()) != len(jobs):
            validation_errors.append(
                "mask strategy counts do not sum to the manifest image count"
            )
        if not set(mask_strategies).issubset(ALLOWED_MASK_STRATEGIES):
            validation_errors.append(
                f"unknown mask strategies: "
                f"{sorted(set(mask_strategies) - ALLOWED_MASK_STRATEGIES)}"
            )
        if mask_strategies.get("nonpadding_lcc_topology_fallback", 0) != 4:
            validation_errors.append("topology fallback mask strategy count is not 4")
        if expected_fallback_ids != set(EXPECTED_TOPOLOGY_FALLBACK_IDS):
            validation_errors.append(
                "configured fallback image IDs differ from the four locked IDs"
            )
        if len(actual_fallback_ids) != 4 or set(actual_fallback_ids) != set(
            EXPECTED_TOPOLOGY_FALLBACK_IDS
        ):
            validation_errors.append(
                "topology fallback image IDs differ from the locked raw-pixel audit: "
                f"actual={sorted(actual_fallback_ids)}"
            )
        actual_split_counts = {
            split: fallback_splits.get(split, 0) for split in EXPECTED_SPLIT_VIEWS
        }
        if actual_split_counts != EXPECTED_TOPOLOGY_FALLBACK_SPLITS:
            validation_errors.append(
                "topology fallback split counts differ from train/val/test=3/1/0: "
                f"actual={actual_split_counts}"
            )
        report = finish_report(
            running_report,
            status="passed" if not failures and not validation_errors else "failed",
            images=len(jobs),
            completed=completed,
            failures=failures,
            mask_strategy_counts=dict(sorted(mask_strategies.items())),
            topology_fallbacks=sorted(
                topology_fallbacks,
                key=lambda item: (
                    item["image_id"],
                    item["laterality"],
                    item["view_name"],
                ),
            ),
            expected_topology_fallback_image_ids=sorted(expected_fallback_ids),
            topology_fallback_split_counts=actual_split_counts,
            topology_fallback_validation={
                "status": "passed" if not validation_errors else "failed",
                "errors": validation_errors,
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(destination, report)
    except BaseException as exc:
        failed_report = finish_report(
            running_report,
            status="failed",
            images=len(jobs),
            completed=completed,
            failures=failures,
            mask_strategy_counts=dict(sorted(mask_strategies.items())),
            topology_fallbacks=topology_fallbacks,
            fatal_error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(destination, failed_report)
        raise
    print(destination.resolve(), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
