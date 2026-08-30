from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_prediction_artifact, sha256_file
from .bootstrap import adjust_comparison_families, paired_cluster_bootstrap


def _classification_payload(path: str | Path) -> dict[str, Any]:
    _, iterator = read_prediction_artifact(path, verify_hash=True)
    rows: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for record in iterator:
        key = (str(record["exam_id"]), str(record["laterality"]))
        if key in rows:
            raise ValueError(f"Duplicate breast prediction key in {path}: {key}")
        labels = np.asarray(record["labels"], dtype=np.int64)
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        if labels.shape != (2,) or probabilities.shape != (2,):
            raise ValueError(f"Prediction labels/probabilities must have length two: {key}")
        rows[key] = (labels, probabilities)
    if not rows:
        raise ValueError(f"Prediction artifact is empty: {path}")
    keys = sorted(rows)
    return {
        "keys": keys,
        "labels": np.stack([rows[key][0] for key in keys]),
        "probabilities": np.stack([rows[key][1] for key in keys]),
        "examinations": np.asarray([key[0] for key in keys]),
        "sha256": sha256_file(path),
        "path": str(Path(path).resolve()),
    }


def compare_prediction_artifacts(
    reference_path: str | Path,
    comparators: Mapping[str, str | Path],
    *,
    reference_name: str = "dagl_mil_full",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Paired examination bootstrap followed by lesion/metric-family Holm."""

    if not comparators:
        raise ValueError("At least one comparator prediction artifact is required")
    reference = _classification_payload(reference_path)
    rows: list[dict[str, Any]] = []
    artifact_records: dict[str, Any] = {
        reference_name: {
            "path": reference["path"],
            "sha256": reference["sha256"],
        }
    }
    for comparator_name, comparator_path in comparators.items():
        comparator = _classification_payload(comparator_path)
        if comparator["keys"] != reference["keys"]:
            raise ValueError(
                f"Breast keys do not align for paired comparison: {comparator_name}"
            )
        if not np.array_equal(comparator["labels"], reference["labels"]):
            raise ValueError(
                f"Targets differ from the reference artifact: {comparator_name}"
            )
        artifact_records[str(comparator_name)] = {
            "path": comparator["path"],
            "sha256": comparator["sha256"],
        }
        for class_index, lesion in enumerate(("mass", "calcification")):
            for metric in ("roc_auc", "pr_auc"):
                result = paired_cluster_bootstrap(
                    reference["labels"][:, class_index],
                    reference["probabilities"][:, class_index],
                    comparator["probabilities"][:, class_index],
                    reference["examinations"],
                    metric=metric,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                rows.append(
                    {
                        "reference": reference_name,
                        "comparator": str(comparator_name),
                        "lesion": lesion,
                        "metric": metric,
                        **result,
                    }
                )
    adjusted = adjust_comparison_families(rows)
    return {
        "reference": reference_name,
        "breasts": len(reference["keys"]),
        "unique_examinations": int(np.unique(reference["examinations"]).size),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "holm_family": "all comparators within each lesion/metric pair",
        "artifacts": artifact_records,
        "comparisons": adjusted,
    }
