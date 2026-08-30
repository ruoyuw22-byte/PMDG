from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FORMAL_EXPERIMENTS = (
    "dagl_mil_full",
    "abmil",
    "set_transformer",
    "fpn_set_transformer",
    "patchgcn_mil",
    "cross_view_transformer_mil",
    "dagl_without_anatomy",
    "dagl_without_prototype",
    "dagl_without_graph",
    "dagl_without_prototype_and_graph",
    "dagl_without_cross_view",
    "dagl_without_topk",
    "dagl_without_diversity",
)
ARTIFACT_FORMAT = "dagl-mil-skipped-step-reconciliation-v1"


class ReconciliationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconciliationError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _summary_target(document: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind in {"training_summary", "train_command_summary"}:
        return document
    if kind == "pipeline_summary":
        target = document.get("training")
        if not isinstance(target, dict):
            raise ReconciliationError("pipeline_summary.training is missing")
        return target
    raise ReconciliationError(f"Unsupported summary kind: {kind}")


def reconcile(project_root: Path, *, execute: bool) -> dict[str, Any]:
    root = project_root.resolve()
    output_root = (root / "outputs").resolve()
    manifest_summary_path = root / "artifacts" / "splits" / "manifest_summary.json"
    manifest_summary = _read_json(manifest_summary_path)
    train_counts = manifest_summary.get("counts", {}).get("train", {})
    train_breasts = int(train_counts.get("breasts", -1))
    if train_breasts != 6400:
        raise ReconciliationError(
            f"Locked formal train count must be 6400, got {train_breasts}"
        )

    matrix_path = output_root / "formal_matrix_status.json"
    matrix = _read_json(matrix_path)
    if (
        matrix.get("status") != "prediction_evaluation_completed"
        or matrix.get("current_experiment") is not None
        or matrix.get("current_stage") is not None
        or matrix.get("stages", {}).get("paired_comparisons", {}).get("status")
        != "completed"
    ):
        raise ReconciliationError("Formal matrix is not in its terminal state")
    matrix_experiments = matrix.get("experiments")
    if not isinstance(matrix_experiments, dict) or set(matrix_experiments) != set(
        FORMAL_EXPERIMENTS
    ):
        raise ReconciliationError("Formal matrix experiment inventory is incomplete")

    planned_documents: dict[Path, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for name in FORMAL_EXPERIMENTS:
        run_dir = output_root / name
        config = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        training = config["training"]
        batch_size = int(training["batch_size"])
        accumulation = int(training["gradient_accumulation"])
        if batch_size != 4 or accumulation != 2:
            raise ReconciliationError(
                f"Unexpected optimizer protocol for {name}: "
                f"batch={batch_size}, accumulation={accumulation}"
            )
        batches_per_epoch = math.ceil(train_breasts / batch_size)
        attempts_per_epoch = math.ceil(batches_per_epoch / accumulation)

        summary_path = run_dir / "training_summary.json"
        summary = _read_json(summary_path)
        epochs = int(summary["epochs_completed"])
        global_steps = int(summary["global_steps"])
        reported = int(summary["skipped_optimizer_steps"])
        derived = epochs * attempts_per_epoch - global_steps
        if derived < 0:
            raise ReconciliationError(f"Negative derived skipped count for {name}")
        audits.append(
            {
                "experiment": name,
                "epochs_completed": epochs,
                "global_steps": global_steps,
                "optimizer_attempts_per_epoch": attempts_per_epoch,
                "reported_skipped_optimizer_steps": reported,
                "derived_skipped_optimizer_steps": derived,
            }
        )

        file_specs = (
            (summary_path, "training_summary"),
            (run_dir / "train_command_summary.json", "train_command_summary"),
            (run_dir / "pipeline_summary.json", "pipeline_summary"),
        )
        for path, kind in file_specs:
            document = summary if path == summary_path else _read_json(path)
            target = _summary_target(document, kind)
            if (
                int(target["epochs_completed"]) != epochs
                or int(target["global_steps"]) != global_steps
                or int(target["skipped_optimizer_steps"]) != reported
            ):
                raise ReconciliationError(
                    f"Summary copies disagree before reconciliation: {path}"
                )
            if reported != derived:
                before_sha256 = _sha256(path)
                target["skipped_optimizer_steps"] = derived
                planned_documents[path] = document
                changes.append(
                    {
                        "experiment": name,
                        "path": str(path.resolve()),
                        "field": (
                            "training.skipped_optimizer_steps"
                            if kind == "pipeline_summary"
                            else "skipped_optimizer_steps"
                        ),
                        "before": reported,
                        "after": derived,
                        "before_sha256": before_sha256,
                    }
                )

        matrix_training = matrix_experiments[name].get("result", {}).get("training")
        if not isinstance(matrix_training, dict):
            raise ReconciliationError(f"Matrix training result is missing: {name}")
        if (
            int(matrix_training["epochs_completed"]) != epochs
            or int(matrix_training["global_steps"]) != global_steps
            or int(matrix_training["skipped_optimizer_steps"]) != reported
        ):
            raise ReconciliationError(f"Matrix summary disagrees for {name}")
        if reported != derived:
            matrix_training["skipped_optimizer_steps"] = derived

    matrix_change: dict[str, Any] | None = None
    if changes:
        matrix_change = {
            "path": str(matrix_path),
            "before_sha256": _sha256(matrix_path),
        }
        planned_documents[matrix_path] = matrix
    if execute:
        for path, document in planned_documents.items():
            _atomic_json(path, document)
        for change in changes:
            path = Path(change["path"])
            change["after_sha256"] = _sha256(path)
        if matrix_change is not None:
            matrix_change["after_sha256"] = _sha256(matrix_path)

    report = {
        "status": "completed" if execute else "dry_run",
        "artifact": ARTIFACT_FORMAT,
        "created_at": _utc_now(),
        "method": (
            "epochs_completed * ceil(ceil(train_breasts / batch_size) / "
            "gradient_accumulation) - global_steps"
        ),
        "train_breasts": train_breasts,
        "matrix_path": str(matrix_path),
        "audits": audits,
        "changes": changes,
        "matrix_change": matrix_change,
        "changed_files": len(planned_documents) if execute else 0,
        "models_checkpoints_predictions_modified": False,
    }
    if execute:
        report_path = output_root / "skipped_optimizer_step_reconciliation.json"
        _atomic_json(report_path, report)
        digest = _sha256(report_path)
        _atomic_json(
            report_path.with_suffix(report_path.suffix + ".sha256.json"),
            {
                "path": report_path.name,
                "sha256": digest,
                "artifact": ARTIFACT_FORMAT,
                "bindings": {
                    "matrix_path": str(matrix_path),
                    "formal_experiments": list(FORMAL_EXPERIMENTS),
                    "summary_change_count": len(changes),
                    "changed_file_count": len(planned_documents),
                },
            },
        )
        report["report_path"] = str(report_path)
        report["report_sha256"] = digest
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile cumulative AMP-skipped optimizer-step diagnostics."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Atomically update inconsistent summaries; default is a dry run.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            reconcile(args.project_root, execute=args.execute),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
