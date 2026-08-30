"""Command-line entry point for reproducible DAGL-MIL experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dagl_mil.config import load_config
from dagl_mil.experiment import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    audit_experiments,
    evaluate_experiment,
    formal_config_paths,
    infer_experiment,
    measure_efficiency_experiment,
    prepare_experiment,
    run_all_experiments,
    smoke_experiment,
    train_experiment,
)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.KEY=VALUE",
        help="Override a config value; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dagl-mil",
        description=(
            "DAGL-MIL manuscript-aligned local-GPU experiment runner. "
            "No manuscript result is embedded in this CLI."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="build and validate the locked seed-2026 manifest"
    )
    _add_config_arguments(prepare)
    prepare.add_argument(
        "--force", action="store_true", help="rebuild an existing generated manifest"
    )
    prepare.set_defaults(handler=_command_prepare)

    train = subparsers.add_parser("train", help="train one formal configuration")
    _add_config_arguments(train)
    train.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="resume last.pt automatically, or resume an explicit checkpoint",
    )
    train.add_argument(
        "--skip-completed",
        action="store_true",
        help="reuse a complete, hash-sidecar-backed training run",
    )
    train.set_defaults(handler=_command_train)

    infer = subparsers.add_parser(
        "infer", help="run the fixed test checkpoint once and save predictions"
    )
    _add_config_arguments(infer)
    infer.add_argument("--checkpoint", type=Path, default=None)
    infer.add_argument("--predictions", type=Path, default=None)
    infer.add_argument("--skip-completed", action="store_true")
    infer.set_defaults(handler=_command_infer)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate an immutable saved prediction artifact"
    )
    _add_config_arguments(evaluate)
    evaluate.add_argument("--predictions", type=Path, default=None)
    evaluate.add_argument("--report", type=Path, default=None)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--efficiency-report", type=Path, default=None)
    evaluate.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="classification/localization only; formal run-all never uses this",
    )
    evaluate.add_argument("--skip-completed", action="store_true")
    evaluate.set_defaults(handler=_command_evaluate)

    efficiency = subparsers.add_parser(
        "efficiency",
        help="measure the median-size test CC/MLO pair on the local RTX 4090",
    )
    _add_config_arguments(efficiency)
    efficiency.add_argument("--checkpoint", type=Path, default=None)
    efficiency.add_argument("--predictions", type=Path, default=None)
    efficiency.add_argument("--report", type=Path, default=None)
    efficiency.add_argument("--skip-completed", action="store_true")
    efficiency.set_defaults(handler=_command_efficiency)

    smoke = subparsers.add_parser(
        "smoke", help="run real DICOM -> GPU -> metrics on a balanced small subset"
    )
    _add_config_arguments(smoke)
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--skip-completed", action="store_true")
    smoke.add_argument(
        "--rows-per-split",
        type=int,
        default=4,
        help="real breasts per train/val/test split (minimum 4)",
    )
    smoke.set_defaults(handler=_command_smoke)

    run_all = subparsers.add_parser(
        "run-all", help="run DAGL + five baselines + seven ablations"
    )
    run_all.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT, help="project containing configs/"
    )
    run_all.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.KEY=VALUE",
    )
    run_all.add_argument("--resume", action="store_true")
    run_all.add_argument("--skip-completed", action="store_true")
    run_all.set_defaults(handler=_command_run_all)

    audit = subparsers.add_parser(
        "audit", help="verify protected inputs, configs, split, checkpoint, and CUDA"
    )
    audit.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    audit.add_argument("--output", type=Path, default=None)
    audit.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.KEY=VALUE",
    )
    audit.set_defaults(handler=_command_audit)
    return parser


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.overrides)


def _command_prepare(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return 0, prepare_experiment(_config(args), overwrite=bool(args.force))


def _command_train(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = _config(args)
    resume = args.resume
    if resume is None:
        resume = config["training"].get("resume")
    return 0, train_experiment(
        config, resume=resume, skip_completed=bool(args.skip_completed)
    )


def _command_infer(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return 0, infer_experiment(
        _config(args),
        checkpoint_path=args.checkpoint,
        prediction_path=args.predictions,
        skip_completed=bool(args.skip_completed),
    )


def _command_evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return 0, evaluate_experiment(
        _config(args),
        prediction_path=args.predictions,
        report_path=args.report,
        skip_completed=bool(args.skip_completed),
        include_efficiency=not bool(args.skip_efficiency),
        checkpoint_path=args.checkpoint,
        efficiency_report_path=args.efficiency_report,
    )


def _command_efficiency(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return 0, measure_efficiency_experiment(
        _config(args),
        checkpoint_path=args.checkpoint,
        prediction_path=args.predictions,
        report_path=args.report,
        skip_completed=bool(args.skip_completed),
    )


def _command_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return 0, smoke_experiment(
        _config(args),
        resume=bool(args.resume),
        skip_completed=bool(args.skip_completed),
        rows_per_split=int(args.rows_per_split),
    )


def _command_run_all(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    paths = formal_config_paths(args.project_root)
    return 0, run_all_experiments(
        config_paths=paths,
        overrides=args.overrides,
        resume=bool(args.resume),
        skip_completed=bool(args.skip_completed),
    )


def _command_audit(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    report = audit_experiments(
        config_paths=formal_config_paths(args.project_root),
        overrides=args.overrides,
        output_path=args.output,
    )
    return (0 if report["status"] == "passed" else 2), report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, result = args.handler(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
