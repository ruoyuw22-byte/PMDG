from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


_PROTOCOL_CONTEXTS = {"formal", "smoke", "exploratory", "test"}
_FORMAL_EVALUATION_DEFAULTS: dict[str, Any] = {
    "bootstrap_samples": 2000,
    "bootstrap_seed": 2026,
    "heatmap_quantile": 0.95,
    "heatmap_fixed_threshold": 0.5,
    "iou_threshold": 0.25,
    "size_thresholds": [16384, 65536],
    "timing_warmup": 20,
    "timing_repetitions": 100,
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_recursive(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular config inheritance: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if parent is None:
        result = current
    else:
        parent_path = (path.parent / parent).resolve()
        result = _merge(_load_recursive(parent_path, seen), current)
    seen.remove(path)
    return result


def _parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"Override must use dotted.key=value: {expression}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = _parse_scalar(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _load_recursive(Path(path), set())
    for expression in overrides or []:
        _apply_override(config, expression)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("experiment", "paths", "data", "model", "training", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    data = config["data"]
    if data["patch_size"] != 512:
        raise ConfigError("Formal protocol requires 512x512 patches")
    if data["train_stride"] != 512 or data["eval_stride"] != 128:
        raise ConfigError("Formal protocol requires train/eval strides 512/128")
    if data["minimum_breast_coverage"] != 0.5:
        raise ConfigError("Formal protocol requires 50% breast coverage")
    if data.get("breast_mask_threshold") != ("otsu_with_nonpadding_topology_guard"):
        raise ConfigError("Formal protocol requires the locked breast-mask strategy")
    if [float(value) for value in data.get("intensity_percentiles", [])] != [
        1.0,
        99.0,
    ]:
        raise ConfigError("Formal protocol requires P1-P99 intensity normalization")
    if bool(data.get("geometric_augmentation", False)):
        raise ConfigError("Formal protocol forbids geometric augmentation")
    if not bool(data.get("cache_preprocessed_images", False)):
        raise ConfigError("Formal protocol requires the deterministic DICOM cache")

    model = config["model"]
    allowed_models = {
        "dagl_mil",
        "abmil",
        "set_transformer",
        "fpn_set_transformer",
        "patchgcn_mil",
        "cross_view_transformer_mil",
    }
    if model["name"] not in allowed_models:
        raise ConfigError(f"Unknown model: {model['name']}")

    training = config["training"]
    if training["seed"] != 2026:
        raise ConfigError("Formal configs must use seed 2026")
    if training["batch_size"] * training["gradient_accumulation"] != 8:
        raise ConfigError("Formal effective batch size must equal 8")

    evaluation = config["evaluation"]
    integer_constraints = {
        "bootstrap_samples": (1, "positive"),
        "bootstrap_seed": (0, "non-negative"),
        "timing_warmup": (0, "non-negative"),
        "timing_repetitions": (1, "positive"),
    }
    for key, (minimum, description) in integer_constraints.items():
        value = evaluation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"evaluation.{key} must be a {description} integer")

    probability_constraints = {
        "heatmap_quantile": (False, False, "strictly between 0 and 1"),
        "heatmap_fixed_threshold": (
            True,
            True,
            "between 0 and 1 inclusive",
        ),
        "iou_threshold": (False, True, "greater than 0 and at most 1"),
    }
    for key, (
        lower_inclusive,
        upper_inclusive,
        description,
    ) in probability_constraints.items():
        value = evaluation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"evaluation.{key} must be {description}")
        numeric = float(value)
        lower_valid = numeric >= 0.0 if lower_inclusive else numeric > 0.0
        upper_valid = numeric <= 1.0 if upper_inclusive else numeric < 1.0
        valid = lower_valid and upper_valid
        if not valid:
            raise ConfigError(f"evaluation.{key} must be {description}")

    size_thresholds = evaluation.get("size_thresholds")
    if (
        not isinstance(size_thresholds, list)
        or len(size_thresholds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in size_thresholds
        )
        or size_thresholds[0] >= size_thresholds[1]
    ):
        raise ConfigError(
            "evaluation.size_thresholds must be two strictly increasing positive integers"
        )

    protocol_context = str(
        config["experiment"].get("protocol_context", "formal")
    ).lower()
    if protocol_context not in _PROTOCOL_CONTEXTS:
        raise ConfigError(
            f"experiment.protocol_context must be one of {sorted(_PROTOCOL_CONTEXTS)}"
        )
    if protocol_context == "formal":
        drift = {
            key: evaluation.get(key)
            for key, expected in _FORMAL_EVALUATION_DEFAULTS.items()
            if evaluation.get(key) != expected
        }
        if drift:
            raise ConfigError(
                "Formal evaluation settings must match the locked manuscript "
                f"protocol; drifted values: {drift}"
            )


def save_resolved_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
