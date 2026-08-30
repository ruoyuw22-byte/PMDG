from pathlib import Path

import pytest

from dagl_mil.config import ConfigError, load_config, validate_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_all_formal_configs_resolve_and_validate():
    paths = [CONFIG_ROOT / "dagl.yaml"]
    paths.extend((CONFIG_ROOT / "baselines").glob("*.yaml"))
    paths.extend((CONFIG_ROOT / "ablations").glob("*.yaml"))
    assert len(paths) == 13
    for path in paths:
        config = load_config(path)
        assert config["training"]["seed"] == 2026
        assert config["data"]["eval_stride"] == 128


def test_ablation_only_changes_requested_switch():
    config = load_config(CONFIG_ROOT / "ablations" / "without_graph.yaml")
    assert config["model"]["ablation"]["graph"] is False
    assert config["model"]["ablation"]["prototypes"] is True


@pytest.mark.parametrize(
    "override",
    [
        "data.breast_mask_threshold=plain_otsu",
        "data.intensity_percentiles=[0,100]",
        "data.geometric_augmentation=true",
        "data.cache_preprocessed_images=false",
    ],
)
def test_formal_preprocessing_protocol_cannot_drift_via_override(override: str):
    with pytest.raises(ConfigError):
        load_config(CONFIG_ROOT / "dagl.yaml", overrides=[override])


@pytest.mark.parametrize(
    "override",
    [
        "evaluation.bootstrap_samples=1999",
        "evaluation.bootstrap_seed=7",
        "evaluation.heatmap_quantile=0.9",
        "evaluation.heatmap_fixed_threshold=0.4",
        "evaluation.iou_threshold=0.5",
        "evaluation.size_thresholds=[4096,16384]",
        "evaluation.timing_warmup=19",
        "evaluation.timing_repetitions=99",
    ],
)
def test_formal_evaluation_protocol_cannot_drift_via_override(override: str):
    with pytest.raises(ConfigError, match="locked manuscript protocol"):
        load_config(CONFIG_ROOT / "dagl.yaml", overrides=[override])


def test_explicit_nonformal_context_allows_valid_smoke_evaluation_settings():
    config = load_config(
        CONFIG_ROOT / "dagl.yaml",
        overrides=[
            "experiment.protocol_context=smoke",
            "evaluation.bootstrap_samples=20",
            "evaluation.bootstrap_seed=7",
            "evaluation.heatmap_quantile=0.9",
            "evaluation.heatmap_fixed_threshold=0.0",
            "evaluation.iou_threshold=1.0",
            "evaluation.size_thresholds=[1,2]",
            "evaluation.timing_warmup=0",
            "evaluation.timing_repetitions=1",
        ],
    )
    assert config["experiment"]["protocol_context"] == "smoke"
    assert config["evaluation"]["bootstrap_samples"] == 20


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bootstrap_samples", 0),
        ("bootstrap_seed", -1),
        ("heatmap_quantile", 0.0),
        ("heatmap_quantile", 1.0),
        ("heatmap_fixed_threshold", 1.1),
        ("iou_threshold", 0.0),
        ("iou_threshold", 1.1),
        ("size_thresholds", [2, 1]),
        ("size_thresholds", [1, 1]),
        ("timing_warmup", -1),
        ("timing_repetitions", 0),
    ],
)
def test_nonformal_evaluation_settings_still_enforce_ranges(key: str, value):
    config = load_config(
        CONFIG_ROOT / "dagl.yaml",
        overrides=["experiment.protocol_context=test"],
    )
    config["evaluation"][key] = value
    with pytest.raises(ConfigError, match=f"evaluation.{key}"):
        validate_config(config)
