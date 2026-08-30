from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

import dagl_mil.data_gates as data_gates
import scripts.validate_data_gates as validation_script
from dagl_mil.data_gates import (
    BBOX_BOUNDARY_PROTOCOL,
    DataGateError,
    EXPECTED_SPLIT_VIEWS,
    EXPECTED_TOPOLOGY_FALLBACK_IDS,
    LOCKED_PREPROCESSING_PROTOCOL,
    atomic_write_json,
    finish_report,
    initial_report,
    require_passed_data_gate,
    validate_bbox_audit,
    validate_patch_report,
    validate_precache_report,
)


CONFIG_SHA = "c" * 64
MANIFEST_SHA = "m" * 64


def _valid_precache_report() -> dict[str, object]:
    return {
        "status": "passed",
        "preprocessing_protocol": LOCKED_PREPROCESSING_PROTOCOL,
        "config_sha256": CONFIG_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "images": 20_000,
        "completed": 20_000,
        "failures": [],
        "mask_strategy_counts": {
            "otsu_lcc": 19_996,
            "nonpadding_lcc_topology_fallback": 4,
        },
        "topology_fallbacks": [
            {"image_id": image_id}
            for image_id in sorted(EXPECTED_TOPOLOGY_FALLBACK_IDS)
        ],
    }


def _image_splits() -> dict[str, str]:
    return {
        "fb8fd3863ad17037894b4171da696251": "train",
        "8ebbb6116d0a6187d9dc5eed5836e196": "train",
        "863bde43e71a55208eda16c6baa2a974": "train",
        "7e1c0fa32a1924fc7f80c5efbc3753d8": "val",
    }


def _data_config() -> dict[str, object]:
    return {
        "patch_size": 512,
        "train_stride": 512,
        "eval_stride": 128,
        "minimum_breast_coverage": 0.5,
    }


def _valid_patch_report() -> dict[str, object]:
    return {
        "status": "passed",
        "preprocessing_protocol": LOCKED_PREPROCESSING_PROTOCOL,
        "config_sha256": CONFIG_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "patch_size": 512,
        "train_stride": 512,
        "eval_stride": 128,
        "minimum_breast_coverage": 0.5,
        "views": 20_000,
        "completed": 20_000,
        "failures": [],
        "fallback_view_count": 0,
        "fallback_breast_count": 0,
        "fallback_views": [],
        "split_summaries": {
            split: {
                "views": views,
                "zero_candidate_views": 0,
                "minimum_candidates": 1,
                "median_candidates": 10.0,
                "maximum_candidates": 20,
            }
            for split, views in EXPECTED_SPLIT_VIEWS.items()
        },
    }


def _valid_bbox_audit() -> dict[str, object]:
    records = [
        {"split": split, "image_id": f"{split}-{index}"}
        for split, count in {"train": 25, "val": 10, "test": 7}.items()
        for index in range(count)
    ]
    return {
        "protocol": BBOX_BOUNDARY_PROTOCOL,
        "total_boxes": 2_254,
        "in_bounds_boxes": 2_212,
        "partially_out_of_canvas_boxes": 42,
        "invalid_or_disjoint_boxes": 0,
        "clipped_split_counts": {"train": 25, "val": 10, "test": 7},
        "clipped_records": records,
    }


def test_atomic_json_uses_unique_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "gate.json"
    replaced_sources: list[Path] = []
    original_replace = data_gates.os.replace

    def tracking_replace(source: str | Path, target: str | Path) -> None:
        replaced_sources.append(Path(source))
        original_replace(source, target)

    monkeypatch.setattr(data_gates.os, "replace", tracking_replace)
    atomic_write_json(destination, {"status": "running", "value": 1})
    atomic_write_json(destination, {"status": "passed", "value": 2})

    assert json.loads(destination.read_text(encoding="utf-8"))["value"] == 2
    assert len(replaced_sources) == 2
    assert replaced_sources[0] != replaced_sources[1]
    assert not list(tmp_path.glob("*.tmp"))


def test_report_lifecycle_binds_hashes_and_preserves_run_id(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "manifest.csv"
    config.write_text("seed: 2026\n", encoding="utf-8")
    manifest.write_text("image_id,split\na,train\n", encoding="utf-8")

    running = initial_report(
        report_kind="unit",
        config_path=config,
        manifest_path=manifest,
        preprocessing_protocol=LOCKED_PREPROCESSING_PROTOCOL,
    )
    terminal = finish_report(running, status="passed", completed=1)

    assert running["status"] == "running"
    assert terminal["status"] == "passed"
    assert terminal["run_id"] == running["run_id"]
    assert terminal["started_at"] == running["started_at"]
    assert terminal["completed_at"]
    assert len(str(running["config_sha256"])) == 64
    assert len(str(running["manifest_sha256"])) == 64


def test_formal_entry_rejects_a_stale_sealed_source_report(tmp_path: Path) -> None:
    config_path = tmp_path / "dagl.yaml"
    manifest_path = tmp_path / "manifest.csv"
    output_root = tmp_path / "outputs"
    precache_path = tmp_path / "precache.json"
    patch_path = tmp_path / "patch.json"
    config_path.write_text("formal: true\n", encoding="utf-8")
    manifest_path.write_text("image_id,split\na,train\n", encoding="utf-8")
    precache_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    patch_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    running = initial_report(
        report_kind="formal_data_gate_validation",
        config_path=config_path,
        manifest_path=manifest_path,
        preprocessing_protocol=LOCKED_PREPROCESSING_PROTOCOL,
    )
    seal = finish_report(
        running,
        status="passed",
        errors=[],
        checks={
            "manifest": "passed",
            "precache": "passed",
            "patch_feasibility": "passed",
        },
        manifest_bbox_audit=_valid_bbox_audit(),
        source_reports={
            "precache": {
                "path": str(precache_path.resolve()),
                "sha256": data_gates.sha256_file(precache_path),
            },
            "patch_feasibility": {
                "path": str(patch_path.resolve()),
                "sha256": data_gates.sha256_file(patch_path),
            },
        },
    )
    atomic_write_json(output_root / "data_gate_validation.json", seal)
    config = {
        "paths": {
            "output_root": str(output_root),
            "split_manifest": str(manifest_path),
        }
    }

    assert require_passed_data_gate(config)["run_id"] == running["run_id"]
    seal_without_bbox_audit = dict(seal)
    seal_without_bbox_audit.pop("manifest_bbox_audit")
    atomic_write_json(
        output_root / "data_gate_validation.json", seal_without_bbox_audit
    )
    with pytest.raises(DataGateError, match="bounding-box audit is missing"):
        require_passed_data_gate(config)

    atomic_write_json(output_root / "data_gate_validation.json", seal)
    patch_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    with pytest.raises(DataGateError, match="source report SHA-256 is stale"):
        require_passed_data_gate(config)


def test_precache_gate_accepts_only_locked_four_with_split_3_1_0() -> None:
    errors = validate_precache_report(
        _valid_precache_report(),
        expected_ids_from_config=set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
        image_split_by_id=_image_splits(),
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert errors == []

    invalid = _valid_precache_report()
    invalid["mask_strategy_counts"] = {
        "otsu_lcc": 19_997,
        "nonpadding_lcc_topology_fallback": 3,
    }
    invalid["topology_fallbacks"] = list(invalid["topology_fallbacks"])[1:]
    errors = validate_precache_report(
        invalid,
        expected_ids_from_config=set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
        image_split_by_id=_image_splits(),
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert any("strategy count != 4" in error for error in errors)
    assert any("record count != 4" in error for error in errors)
    assert any("four locked IDs" in error for error in errors)


def test_precache_gate_derives_split_for_legacy_running_process_report() -> None:
    report = _valid_precache_report()
    report.pop("config_sha256")
    report.pop("manifest_sha256")
    assert (
        validate_precache_report(
            report,
            expected_ids_from_config=set(EXPECTED_TOPOLOGY_FALLBACK_IDS),
            image_split_by_id=_image_splits(),
            config_sha256=CONFIG_SHA,
            manifest_sha256=MANIFEST_SHA,
        )
        == []
    )


def test_patch_gate_locks_all_split_counts_and_positive_minimum() -> None:
    errors = validate_patch_report(
        _valid_patch_report(),
        data_config=_data_config(),
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert errors == []

    invalid = _valid_patch_report()
    summaries = dict(invalid["split_summaries"])
    summaries["val"] = dict(summaries["val"], views=3_199, minimum_candidates=0)
    invalid["split_summaries"] = summaries
    invalid["fallback_view_count"] = 1
    errors = validate_patch_report(
        invalid,
        data_config=_data_config(),
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert any("fallback_view_count != 0" in error for error in errors)
    assert any("val views != 3200" in error for error in errors)
    assert any("minimum_candidates is not positive" in error for error in errors)


def test_common_gate_rejects_running_protocol_or_hash_drift() -> None:
    report = _valid_patch_report()
    report["status"] = "running"
    report["preprocessing_protocol"] = "v1"
    report["manifest_sha256"] = "wrong"
    errors = validate_patch_report(
        report,
        data_config=_data_config(),
        config_sha256=CONFIG_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert any("status is not passed" in error for error in errors)
    assert any("locked v2 protocol" in error for error in errors)
    assert any("manifest SHA-256 mismatch" in error for error in errors)


def test_final_validator_seals_legacy_precache_with_current_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.csv"
    config_path.write_text("formal: true\n", encoding="utf-8")
    manifest_path.write_text("formal-manifest\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    precache_path = cache_dir / "precache_report.json"
    patch_path = tmp_path / "patch.json"
    output_path = tmp_path / "sealed.json"
    precache = _valid_precache_report()
    patch = _valid_patch_report()
    for report in (precache, patch):
        report.pop("config_sha256")
        report.pop("manifest_sha256")
    atomic_write_json(precache_path, precache)
    atomic_write_json(patch_path, patch)

    config = {
        "paths": {
            "split_manifest": str(manifest_path),
            "preprocessing_cache": str(cache_dir),
        },
        "data": {
            **_data_config(),
            "expected_topology_fallback_image_ids": sorted(
                EXPECTED_TOPOLOGY_FALLBACK_IDS
            ),
        },
    }
    monkeypatch.setattr(validation_script, "load_config", lambda _: config)
    monkeypatch.setattr(validation_script, "load_manifest", lambda *_, **__: object())
    monkeypatch.setattr(
        validation_script,
        "_manifest_image_splits",
        lambda _: (
            _image_splits(),
            Counter(EXPECTED_SPLIT_VIEWS),
            [],
            _valid_bbox_audit(),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_data_gates.py",
            "--config",
            str(config_path),
            "--precache-report",
            str(precache_path),
            "--patch-report",
            str(patch_path),
            "--output",
            str(output_path),
        ],
    )

    assert validation_script.main() == 0
    sealed = json.loads(output_path.read_text(encoding="utf-8"))
    assert sealed["status"] == "passed"
    assert sealed["checks"] == {
        "manifest": "passed",
        "patch_feasibility": "passed",
        "precache": "passed",
    }
    assert sealed["source_reports"]["precache"]["sha256"]
    assert sealed["source_reports"]["patch_feasibility"]["sha256"]
    assert sealed["manifest_bbox_audit"]["invalid_or_disjoint_boxes"] == 0


def test_bbox_audit_rejects_missing_or_internally_inconsistent_records() -> None:
    assert validate_bbox_audit(_valid_bbox_audit()) == []
    assert validate_bbox_audit(None)

    invalid = _valid_bbox_audit()
    invalid["invalid_or_disjoint_boxes"] = 1
    errors = validate_bbox_audit(invalid)
    assert any("invalid_or_disjoint_boxes != 0" in error for error in errors)
    assert any("do not sum" in error for error in errors)
