from pathlib import Path

import torch
import pytest

from dagl_mil.config import load_config
from dagl_mil.engine.feature_cache import FrozenGAPFeatureCache
from dagl_mil.engine.features import (
    FeatureEncodingError,
    build_evaluation_feature_cache,
    encode_collated_breasts,
    normalise_mammoclip_patches,
)
from dagl_mil.models import DAGLMIL


class _FrozenBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.chunk_sizes = []

    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        self.chunk_sizes.append(int(patches.shape[0]))
        mean = patches.mean(dim=(2, 3))
        features = mean.repeat(1, 2)[:, :4]
        if not return_pyramid:
            return features
        base = patches.mean(dim=1, keepdim=True)
        return {
            8: base.repeat(1, 48, 1, 1),
            16: base.repeat(1, 120, 1, 1),
            32: base.repeat(1, 352, 1, 1),
        }


class _OrderedBackbone(torch.nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.feature_dim = feature_dim
        self.call_sizes = []
        self.call_values = []

    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        del micro_batch_size
        marker = patches[:, 0].mean(dim=(1, 2))
        self.call_sizes.append(int(patches.shape[0]))
        self.call_values.extend(marker.detach().cpu().tolist())
        if return_pyramid:
            base = marker[:, None, None, None].expand(-1, 1, 2, 2)
            return {8: base, 16: base + 10, 32: base + 20}
        offsets = torch.arange(
            self.feature_dim, device=patches.device, dtype=patches.dtype
        )
        return marker[:, None] + offsets[None, :]


class _BrokenLengthBackbone(_OrderedBackbone):
    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        output = super().encode_patches(
            patches,
            micro_batch_size=micro_batch_size,
            return_pyramid=return_pyramid,
        )
        if return_pyramid:
            return {stride: value[:-1] for stride, value in output.items()}
        return output[:-1]


class _BatchShapeBackbone(_OrderedBackbone):
    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        output = super().encode_patches(
            patches,
            micro_batch_size=micro_batch_size,
            return_pyramid=return_pyramid,
        )
        offset = float(patches.shape[0]) * 100.0
        if return_pyramid:
            return {stride: value + offset for stride, value in output.items()}
        return output + offset


def _batch(source_path="synthetic.dcm"):
    grayscale = torch.ones(5, 1, 4, 4)
    return {
        "exam_id": ["e"],
        "breast_id": ["e_L"],
        "laterality": ["L"],
        "labels": torch.tensor([[1.0, 0.0]]),
        "cross_view_valid": torch.tensor([True]),
        "views": [
            {
                "CC": {
                    "view_name": "CC",
                    "image_id": "image-cc",
                    "source_path": str(source_path),
                    "patches": grayscale.expand(-1, 3, -1, -1),
                    "anatomy_base": torch.zeros(5, 5),
                    "coords": torch.tensor(
                        [
                            [0, 0, 4, 4],
                            [1, 0, 5, 4],
                            [2, 0, 6, 4],
                            [3, 0, 7, 4],
                            [4, 0, 8, 4],
                        ]
                    ),
                    "coverage": torch.ones(5),
                    "image_shape": (4, 8),
                }
            }
        ],
    }


def _ordered_ragged_batch():
    counts = (3, 2, 5, 1)
    cursor = 0

    def view(count, name):
        nonlocal cursor
        values = torch.arange(cursor, cursor + count, dtype=torch.float32)
        cursor += count
        return {
            "view_name": name,
            "patches": values[:, None, None, None].expand(-1, 1, 4, 4),
            "anatomy_base": torch.zeros(count, 5),
            "coords": torch.stack(
                (values, values, values + 4, values + 4), dim=1
            ).long(),
            "coverage": torch.ones(count),
            "image_shape": (4, 4),
        }

    return {
        "exam_id": ["e0", "e1"],
        "breast_id": ["e0_L", "e1_R"],
        "laterality": ["L", "R"],
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "cross_view_valid": torch.tensor([True, True]),
        "views": [
            {"CC": view(counts[0], "CC"), "MLO": view(counts[1], "MLO")},
            {"CC": view(counts[2], "CC"), "MLO": view(counts[3], "MLO")},
        ],
    }


def test_patch_normalisation():
    values = torch.tensor([[[[0.3089]]]])
    torch.testing.assert_close(
        normalise_mammoclip_patches(values), torch.zeros_like(values)
    )


def test_online_encoding_is_micro_batched_and_hierarchical():
    backbone = _FrozenBackbone()
    encoded = encode_collated_breasts(
        _batch(), backbone, device="cpu", micro_batch_size=2
    )
    assert backbone.chunk_sizes == [2, 2, 1]
    assert encoded[0]["views"]["CC"]["features"].shape == (5, 4)
    assert encoded[0]["labels"].tolist() == [1.0, 0.0]
    assert encoded[0]["views"]["CC"]["cross_view_valid"] is True


def _cache(tmp_path):
    return FrozenGAPFeatureCache(
        tmp_path,
        provenance={
            "backbone_sha256": "a" * 64,
            "preprocessing_protocol": "unit-preprocess-v1",
            "mean": 0.3089,
            "std": 0.2505,
            "patch_size": 4,
            "eval_stride": 1,
            "minimum_breast_coverage": 0.5,
            "amp_dtype": "float32",
        },
        storage_dtype=torch.float32,
        feature_dim=4,
    )


def test_deterministic_gap_cache_miss_write_hit_and_exact_order(tmp_path):
    source = tmp_path / "source.dcm"
    source.write_bytes(b"fixed source")
    batch = _batch(source)
    cache = _cache(tmp_path / "features")
    first_backbone = _FrozenBackbone()
    first = encode_collated_breasts(
        batch,
        first_backbone,
        device="cpu",
        micro_batch_size=2,
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert first_backbone.chunk_sizes == [2, 2, 1]
    assert cache.stats.misses == 1 and cache.stats.writes == 1

    second_backbone = _FrozenBackbone()
    second = encode_collated_breasts(
        batch,
        second_backbone,
        device="cpu",
        micro_batch_size=2,
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert second_backbone.chunk_sizes == []
    assert cache.stats.hits == 1
    torch.testing.assert_close(
        first[0]["views"]["CC"]["features"],
        second[0]["views"]["CC"]["features"],
        rtol=0,
        atol=0,
    )

    # Candidate order participates in the item identity: reversing geometry
    # is a miss, never a silently misaligned hit.
    reordered = _batch(source)
    reordered["views"][0]["CC"]["coords"] = torch.flip(
        reordered["views"][0]["CC"]["coords"], dims=(0,)
    )
    third_backbone = _FrozenBackbone()
    encode_collated_breasts(
        reordered,
        third_backbone,
        device="cpu",
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert third_backbone.chunk_sizes == [5]
    assert cache.stats.misses == 2 and cache.stats.writes == 2


def test_cache_hash_mismatch_is_a_recoverable_online_miss(tmp_path):
    source = tmp_path / "source.dcm"
    source.write_bytes(b"fixed source")
    batch = _batch(source)
    cache = _cache(tmp_path / "features")
    encode_collated_breasts(
        batch,
        _FrozenBackbone(),
        device="cpu",
        feature_cache=cache,
        deterministic_eval=True,
    )
    view = batch["views"][0]["CC"]
    identity = cache.identify(
        exam_id="e",
        breast_id="e_L",
        laterality="L",
        view_key="CC",
        view=view,
    )
    assert identity is not None
    payload = torch.load(identity.path, weights_only=True)
    payload["feature_sha256"] = "0" * 64
    torch.save(payload, identity.path)

    backbone = _FrozenBackbone()
    encode_collated_breasts(
        batch,
        backbone,
        device="cpu",
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert backbone.chunk_sizes == [5]
    assert cache.stats.invalid_entries == 1
    assert cache.stats.writes == 2


def test_training_and_fpn_paths_always_bypass_gap_cache(tmp_path):
    cache = _cache(tmp_path / "features")
    batch = _batch()
    training_backbone = _FrozenBackbone()
    for _ in range(2):
        encode_collated_breasts(
            batch,
            training_backbone,
            device="cpu",
            feature_cache=cache,
            deterministic_eval=False,
        )
    assert training_backbone.chunk_sizes == [5, 5]
    assert not list(tmp_path.rglob("*.pt"))

    fpn_backbone = _FrozenBackbone()
    encoded = encode_collated_breasts(
        batch,
        fpn_backbone,
        device="cpu",
        feature_cache=cache,
        deterministic_eval=True,
        return_pyramid=True,
    )
    assert fpn_backbone.chunk_sizes == [5]
    assert set(encoded[0]["views"]["CC"]["pyramid"]) == {8, 16, 32}
    assert not list(tmp_path.rglob("*.pt"))


def test_factory_namespace_is_shared_across_non_fpn_configs_only(tmp_path):
    config_root = Path(__file__).parents[1] / "configs"
    common = {
        "backbone_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "device": "cpu",
        "cache_dir": tmp_path,
    }
    dagl = build_evaluation_feature_cache(
        load_config(config_root / "dagl.yaml"), **common
    )
    abmil = build_evaluation_feature_cache(
        load_config(config_root / "baselines" / "abmil.yaml"), **common
    )
    fpn = build_evaluation_feature_cache(
        load_config(config_root / "baselines" / "fpn_set_transformer.yaml"),
        **common,
    )
    assert dagl is not None and abmil is not None
    assert dagl.namespace_digest == abmil.namespace_digest
    assert fpn is None


@pytest.mark.parametrize("return_pyramid", [False, True])
def test_global_micro_batches_preserve_legacy_view_and_patch_order(return_pyramid):
    batch = _ordered_ragged_batch()
    global_backbone = _OrderedBackbone()
    legacy_backbone = _OrderedBackbone()
    global_output = encode_collated_breasts(
        batch,
        global_backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
        return_pyramid=return_pyramid,
        global_micro_batching=True,
    )
    legacy_output = encode_collated_breasts(
        batch,
        legacy_backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
        return_pyramid=return_pyramid,
        global_micro_batching=False,
    )
    assert global_backbone.call_sizes == [4, 4, 3]
    assert legacy_backbone.call_sizes == [3, 2, 4, 1, 1]
    assert global_backbone.call_values == list(range(11))
    assert list(global_output[0]["views"]) == ["CC", "MLO"]
    assert list(global_output[1]["views"]) == ["CC", "MLO"]
    for global_sample, legacy_sample in zip(global_output, legacy_output, strict=True):
        for key in global_sample["views"]:
            current = global_sample["views"][key]
            reference = legacy_sample["views"][key]
            torch.testing.assert_close(
                current["features"], reference["features"], rtol=0, atol=0
            )
            if return_pyramid:
                for stride in (8, 16, 32):
                    torch.testing.assert_close(
                        current["pyramid"][stride],
                        reference["pyramid"][stride],
                        rtol=0,
                        atol=0,
                    )


def test_global_scheduler_batches_only_cache_misses_without_reordering(tmp_path):
    batch = _ordered_ragged_batch()
    cache = _cache(tmp_path / "features")

    first_backbone = _BatchShapeBackbone()
    reference = encode_collated_breasts(
        batch,
        first_backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert first_backbone.call_sizes == [4, 4, 3]
    assert cache.stats.misses == 4
    assert cache.stats.writes == 4

    hit_backbone = _BatchShapeBackbone()
    cached = encode_collated_breasts(
        batch,
        hit_backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert hit_backbone.call_sizes == []
    assert cache.stats.hits == 4

    # Invalidate two non-adjacent views.  A partial-cache recovery must replay
    # the complete canonical batch, including the two hits, because a
    # batch-shape-sensitive backbone can otherwise produce different FP16
    # values from the original cold-cache pass.
    partially_changed = _ordered_ragged_batch()
    partially_changed["views"][0]["MLO"]["coords"][0, 0] += 100
    partially_changed["views"][1]["CC"]["coords"][0, 0] += 100
    miss_backbone = _BatchShapeBackbone()
    partial = encode_collated_breasts(
        partially_changed,
        miss_backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
        feature_cache=cache,
        deterministic_eval=True,
    )
    assert miss_backbone.call_sizes == [4, 4, 3]
    assert miss_backbone.call_values == list(range(11))
    assert cache.stats.hits == 6
    assert cache.stats.misses == 6
    assert cache.stats.writes == 8

    for expected, entirely_cached, mixed in zip(
        reference, cached, partial, strict=True
    ):
        assert (
            list(expected["views"])
            == list(entirely_cached["views"])
            == list(mixed["views"])
        )
        for view_key in expected["views"]:
            torch.testing.assert_close(
                expected["views"][view_key]["features"],
                entirely_cached["views"][view_key]["features"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                expected["views"][view_key]["features"],
                mixed["views"][view_key]["features"],
                rtol=0,
                atol=0,
            )


def test_global_encoding_stops_backbone_gradient_and_keeps_mil_gradient():
    backbone = _OrderedBackbone(feature_dim=352)
    encoded = encode_collated_breasts(
        _ordered_ragged_batch(),
        backbone,
        device="cpu",
        micro_batch_size=4,
        mean=0.0,
        std=1.0,
    )
    assert all(
        not view["features"].requires_grad
        for sample in encoded
        for view in sample["views"].values()
    )
    model = DAGLMIL()
    output = model(encoded)
    output["loss"].backward()
    assert backbone.anchor.grad is None
    assert model.token_encoder.projection.weight.grad is not None
    assert torch.isfinite(model.token_encoder.projection.weight.grad).all()
    assert [
        int(record["probabilities"].shape[0]) for record in output["candidate_records"]
    ] == [3, 2, 5, 1]


def test_global_scheduler_fails_closed_on_length_or_shape_corruption():
    batch = _ordered_ragged_batch()
    with pytest.raises(FeatureEncodingError, match="must return \\[4,D\\]"):
        encode_collated_breasts(
            batch,
            _BrokenLengthBackbone(),
            device="cpu",
            micro_batch_size=4,
            mean=0.0,
            std=1.0,
        )

    mismatched = _ordered_ragged_batch()
    mismatched["views"][1]["MLO"]["patches"] = torch.zeros(1, 1, 5, 5)
    with pytest.raises(FeatureEncodingError, match="identical patch shapes"):
        encode_collated_breasts(
            mismatched,
            _OrderedBackbone(),
            device="cpu",
            micro_batch_size=4,
            mean=0.0,
            std=1.0,
        )
