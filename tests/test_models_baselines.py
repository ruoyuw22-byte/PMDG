from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dagl_mil.config import load_config
from dagl_mil.models import (
    AbMIL,
    CrossViewTransformerMIL,
    DAGLMIL,
    FPNSetTransformerMIL,
    PatchGCNMIL,
    SetTransformerMIL,
    build_model,
)


def _flat_features() -> dict[str, torch.Tensor]:
    return {
        "features": torch.randn(6, 352, requires_grad=True),
        "anatomy": torch.rand(6, 5),
        "sample_index": torch.tensor([0, 0, 0, 0, 0, 1]),
        "view_index": torch.tensor([0, 0, 1, 1, 1, 0]),
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }


def _small_fpn_batch(*, requires_grad: bool, seed: int = 17) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    pyramid = {
        8: torch.randn(5, 48, 4, 4, generator=generator).requires_grad_(requires_grad),
        16: torch.randn(5, 120, 2, 2, generator=generator).requires_grad_(requires_grad),
        32: torch.randn(5, 352, 1, 1, generator=generator).requires_grad_(requires_grad),
    }
    return {
        "pyramid": pyramid,
        "anatomy": torch.rand(5, 5, generator=generator),
        "sample_index": torch.zeros(5, dtype=torch.long),
        "view_index": torch.tensor([0, 0, 1, 1, 1]),
        "labels": torch.tensor([[1.0, 1.0]]),
    }


def _small_fpn_model() -> FPNSetTransformerMIL:
    return FPNSetTransformerMIL(
        token_dim=8,
        view_attention_hidden_dim=4,
        scale_attention_hidden_dim=4,
    )


def _assert_nested_close(actual: object, expected: object, path: str = "output") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        # CPU convolution kernels may differ by a few float32 ULPs when their
        # batch dimension changes, while candidate ordering and model semantics
        # remain identical.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-7, msg=path)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(actual_item, expected_item, f"{path}[{index}]")
        return
    assert actual == expected, path


@pytest.mark.parametrize(
    "model_class",
    [AbMIL, SetTransformerMIL, PatchGCNMIL, CrossViewTransformerMIL],
)
def test_single_scale_baseline_shapes_gradients_and_singleton_bag(model_class) -> None:
    batch = _flat_features()
    model = model_class()
    output = model(batch)
    assert output["logits"].shape == (2, 2)
    assert output["candidate_logits"].shape == (6, 2)
    assert output["bag_representations"].shape == (3, 256)
    assert output["view_mask"].tolist() == [[True, True], [True, False]]
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert batch["features"].grad is not None
    assert torch.isfinite(batch["features"].grad).all()


def test_baseline_candidate_responses_reuse_the_trained_classifier() -> None:
    model = AbMIL()
    output = model(_flat_features())
    torch.testing.assert_close(
        output["candidate_probabilities"],
        torch.sigmoid(model.classifier(output["tokens"])),
    )
    assert not hasattr(model, "candidate_head")


def test_patchgcn_is_spatial_k16_self_excluding() -> None:
    n = 18
    batch = {
        "features": torch.randn(n, 352),
        "anatomy": torch.rand(n, 5),
        "sample_index": torch.zeros(n, dtype=torch.long),
        "view_index": torch.zeros(n, dtype=torch.long),
    }
    output = PatchGCNMIL()(batch)
    neighbors = output["graph_records"][0]["neighbors"]
    assert neighbors.shape == (n, 16)
    assert not bool((neighbors == torch.arange(n)[:, None]).any())


def test_cross_view_transformer_has_two_blocks_and_directional_attention() -> None:
    output = CrossViewTransformerMIL()(_flat_features())
    assert len(output["cross_view_records"]) == 1
    layers = output["cross_view_records"][0]["layers"]
    assert len(layers) == 2
    assert layers[0]["cc_to_mlo"].shape == (4, 2, 3)
    assert layers[0]["mlo_to_cc"].shape == (4, 3, 2)


def test_fpn_settransformer_three_scales_and_gradient() -> None:
    n = 5
    pyramid = {
        8: torch.randn(n, 48, 8, 8, requires_grad=True),
        16: torch.randn(n, 120, 4, 4, requires_grad=True),
        32: torch.randn(n, 352, 2, 2, requires_grad=True),
    }
    output = FPNSetTransformerMIL()(
        {
            "pyramid": pyramid,
            "anatomy": torch.rand(n, 5),
            "sample_index": torch.zeros(n, dtype=torch.long),
            "view_index": torch.tensor([0, 0, 1, 1, 1]),
            "labels": torch.tensor([[1.0, 1.0]]),
        }
    )
    assert [values.shape for values in output["scale_tokens"]] == [
        (n, 256),
        (n, 256),
        (n, 256),
    ]
    assert output["scale_representations"].shape == (2, 3, 256)
    torch.testing.assert_close(output["scale_attention"].sum(dim=1), torch.ones(2))
    output["loss"].backward()
    assert all(pyramid[stride].grad is not None for stride in pyramid)


def test_fpn_eval_chunking_preserves_every_retained_output_and_order() -> None:
    torch.manual_seed(23)
    reference = _small_fpn_model().eval()
    chunked = _small_fpn_model().eval()
    chunked.load_state_dict(reference.state_dict())
    reference._EVAL_FPN_CHUNK_SIZE = 5
    chunked._EVAL_FPN_CHUNK_SIZE = 2
    batch = _small_fpn_batch(requires_grad=False)

    with torch.no_grad():
        expected = reference(batch)
        actual = chunked(batch)

    assert "fpn_maps" not in expected
    assert "fpn_maps" not in actual
    _assert_nested_close(actual, expected)


def test_fpn_grad_enabled_path_ignores_eval_chunking_and_preserves_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    reference = _small_fpn_model().train()
    chunked = _small_fpn_model().train()
    chunked.load_state_dict(reference.state_dict())
    reference._EVAL_FPN_CHUNK_SIZE = 5
    chunked._EVAL_FPN_CHUNK_SIZE = 2
    reference_batch = _small_fpn_batch(requires_grad=True, seed=31)
    chunked_batch = _small_fpn_batch(requires_grad=True, seed=31)

    fpn_batch_sizes: list[int] = []
    original_fpn = chunked._fpn

    def recording_fpn(maps: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        fpn_batch_sizes.append(int(maps[32].shape[0]))
        return original_fpn(maps)

    monkeypatch.setattr(chunked, "_fpn", recording_fpn)
    expected = reference(reference_batch)
    actual = chunked(chunked_batch)
    _assert_nested_close(actual, expected)
    assert fpn_batch_sizes == [5]

    expected["loss"].backward()
    actual["loss"].backward()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), chunked.named_parameters()
    ):
        assert actual_name == expected_name
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None, actual_name
        else:
            assert actual_parameter.grad is not None, actual_name
            torch.testing.assert_close(
                actual_parameter.grad,
                expected_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
                msg=actual_name,
            )

    expected_pyramid = reference_batch["pyramid"]
    actual_pyramid = chunked_batch["pyramid"]
    assert isinstance(expected_pyramid, dict)
    assert isinstance(actual_pyramid, dict)
    for stride in (8, 16, 32):
        torch.testing.assert_close(
            actual_pyramid[stride].grad,
            expected_pyramid[stride].grad,
            rtol=1e-6,
            atol=1e-7,
            msg=f"OS{stride} input gradient",
        )


def test_fpn_no_grad_calls_fpn_in_bounded_candidate_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_fpn_model().eval()
    model._EVAL_FPN_CHUNK_SIZE = 2
    batch = _small_fpn_batch(requires_grad=False)
    pyramid = batch["pyramid"]
    assert isinstance(pyramid, dict)
    fpn_batch_sizes: list[int] = []
    original_fpn = model._fpn

    def recording_fpn(maps: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        fpn_batch_sizes.append(int(maps[32].shape[0]))
        return original_fpn(maps)

    monkeypatch.setattr(model, "_fpn", recording_fpn)
    with torch.no_grad():
        scale_tokens = model._scale_tokens(pyramid)

    assert fpn_batch_sizes == [2, 2, 1]
    assert len(fpn_batch_sizes) == (5 + model._EVAL_FPN_CHUNK_SIZE - 1) // 2
    assert max(fpn_batch_sizes) <= model._EVAL_FPN_CHUNK_SIZE
    assert [tokens.shape for tokens in scale_tokens] == [(5, 8), (5, 8), (5, 8)]


def test_factory_covers_main_baselines_and_all_seven_ablations() -> None:
    config_root = Path(__file__).parents[1] / "configs"
    expected = {
        "dagl.yaml": DAGLMIL,
        "abmil.yaml": AbMIL,
        "set_transformer.yaml": SetTransformerMIL,
        "fpn_set_transformer.yaml": FPNSetTransformerMIL,
        "patchgcn_mil.yaml": PatchGCNMIL,
        "cross_view_transformer_mil.yaml": CrossViewTransformerMIL,
    }
    paths = [config_root / "dagl.yaml", *sorted((config_root / "baselines").glob("*.yaml"))]
    for path in paths:
        model = build_model(load_config(path))
        assert isinstance(model, expected[path.name])

    ablations = sorted((config_root / "ablations").glob("*.yaml"))
    assert len(ablations) == 7
    for path in ablations:
        assert isinstance(build_model(load_config(path)), DAGLMIL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="formal AMP test requires CUDA")
def test_all_trainable_models_support_fp16_amp_on_local_gpu() -> None:
    device = torch.device("cuda")
    n = 5
    features = torch.randn(n, 352, device=device)
    anatomy = torch.rand(n, 5, device=device)
    sample_index = torch.zeros(n, dtype=torch.long, device=device)
    view_index = torch.tensor([0, 0, 1, 1, 1], device=device)
    labels = torch.tensor([[1.0, 0.0]], device=device)
    for model in (
        DAGLMIL(),
        AbMIL(),
        SetTransformerMIL(),
        PatchGCNMIL(),
        CrossViewTransformerMIL(),
    ):
        model = model.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(features, anatomy, sample_index, view_index, labels)
        output["loss"].backward()
        assert torch.isfinite(output["loss"])

    pyramid = {
        8: torch.randn(n, 48, 8, 8, device=device),
        16: torch.randn(n, 120, 4, 4, device=device),
        32: torch.randn(n, 352, 2, 2, device=device),
    }
    model = FPNSetTransformerMIL().to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(
            {
                "pyramid": pyramid,
                "anatomy": anatomy,
                "sample_index": sample_index,
                "view_index": view_index,
                "labels": labels,
            }
        )
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
