from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from dagl_mil.models.mammoclip import (
    MammoCLIPB2Backbone,
    MammoCLIPCheckpointError,
)


def test_frozen_backbone_eval_and_published_interfaces() -> None:
    model = MammoCLIPB2Backbone(strict_input_size=False, micro_batch_size=1)
    model.train()
    assert not model.training
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.parameters())

    patches = torch.randn(1, 1, 64, 64, requires_grad=True)
    maps = model.encode_patches(patches, return_pyramid=True)
    assert {stride: value.shape for stride, value in maps.items()} == {
        8: torch.Size([1, 48, 8, 8]),
        16: torch.Size([1, 120, 4, 4]),
        32: torch.Size([1, 352, 2, 2]),
    }
    features = model.encode_patches(patches)
    assert features.shape == (1, 352)
    assert not features.requires_grad


def _synthetic_checkpoint(model: MammoCLIPB2Backbone) -> dict[str, object]:
    state = OrderedDict(
        (f"image_encoder.{key}", value.clone())
        for key, value in model.encoder.state_dict().items()
    )
    for key, shape in model._known_head_shapes.items():
        state[f"image_encoder.{key}"] = (
            torch.tensor(0, dtype=torch.long) if shape == () else torch.randn(shape)
        )
    return {"model": state, "epoch": 10}


def test_checkpoint_loading_is_exact_not_permissive(tmp_path) -> None:
    source = MammoCLIPB2Backbone(strict_input_size=False)
    checkpoint = _synthetic_checkpoint(source)
    valid_path = tmp_path / "valid.pt"
    torch.save(checkpoint, valid_path)
    loaded = MammoCLIPB2Backbone(valid_path, strict_input_size=False)
    assert loaded.checkpoint_path == str(valid_path.resolve())
    for key, value in source.encoder.state_dict().items():
        torch.testing.assert_close(loaded.encoder.state_dict()[key], value)

    invalid = _synthetic_checkpoint(source)
    invalid["model"].pop("image_encoder._conv_stem.weight")
    invalid_path = tmp_path / "invalid.pt"
    torch.save(invalid, invalid_path)
    with pytest.raises(MammoCLIPCheckpointError, match="missing"):
        loaded.load_mammoclip_checkpoint(invalid_path)


def test_formal_backbone_rejects_non_512_input() -> None:
    model = MammoCLIPB2Backbone(strict_input_size=True)
    with pytest.raises(ValueError, match="512x512"):
        model(torch.randn(1, 3, 64, 64))
