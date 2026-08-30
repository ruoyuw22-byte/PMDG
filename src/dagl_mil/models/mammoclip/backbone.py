from __future__ import annotations

import math
import pickle
import typing
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .efficientnet_custom import EfficientNet


class MammoCLIPCheckpointError(RuntimeError):
    pass


class _SafeOmegaObject:
    """Inert target for OmegaConf metadata in the published checkpoint.

    The checkpoint stores weights and an OmegaConf training configuration in
    one archive.  We never execute arbitrary pickle globals and never use that
    configuration; these inert instances merely let PyTorch's weights-only
    unpickler walk past the metadata to reach the state dictionary.
    """

    def __new__(cls, *_: Any, **__: Any) -> "_SafeOmegaObject":
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


def _safe_defaultdict(*_: Any) -> dict[Any, Any]:
    return {}


def _weights_only_load(path: Path, map_location: str | torch.device) -> Any:
    kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": True,
    }
    # mmap avoids a second 1.4 GB resident copy of the official archive.
    if path.is_file():
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except (pickle.UnpicklingError, RuntimeError) as first_error:
        if not hasattr(torch.serialization, "safe_globals"):
            raise MammoCLIPCheckpointError(
                "This PyTorch version cannot safely read the checkpoint metadata"
            ) from first_error
        safe = [
            (_SafeOmegaObject, "omegaconf.listconfig.ListConfig"),
            (_SafeOmegaObject, "omegaconf.dictconfig.DictConfig"),
            (_SafeOmegaObject, "omegaconf.base.ContainerMetadata"),
            (_SafeOmegaObject, "omegaconf.base.Metadata"),
            (_SafeOmegaObject, "omegaconf.nodes.AnyNode"),
            (_safe_defaultdict, "collections.defaultdict"),
            typing.Any,
            list,
            dict,
            int,
        ]
        try:
            with torch.serialization.safe_globals(safe):
                return torch.load(path, **kwargs)
        except Exception as error:  # pragma: no cover - message depends on torch
            raise MammoCLIPCheckpointError(
                f"Could not safely read Mammo-CLIP checkpoint: {path}"
            ) from error


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise MammoCLIPCheckpointError("Checkpoint root must be a mapping")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise MammoCLIPCheckpointError("Checkpoint does not contain a state dictionary")
    tensor_values = {key: value for key, value in checkpoint.items() if isinstance(value, Tensor)}
    if not tensor_values:
        raise MammoCLIPCheckpointError("Checkpoint state dictionary contains no tensors")
    return tensor_values


class MammoCLIPB2Backbone(nn.Module):
    """Frozen Mammo-CLIP EfficientNet-B2 feature interface used in the paper.

    The 1408-channel EfficientNet head is intentionally not evaluated.  The
    formal single-scale interface is the final MBConv map (352 channels,
    16x16 for a 512x512 patch) followed by global average pooling.  The
    pyramid interface exposes the final maps at output strides 8, 16, and 32.
    """

    feature_dim = 352
    pyramid_channels = {8: 48, 16: 120, 32: 352}
    _pyramid_blocks = {7: 8, 15: 16, 22: 32}
    _known_head_shapes = {
        "_conv_head.weight": (1408, 352, 1, 1),
        "_bn1.weight": (1408,),
        "_bn1.bias": (1408,),
        "_bn1.running_mean": (1408,),
        "_bn1.running_var": (1408,),
        "_bn1.num_batches_tracked": (),
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        micro_batch_size: int = 16,
        strict_input_size: bool = True,
    ) -> None:
        super().__init__()
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        # Dynamic same padding preserves the published weights and permits
        # inexpensive small-input unit tests; formal use is still checked at
        # the wrapper boundary when strict_input_size=True.
        self.encoder = EfficientNet.from_name(
            "efficientnet-b2", num_classes=1, image_size=None
        )
        self.micro_batch_size = int(micro_batch_size)
        self.strict_input_size = bool(strict_input_size)
        self.checkpoint_path: str | None = None
        if checkpoint_path is not None:
            self.load_mammoclip_checkpoint(checkpoint_path)
        self.freeze()

    def freeze(self) -> "MammoCLIPB2Backbone":
        nn.Module.train(self, False)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    def train(self, mode: bool = True) -> "MammoCLIPB2Backbone":
        # Calling parent_model.train() must never reactivate BN or drop-connect.
        super().train(False)
        self.encoder.eval()
        return self

    def load_mammoclip_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> None:
        path = Path(checkpoint_path)
        if path.is_dir():
            repacked = path.parent / f"{path.name}.repacked.pt"
            if not repacked.is_file():
                raise MammoCLIPCheckpointError(
                    "Checkpoint is an extracted archive and no read-only repacked file exists: "
                    f"{repacked}"
                )
            path = repacked
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = _extract_state_dict(_weights_only_load(path, map_location))

        prefixes = (
            "image_encoder.",
            "module.image_encoder.",
            "model.image_encoder.",
        )
        selected: dict[str, Tensor] = {}
        for key, value in raw.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    selected[key[len(prefix) :]] = value
                    break
        if not selected:
            # Pure image-encoder state dictionaries are also accepted.
            selected = dict(raw)

        expected = self.encoder.state_dict()
        expected_keys = set(expected)
        selected_keys = set(selected)
        missing = sorted(expected_keys - selected_keys)
        unexpected = sorted(selected_keys - expected_keys - set(self._known_head_shapes))
        missing_known_head = sorted(set(self._known_head_shapes) - selected_keys)
        mismatched = sorted(
            (key, tuple(expected[key].shape), tuple(selected[key].shape))
            for key in expected_keys & selected_keys
            if expected[key].shape != selected[key].shape
        )
        bad_head = sorted(
            (key, shape, tuple(selected[key].shape))
            for key, shape in self._known_head_shapes.items()
            if key in selected and tuple(selected[key].shape) != shape
        )
        if missing or unexpected or mismatched or missing_known_head or bad_head:
            raise MammoCLIPCheckpointError(
                "Mammo-CLIP B2 checkpoint does not exactly match the published encoder: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}, "
                f"missing_head={missing_known_head}, head_shape_mismatch={bad_head}"
            )
        # There is no permissive load: every trunk key and shape was checked,
        # and the known 1408-D head is explicitly validated then omitted because
        # the manuscript takes the final 352-D MBConv output.
        trunk = OrderedDict((key, selected[key]) for key in expected)
        self.encoder.load_state_dict(trunk, strict=True)
        self.checkpoint_path = str(path.resolve())
        self.freeze()

    def _validate_patches(self, patches: Tensor) -> Tensor:
        if not isinstance(patches, Tensor) or patches.ndim != 4:
            raise ValueError("patches must have shape [N,C,H,W]")
        if patches.shape[1] == 1:
            patches = patches.expand(-1, 3, -1, -1)
        elif patches.shape[1] != 3:
            raise ValueError("Mammo-CLIP accepts one-channel or replicated three-channel patches")
        if self.strict_input_size and tuple(patches.shape[-2:]) != (512, 512):
            raise ValueError("The formal Mammo-CLIP interface requires 512x512 patches")
        return patches

    def _extract_maps_chunk(self, patches: Tensor) -> dict[int, Tensor]:
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(patches)))
        maps: dict[int, Tensor] = {}
        for block_index, block in enumerate(self.encoder._blocks):
            # encoder is permanently eval, hence stochastic depth is inactive.
            x = block(x, drop_connect_rate=0.0)
            stride = self._pyramid_blocks.get(block_index)
            if stride is not None:
                maps[stride] = x
        if set(maps) != {8, 16, 32}:
            raise RuntimeError("Unexpected EfficientNet-B2 stage layout")
        return maps

    def forward_pyramid(self, patches: Tensor) -> dict[int, Tensor]:
        patches = self._validate_patches(patches)
        if patches.shape[0] == 0:
            h, w = patches.shape[-2:]
            return {
                stride: patches.new_empty(
                    (0, channels, math.ceil(h / stride), math.ceil(w / stride))
                )
                for stride, channels in self.pyramid_channels.items()
            }
        chunks: dict[int, list[Tensor]] = {8: [], 16: [], 32: []}
        with torch.no_grad():
            for patch_chunk in patches.split(self.micro_batch_size, dim=0):
                maps = self._extract_maps_chunk(patch_chunk)
                for stride in chunks:
                    chunks[stride].append(maps[stride].detach())
        return {stride: torch.cat(values, dim=0) for stride, values in chunks.items()}

    def forward(self, patches: Tensor) -> Tensor:
        final_map = self.forward_pyramid(patches)[32]
        return F.adaptive_avg_pool2d(final_map, 1).flatten(1).detach()

    def encode_patches(
        self,
        patches: Tensor,
        micro_batch_size: int | None = None,
        *,
        return_pyramid: bool = False,
    ) -> Tensor | dict[int, Tensor]:
        """Engine-facing online encoder with an optional temporary chunk size."""

        original = self.micro_batch_size
        if micro_batch_size is not None:
            if micro_batch_size <= 0:
                raise ValueError("micro_batch_size must be positive")
            self.micro_batch_size = int(micro_batch_size)
        try:
            if return_pyramid:
                return self.forward_pyramid(patches)
            return self.forward(patches)
        finally:
            self.micro_batch_size = original

    def forward_all(self, patches: Tensor) -> dict[str, Any]:
        maps = self.forward_pyramid(patches)
        return {
            "features": F.adaptive_avg_pool2d(maps[32], 1).flatten(1).detach(),
            "pyramid": maps,
        }


# Clear alias for callers that use the manuscript's spelling.
StrictMammoCLIPB2 = MammoCLIPB2Backbone
