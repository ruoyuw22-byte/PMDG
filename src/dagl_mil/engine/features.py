from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dagl_mil.data.preprocessing import _CACHE_PROTOCOL_VERSION

from .feature_cache import CacheEntryIdentity, FrozenGAPFeatureCache


GLOBAL_BATCHING_PROTOCOL = "stable-cross-view-global-v1"


class FeatureEncodingError(RuntimeError):
    """Raised when a collated raw-patch batch cannot be encoded safely."""


def normalise_mammoclip_patches(
    patches: Tensor,
    *,
    mean: float = 0.3089,
    std: float = 0.2505,
) -> Tensor:
    if patches.ndim != 4 or patches.shape[1] not in (1, 3):
        raise ValueError("patches must have shape [N,1|3,H,W]")
    if std <= 0:
        raise ValueError("normalisation std must be positive")
    return (patches - float(mean)) / float(std)


def _encode_patch_micro_batch(
    backbone: nn.Module,
    patch_parts: Sequence[Tensor],
    *,
    device: torch.device,
    micro_batch_size: int,
    mean: float,
    std: float,
    return_pyramid: bool,
) -> tuple[Tensor, dict[int, Tensor] | None]:
    if not patch_parts:
        raise FeatureEncodingError("A backbone micro-batch cannot be empty")
    count = sum(int(part.shape[0]) for part in patch_parts)
    if count <= 0 or count > micro_batch_size:
        raise FeatureEncodingError(
            f"Backbone micro-batch has {count} patches; expected 1..{micro_batch_size}"
        )
    expected_tail = tuple(patch_parts[0].shape[1:])
    if any(
        part.ndim != 4 or tuple(part.shape[1:]) != expected_tail for part in patch_parts
    ):
        raise FeatureEncodingError(
            "All globally batched patches must share channel and spatial dimensions"
        )
    with torch.no_grad():
        # Transfer the original one-channel segments first and concatenate only
        # this temporary GPU micro-batch.  No complete RGB batch or padded bag
        # is materialised on either host or device.
        device_parts = [
            part.contiguous().to(device=device, non_blocking=True)
            for part in patch_parts
        ]
        chunk = (
            device_parts[0]
            if len(device_parts) == 1
            else torch.cat(device_parts, dim=0)
        )
        chunk = normalise_mammoclip_patches(chunk, mean=mean, std=std)
        if return_pyramid:
            maps = backbone.encode_patches(
                chunk,
                micro_batch_size=micro_batch_size,
                return_pyramid=True,
            )
            if not isinstance(maps, Mapping) or set(maps) != {8, 16, 32}:
                raise FeatureEncodingError("Backbone pyramid must expose OS8/16/32")
            detached_maps: dict[int, Tensor] = {}
            for stride in (8, 16, 32):
                value = maps[stride]
                if (
                    not isinstance(value, Tensor)
                    or value.ndim != 4
                    or value.shape[0] != count
                ):
                    raise FeatureEncodingError(
                        f"Backbone OS{stride} map must preserve micro-batch length {count}"
                    )
                detached_maps[stride] = value.detach()
            features = F.adaptive_avg_pool2d(detached_maps[32], 1).flatten(1).detach()
            if features.shape[0] != count:
                raise FeatureEncodingError(
                    "Backbone pooled features changed patch order/length"
                )
            return features, detached_maps

        encoded = backbone.encode_patches(chunk, micro_batch_size=micro_batch_size)
        if (
            not isinstance(encoded, Tensor)
            or encoded.ndim != 2
            or encoded.shape[0] != count
        ):
            raise FeatureEncodingError(
                f"Backbone must return [{count},D] features for each global micro-batch"
            )
        return encoded.detach(), None


def _encode_patch_bag(
    backbone: nn.Module,
    patches: Tensor,
    *,
    device: torch.device,
    micro_batch_size: int,
    mean: float,
    std: float,
    return_pyramid: bool,
) -> tuple[Tensor, dict[int, Tensor] | None]:
    """Legacy per-view scheduler retained for equivalence measurements."""

    if patches.ndim != 4 or not patches.shape[0]:
        raise FeatureEncodingError("Every view must contain a non-empty patch tensor")
    feature_parts: list[Tensor] = []
    pyramid_parts: dict[int, list[Tensor]] = {8: [], 16: [], 32: []}
    for chunk in patches.split(micro_batch_size, dim=0):
        features, maps = _encode_patch_micro_batch(
            backbone,
            [chunk],
            device=device,
            micro_batch_size=micro_batch_size,
            mean=mean,
            std=std,
            return_pyramid=return_pyramid,
        )
        feature_parts.append(features)
        if maps is not None:
            for stride in pyramid_parts:
                pyramid_parts[stride].append(maps[stride])
    combined = torch.cat(feature_parts, dim=0)
    if return_pyramid:
        return combined, {
            stride: torch.cat(parts, dim=0) for stride, parts in pyramid_parts.items()
        }
    return combined, None


def _encode_patch_views_globally(
    backbone: nn.Module,
    patch_views: Sequence[Tensor],
    *,
    device: torch.device,
    micro_batch_size: int,
    mean: float,
    std: float,
    return_pyramid: bool,
) -> list[tuple[Tensor, dict[int, Tensor] | None]]:
    """Stream ordered view bags through full cross-view micro-batches."""

    if not patch_views:
        return []
    expected_tail: tuple[int, ...] | None = None
    for patches in patch_views:
        if (
            not isinstance(patches, Tensor)
            or patches.ndim != 4
            or patches.shape[0] <= 0
        ):
            raise FeatureEncodingError(
                "Every view must contain a non-empty patch tensor"
            )
        tail = tuple(int(value) for value in patches.shape[1:])
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise FeatureEncodingError(
                "All views in a global backbone batch must use identical patch shapes"
            )

    # Allocate the final ragged outputs lazily and copy each encoded segment
    # into its stable view offset.  Retaining slice views from every backbone
    # output until a final ``cat`` would transiently keep a second complete
    # FPN pyramid alive, which becomes material for stride-128 evaluation.
    feature_outputs: list[Tensor | None] = [None for _ in patch_views]
    pyramid_outputs: list[dict[int, Tensor] | None] = [None for _ in patch_views]
    written = [0 for _ in patch_views]
    pending_tensors: list[Tensor] = []
    pending_destinations: list[tuple[int, int]] = []
    pending_count = 0

    def flush() -> None:
        nonlocal pending_count
        if pending_count == 0:
            return
        features, maps = _encode_patch_micro_batch(
            backbone,
            pending_tensors,
            device=device,
            micro_batch_size=micro_batch_size,
            mean=mean,
            std=std,
            return_pyramid=return_pyramid,
        )
        cursor = 0
        for view_number, length in pending_destinations:
            end = cursor + length
            total = int(patch_views[view_number].shape[0])
            destination = feature_outputs[view_number]
            if destination is None:
                destination = features.new_empty((total, features.shape[1]))
                feature_outputs[view_number] = destination
            elif destination.shape[1] != features.shape[1]:
                raise FeatureEncodingError(
                    "Backbone feature dimension changed between global micro-batches"
                )
            destination_start = written[view_number]
            destination_end = destination_start + length
            destination[destination_start:destination_end].copy_(features[cursor:end])
            if maps is not None:
                view_pyramid = pyramid_outputs[view_number]
                if view_pyramid is None:
                    view_pyramid = {
                        stride: maps[stride].new_empty((total, *maps[stride].shape[1:]))
                        for stride in (8, 16, 32)
                    }
                    pyramid_outputs[view_number] = view_pyramid
                for stride in (8, 16, 32):
                    if view_pyramid[stride].shape[1:] != maps[stride].shape[1:]:
                        raise FeatureEncodingError(
                            f"Backbone OS{stride} shape changed between global micro-batches"
                        )
                    view_pyramid[stride][destination_start:destination_end].copy_(
                        maps[stride][cursor:end]
                    )
            written[view_number] = destination_end
            cursor = end
        if cursor != pending_count or cursor != features.shape[0]:
            raise FeatureEncodingError(
                "Global backbone output could not be split by view"
            )
        pending_tensors.clear()
        pending_destinations.clear()
        pending_count = 0

    for view_number, patches in enumerate(patch_views):
        offset = 0
        while offset < patches.shape[0]:
            capacity = micro_batch_size - pending_count
            length = min(capacity, int(patches.shape[0]) - offset)
            pending_tensors.append(patches[offset : offset + length])
            pending_destinations.append((view_number, length))
            pending_count += length
            offset += length
            if pending_count == micro_batch_size:
                flush()
    flush()

    results: list[tuple[Tensor, dict[int, Tensor] | None]] = []
    for view_number in range(len(patch_views)):
        features = feature_outputs[view_number]
        if (
            features is None
            or written[view_number] != patch_views[view_number].shape[0]
        ):
            raise FeatureEncodingError("Global scheduler changed a view bag length")
        maps = pyramid_outputs[view_number]
        if return_pyramid and maps is None:
            raise FeatureEncodingError("Global scheduler lost a pyramid output")
        results.append((features, maps))
    return results


@dataclass
class _ViewEncodingWork:
    sample_index: int
    view_key: str
    view: Mapping[str, Any]
    patches: Tensor
    anatomy: Tensor
    cache_identity: CacheEntryIdentity | None = None
    features: Tensor | None = None
    pyramid: dict[int, Tensor] | None = None


def encode_collated_breasts(
    batch: Mapping[str, Any],
    backbone: nn.Module,
    *,
    device: str | torch.device,
    micro_batch_size: int = 16,
    mean: float = 0.3089,
    std: float = 0.2505,
    return_pyramid: bool = False,
    feature_cache: FrozenGAPFeatureCache | None = None,
    deterministic_eval: bool = False,
    global_micro_batching: bool = True,
) -> list[dict[str, Any]]:
    """Encode ragged CC/MLO bags online without leaking localization labels.

    The input is the hierarchical output of ``RaggedBreastCollator`` with
    ``flatten_patches=False``.  Only patches and the five anatomy descriptors
    are passed into the trainable model; boxes, masks, and source metadata stay
    outside this return value and are used only by post-hoc inference export.
    """

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    views = batch.get("views")
    labels = batch.get("labels")
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
        raise FeatureEncodingError(
            "Collated batch must contain a list of view mappings"
        )
    if not isinstance(labels, Tensor) or labels.shape != (len(views), 2):
        raise FeatureEncodingError("Collated labels must have shape [B,2]")
    cross_view = batch.get("cross_view_valid")
    if cross_view is None:
        cross_view = torch.ones(len(views), dtype=torch.bool)
    cross_view = torch.as_tensor(cross_view, dtype=torch.bool)
    if cross_view.shape != (len(views),):
        raise FeatureEncodingError("cross_view_valid must have shape [B]")

    backbone.eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise FeatureEncodingError("The formal Mammo-CLIP backbone must be frozen")
    if feature_cache is not None and deterministic_eval and not global_micro_batching:
        raise FeatureEncodingError(
            "Evaluation cache provenance requires stable global micro-batching"
        )

    # Both guards are intentional.  Randomly augmented training patches must
    # always be encoded online, while FPN-SetTrans requires full OS8/16/32 maps
    # and can never be reconstructed from cached GAP vectors.
    active_cache = feature_cache if deterministic_eval and not return_pyramid else None
    work_items: list[_ViewEncodingWork] = []
    for sample_index, sample_views in enumerate(views):
        if not isinstance(sample_views, Mapping) or not sample_views:
            raise FeatureEncodingError(f"Breast {sample_index} has no view bags")
        for view_key, view in sample_views.items():
            if not isinstance(view, Mapping):
                raise FeatureEncodingError(f"Invalid view payload: {view_key}")
            patches = view.get("patches")
            anatomy = view.get("anatomy_base", view.get("anatomy"))
            if not isinstance(patches, Tensor) or not isinstance(anatomy, Tensor):
                raise FeatureEncodingError(
                    f"View {view_key} must provide patches and anatomy_base"
                )
            if (
                patches.ndim != 4
                or patches.shape[0] <= 0
                or anatomy.ndim != 2
                or anatomy.shape[0] != patches.shape[0]
                or anatomy.shape[1] < 5
            ):
                raise FeatureEncodingError(
                    f"View {view_key} patches/anatomy must be non-empty [N,C,H,W]/[N,A>=5]"
                )
            item = _ViewEncodingWork(
                sample_index=sample_index,
                view_key=str(view_key),
                view=view,
                patches=patches,
                anatomy=anatomy,
            )
            item.cache_identity = (
                active_cache.identify(
                    exam_id=str(batch.get("exam_id", [sample_index])[sample_index]),
                    breast_id=str(batch.get("breast_id", [sample_index])[sample_index]),
                    laterality=str(batch.get("laterality", [""])[sample_index]),
                    view_key=str(view_key),
                    view=view,
                )
                if active_cache is not None
                else None
            )
            item.features = (
                active_cache.load(item.cache_identity, device=resolved_device)
                if active_cache is not None
                else None
            )
            work_items.append(item)

    missing = [item for item in work_items if item.features is None]
    if active_cache is not None and missing:
        # A partial hit must not change the global FP16 micro-batch packing
        # relative to a cold-cache pass.  Evaluation loaders use batch size 1,
        # and recomputing the complete collated breast preserves the canonical
        # sample->view->candidate schedule after an interruption or corrupt
        # entry.  Reusing some hit tensors while packing only misses can change
        # cuDNN rounding through different batch shapes.
        missing = work_items
    if missing:
        if global_micro_batching:
            encoded_missing = _encode_patch_views_globally(
                backbone,
                [item.patches for item in missing],
                device=resolved_device,
                micro_batch_size=micro_batch_size,
                mean=mean,
                std=std,
                return_pyramid=return_pyramid,
            )
        else:
            encoded_missing = [
                _encode_patch_bag(
                    backbone,
                    item.patches,
                    device=resolved_device,
                    micro_batch_size=micro_batch_size,
                    mean=mean,
                    std=std,
                    return_pyramid=return_pyramid,
                )
                for item in missing
            ]
        for item, (features, pyramid) in zip(missing, encoded_missing, strict=True):
            item.features = features
            item.pyramid = pyramid
            if active_cache is not None:
                active_cache.store(item.cache_identity, features)

    encoded_samples: list[dict[str, Any]] = []
    for sample_index, sample_views in enumerate(views):
        encoded_views: dict[str, dict[str, Any]] = {}
        sample_items = [
            item for item in work_items if item.sample_index == sample_index
        ]
        if len(sample_items) != len(sample_views):
            raise FeatureEncodingError("Internal view scheduling mismatch")
        for item in sample_items:
            features = item.features
            if features is None:
                raise FeatureEncodingError("Internal backbone output is missing")
            encoded_view: dict[str, Any] = {
                "view_name": str(item.view.get("view_name", item.view_key)).upper(),
                "features": features,
                "anatomy_base": item.anatomy[:, :5].to(
                    device=resolved_device,
                    dtype=features.dtype,
                    non_blocking=True,
                ),
                "cross_view_valid": bool(cross_view[sample_index]),
            }
            if item.pyramid is not None:
                encoded_view["pyramid"] = item.pyramid
            encoded_views[item.view_key] = encoded_view
        encoded_samples.append(
            {
                "exam_id": str(batch.get("exam_id", [sample_index])[sample_index]),
                "breast_id": str(batch.get("breast_id", [sample_index])[sample_index]),
                "laterality": str(batch.get("laterality", [""])[sample_index]),
                "labels": labels[sample_index].to(
                    device=resolved_device,
                    dtype=encoded_views[next(iter(encoded_views))]["features"].dtype,
                    non_blocking=True,
                ),
                "views": encoded_views,
                "cross_view_valid": bool(cross_view[sample_index]),
            }
        )
    return encoded_samples


def evaluation_cache_provenance(
    config: Mapping[str, Any],
    *,
    backbone_sha256: str,
    manifest_sha256: str | None = None,
    device: str | torch.device,
) -> tuple[dict[str, Any], torch.dtype]:
    """Select only frozen-feature inputs shared by the 12 non-FPN models."""

    data = config["data"]
    mixed_precision = str(config["training"].get("mixed_precision", "fp16")).lower()
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif mixed_precision == "none":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed precision mode: {mixed_precision}")
    resolved = torch.device(device)
    runtime = {
        "device_type": resolved.type,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    provenance = {
        "backbone_sha256": str(backbone_sha256).lower(),
        "manifest_sha256": (
            str(manifest_sha256).lower() if manifest_sha256 is not None else None
        ),
        "preprocessing_protocol": _CACHE_PROTOCOL_VERSION,
        "breast_mask_threshold": data.get("breast_mask_threshold"),
        "intensity_percentiles": list(data.get("intensity_percentiles", (1.0, 99.0))),
        "mean": float(data["mean"]),
        "std": float(data["std"]),
        "patch_size": int(data["patch_size"]),
        "eval_stride": int(data["eval_stride"]),
        "minimum_breast_coverage": float(data["minimum_breast_coverage"]),
        "backbone_batching_protocol": GLOBAL_BATCHING_PROTOCOL,
        "patch_micro_batch_size": int(config["training"]["patch_micro_batch_size"]),
        "amp_dtype": {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[dtype],
        "runtime": runtime,
    }
    return provenance, dtype


def build_evaluation_feature_cache(
    config: Mapping[str, Any],
    *,
    backbone_sha256: str,
    manifest_sha256: str | None = None,
    device: str | torch.device,
    cache_dir: str | Path | None = None,
) -> FrozenGAPFeatureCache | None:
    """Build the shared GAP cache, or ``None`` for the FPN configuration."""

    if str(config["model"]["name"]) == "fpn_set_transformer":
        return None
    if cache_dir is None:
        paths = config.get("paths", {})
        cache_dir = paths.get("feature_cache")
        if cache_dir is None:
            preprocessing = paths.get("preprocessing_cache")
            if preprocessing is None:
                # The cache is an optimization, never a prerequisite for a
                # correct online evaluation path.
                return None
            cache_dir = Path(preprocessing).parent / "mammoclip_b2_gap_eval"
    provenance, dtype = evaluation_cache_provenance(
        config,
        backbone_sha256=backbone_sha256,
        manifest_sha256=manifest_sha256,
        device=device,
    )
    return FrozenGAPFeatureCache(
        cache_dir,
        provenance=provenance,
        storage_dtype=dtype,
        feature_dim=int(config["model"].get("feature_dim", 352)),
    )
