from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


VIEW_TO_INDEX = {"CC": 0, "MLO": 1}


@dataclass
class FlatFeatureBatch:
    """Canonical ragged representation used by all MIL heads.

    Instances are kept flat, so no padded candidate can accidentally enter a
    prototype match, graph, correspondence matrix, or Top-K operation.
    """

    features: Tensor
    anatomy: Tensor
    sample_index: Tensor
    view_index: Tensor
    bag_index: Tensor
    bag_sample_index: Tensor
    bag_view_index: Tensor
    bag_cross_view_valid: Tensor
    labels: Tensor | None
    num_samples: int
    num_bags: int


def xavier_init(module: nn.Module) -> None:
    """Apply the initialization stated in the supplement to linear layers."""

    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _view_number(value: Any, fallback: int | None = None) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("A view identity tensor must contain one value")
        value = value.item()
    if isinstance(value, str):
        key = value.upper()
        if key not in VIEW_TO_INDEX:
            raise ValueError(f"Unknown mammographic view: {value!r}")
        return VIEW_TO_INDEX[key]
    if value is None:
        if fallback is None:
            raise ValueError("Missing view identity (expected CC or MLO)")
        return fallback
    number = int(value)
    if number not in (0, 1):
        raise ValueError(f"view_index must be 0 (CC) or 1 (MLO), got {number}")
    return number


def _normalise_views(views: Any) -> list[tuple[int, Mapping[str, Any]]]:
    if isinstance(views, Mapping):
        result: list[tuple[int, Mapping[str, Any]]] = []
        for key, value in views.items():
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError("Each view must be a mapping")
            result.append((_view_number(value.get("view_name", key)), value))
        return result
    if isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
        result = []
        for fallback, value in enumerate(views):
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError("Each view must be a mapping")
            result.append(
                (
                    _view_number(
                        value.get("view_name", value.get("view_index")),
                        fallback=fallback,
                    ),
                    value,
                )
            )
        return result
    raise TypeError("views must be a mapping or a sequence of view mappings")


def _as_samples(data: Any) -> tuple[list[Mapping[str, Any]], Tensor | None]:
    if isinstance(data, Mapping) and "samples" in data:
        samples = data["samples"]
        labels = data.get("labels")
    elif isinstance(data, Mapping) and "views" in data:
        views = data["views"]
        labels = data.get("labels")
        # A collate function may return one dictionary per sample at top level,
        # or a single sample whose top-level mapping directly owns its views.
        if isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
            if len(views) == 0 or all(
                isinstance(item, Mapping) and "views" in item for item in views
            ):
                samples = views
            elif all(isinstance(item, Mapping) for item in views) and isinstance(
                labels, Tensor
            ) and labels.ndim == 2 and labels.shape[0] == len(views):
                cross_valid = data.get(
                    "cross_view_valid", torch.ones(len(views), dtype=torch.bool)
                )
                samples = [
                    {
                        "views": item,
                        "labels": labels[index],
                        "cross_view_valid": bool(torch.as_tensor(cross_valid)[index]),
                    }
                    for index, item in enumerate(views)
                ]
            else:
                samples = [data]
        else:
            samples = [data]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, Tensor)):
        samples = data
        labels = None
    else:
        raise TypeError("Expected a flat feature mapping or list of breast samples")
    if not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    return list(samples), labels


def flatten_feature_batch(
    data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    anatomy: Tensor | None = None,
    sample_index: Tensor | None = None,
    view_index: Tensor | None = None,
    labels: Tensor | None = None,
    *,
    feature_key: str = "features",
) -> FlatFeatureBatch:
    """Convert supported flat or nested feature inputs to a ragged flat batch.

    Nested samples use ``{"views": {"CC": {...}, "MLO": {...}}}``.  A view
    may store its five continuous descriptors under ``anatomy_base`` or
    ``anatomy``.  The latter may contain a sixth, discrete view-id column; it
    is deliberately discarded because the model owns the learnable embedding.
    """

    if isinstance(data, Tensor):
        features = data
    elif isinstance(data, Mapping) and feature_key in data:
        features = data[feature_key]
        anatomy = data.get("anatomy_base", data.get("anatomy", anatomy))
        sample_index = data.get("sample_index", sample_index)
        view_index = data.get("view_index", view_index)
        bag_index = data.get("bag_index")
        bag_cross_view_valid = data.get("bag_cross_view_valid")
        labels = data.get("labels", labels)
    else:
        samples, nested_labels = _as_samples(data)
        if labels is None:
            labels = nested_labels
        feature_parts: list[Tensor] = []
        anatomy_parts: list[Tensor] = []
        sample_parts: list[Tensor] = []
        view_parts: list[Tensor] = []
        bag_parts: list[Tensor] = []
        bag_cross_view_flags: list[bool] = []
        inferred_labels: list[Tensor] = []
        next_bag = 0
        for sample_number, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or "views" not in sample:
                raise TypeError("Each sample must contain a 'views' mapping")
            if labels is None and "labels" in sample:
                inferred_labels.append(torch.as_tensor(sample["labels"]))
            sample_cross_view_valid = bool(sample.get("cross_view_valid", True))
            for view_number, view in _normalise_views(sample["views"]):
                if feature_key not in view:
                    raise KeyError(
                        f"View {view_number} has no {feature_key!r}; encode patches first"
                    )
                part = torch.as_tensor(view[feature_key])
                if part.ndim != 2:
                    raise ValueError(f"{feature_key} must have shape [N,D]")
                n_instances = part.shape[0]
                if n_instances == 0:
                    continue
                descriptors = view.get("anatomy_base", view.get("anatomy"))
                if descriptors is None:
                    descriptors = part.new_zeros((n_instances, 5))
                descriptors = torch.as_tensor(
                    descriptors, device=part.device, dtype=part.dtype
                )
                if descriptors.ndim != 2 or descriptors.shape[0] != n_instances:
                    raise ValueError("anatomy descriptors must have shape [N,A]")
                feature_parts.append(part)
                anatomy_parts.append(descriptors[:, :5])
                sample_parts.append(
                    torch.full(
                        (n_instances,), sample_number, dtype=torch.long, device=part.device
                    )
                )
                view_parts.append(
                    torch.full(
                        (n_instances,), view_number, dtype=torch.long, device=part.device
                    )
                )
                bag_parts.append(
                    torch.full(
                        (n_instances,), next_bag, dtype=torch.long, device=part.device
                    )
                )
                bag_cross_view_flags.append(
                    bool(view.get("cross_view_valid", sample_cross_view_valid))
                )
                next_bag += 1
        if not feature_parts:
            raise ValueError("A breast batch must contain at least one valid candidate")
        features = torch.cat(feature_parts, dim=0)
        anatomy = torch.cat(anatomy_parts, dim=0)
        sample_index = torch.cat(sample_parts, dim=0)
        view_index = torch.cat(view_parts, dim=0)
        bag_index = torch.cat(bag_parts, dim=0)
        bag_cross_view_valid = torch.tensor(
            bag_cross_view_flags, dtype=torch.bool, device=features.device
        )
        if labels is None and inferred_labels:
            if len(inferred_labels) != len(samples):
                raise ValueError("Either every nested sample must provide labels or none may")
            labels = torch.stack(inferred_labels)
        num_samples = len(samples)
        return validate_flat_batch(
            features,
            anatomy,
            sample_index,
            view_index,
            labels,
            num_samples=num_samples,
            bag_index=bag_index,
            bag_cross_view_valid=bag_cross_view_valid,
        )

    if not isinstance(features, Tensor):
        raise TypeError(f"{feature_key} must be a tensor")
    n_instances = features.shape[0] if features.ndim else 0
    if anatomy is None:
        anatomy = features.new_zeros((n_instances, 5))
    if sample_index is None:
        sample_index = torch.zeros(n_instances, dtype=torch.long, device=features.device)
    if view_index is None:
        view_index = torch.zeros(n_instances, dtype=torch.long, device=features.device)
    if labels is not None:
        num_samples = int(labels.shape[0])
    elif n_instances:
        num_samples = int(torch.as_tensor(sample_index).max().item()) + 1
    else:
        num_samples = 0
    bag_index = None
    bag_cross_view_valid = None
    if isinstance(data, Mapping):
        bag_index = data.get("bag_index")
        bag_cross_view_valid = data.get("bag_cross_view_valid")
        if bag_index is None and "bag_slices" in data:
            slices = data["bag_slices"]
            if not isinstance(slices, Sequence) or len(slices) != num_samples:
                raise ValueError("bag_slices must contain one slot mapping per sample")
            derived = torch.full(
                (n_instances,), -1, dtype=torch.long, device=features.device
            )
            flags: list[bool] = []
            sample_flags = data.get(
                "cross_view_valid", torch.ones(num_samples, dtype=torch.bool)
            )
            sample_flags = torch.as_tensor(sample_flags, dtype=torch.bool)
            sample_index_for_slices = torch.as_tensor(sample_index)
            next_bag = 0
            for sample, slot_mapping in enumerate(slices):
                if not isinstance(slot_mapping, Mapping):
                    raise ValueError("Each bag_slices entry must be a mapping")
                for start_end in slot_mapping.values():
                    if len(start_end) != 2:
                        raise ValueError("Each bag slice must be a (start,end) pair")
                    start, end = (int(start_end[0]), int(start_end[1]))
                    if start < 0 or end <= start or end > n_instances:
                        raise ValueError("bag_slices contains an invalid candidate range")
                    if bool((derived[start:end] >= 0).any()):
                        raise ValueError("bag_slices candidate ranges overlap")
                    if not bool((sample_index_for_slices[start:end] == sample).all()):
                        raise ValueError("bag_slices disagrees with sample_index")
                    derived[start:end] = next_bag
                    flags.append(bool(sample_flags[sample]))
                    next_bag += 1
            if bool((derived < 0).any()):
                raise ValueError("bag_slices does not cover every candidate")
            bag_index = derived
            bag_cross_view_valid = torch.tensor(
                flags, dtype=torch.bool, device=features.device
            )
    return validate_flat_batch(
        features,
        anatomy,
        sample_index,
        view_index,
        labels,
        num_samples=num_samples,
        bag_index=bag_index,
        bag_cross_view_valid=bag_cross_view_valid,
    )


def validate_flat_batch(
    features: Tensor,
    anatomy: Tensor,
    sample_index: Tensor,
    view_index: Tensor,
    labels: Tensor | None,
    *,
    num_samples: int,
    bag_index: Tensor | None = None,
    bag_cross_view_valid: Tensor | None = None,
) -> FlatFeatureBatch:
    if features.ndim != 2:
        raise ValueError("features must have shape [N,D]")
    n_instances = features.shape[0]
    anatomy = torch.as_tensor(anatomy, device=features.device, dtype=features.dtype)
    sample_index = torch.as_tensor(sample_index, device=features.device, dtype=torch.long)
    view_index = torch.as_tensor(view_index, device=features.device, dtype=torch.long)
    if anatomy.ndim != 2 or anatomy.shape[0] != n_instances or anatomy.shape[1] < 5:
        raise ValueError("anatomy must have shape [N,A] with A >= 5")
    if sample_index.shape != (n_instances,) or view_index.shape != (n_instances,):
        raise ValueError("sample_index and view_index must have shape [N]")
    if n_instances == 0:
        raise ValueError("A breast batch must contain at least one valid candidate")
    if int(sample_index.min()) < 0 or int(sample_index.max()) >= num_samples:
        raise ValueError("sample_index contains an out-of-range breast index")
    if not bool(((view_index == 0) | (view_index == 1)).all()):
        raise ValueError("view_index must contain only 0 (CC) and 1 (MLO)")
    present = torch.bincount(sample_index, minlength=num_samples)
    if bool((present == 0).any()):
        raise ValueError("Every breast sample must contain at least one candidate")
    if bag_index is None:
        # The legacy flat form cannot express repeated CC/MLO acquisitions; it
        # deterministically creates one bag per observed sample/view pair.
        bag_index = torch.empty(n_instances, dtype=torch.long, device=features.device)
        next_bag = 0
        for sample in range(num_samples):
            for view in range(2):
                selected = (sample_index == sample) & (view_index == view)
                if bool(selected.any()):
                    bag_index[selected] = next_bag
                    next_bag += 1
    else:
        bag_index = torch.as_tensor(bag_index, device=features.device, dtype=torch.long)
        if bag_index.shape != (n_instances,) or int(bag_index.min()) < 0:
            raise ValueError("bag_index must be a non-negative [N] tensor")
        unique_bags = torch.unique(bag_index, sorted=True)
        if not torch.equal(
            unique_bags,
            torch.arange(unique_bags.numel(), device=features.device),
        ):
            raise ValueError("bag_index values must be contiguous from zero")
    num_bags = int(bag_index.max().item()) + 1
    bag_sample_index = torch.empty(num_bags, dtype=torch.long, device=features.device)
    bag_view_index = torch.empty(num_bags, dtype=torch.long, device=features.device)
    for bag in range(num_bags):
        members = torch.nonzero(bag_index == bag, as_tuple=False).flatten()
        sample_values = torch.unique(sample_index[members])
        view_values = torch.unique(view_index[members])
        if sample_values.numel() != 1 or view_values.numel() != 1:
            raise ValueError("Each bag must belong to exactly one breast and one view type")
        bag_sample_index[bag] = sample_values[0]
        bag_view_index[bag] = view_values[0]
    if bag_cross_view_valid is None:
        bag_cross_view_valid = torch.ones(
            num_bags, dtype=torch.bool, device=features.device
        )
    else:
        bag_cross_view_valid = torch.as_tensor(
            bag_cross_view_valid, device=features.device, dtype=torch.bool
        )
        if bag_cross_view_valid.shape == (n_instances,):
            per_bag = torch.empty(num_bags, dtype=torch.bool, device=features.device)
            for bag in range(num_bags):
                flags = torch.unique(bag_cross_view_valid[bag_index == bag])
                if flags.numel() != 1:
                    raise ValueError("cross_view_valid must be constant within a bag")
                per_bag[bag] = flags[0]
            bag_cross_view_valid = per_bag
        elif bag_cross_view_valid.shape != (num_bags,):
            raise ValueError("bag_cross_view_valid must have shape [G] or [N]")
    if labels is not None:
        labels = torch.as_tensor(labels, device=features.device, dtype=features.dtype)
        if labels.shape != (num_samples, 2):
            raise ValueError("labels must have shape [B,2] in Mass/Calcification order")
    return FlatFeatureBatch(
        features=features,
        anatomy=anatomy[:, :5],
        sample_index=sample_index,
        view_index=view_index,
        bag_index=bag_index,
        bag_sample_index=bag_sample_index,
        bag_view_index=bag_view_index,
        bag_cross_view_valid=bag_cross_view_valid,
        labels=labels,
        num_samples=num_samples,
        num_bags=num_bags,
    )


def iter_bags(batch: FlatFeatureBatch) -> Iterable[tuple[int, int, int, Tensor]]:
    """Yield ``(sample, view_type, bag, global_indices)`` groups."""

    for bag in range(batch.num_bags):
        indices = torch.nonzero(batch.bag_index == bag, as_tuple=False).flatten()
        yield (
            int(batch.bag_sample_index[bag]),
            int(batch.bag_view_index[bag]),
            bag,
            indices,
        )


class AttentionScorer(nn.Module):
    """Small tanh attention network used for view and scale fusion."""

    def __init__(self, dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).squeeze(-1)


class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.tanh = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, instances: Tensor) -> tuple[Tensor, Tensor]:
        if instances.ndim != 2 or instances.shape[0] == 0:
            raise ValueError("Gated attention requires a non-empty [N,D] bag")
        logits = self.score(
            torch.tanh(self.tanh(instances)) * torch.sigmoid(self.gate(instances))
        ).squeeze(-1)
        weights = torch.softmax(logits, dim=0).to(dtype=instances.dtype)
        return torch.sum(weights[:, None] * instances, dim=0), weights
