from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (
    AttentionScorer,
    FlatFeatureBatch,
    GatedAttentionPool,
    _as_samples,
    _normalise_views,
    flatten_feature_batch,
    iter_bags,
    xavier_init,
)


class MultiheadAttentionBlock(nn.Module):
    """Mask-free MAB used on one true ragged bag at a time."""

    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        xavier_init(self.feedforward)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        attended, _ = self.attention(
            query, key_value, key_value, need_weights=False
        )
        hidden = self.norm1(query + attended)
        return self.norm2(hidden + self.feedforward(hidden))


class ISAB(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        inducing_points: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if inducing_points <= 0:
            raise ValueError("inducing_points must be positive")
        self.inducing = nn.Parameter(torch.empty(1, inducing_points, dim))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing = MultiheadAttentionBlock(dim, heads, 2 * dim, dropout)
        self.to_instances = MultiheadAttentionBlock(dim, heads, 2 * dim, dropout)

    def forward(self, values: Tensor) -> Tensor:
        inducing = self.inducing.expand(values.shape[0], -1, -1)
        hidden = self.to_inducing(inducing, values)
        return self.to_instances(values, hidden)


class PMA(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        seeds: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if seeds <= 0:
            raise ValueError("PMA seeds must be positive")
        self.seeds = nn.Parameter(torch.empty(1, seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.attention = MultiheadAttentionBlock(dim, heads, 2 * dim, dropout)

    def forward(self, values: Tensor) -> Tensor:
        seeds = self.seeds.expand(values.shape[0], -1, -1)
        return self.attention(seeds, values)


class SetTransformerPool(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        inducing_points: int = 64,
        isab_layers: int = 2,
        pma_seeds: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isab_layers != 2:
            raise ValueError("The formal SetTransformer protocol uses exactly two ISAB layers")
        if pma_seeds != 1:
            raise ValueError("The formal SetTransformer protocol uses exactly one PMA seed")
        self.encoder = nn.ModuleList(
            ISAB(dim, heads, inducing_points, dropout) for _ in range(isab_layers)
        )
        self.pool = PMA(dim, heads, pma_seeds, dropout)

    def forward(self, instances: Tensor) -> Tensor:
        if instances.ndim != 2 or not instances.shape[0]:
            raise ValueError("SetTransformer requires a non-empty [N,D] bag")
        hidden = instances.unsqueeze(0)
        for block in self.encoder:
            hidden = block(hidden)
        return self.pool(hidden).squeeze(0).squeeze(0)


class _DualViewBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int = 352,
        token_dim: int = 256,
        view_attention_hidden_dim: int = 128,
        positive_weights: Sequence[float] = (6007 / 393, 6261 / 139),
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.token_dim), nn.GELU()
        )
        # The comparison protocol specifies shared (not class-specific) view
        # attention for the baselines.
        self.view_attention = AttentionScorer(
            self.token_dim, view_attention_hidden_dim
        )
        self.classifier = nn.Linear(self.token_dim, 2)
        self.register_buffer("positive_weights", torch.as_tensor(positive_weights).float())
        xavier_init(self)

    def _prepare(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None,
        sample_index: Tensor | None,
        view_index: Tensor | None,
        labels: Tensor | None,
    ) -> tuple[FlatFeatureBatch, Tensor]:
        batch = flatten_feature_batch(
            data, anatomy, sample_index, view_index, labels, feature_key="features"
        )
        if batch.features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim}-D patch features, got {batch.features.shape[1]}"
            )
        return batch, self.projection(batch.features)

    def _finish(
        self,
        batch: FlatFeatureBatch,
        instance_tokens: Tensor,
        bag_representations: Tensor,
        candidate_attention: Tensor,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        bag_representations = bag_representations.to(dtype=instance_tokens.dtype)
        breast_representations = instance_tokens.new_zeros(
            (batch.num_samples, self.token_dim)
        )
        bag_attention = instance_tokens.new_zeros(batch.num_bags)
        counts = torch.bincount(batch.bag_sample_index, minlength=batch.num_samples)
        max_bags = int(counts.max().item())
        padded_views = instance_tokens.new_zeros(
            (batch.num_samples, max_bags, self.token_dim)
        )
        view_mask = torch.zeros(
            (batch.num_samples, max_bags), dtype=torch.bool, device=instance_tokens.device
        )
        view_types = torch.full(
            (batch.num_samples, max_bags),
            -1,
            dtype=torch.long,
            device=instance_tokens.device,
        )
        padded_attention = instance_tokens.new_zeros((batch.num_samples, max_bags))
        for sample in range(batch.num_samples):
            bags = torch.nonzero(
                batch.bag_sample_index == sample, as_tuple=False
            ).flatten()
            scores = self.view_attention(bag_representations[bags])
            weights = torch.softmax(scores, dim=0).to(
                dtype=bag_representations.dtype
            )
            bag_attention[bags] = weights
            breast_representations[sample] = torch.sum(
                weights[:, None] * bag_representations[bags], dim=0
            )
            padded_views[sample, : bags.numel()] = bag_representations[bags]
            view_mask[sample, : bags.numel()] = True
            view_types[sample, : bags.numel()] = batch.bag_view_index[bags]
            padded_attention[sample, : bags.numel()] = weights
        logits = self.classifier(breast_representations)
        probabilities = torch.sigmoid(logits)
        # The class decision function is shared between breast and candidate
        # levels.  A separate candidate head would never receive a breast-loss
        # gradient and would make baseline localization random.
        candidate_logits = self.classifier(instance_tokens)
        candidate_probabilities = torch.sigmoid(candidate_logits)
        zero = logits.sum() * 0.0
        classification = (
            F.binary_cross_entropy_with_logits(
                logits,
                batch.labels,
                pos_weight=self.positive_weights.to(logits),
            )
            if batch.labels is not None
            else zero
        )
        candidate_records = []
        for sample, view, bag, indices in iter_bags(batch):
            candidate_records.append(
                {
                    "sample_index": sample,
                    "view_index": view,
                    "bag_index": bag,
                    "indices": indices,
                    "logits": candidate_logits[indices],
                    "probabilities": candidate_probabilities[indices],
                }
            )
        result: dict[str, Any] = {
            "logits": logits,
            "probabilities": probabilities,
            "breast_logits": logits,
            "breast_probabilities": probabilities,
            "candidate_logits": candidate_logits,
            "candidate_probabilities": candidate_probabilities,
            "candidate_scores": candidate_probabilities,
            "candidate_records": candidate_records,
            "candidate_attention": candidate_attention,
            "tokens": instance_tokens,
            "bag_representations": bag_representations,
            "bag_attention": bag_attention,
            "view_representations": padded_views,
            "view_attention": padded_attention,
            "view_mask": view_mask,
            "view_types": view_types,
            "breast_representations": breast_representations,
            "sample_index": batch.sample_index,
            "view_index": batch.view_index,
            "bag_index": batch.bag_index,
            "bag_sample_index": batch.bag_sample_index,
            "bag_view_index": batch.bag_view_index,
            "losses": {"classification": classification, "total": classification},
            "loss": classification,
        }
        if extra:
            result.update(extra)
        return result

    def _pool_bags(
        self,
        batch: FlatFeatureBatch,
        tokens: Tensor,
        pool: Callable[[Tensor], tuple[Tensor, Tensor | None]],
    ) -> tuple[Tensor, Tensor]:
        representations = tokens.new_zeros((batch.num_bags, self.token_dim))
        attention = tokens.new_zeros(tokens.shape[0])
        for _, _, bag, indices in iter_bags(batch):
            representation, weights = pool(tokens[indices])
            representations[bag] = representation
            if weights is not None:
                attention[indices] = weights
        return representations, attention


class AbMIL(_DualViewBaseline):
    """Gated-attention MIL with hidden dimension 128."""

    def __init__(self, *args: Any, attention_hidden_dim: int = 128, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if attention_hidden_dim != 128:
            raise ValueError("The formal AbMIL attention hidden dimension is 128")
        self.pool = GatedAttentionPool(self.token_dim, attention_hidden_dim)
        xavier_init(self.pool)

    def forward(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, tokens = self._prepare(data, anatomy, sample_index, view_index, labels)
        representations, attention = self._pool_bags(batch, tokens, self.pool)
        return self._finish(batch, tokens, representations, attention)


class SetTransformerMIL(_DualViewBaseline):
    def __init__(
        self,
        *args: Any,
        attention_heads: int = 4,
        inducing_points: int = 64,
        isab_layers: int = 2,
        pma_seeds: int = 1,
        transformer_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if attention_heads != 4 or inducing_points != 64:
            raise ValueError("Formal SetTrans uses four heads and 64 inducing points")
        self.set_pool = SetTransformerPool(
            self.token_dim,
            attention_heads,
            inducing_points,
            isab_layers,
            pma_seeds,
            transformer_dropout,
        )

    def forward(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, tokens = self._prepare(data, anatomy, sample_index, view_index, labels)

        def pool(values: Tensor) -> tuple[Tensor, None]:
            return self.set_pool(values), None

        representations, attention = self._pool_bags(batch, tokens, pool)
        return self._finish(batch, tokens, representations, attention)


class PatchGCNMIL(_DualViewBaseline):
    """Spatial kNN PatchGCN adaptation used by the unified protocol."""

    def __init__(
        self,
        *args: Any,
        baseline_graph_neighbors: int = 16,
        graph_layers: int = 2,
        graph_dropout: float = 0.1,
        attention_hidden_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if baseline_graph_neighbors != 16 or graph_layers != 2:
            raise ValueError("Formal PatchGCN-MIL uses k=16 and two graph layers")
        if abs(graph_dropout - 0.1) > 1e-12:
            raise ValueError("Formal PatchGCN-MIL uses graph dropout 0.1")
        self.neighbors = baseline_graph_neighbors
        self.self_layers = nn.ModuleList(
            nn.Linear(self.token_dim, self.token_dim) for _ in range(graph_layers)
        )
        self.neighbor_layers = nn.ModuleList(
            nn.Linear(self.token_dim, self.token_dim) for _ in range(graph_layers)
        )
        self.dropout = nn.Dropout(graph_dropout)
        self.pool = GatedAttentionPool(self.token_dim, attention_hidden_dim)
        xavier_init(self.self_layers)
        xavier_init(self.neighbor_layers)
        xavier_init(self.pool)

    def _graph_bag(self, tokens: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        n = tokens.shape[0]
        k = min(self.neighbors, max(0, n - 1))
        if k:
            distance = torch.cdist(positions, positions)
            distance.fill_diagonal_(torch.inf)
            neighbors = torch.topk(distance, k=k, largest=False, dim=1).indices
        else:
            neighbors = torch.empty((n, 0), dtype=torch.long, device=tokens.device)
        hidden = tokens
        for self_layer, neighbor_layer in zip(self.self_layers, self.neighbor_layers):
            if k:
                aggregate = neighbor_layer(hidden)[neighbors].mean(dim=1)
            else:
                aggregate = torch.zeros_like(hidden)
            hidden = self.dropout(F.gelu(self_layer(hidden) + aggregate))
        return hidden, neighbors

    def forward(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, tokens = self._prepare(data, anatomy, sample_index, view_index, labels)
        graph_tokens = tokens.clone()
        graph_records: list[dict[str, Any]] = []
        for sample, view, bag, indices in iter_bags(batch):
            enhanced, local_neighbors = self._graph_bag(
                tokens[indices], batch.anatomy[indices, :2]
            )
            graph_tokens[indices] = enhanced
            graph_records.append(
                {
                    "sample_index": sample,
                    "view_index": view,
                    "bag_index": bag,
                    "node_indices": indices,
                    "neighbors": local_neighbors,
                    "global_neighbors": indices[local_neighbors]
                    if local_neighbors.numel()
                    else local_neighbors,
                }
            )
        representations, attention = self._pool_bags(batch, graph_tokens, self.pool)
        return self._finish(
            batch,
            graph_tokens,
            representations,
            attention,
            extra={"graph_records": graph_records, "pre_graph_tokens": tokens},
        )


class CrossViewTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout),
        )
        xavier_init(self.feedforward)

    def _finish(self, original: Tensor, attended: Tensor) -> Tensor:
        hidden = self.norm1(original + attended)
        return self.norm2(hidden + self.feedforward(hidden)).to(dtype=original.dtype)

    def forward(self, cc: Tensor, mlo: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cc_b = cc.unsqueeze(0)
        mlo_b = mlo.unsqueeze(0)
        cc_message, cc_weights = self.attention(
            cc_b,
            mlo_b,
            mlo_b,
            need_weights=True,
            average_attn_weights=False,
        )
        mlo_message, mlo_weights = self.attention(
            mlo_b,
            cc_b,
            cc_b,
            need_weights=True,
            average_attn_weights=False,
        )
        return (
            self._finish(cc_b, cc_message).squeeze(0),
            self._finish(mlo_b, mlo_message).squeeze(0),
            cc_weights.squeeze(0),
            mlo_weights.squeeze(0),
        )


class CrossViewTransformerMIL(_DualViewBaseline):
    def __init__(
        self,
        *args: Any,
        cross_view_blocks: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 512,
        transformer_dropout: float = 0.1,
        attention_hidden_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            cross_view_blocks != 2
            or attention_heads != 4
            or feedforward_dim != 512
            or abs(transformer_dropout - 0.1) > 1e-12
        ):
            raise ValueError("Formal Cross-view Transformer uses 2 blocks/4 heads/FF512/dropout0.1")
        self.blocks = nn.ModuleList(
            CrossViewTransformerBlock(
                self.token_dim,
                attention_heads,
                feedforward_dim,
                transformer_dropout,
            )
            for _ in range(cross_view_blocks)
        )
        self.pool = GatedAttentionPool(self.token_dim, attention_hidden_dim)
        xavier_init(self.pool)

    def forward(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, tokens = self._prepare(data, anatomy, sample_index, view_index, labels)
        exchanged = tokens.clone()
        records: list[dict[str, Any]] = []
        for sample in range(batch.num_samples):
            sample_bags = torch.nonzero(
                batch.bag_sample_index == sample, as_tuple=False
            ).flatten()
            cc_bags = sample_bags[batch.bag_view_index[sample_bags] == 0]
            mlo_bags = sample_bags[batch.bag_view_index[sample_bags] == 1]
            if cc_bags.numel() != 1 or mlo_bags.numel() != 1:
                continue
            pair = torch.cat((cc_bags, mlo_bags))
            if not bool(batch.bag_cross_view_valid[pair].all()):
                continue
            cc_indices = torch.nonzero(
                batch.bag_index == cc_bags[0], as_tuple=False
            ).flatten()
            mlo_indices = torch.nonzero(
                batch.bag_index == mlo_bags[0], as_tuple=False
            ).flatten()
            cc, mlo = exchanged[cc_indices], exchanged[mlo_indices]
            layer_weights = []
            for block in self.blocks:
                cc, mlo, cc_to_mlo, mlo_to_cc = block(cc, mlo)
                layer_weights.append(
                    {"cc_to_mlo": cc_to_mlo, "mlo_to_cc": mlo_to_cc}
                )
            exchanged[cc_indices] = cc
            exchanged[mlo_indices] = mlo
            records.append(
                {
                    "sample_index": sample,
                    "cc_indices": cc_indices,
                    "mlo_indices": mlo_indices,
                    "layers": layer_weights,
                }
            )
        representations, attention = self._pool_bags(batch, exchanged, self.pool)
        return self._finish(
            batch,
            exchanged,
            representations,
            attention,
            extra={"cross_view_records": records, "pre_cross_view_tokens": tokens},
        )


def _pyramid_value(mapping: Mapping[Any, Tensor], stride: int) -> Tensor:
    for key in (stride, str(stride), f"os{stride}", f"OS{stride}"):
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Pyramid has no output-stride {stride} map")


def _flatten_pyramid_batch(
    data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    labels: Tensor | None,
) -> tuple[dict[int, Tensor], FlatFeatureBatch]:
    if isinstance(data, Mapping) and "pyramid" in data and isinstance(
        data["pyramid"], Mapping
    ):
        pyramid = {stride: _pyramid_value(data["pyramid"], stride) for stride in (8, 16, 32)}
        dummy = F.adaptive_avg_pool2d(pyramid[32], 1).flatten(1)
        flat = dict(data)
        flat["features"] = dummy
        batch = flatten_feature_batch(flat, labels=labels)
        return pyramid, batch

    samples, top_labels = _as_samples(data)
    if labels is None:
        labels = top_labels
    pyramid_parts: dict[int, list[Tensor]] = {8: [], 16: [], 32: []}
    synthetic_samples: list[dict[str, Any]] = []
    for sample in samples:
        synthetic_views: list[dict[str, Any]] = []
        for view_number, view in _normalise_views(sample["views"]):
            source = view.get("pyramid", view.get("feature_maps"))
            if not isinstance(source, Mapping):
                raise KeyError("FPN-SetTrans views must provide a pyramid mapping")
            maps = {stride: _pyramid_value(source, stride) for stride in (8, 16, 32)}
            n = maps[32].shape[0]
            if n == 0:
                continue
            if any(value.ndim != 4 or value.shape[0] != n for value in maps.values()):
                raise ValueError("Every pyramid map must have shape [N,C,H,W]")
            for stride, value in maps.items():
                pyramid_parts[stride].append(value)
            synthetic_views.append(
                {
                    "features": F.adaptive_avg_pool2d(maps[32], 1).flatten(1),
                    "anatomy_base": view.get("anatomy_base", view.get("anatomy")),
                    "view_index": view_number,
                    "cross_view_valid": view.get("cross_view_valid", True),
                }
            )
        synthetic: dict[str, Any] = {"views": synthetic_views}
        if "labels" in sample:
            synthetic["labels"] = sample["labels"]
        synthetic_samples.append(synthetic)
    if any(not parts for parts in pyramid_parts.values()):
        raise ValueError("Pyramid batch contains no candidates")
    pyramid = {
        stride: torch.cat(parts, dim=0) for stride, parts in pyramid_parts.items()
    }
    batch = flatten_feature_batch(synthetic_samples, labels=labels)
    return pyramid, batch


class FPNSetTransformerMIL(_DualViewBaseline):
    """Three-level OS8/16/32 FPN followed by scale-specific SetTransformers."""

    _EVAL_FPN_CHUNK_SIZE = 16

    def __init__(
        self,
        feature_dim: int = 352,
        token_dim: int = 256,
        view_attention_hidden_dim: int = 128,
        positive_weights: Sequence[float] = (6007 / 393, 6261 / 139),
        *,
        attention_heads: int = 4,
        inducing_points_by_scale: Sequence[int] = (64, 32, 16),
        isab_layers: int = 2,
        pma_seeds: int = 1,
        scale_attention_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            feature_dim,
            token_dim,
            view_attention_hidden_dim,
            positive_weights,
        )
        if attention_heads != 4 or tuple(inducing_points_by_scale) != (64, 32, 16):
            raise ValueError("Formal FPN-SetTrans uses 4 heads and inducing points 64/32/16")
        self.lateral = nn.ModuleDict(
            {
                "8": nn.Conv2d(48, token_dim, 1),
                "16": nn.Conv2d(120, token_dim, 1),
                "32": nn.Conv2d(352, token_dim, 1),
            }
        )
        self.smooth = nn.ModuleDict(
            {str(stride): nn.Conv2d(token_dim, token_dim, 3, padding=1) for stride in (8, 16, 32)}
        )
        self.scale_pools = nn.ModuleList(
            SetTransformerPool(
                token_dim,
                attention_heads,
                inducing,
                isab_layers,
                pma_seeds,
            )
            for inducing in inducing_points_by_scale
        )
        self.scale_attention = AttentionScorer(token_dim, scale_attention_hidden_dim)
        # FPN inputs bypass the single-scale projection inherited by the base.
        del self.projection
        xavier_init(self.scale_attention)

    def _fpn(self, maps: Mapping[int, Tensor]) -> dict[int, Tensor]:
        expected_channels = {8: 48, 16: 120, 32: 352}
        for stride, channels in expected_channels.items():
            if maps[stride].ndim != 4 or maps[stride].shape[1] != channels:
                raise ValueError(
                    f"OS{stride} map must have shape [N,{channels},H,W]"
                )
        p32 = self.lateral["32"](maps[32])
        p16 = self.lateral["16"](maps[16]) + F.interpolate(
            p32, size=maps[16].shape[-2:], mode="nearest"
        )
        p8 = self.lateral["8"](maps[8]) + F.interpolate(
            p16, size=maps[8].shape[-2:], mode="nearest"
        )
        return {
            8: self.smooth["8"](p8),
            16: self.smooth["16"](p16),
            32: self.smooth["32"](p32),
        }

    @staticmethod
    def _pool_fpn_maps(maps: Mapping[int, Tensor]) -> list[Tensor]:
        return [
            F.adaptive_avg_pool2d(maps[stride], 1).flatten(1)
            for stride in (8, 16, 32)
        ]

    def _scale_tokens(self, raw_maps: Mapping[int, Tensor]) -> list[Tensor]:
        # Preserve the original full-bag graph for every gradient-enabled call.
        # Validation and inference only need the pooled candidate tokens, so their
        # FPN maps can be produced in bounded chunks instead of retaining a full
        # high-resolution breast pyramid on the GPU.
        if torch.is_grad_enabled():
            return self._pool_fpn_maps(self._fpn(raw_maps))

        chunk_size = self._EVAL_FPN_CHUNK_SIZE
        if chunk_size <= 0:
            raise ValueError("_EVAL_FPN_CHUNK_SIZE must be positive")
        candidate_count = raw_maps[32].shape[0]
        pooled_parts: list[list[Tensor]] = [[], [], []]
        for start in range(0, candidate_count, chunk_size):
            chunk = {
                stride: raw_maps[stride][start : start + chunk_size]
                for stride in (8, 16, 32)
            }
            pooled = self._pool_fpn_maps(self._fpn(chunk))
            for parts, tokens in zip(pooled_parts, pooled):
                parts.append(tokens)
        return [torch.cat(parts, dim=0) for parts in pooled_parts]

    def forward(
        self,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        if anatomy is not None or sample_index is not None or view_index is not None:
            if not isinstance(data, Mapping) or "pyramid" not in data:
                raise TypeError("Flat FPN metadata must accompany a {'pyramid': ...} mapping")
            data = dict(data)
            data.update(
                {
                    "anatomy": anatomy,
                    "sample_index": sample_index,
                    "view_index": view_index,
                }
            )
        raw_maps, batch = _flatten_pyramid_batch(data, labels)
        scale_tokens = self._scale_tokens(raw_maps)
        scale_bag_representations = scale_tokens[0].new_zeros(
            (batch.num_bags, 3, self.token_dim)
        )
        for _, _, bag, indices in iter_bags(batch):
            for scale_number, (tokens, pool) in enumerate(
                zip(scale_tokens, self.scale_pools)
            ):
                scale_bag_representations[bag, scale_number] = pool(tokens[indices])
        scale_scores = self.scale_attention(scale_bag_representations)
        scale_weights = torch.softmax(scale_scores, dim=1).to(
            dtype=scale_bag_representations.dtype
        )
        bag_representations = torch.sum(
            scale_weights[..., None] * scale_bag_representations, dim=1
        )
        instance_tokens = scale_tokens[0]
        attention = instance_tokens.new_zeros(instance_tokens.shape[0])
        return self._finish(
            batch,
            instance_tokens,
            bag_representations,
            attention,
            extra={
                "scale_tokens": scale_tokens,
                "scale_representations": scale_bag_representations,
                "scale_attention": scale_weights,
            },
        )


# Common spelling variants retained at the package boundary.
ABMIL = AbMIL
SetTransformer = SetTransformerMIL
FPNSetTransformer = FPNSetTransformerMIL
PatchGCN = PatchGCNMIL
CrossViewTransformer = CrossViewTransformerMIL
