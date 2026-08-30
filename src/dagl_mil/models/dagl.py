from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (
    AttentionScorer,
    FlatFeatureBatch,
    flatten_feature_batch,
    iter_bags,
    xavier_init,
)


LESION_TO_PROTOTYPE = (1, 2)  # Mass, Calcification in N/M/C memory order.


@dataclass(frozen=True)
class DAGLAblation:
    anatomy: bool = True
    prototypes: bool = True
    graph: bool = True
    cross_view: bool = True
    topk: bool = True
    diversity: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "DAGLAblation":
        if values is None:
            return cls()
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown DAGL ablation switches: {sorted(unknown)}")
        defaults = cls()
        return cls(
            **{
                name: bool(values.get(name, getattr(defaults, name)))
                for name in cls.__dataclass_fields__
            }
        )


class AnatomyAwareTokenEncoder(nn.Module):
    """Equation (1): visual feature plus five anatomy values and view embedding."""

    def __init__(
        self,
        feature_dim: int = 352,
        token_dim: int = 256,
        view_embedding_dim: int = 8,
        *,
        use_anatomy: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.view_embedding_dim = int(view_embedding_dim)
        self.use_anatomy = bool(use_anatomy)
        if self.use_anatomy:
            if self.view_embedding_dim <= 0:
                raise ValueError("view_embedding_dim must be positive when anatomy is enabled")
            self.view_embedding = nn.Embedding(2, self.view_embedding_dim)
            input_dim = self.feature_dim + 5 + self.view_embedding_dim
        else:
            self.view_embedding = None
            input_dim = self.feature_dim
        self.projection = nn.Linear(input_dim, self.token_dim)
        self.activation = nn.GELU()
        xavier_init(self)

    def forward(self, features: Tensor, anatomy: Tensor, view_index: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape [N,{self.feature_dim}]")
        if self.use_anatomy:
            if anatomy.ndim != 2 or anatomy.shape != (features.shape[0], 5):
                raise ValueError("anatomy must have shape [N,5]")
            embedded_view = self.view_embedding(view_index)
            values = torch.cat((features, anatomy, embedded_view), dim=-1)
        else:
            # The w/o Anatomy ablation removes both continuous descriptors and
            # the view embedding from equation (1), while graph coordinates
            # remain available for the otherwise unchanged relation module.
            values = features
        return self.activation(self.projection(values))


class LesionPrototypeMemory(nn.Module):
    """Equation (2): three groups of learnable unit-vector prototypes."""

    def __init__(
        self,
        token_dim: int = 256,
        prototypes_per_group: int = 8,
        diversity_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if prototypes_per_group < 2:
            raise ValueError("At least two prototypes per group are required for diversity")
        self.token_dim = int(token_dim)
        self.prototypes_per_group = int(prototypes_per_group)
        self.diversity_threshold = float(diversity_threshold)
        self.projection = nn.Linear(self.token_dim, self.token_dim)
        self.activation = nn.GELU()
        prototypes = torch.randn(3, self.prototypes_per_group, self.token_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = nn.Parameter(prototypes)
        xavier_init(self.projection)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        projected = self.activation(self.projection(tokens))
        # ``eps=1e-8`` underflows to zero in fp16.  A temporarily zero
        # projected token would therefore turn every cosine response into
        # NaN during a long AMP run.  Similarities are inexpensive compared
        # with the backbone, so keep this numerically sensitive operation in
        # fp32 even when the surrounding forward is autocast.
        with torch.autocast(device_type=projected.device.type, enabled=False):
            projected_norm = F.normalize(projected.float(), dim=-1, eps=1e-8)
            prototype_norm = F.normalize(
                self.prototypes.float(), dim=-1, eps=1e-8
            )
            similarities = torch.einsum(
                "nd,gmd->ngm", projected_norm, prototype_norm
            )
        responses, assignments = similarities.max(dim=-1)
        return {
            "projected": projected,
            "similarities": similarities,
            "responses": responses,
            "assignments": assignments,
        }

    def diversity_loss(self) -> Tensor:
        with torch.autocast(device_type=self.prototypes.device.type, enabled=False):
            normalised = F.normalize(self.prototypes.float(), dim=-1, eps=1e-8)
            similarities = torch.matmul(
                normalised, normalised.transpose(-1, -2)
            )
        upper = torch.triu(
            torch.ones_like(similarities, dtype=torch.bool), diagonal=1
        )
        pairs = similarities[upper]
        return F.relu(pairs - self.diversity_threshold).mean()


class SparseLesionGraph(nn.Module):
    """Equations (3)-(4), with a self-excluding neighborhood fixed per forward."""

    def __init__(
        self,
        token_dim: int = 256,
        neighbors: int = 16,
        layers: int = 2,
        spatial_temperature: float = 0.1,
        relation_weights: Sequence[float] = (1.0, 1.0, 1.0),
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if neighbors <= 0 or layers <= 0:
            raise ValueError("Graph neighbors and layers must be positive")
        if spatial_temperature <= 0:
            raise ValueError("spatial_temperature must be positive")
        if len(relation_weights) != 3:
            raise ValueError("relation_weights must contain spatial/visual/prototype values")
        self.token_dim = int(token_dim)
        self.neighbors = int(neighbors)
        self.layers = int(layers)
        self.spatial_temperature = float(spatial_temperature)
        self.relation_weights = tuple(float(value) for value in relation_weights)
        self.self_layers = nn.ModuleList(
            nn.Linear(self.token_dim, self.token_dim) for _ in range(self.layers)
        )
        self.message_layers = nn.ModuleList(
            nn.Linear(self.token_dim, self.token_dim) for _ in range(self.layers)
        )
        if activation.lower() != "gelu":
            raise ValueError("The formal graph activation is GELU")
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        xavier_init(self)

    @staticmethod
    def _cosine_matrix(values: Tensor) -> Tensor:
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = F.normalize(values.float(), dim=-1, eps=1e-8)
            return values @ values.transpose(0, 1)

    def relation_scores(
        self,
        hidden: Tensor,
        positions: Tensor,
        prototype_responses: Tensor | None,
    ) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            positions_fp32 = positions.float()
            squared_distance = torch.cdist(
                positions_fp32, positions_fp32, p=2
            ).square()
            spatial = torch.exp(-squared_distance / self.spatial_temperature)
            visual = self._cosine_matrix(hidden)
            scores = (
                self.relation_weights[0] * spatial
                + self.relation_weights[1] * visual
            )
            if prototype_responses is not None:
                semantic = self._cosine_matrix(prototype_responses)
                scores = scores + self.relation_weights[2] * semantic
            return scores

    def selected_relation_scores(
        self,
        hidden: Tensor,
        positions: Tensor,
        prototype_responses: Tensor | None,
        neighbors: Tensor,
    ) -> Tensor:
        """Return equation (3) values on the already fixed sparse edges.

        Top-K membership is discrete and is fixed from the initial tokens.
        Rebuilding a dense differentiable N-by-N matrix at every graph layer
        *and returning it* is therefore unnecessary.  A transient dense GEMM
        is nevertheless substantially more memory-efficient than materialising
        an [N,K,D] neighbor-feature tensor for the manuscript's bag sizes; the
        dense result is immediately gathered and is not retained in outputs.
        Thus this routine preserves exact gradients while persistent relation
        storage after neighborhood selection is N-by-K.
        """

        if neighbors.ndim != 2 or neighbors.shape[0] != hidden.shape[0]:
            raise ValueError("neighbors must have shape [N,K]")
        if neighbors.shape[1] == 0:
            return torch.empty(
                neighbors.shape,
                dtype=torch.float32,
                device=hidden.device,
            )
        dense = self.relation_scores(hidden, positions, prototype_responses)
        return dense.gather(1, neighbors)

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        prototype_responses: Tensor | None,
    ) -> tuple[Tensor, dict[str, Any]]:
        if tokens.ndim != 2 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"tokens must have shape [N,{self.token_dim}]")
        n_nodes = tokens.shape[0]
        if positions.shape != (n_nodes, 2):
            raise ValueError("positions must have shape [N,2]")
        if prototype_responses is not None and prototype_responses.shape != (n_nodes, 3):
            raise ValueError("prototype_responses must have shape [N,3]")

        # Neighbor selection itself is non-differentiable.  Detaching it avoids
        # retaining a dense N-by-N autograd graph; each selected edge is
        # recomputed differentiably below.
        with torch.no_grad():
            initial_scores = self.relation_scores(
                tokens.detach(),
                positions.detach(),
                None
                if prototype_responses is None
                else prototype_responses.detach(),
            )
        k = min(self.neighbors, max(0, n_nodes - 1))
        if k:
            selectable = initial_scores.masked_fill(
                torch.eye(n_nodes, dtype=torch.bool, device=tokens.device), -torch.inf
            )
            neighbors = torch.topk(selectable, k=k, dim=1, largest=True).indices
        else:
            neighbors = torch.empty((n_nodes, 0), dtype=torch.long, device=tokens.device)

        hidden = tokens
        edge_weights: list[Tensor] = []
        layer_relations: list[Tensor] = []
        for self_layer, message_layer in zip(self.self_layers, self.message_layers):
            selected_scores = self.selected_relation_scores(
                hidden, positions, prototype_responses, neighbors
            )
            # Layer relations now correspond one-to-one with ``neighbors`` and
            # have shape [N,K], rather than retaining unused dense matrices.
            layer_relations.append(selected_scores)
            if k:
                # Softmax itself stays in fp32, while its bounded [0,1]
                # result can safely follow the hidden dtype for message
                # aggregation without materialising [N,K,D] in fp32.
                alpha = torch.softmax(selected_scores, dim=1).to(
                    dtype=hidden.dtype
                )
                transformed = message_layer(hidden)
                aggregate = torch.sum(
                    alpha[..., None] * transformed[neighbors], dim=1
                )
            else:
                alpha = torch.empty(
                    (n_nodes, 0), dtype=torch.float32, device=hidden.device
                )
                aggregate = torch.zeros_like(hidden)
            edge_weights.append(alpha)
            hidden = self.dropout(
                self.activation(self_layer(hidden) + aggregate)
            ).to(dtype=tokens.dtype)

        return hidden, {
            "neighbors": neighbors,
            "initial_relations": initial_scores,
            "layer_relations": layer_relations,
            "edge_weights": edge_weights,
        }


class CrossViewConsistency(nn.Module):
    """Equation (5): non-rigid, directional row-softmax correspondence."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.bilinear = nn.Parameter(torch.empty(self.token_dim, self.token_dim))
        nn.init.xavier_uniform_(self.bilinear)

    def forward(
        self,
        first_tokens: Tensor,
        second_tokens: Tensor,
        first_probabilities: Tensor,
        second_probabilities: Tensor,
    ) -> dict[str, Tensor]:
        scale = math.sqrt(self.token_dim)
        # Bilinear logits can overflow fp16 before softmax.  Correspondence
        # and its L1 consistency loss are kept in fp32; gradients still flow
        # to both token streams and the bilinear parameter.
        with torch.autocast(device_type=first_tokens.device.type, enabled=False):
            first_fp32 = first_tokens.float()
            second_fp32 = second_tokens.float()
            bilinear_fp32 = self.bilinear.float()
            first_to_second = torch.softmax(
                first_fp32
                @ bilinear_fp32
                @ second_fp32.transpose(0, 1)
                / scale,
                dim=1,
            )
            second_to_first = torch.softmax(
                second_fp32
                @ bilinear_fp32
                @ first_fp32.transpose(0, 1)
                / scale,
                dim=1,
            )
            # This is exactly 1/2 * sum_c of the two length-normalised L1 terms.
            first_error = torch.abs(
                first_probabilities.float()
                - first_to_second @ second_probabilities.float()
            ).mean(dim=0).sum()
            second_error = torch.abs(
                second_probabilities.float()
                - second_to_first @ first_probabilities.float()
            ).mean(dim=0).sum()
        return {
            "first_to_second": first_to_second,
            "second_to_first": second_to_first,
            "loss": 0.5 * (first_error + second_error),
        }


class SparseTopKAggregator(nn.Module):
    """Per-class, per-view sparse aggregation from equation (6)."""

    def __init__(self, fraction: float = 0.05, minimum: int = 8, enabled: bool = True) -> None:
        super().__init__()
        if not 0 < fraction <= 1:
            raise ValueError("Top-K fraction must lie in (0,1]")
        if minimum <= 0:
            raise ValueError("Top-K minimum must be positive")
        self.fraction = float(fraction)
        self.minimum = int(minimum)
        self.enabled = bool(enabled)

    def retained_count(self, n_instances: int) -> int:
        if n_instances <= 0:
            raise ValueError("Cannot aggregate an empty view")
        if not self.enabled:
            return n_instances
        return min(n_instances, max(math.ceil(self.fraction * n_instances), self.minimum))

    def forward(self, tokens: Tensor, logits: Tensor) -> dict[str, Tensor | int]:
        if tokens.ndim != 2 or logits.ndim != 1 or tokens.shape[0] != logits.shape[0]:
            raise ValueError("tokens/logits must have shapes [N,D] and [N]")
        k = self.retained_count(tokens.shape[0])
        indices = torch.topk(logits, k=k, dim=0, largest=True).indices
        weights = torch.softmax(logits[indices], dim=0).to(dtype=logits.dtype)
        representation = torch.sum(
            weights.to(dtype=tokens.dtype)[:, None] * tokens[indices], dim=0
        )
        return {
            "representation": representation,
            "indices": indices,
            "weights": weights,
            "k": k,
        }


class DAGLMIL(nn.Module):
    """Dual-view Anatomy-guided Lesion Graph MIL.

    ``forward`` accepts either the canonical flat tensors or a list of nested
    breast dictionaries with already extracted features.  It returns all
    candidate-level intermediates and each of the six manuscript losses; no
    mutable ``self.patch_scores`` side channel is used.
    """

    def __init__(
        self,
        feature_dim: int = 352,
        token_dim: int = 256,
        view_embedding_dim: int = 8,
        prototypes_per_group: int = 8,
        graph_neighbors: int = 16,
        graph_layers: int = 2,
        graph_activation: str = "gelu",
        graph_dropout: float = 0.0,
        spatial_temperature: float = 0.1,
        relation_weights: Sequence[float] = (1.0, 1.0, 1.0),
        prototype_response_coefficient: float = 0.5,
        prototype_margin: float = 0.2,
        prototype_diversity_threshold: float = 0.5,
        topk_fraction: float = 0.05,
        topk_minimum: int = 8,
        view_attention_hidden_dim: int = 128,
        positive_weights: Sequence[float] = (6007 / 393, 6261 / 139),
        loss_weights: Mapping[str, float] | None = None,
        ablation: Mapping[str, Any] | DAGLAblation | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.prototype_margin = float(prototype_margin)
        self.ablation = (
            ablation if isinstance(ablation, DAGLAblation) else DAGLAblation.from_mapping(ablation)
        )
        self.backbone = backbone
        self.token_encoder = AnatomyAwareTokenEncoder(
            self.feature_dim,
            self.token_dim,
            view_embedding_dim,
            use_anatomy=self.ablation.anatomy,
        )
        self.prototype_memory = (
            LesionPrototypeMemory(
                self.token_dim,
                prototypes_per_group,
                prototype_diversity_threshold,
            )
            if self.ablation.prototypes
            else None
        )
        self.graph = (
            SparseLesionGraph(
                self.token_dim,
                graph_neighbors,
                graph_layers,
                spatial_temperature,
                relation_weights,
                graph_activation,
                graph_dropout,
            )
            if self.ablation.graph
            else None
        )
        self.candidate_heads = nn.ModuleList(
            nn.Linear(self.token_dim, 1) for _ in range(2)
        )
        self.register_buffer(
            "prototype_response_coefficient",
            torch.full((2,), float(prototype_response_coefficient)),
        )
        self.cross_view = (
            CrossViewConsistency(self.token_dim) if self.ablation.cross_view else None
        )
        self.topk = SparseTopKAggregator(
            topk_fraction, topk_minimum, enabled=self.ablation.topk
        )
        self.view_attention = nn.ModuleList(
            AttentionScorer(self.token_dim, view_attention_hidden_dim) for _ in range(2)
        )
        self.breast_heads = nn.ModuleList(
            nn.Linear(self.token_dim, 1) for _ in range(2)
        )
        if len(positive_weights) != 2:
            raise ValueError("positive_weights must contain Mass and Calcification")
        self.register_buffer("positive_weights", torch.as_tensor(positive_weights).float())
        defaults = {
            "classification": 1.0,
            "prototype": 0.1,
            "diversity": 0.01,
            "graph": 0.05,
            "cross_view": 0.1,
            "sparsity": 0.01,
        }
        if loss_weights is not None:
            unknown = set(loss_weights) - set(defaults)
            if unknown:
                raise ValueError(f"Unknown DAGL loss weights: {sorted(unknown)}")
            defaults.update({key: float(value) for key, value in loss_weights.items()})
        self.loss_weights = defaults
        # Prototype parameters were initialized separately as random unit
        # vectors; initialize all remaining linear layers after construction.
        for module in (
            self.candidate_heads,
            self.view_attention,
            self.breast_heads,
        ):
            xavier_init(module)

    def encode_patches(self, patches: Tensor) -> Tensor:
        if self.backbone is None:
            raise RuntimeError("No frozen backbone is attached to this DAGL model")
        features = self.backbone(patches)
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                f"Backbone returned {features.shape[-1]} features, expected {self.feature_dim}"
            )
        return features

    def _candidate_outputs(
        self, graph_tokens: Tensor, prototype_responses: Tensor
    ) -> tuple[Tensor, Tensor]:
        token_logits = torch.cat(
            [head(graph_tokens) for head in self.candidate_heads], dim=1
        )
        if self.prototype_memory is not None:
            lesion_responses = prototype_responses[:, 1:3]
            coefficient = self.prototype_response_coefficient.to(
                device=token_logits.device, dtype=token_logits.dtype
            )
            logits = token_logits + coefficient * lesion_responses.to(token_logits.dtype)
        else:
            logits = token_logits
        return logits, torch.sigmoid(logits)

    def _run_graph(
        self,
        batch: FlatFeatureBatch,
        tokens: Tensor,
        prototype_responses: Tensor,
    ) -> tuple[Tensor, list[dict[str, Any]]]:
        if self.graph is None:
            return tokens, []
        enhanced = tokens.clone()
        records: list[dict[str, Any]] = []
        semantic = prototype_responses if self.prototype_memory is not None else None
        for sample, view, bag, indices in iter_bags(batch):
            bag_tokens, detail = self.graph(
                tokens[indices], batch.anatomy[indices, :2], None if semantic is None else semantic[indices]
            )
            enhanced[indices] = bag_tokens
            local_neighbors = detail["neighbors"]
            detail.update(
                {
                    "sample_index": sample,
                    "view_index": view,
                    "bag_index": bag,
                    "node_indices": indices,
                    "global_neighbors": indices[local_neighbors]
                    if local_neighbors.numel()
                    else local_neighbors,
                }
            )
            records.append(detail)
        return enhanced, records

    def _aggregate(
        self,
        batch: FlatFeatureBatch,
        tokens: Tensor,
        candidate_logits: Tensor,
    ) -> dict[str, Any]:
        bsz, dim = batch.num_samples, tokens.shape[1]
        bag_representations = tokens.new_zeros((batch.num_bags, 2, dim))
        topk_mask = torch.zeros(
            (tokens.shape[0], 2), dtype=torch.bool, device=tokens.device
        )
        candidate_attention = torch.zeros_like(candidate_logits)
        selections: list[dict[str, Any]] = []
        for sample, view, bag, indices in iter_bags(batch):
            for lesion_class in range(2):
                result = self.topk(tokens[indices], candidate_logits[indices, lesion_class])
                local = result["indices"]
                global_indices = indices[local]
                bag_representations[bag, lesion_class] = result["representation"]
                topk_mask[global_indices, lesion_class] = True
                candidate_attention[global_indices, lesion_class] = result["weights"]
                selections.append(
                    {
                        "sample_index": sample,
                        "view_index": view,
                        "bag_index": bag,
                        "class_index": lesion_class,
                        "indices": global_indices,
                        "weights": result["weights"],
                        "k": result["k"],
                    }
                )

        breast_representations = tokens.new_zeros((bsz, 2, dim))
        bag_weights = tokens.new_zeros((batch.num_bags, 2))
        counts = torch.bincount(batch.bag_sample_index, minlength=bsz)
        max_bags = int(counts.max().item())
        view_representations = tokens.new_zeros((bsz, max_bags, 2, dim))
        view_mask = torch.zeros((bsz, max_bags), dtype=torch.bool, device=tokens.device)
        view_types = torch.full(
            (bsz, max_bags), -1, dtype=torch.long, device=tokens.device
        )
        view_weights = tokens.new_zeros((bsz, 2, max_bags))
        for sample in range(bsz):
            available = torch.nonzero(
                batch.bag_sample_index == sample, as_tuple=False
            ).flatten()
            view_representations[sample, : available.numel()] = bag_representations[
                available
            ]
            view_mask[sample, : available.numel()] = True
            view_types[sample, : available.numel()] = batch.bag_view_index[available]
            for lesion_class in range(2):
                representations = bag_representations[available, lesion_class]
                scores = self.view_attention[lesion_class](representations)
                beta = torch.softmax(scores, dim=0).to(dtype=representations.dtype)
                bag_weights[available, lesion_class] = beta
                view_weights[sample, lesion_class, : available.numel()] = beta
                breast_representations[sample, lesion_class] = torch.sum(
                    beta[:, None] * representations, dim=0
                )
        breast_logits = torch.cat(
            [
                self.breast_heads[lesion_class](
                    breast_representations[:, lesion_class]
                )
                for lesion_class in range(2)
            ],
            dim=1,
        )
        return {
            "view_representations": view_representations,
            "view_mask": view_mask,
            "view_types": view_types,
            "view_attention": view_weights,
            "bag_representations": bag_representations,
            "bag_attention": bag_weights,
            "breast_representations": breast_representations,
            "breast_logits": breast_logits,
            "topk_mask": topk_mask,
            "candidate_attention": candidate_attention,
            "topk_selections": selections,
        }

    def _cross_view_outputs(
        self,
        batch: FlatFeatureBatch,
        tokens: Tensor,
        candidate_probabilities: Tensor,
    ) -> tuple[list[dict[str, Any]], Tensor]:
        zero = tokens.sum() * 0.0
        if self.cross_view is None:
            return [], zero
        records: list[dict[str, Any]] = []
        losses: list[Tensor] = []
        for sample in range(batch.num_samples):
            sample_bags = torch.nonzero(
                batch.bag_sample_index == sample, as_tuple=False
            ).flatten()
            cc_bags = sample_bags[batch.bag_view_index[sample_bags] == 0]
            mlo_bags = sample_bags[batch.bag_view_index[sample_bags] == 1]
            # Abnormal duplicate-view records are still independently pooled
            # and fused, but never masquerade as a valid CC/MLO pair.
            if cc_bags.numel() != 1 or mlo_bags.numel() != 1:
                continue
            pair_bags = torch.cat((cc_bags, mlo_bags))
            if not bool(batch.bag_cross_view_valid[pair_bags].all()):
                continue
            cc = torch.nonzero(batch.bag_index == cc_bags[0], as_tuple=False).flatten()
            mlo = torch.nonzero(batch.bag_index == mlo_bags[0], as_tuple=False).flatten()
            record = self.cross_view(
                tokens[cc],
                tokens[mlo],
                candidate_probabilities[cc],
                candidate_probabilities[mlo],
            )
            record.update(
                {
                    "sample_index": sample,
                    "cc_indices": cc,
                    "mlo_indices": mlo,
                    "cc_to_mlo": record.pop("first_to_second"),
                    "mlo_to_cc": record.pop("second_to_first"),
                }
            )
            losses.append(record["loss"])
            records.append(record)
        return records, torch.stack(losses).mean() if losses else zero

    def _prototype_loss(
        self,
        batch: FlatFeatureBatch,
        prototype_responses: Tensor,
        topk_mask: Tensor,
    ) -> Tensor:
        zero = prototype_responses.sum() * 0.0
        if self.prototype_memory is None or batch.labels is None:
            return zero
        terms: list[Tensor] = []
        for sample in range(batch.num_samples):
            sample_mask = batch.sample_index == sample
            for lesion_class, prototype_class in enumerate(LESION_TO_PROTOTYPE):
                if batch.labels[sample, lesion_class] > 0.5:
                    selected = sample_mask & topk_mask[:, lesion_class]
                    if bool(selected.any()):
                        lesion = prototype_responses[selected, prototype_class]
                        normal = prototype_responses[selected, 0]
                        terms.append(F.relu(self.prototype_margin - (lesion - normal)))
            # Crucially, this is a whole-breast normal condition; a Mass-positive
            # breast is never pushed to Normal because Calcification is absent.
            if bool((batch.labels[sample] <= 0.5).all()):
                normal = prototype_responses[sample_mask, 0]
                strongest_lesion = prototype_responses[sample_mask, 1:3].max(dim=1).values
                terms.append(
                    F.relu(self.prototype_margin - (normal - strongest_lesion))
                )
        return torch.cat(terms).mean() if terms else zero

    @staticmethod
    def _graph_loss(
        candidate_probabilities: Tensor, graph_records: list[dict[str, Any]]
    ) -> Tensor:
        zero = candidate_probabilities.sum() * 0.0
        numerator = zero
        denominator = 0
        for record in graph_records:
            neighbors = record["global_neighbors"]
            if not neighbors.numel():
                continue
            nodes = record["node_indices"]
            alpha = record["edge_weights"][-1]
            difference = torch.abs(
                candidate_probabilities[nodes, None, :]
                - candidate_probabilities[neighbors]
            )
            numerator = numerator + torch.sum(alpha[..., None] * difference)
            denominator += int(nodes.numel()) * candidate_probabilities.shape[1]
        return numerator / denominator if denominator else zero

    def forward(
        self,
        data: Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        anatomy: Tensor | None = None,
        sample_index: Tensor | None = None,
        view_index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        batch = flatten_feature_batch(
            data, anatomy, sample_index, view_index, labels, feature_key="features"
        )
        if batch.features.shape[1] != self.feature_dim:
            raise ValueError(
                f"DAGL expects {self.feature_dim}-D Mammo-CLIP features, "
                f"got {batch.features.shape[1]}"
            )
        tokens = self.token_encoder(
            batch.features, batch.anatomy, batch.view_index
        )
        if self.prototype_memory is not None:
            prototype = self.prototype_memory(tokens)
            prototype_responses = prototype["responses"]
        else:
            prototype = {
                "projected": tokens.new_empty((tokens.shape[0], 0)),
                "similarities": tokens.new_empty((tokens.shape[0], 0, 0)),
                "responses": tokens.new_zeros((tokens.shape[0], 3)),
                "assignments": torch.full(
                    (tokens.shape[0], 3), -1, dtype=torch.long, device=tokens.device
                ),
            }
            prototype_responses = prototype["responses"]

        graph_tokens, graph_records = self._run_graph(
            batch, tokens, prototype_responses
        )
        candidate_logits, candidate_probabilities = self._candidate_outputs(
            graph_tokens, prototype_responses
        )
        aggregate = self._aggregate(batch, graph_tokens, candidate_logits)
        correspondence, cross_view_loss = self._cross_view_outputs(
            batch, graph_tokens, candidate_probabilities
        )

        zero = graph_tokens.sum() * 0.0
        classification_loss = (
            F.binary_cross_entropy_with_logits(
                aggregate["breast_logits"],
                batch.labels,
                pos_weight=self.positive_weights.to(
                    device=graph_tokens.device, dtype=graph_tokens.dtype
                ),
            )
            if batch.labels is not None
            else zero
        )
        prototype_loss = self._prototype_loss(
            batch, prototype_responses, aggregate["topk_mask"]
        )
        diversity_loss = (
            self.prototype_memory.diversity_loss()
            if self.prototype_memory is not None and self.ablation.diversity
            else zero
        )
        graph_loss = (
            self._graph_loss(candidate_probabilities, graph_records)
            if self.graph is not None
            else zero
        )
        if self.ablation.topk:
            outside = ~aggregate["topk_mask"]
            sparsity_loss = (
                candidate_probabilities[outside].mean() if bool(outside.any()) else zero
            )
        else:
            sparsity_loss = zero
        losses = {
            "classification": classification_loss,
            "prototype": prototype_loss,
            "diversity": diversity_loss,
            "graph": graph_loss,
            "cross_view": cross_view_loss,
            "sparsity": sparsity_loss,
        }
        total = sum(self.loss_weights[name] * value for name, value in losses.items())
        losses["total"] = total

        breast_logits = aggregate["breast_logits"]
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
            "logits": breast_logits,
            "probabilities": torch.sigmoid(breast_logits),
            "breast_logits": breast_logits,
            "breast_probabilities": torch.sigmoid(breast_logits),
            "candidate_logits": candidate_logits,
            "candidate_probabilities": candidate_probabilities,
            "candidate_scores": candidate_probabilities,
            "candidate_records": candidate_records,
            "features": batch.features,
            "tokens": tokens,
            "graph_tokens": graph_tokens,
            "prototype_projected": prototype["projected"],
            "prototype_similarities": prototype["similarities"],
            "prototype_responses": prototype_responses,
            "prototype_assignments": prototype["assignments"],
            "graph_records": graph_records,
            "correspondence": correspondence,
            "losses": losses,
            "loss": total,
            "sample_index": batch.sample_index,
            "view_index": batch.view_index,
            "bag_index": batch.bag_index,
            "bag_sample_index": batch.bag_sample_index,
            "bag_view_index": batch.bag_view_index,
            "bag_cross_view_valid": batch.bag_cross_view_valid,
        }
        result.update(aggregate)
        return result


# Friendly aliases used in older experiment drafts.
DAGL = DAGLMIL
DAGL_MIL = DAGLMIL
