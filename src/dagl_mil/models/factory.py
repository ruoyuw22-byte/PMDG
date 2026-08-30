from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch import nn

from .baselines import (
    AbMIL,
    CrossViewTransformerMIL,
    FPNSetTransformerMIL,
    PatchGCNMIL,
    SetTransformerMIL,
)
from .dagl import DAGLMIL
from .mammoclip import MammoCLIPB2Backbone


def _sections(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if "model" in config:
        model = config["model"]
        loss = config.get("loss", {})
    else:
        model = config
        loss = config.get("loss", {}) if isinstance(config.get("loss"), Mapping) else {}
    if not isinstance(model, Mapping):
        raise TypeError("config['model'] must be a mapping")
    if not isinstance(loss, Mapping):
        raise TypeError("config['loss'] must be a mapping")
    return model, loss


def build_backbone(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> MammoCLIPB2Backbone:
    """Build and strictly load the frozen formal Mammo-CLIP backbone."""

    model, _ = _sections(config)
    paths = config.get("paths", {}) if "model" in config else {}
    training = config.get("training", {}) if "model" in config else {}
    if checkpoint_path is None:
        checkpoint_path = paths.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("A Mammo-CLIP B2 checkpoint path is required")
    return MammoCLIPB2Backbone(
        checkpoint_path,
        micro_batch_size=int(training.get("patch_micro_batch_size", 16)),
        strict_input_size=True,
    )


def build_model(
    config: Mapping[str, Any],
    *,
    backbone: nn.Module | None = None,
) -> nn.Module:
    """Unified factory for DAGL-MIL, five baselines, and all ablations."""

    model, loss = _sections(config)
    name = str(model.get("name", "dagl_mil")).lower().replace("-", "_")
    common = {
        "feature_dim": int(model.get("feature_dim", 352)),
        "token_dim": int(model.get("token_dim", 256)),
        "view_attention_hidden_dim": int(model.get("view_attention_hidden_dim", 128)),
        "positive_weights": tuple(loss.get("positive_weights", (6007 / 393, 6261 / 139))),
    }
    if name in {"dagl", "dagl_mil"}:
        loss_weights = {
            key: float(loss[key])
            for key in (
                "classification",
                "prototype",
                "diversity",
                "graph",
                "cross_view",
                "sparsity",
            )
            if key in loss
        }
        return DAGLMIL(
            **common,
            view_embedding_dim=int(model.get("view_embedding_dim", 8)),
            prototypes_per_group=int(model.get("prototypes_per_group", 8)),
            graph_neighbors=int(model.get("graph_neighbors", 16)),
            graph_layers=int(model.get("graph_layers", 2)),
            graph_activation=str(model.get("graph_activation", "gelu")),
            graph_dropout=float(model.get("graph_dropout", 0.0)),
            spatial_temperature=float(model.get("spatial_temperature", 0.1)),
            relation_weights=tuple(model.get("relation_weights", (1.0, 1.0, 1.0))),
            prototype_response_coefficient=float(
                model.get("prototype_response_coefficient", 0.5)
            ),
            prototype_margin=float(model.get("prototype_margin", 0.2)),
            prototype_diversity_threshold=float(
                model.get("prototype_diversity_threshold", 0.5)
            ),
            topk_fraction=float(model.get("topk_fraction", 0.05)),
            topk_minimum=int(model.get("topk_minimum", 8)),
            loss_weights=loss_weights,
            ablation=model.get("ablation"),
            backbone=backbone,
        )
    if name in {"abmil", "ab_mil"}:
        return AbMIL(**common, attention_hidden_dim=128)
    if name in {"set_transformer", "settrans", "set_transformer_mil"}:
        return SetTransformerMIL(
            **common,
            attention_heads=int(model.get("attention_heads", 4)),
            inducing_points=int(model.get("inducing_points", 64)),
            isab_layers=int(model.get("isab_layers", 2)),
            pma_seeds=int(model.get("pma_seeds", 1)),
            transformer_dropout=float(model.get("transformer_dropout", 0.0)),
        )
    if name in {"fpn_set_transformer", "fpn_settrans"}:
        return FPNSetTransformerMIL(
            **common,
            attention_heads=int(model.get("attention_heads", 4)),
            inducing_points_by_scale=tuple(
                model.get("inducing_points_by_scale", (64, 32, 16))
            ),
            isab_layers=int(model.get("isab_layers", 2)),
            pma_seeds=int(model.get("pma_seeds", 1)),
            scale_attention_hidden_dim=int(
                model.get("scale_attention_hidden_dim", 128)
            ),
        )
    if name in {"patchgcn", "patchgcn_mil", "patch_gcn_mil"}:
        return PatchGCNMIL(
            **common,
            baseline_graph_neighbors=int(model.get("baseline_graph_neighbors", 16)),
            graph_layers=int(model.get("graph_layers", 2)),
            graph_dropout=float(model.get("graph_dropout", 0.1)),
        )
    if name in {
        "cross_view_transformer",
        "cross_view_transformer_mil",
        "crossview_transformer_mil",
    }:
        return CrossViewTransformerMIL(
            **common,
            cross_view_blocks=int(model.get("cross_view_blocks", 2)),
            attention_heads=int(model.get("attention_heads", 4)),
            feedforward_dim=int(model.get("feedforward_dim", 512)),
            transformer_dropout=float(model.get("transformer_dropout", 0.1)),
        )
    raise ValueError(f"Unknown MIL model: {model.get('name')!r}")


def build_model_with_backbone(config: Mapping[str, Any]) -> nn.Module:
    """Convenience constructor for the online formal DAGL route."""

    return build_model(config, backbone=build_backbone(config))
