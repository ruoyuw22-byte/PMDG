from __future__ import annotations

import pytest
import torch

from dagl_mil.models import (
    CrossViewConsistency,
    DAGLMIL,
    LesionPrototypeMemory,
    SparseLesionGraph,
    SparseTopKAggregator,
)


def _ragged_batch() -> dict[str, torch.Tensor]:
    # Two breasts: unequal CC/MLO bags plus a single-view breast.
    counts = (18, 3, 1)
    n = sum(counts)
    return {
        "features": torch.randn(n, 352, requires_grad=True),
        "anatomy": torch.rand(n, 5),
        "sample_index": torch.tensor([0] * 21 + [1]),
        "view_index": torch.tensor([0] * 18 + [1] * 3 + [0]),
        "labels": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }


def test_dagl_shapes_fixed_graph_and_gradients() -> None:
    batch = _ragged_batch()
    model = DAGLMIL()
    output = model(batch)
    assert output["logits"].shape == (2, 2)
    assert output["candidate_logits"].shape == (22, 2)
    assert output["prototype_similarities"].shape == (22, 3, 8)
    assert output["prototype_responses"].shape == (22, 3)
    assert output["graph_tokens"].shape == (22, 256)
    assert output["view_representations"].shape == (2, 2, 2, 256)
    assert set(output["losses"]) == {
        "classification",
        "prototype",
        "diversity",
        "graph",
        "cross_view",
        "sparsity",
        "total",
    }
    assert torch.isfinite(output["loss"])

    cc_graph = next(record for record in output["graph_records"] if record["node_indices"].numel() == 18)
    assert cc_graph["neighbors"].shape == (18, 16)
    local_nodes = torch.arange(18)[:, None]
    assert not bool((cc_graph["neighbors"] == local_nodes).any())
    assert len(cc_graph["edge_weights"]) == 2
    assert not cc_graph["initial_relations"].requires_grad
    for relations, weights in zip(
        cc_graph["layer_relations"], cc_graph["edge_weights"]
    ):
        assert relations.shape == (18, 16)
        assert weights.shape == (18, 16)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(18))
    # Neighborhood indices are one fixed tensor reused by both layers.
    assert "neighbors" in cc_graph and len(cc_graph["layer_relations"]) == 2

    assert len(output["correspondence"]) == 1
    correspondence = output["correspondence"][0]
    assert correspondence["cc_to_mlo"].shape == (18, 3)
    assert correspondence["mlo_to_cc"].shape == (3, 18)
    torch.testing.assert_close(
        correspondence["cc_to_mlo"].sum(dim=1), torch.ones(18)
    )
    torch.testing.assert_close(
        correspondence["mlo_to_cc"].sum(dim=1), torch.ones(3)
    )

    output["loss"].backward()
    assert batch["features"].grad is not None
    assert torch.isfinite(batch["features"].grad).all()
    assert model.prototype_memory.prototypes.grad is not None


@pytest.mark.parametrize(
    ("n_instances", "expected"),
    [(1, 1), (7, 7), (8, 8), (159, 8), (160, 8), (161, 9), (200, 10)],
)
def test_sparse_topk_exact_ceiling_rule(n_instances: int, expected: int) -> None:
    module = SparseTopKAggregator(fraction=0.05, minimum=8)
    assert module.retained_count(n_instances) == expected
    result = module(torch.randn(n_instances, 256), torch.randn(n_instances))
    assert result["indices"].numel() == expected
    torch.testing.assert_close(result["weights"].sum(), torch.tensor(1.0))


def test_prototype_response_is_groupwise_max_and_diversity_is_intra_class() -> None:
    memory = LesionPrototypeMemory(token_dim=4, prototypes_per_group=2, diversity_threshold=0.5)
    with torch.no_grad():
        memory.projection.weight.copy_(torch.eye(4))
        memory.projection.bias.zero_()
        memory.prototypes.copy_(
            torch.tensor(
                [
                    [[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
                    [[0, 1.0, 0, 0], [0, -1.0, 0, 0]],
                    [[0, 0, 1.0, 0], [0, 0, -1.0, 0]],
                ]
            )
        )
    output = memory(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(
        output["responses"], output["similarities"].max(dim=-1).values
    )
    # Only the identical pair in the Normal group violates delta=0.5:
    # mean over 3 groups * one pair = (1 - .5) / 3.
    torch.testing.assert_close(memory.diversity_loss(), torch.tensor(1 / 6))


def test_cross_view_formula_supports_unequal_bags() -> None:
    module = CrossViewConsistency(token_dim=8)
    output = module(
        torch.randn(2, 8),
        torch.randn(5, 8),
        torch.rand(2, 2),
        torch.rand(5, 2),
    )
    assert output["first_to_second"].shape == (2, 5)
    assert output["second_to_first"].shape == (5, 2)
    torch.testing.assert_close(output["first_to_second"].sum(1), torch.ones(2))
    torch.testing.assert_close(output["second_to_first"].sum(1), torch.ones(5))
    assert output["loss"].ndim == 0


def test_custom_similarity_paths_are_stable_for_fp16_edge_values() -> None:
    memory = LesionPrototypeMemory(token_dim=8, prototypes_per_group=2).half()
    with torch.no_grad():
        memory.projection.weight.zero_()
        memory.projection.bias.zero_()
    prototype = memory(torch.zeros(3, 8, dtype=torch.float16))
    assert prototype["similarities"].dtype == torch.float32
    assert torch.isfinite(prototype["similarities"]).all()
    torch.testing.assert_close(
        prototype["responses"], torch.zeros(3, 3), atol=0, rtol=0
    )

    consistency = CrossViewConsistency(token_dim=8).half()
    first = torch.full((2, 8), 1e4, dtype=torch.float16, requires_grad=True)
    second = torch.full((3, 8), -1e4, dtype=torch.float16, requires_grad=True)
    output = consistency(
        first,
        second,
        torch.rand(2, 2, dtype=torch.float16),
        torch.rand(3, 2, dtype=torch.float16),
    )
    assert output["first_to_second"].dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in output.values())
    output["loss"].backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert consistency.bilinear.grad is not None


def test_sparse_edge_scores_equal_the_dense_formula_on_fixed_neighbors() -> None:
    graph = SparseLesionGraph(token_dim=8, neighbors=3)
    hidden = torch.randn(7, 8)
    positions = torch.rand(7, 2)
    prototype_responses = torch.randn(7, 3)
    dense = graph.relation_scores(hidden, positions, prototype_responses)
    selectable = dense.masked_fill(torch.eye(7, dtype=torch.bool), -torch.inf)
    neighbors = torch.topk(selectable, k=3, dim=1).indices
    sparse = graph.selected_relation_scores(
        hidden, positions, prototype_responses, neighbors
    )
    torch.testing.assert_close(sparse, dense.gather(1, neighbors))


@pytest.mark.parametrize(
    ("switches", "zero_losses", "absent"),
    [
        ({"anatomy": False}, (), "view_embedding"),
        ({"prototypes": False}, ("prototype", "diversity"), "prototype_memory"),
        ({"graph": False}, ("graph",), "graph"),
        ({"cross_view": False}, ("cross_view",), "cross_view"),
        ({"topk": False}, ("sparsity",), "topk_all"),
        ({"diversity": False}, ("diversity",), "diversity_only"),
        (
            {"prototypes": False, "graph": False},
            ("prototype", "diversity", "graph"),
            "both",
        ),
    ],
)
def test_ablation_switches_remove_only_named_mechanism(
    switches: dict[str, bool], zero_losses: tuple[str, ...], absent: str
) -> None:
    batch = _ragged_batch()
    model = DAGLMIL(ablation=switches)
    output = model(batch)
    for name in zero_losses:
        torch.testing.assert_close(output["losses"][name], torch.tensor(0.0))
    if absent == "view_embedding":
        assert model.token_encoder.view_embedding is None
        assert model.graph is not None and model.prototype_memory is not None
    elif absent == "prototype_memory":
        assert model.prototype_memory is None and model.graph is not None
        assert bool((output["prototype_assignments"] == -1).all())
    elif absent == "graph":
        assert model.graph is None and output["graph_records"] == []
        assert model.prototype_memory is not None
    elif absent == "cross_view":
        assert model.cross_view is None and output["correspondence"] == []
    elif absent == "topk_all":
        assert bool(output["topk_mask"].all())
        assert model.prototype_memory is not None and model.graph is not None
    elif absent == "diversity_only":
        assert model.prototype_memory is not None and model.graph is not None
    elif absent == "both":
        assert model.prototype_memory is None and model.graph is None


def test_duplicate_cc_views_are_fused_but_not_cross_matched() -> None:
    samples = [
        {
            "labels": torch.tensor([0.0, 1.0]),
            "views": {
                "CC": {
                    "view_name": "CC",
                    "cross_view_valid": False,
                    "features": torch.randn(2, 352),
                    "anatomy_base": torch.rand(2, 5),
                },
                "CC_2": {
                    "view_name": "CC",
                    "cross_view_valid": False,
                    "features": torch.randn(3, 352),
                    "anatomy_base": torch.rand(3, 5),
                },
            },
        }
    ]
    output = DAGLMIL()(samples)
    assert output["bag_representations"].shape == (2, 2, 256)
    assert output["view_types"].tolist() == [[0, 0]]
    assert output["correspondence"] == []
    torch.testing.assert_close(output["bag_attention"].sum(dim=0), torch.ones(2))


def test_top_level_collated_feature_schema_is_supported() -> None:
    collated = {
        "labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "cross_view_valid": torch.tensor([True, False]),
        "views": [
            {
                "CC": {
                    "view_name": "CC",
                    "features": torch.randn(2, 352),
                    "anatomy_base": torch.rand(2, 5),
                },
                "MLO": {
                    "view_name": "MLO",
                    "features": torch.randn(3, 352),
                    "anatomy_base": torch.rand(3, 5),
                },
            },
            {
                "CC": {
                    "view_name": "CC",
                    "features": torch.randn(1, 352),
                    "anatomy_base": torch.rand(1, 5),
                }
            },
        ],
    }
    output = DAGLMIL()(collated)
    assert output["logits"].shape == (2, 2)
    assert output["bag_sample_index"].tolist() == [0, 0, 1]
    assert output["bag_cross_view_valid"].tolist() == [True, True, False]
    assert len(output["correspondence"]) == 1


def test_flat_duplicate_views_use_collate_bag_slices() -> None:
    output = DAGLMIL()(
        {
            "features": torch.randn(5, 352),
            "anatomy_base": torch.rand(5, 5),
            "sample_index": torch.zeros(5, dtype=torch.long),
            "view_index": torch.zeros(5, dtype=torch.long),
            "labels": torch.tensor([[0.0, 0.0]]),
            "cross_view_valid": torch.tensor([False]),
            "bag_slices": [{"CC": (0, 2), "CC_2": (2, 5)}],
        }
    )
    assert output["bag_representations"].shape[0] == 2
    assert output["bag_view_index"].tolist() == [0, 0]
    assert output["correspondence"] == []
