import torch
from torch.nn import functional as F

from dagl_mil.engine.feature_cache import FrozenGAPFeatureCache
from dagl_mil.engine.trainer import Trainer, _restored_skipped_optimizer_steps


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def encode_patches(self, patches, micro_batch_size=None, return_pyramid=False):
        del micro_batch_size
        assert not return_pyramid
        self.calls += 1
        means = patches.mean(dim=(1, 2, 3), keepdim=False)
        return torch.stack((means, means.square(), means * 0 + 1, means * 0), dim=1)


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, samples):
        representations = []
        labels = []
        for sample in samples:
            features = torch.cat(
                [view["features"] for view in sample["views"].values()], dim=0
            )
            representations.append(features.mean(dim=0))
            labels.append(sample["labels"])
        logits = self.classifier(torch.stack(representations))
        targets = torch.stack(labels).to(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return {
            "breast_probabilities": torch.sigmoid(logits),
            "losses": {"classification": loss, "total": loss},
        }


def _config():
    return {
        "runtime": {"require_cuda": False},
        "data": {"mean": 0.0, "std": 1.0},
        "model": {"name": "dagl_mil"},
        "training": {
            "mixed_precision": "none",
            "learning_rate": 1e-3,
            "minimum_learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-8,
            "epochs": 2,
            "gradient_accumulation": 2,
            "patch_micro_batch_size": 2,
            "gradient_clip_norm": 5.0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
        },
    }


def _batch(label):
    value = 0.2 + 0.6 * label
    return {
        "exam_id": [f"e{label}"],
        "breast_id": [f"e{label}_L"],
        "laterality": ["L"],
        "labels": torch.tensor([[float(label), float(label)]]),
        "cross_view_valid": torch.tensor([False]),
        "views": [
            {
                "CC": {
                    "view_name": "CC",
                    "image_id": f"image-{label}",
                    "source_path": f"synthetic-{label}.dcm",
                    "patches": torch.full((3, 3, 4, 4), value),
                    "anatomy_base": torch.zeros(3, 5),
                    "coords": torch.tensor(
                        [[0, 0, 4, 4], [1, 0, 5, 4], [2, 0, 6, 4]]
                    ),
                    "coverage": torch.ones(3),
                    "image_shape": (4, 6),
                }
            }
        ],
    }


def test_cpu_unit_trainer_accumulates_and_validates():
    trainer = Trainer(
        model=_Head(), backbone=_Backbone(), config=_config(), device="cpu"
    )
    metrics = trainer.run_epoch([_batch(0), _batch(1)], training=True)
    assert metrics["breasts"] == 2
    assert trainer.global_step == 1
    assert 0.0 <= metrics["mean_roc_auc"] <= 1.0


def test_resume_restores_or_derives_cumulative_skipped_optimizer_steps():
    loader = [_batch(0)] * 1600
    current = {
        "global_step": 2398,
        "skipped_optimizer_steps": 2,
        "history": [],
    }
    assert (
        _restored_skipped_optimizer_steps(
            current, loader, start_epoch=3, accumulation=2
        )
        == 2
    )
    legacy = {"global_step": 2397, "history": []}
    assert (
        _restored_skipped_optimizer_steps(
            legacy, loader, start_epoch=3, accumulation=2
        )
        == 3
    )


def test_trainer_reuses_eval_cache_but_never_training_features(tmp_path):
    backbone = _Backbone()
    trainer = Trainer(
        model=_Head(), backbone=backbone, config=_config(), device="cpu"
    )
    trainer.feature_cache = FrozenGAPFeatureCache(
        tmp_path,
        provenance={
            "backbone_sha256": "b" * 64,
            "preprocessing_protocol": "unit-v1",
            "mean": 0.0,
            "std": 1.0,
            "patch_size": 4,
            "eval_stride": 1,
            "minimum_breast_coverage": 0.5,
            "amp_dtype": "float32",
        },
        storage_dtype=torch.float32,
        feature_dim=4,
    )
    loader = [_batch(0), _batch(1)]
    trainer.run_epoch(loader, training=False)
    assert backbone.calls == 4
    trainer.run_epoch(loader, training=False)
    assert backbone.calls == 4
    assert trainer.feature_cache.stats.hits == 2

    writes = trainer.feature_cache.stats.writes
    trainer.run_epoch(loader, training=True)
    assert backbone.calls == 8
    assert trainer.feature_cache.stats.writes == writes
