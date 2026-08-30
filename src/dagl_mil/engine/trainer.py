from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from dagl_mil.reproducibility import sha256_mapping

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .feature_cache import FrozenGAPFeatureCache
from .features import build_evaluation_feature_cache, encode_collated_breasts


class TrainingError(RuntimeError):
    """Raised when a formal run cannot produce a valid optimization step."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _amp_context(device: torch.device, mixed_precision: str):
    enabled = device.type == "cuda" and mixed_precision.lower() != "none"
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if mixed_precision.lower() == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _classification_summary(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if targets.ndim != 2 or targets.shape[1] != 2 or probabilities.shape != targets.shape:
        raise TrainingError("Classification arrays must have shape [B,2]")
    result: dict[str, float] = {}
    aucs: list[float] = []
    for index, name in enumerate(("mass", "calcification")):
        if np.unique(targets[:, index]).size < 2:
            raise TrainingError(f"Validation split has only one {name} target class")
        auc = float(roc_auc_score(targets[:, index], probabilities[:, index]))
        result[f"{name}_roc_auc"] = auc
        result[f"{name}_pr_auc"] = float(
            average_precision_score(targets[:, index], probabilities[:, index])
        )
        aucs.append(auc)
    result["mean_roc_auc"] = float(np.mean(aucs))
    return result


def _restored_skipped_optimizer_steps(
    payload: Mapping[str, Any],
    train_loader: Iterable[Mapping[str, Any]],
    *,
    start_epoch: int,
    accumulation: int,
) -> int:
    saved = payload.get("skipped_optimizer_steps")
    if saved is not None:
        restored = int(saved)
    else:
        history = payload.get("history", [])
        last_history = history[-1] if history else {}
        if isinstance(last_history, Mapping) and last_history.get(
            "skipped_optimizer_steps"
        ) is not None:
            restored = int(last_history["skipped_optimizer_steps"])
        else:
            try:
                batches_per_epoch = len(train_loader)  # type: ignore[arg-type]
            except (TypeError, AttributeError) as error:
                raise TrainingError(
                    "Legacy checkpoint lacks skipped-step state and the training "
                    "loader has no deterministic length"
                ) from error
            if batches_per_epoch <= 0 or accumulation <= 0:
                raise TrainingError("Cannot derive skipped steps from an empty loader")
            attempts_per_epoch = math.ceil(batches_per_epoch / accumulation)
            restored = (
                start_epoch * attempts_per_epoch - int(payload["global_step"])
            )
    if restored < 0:
        raise TrainingError("Checkpoint skipped optimizer-step count is invalid")
    return restored


class Trainer:
    """Formal batch-4/accumulation-2 trainer with an always-frozen online backbone."""

    def __init__(
        self,
        *,
        model: nn.Module,
        backbone: nn.Module,
        config: Mapping[str, Any],
        device: str | torch.device = "cuda",
    ) -> None:
        self.config = dict(config)
        self.device = torch.device(device)
        runtime = config.get("runtime", {})
        if bool(runtime.get("require_cuda", True)) and self.device.type != "cuda":
            raise TrainingError("The manuscript run is restricted to the local CUDA GPU")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA is required but torch.cuda.is_available() is false")
        self.model = model.to(self.device)
        self.backbone = backbone.to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        training = config["training"]
        self.mixed_precision = str(training.get("mixed_precision", "fp16"))
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise TrainingError("The MIL model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=tuple(float(value) for value in training["adam_betas"]),
            eps=float(training["adam_epsilon"]),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(training["epochs"]),
            eta_min=float(training["minimum_learning_rate"]),
        )
        amp_enabled = self.device.type == "cuda" and self.mixed_precision == "fp16"
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
            init_scale=float(training.get("amp_initial_scale", 4096.0)),
        )
        self.accumulation = int(training["gradient_accumulation"])
        self.micro_batch_size = int(training["patch_micro_batch_size"])
        self.gradient_clip = float(training["gradient_clip_norm"])
        self.data_mean = float(config["data"]["mean"])
        self.data_std = float(config["data"]["std"])
        self.return_pyramid = str(config["model"]["name"]) == "fpn_set_transformer"
        self.feature_cache: FrozenGAPFeatureCache | None = None
        self.global_step = 0
        self.skipped_optimizer_steps = 0

    def _optimizer_step(self, *, context: str) -> None:
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip
        )
        finite = bool(torch.isfinite(norm))
        if not finite and not self.scaler.is_enabled():
            raise TrainingError(f"Non-finite gradient norm {context}")
        previous_scale = float(self.scaler.get_scale())
        # With AMP, unscale_ has already recorded non-finite gradients.  The
        # scaler must be allowed to skip this step and lower its scale; treating
        # a recoverable first-iteration overflow as fatal defeats dynamic AMP.
        self.scaler.step(self.optimizer)
        self.scaler.update()
        current_scale = float(self.scaler.get_scale())
        stepped = not self.scaler.is_enabled() or current_scale >= previous_scale
        if stepped:
            self.global_step += 1
        else:
            self.skipped_optimizer_steps += 1
        self.optimizer.zero_grad(set_to_none=True)

    def _encode(
        self,
        batch: Mapping[str, Any],
        *,
        deterministic_eval: bool,
    ) -> list[dict[str, Any]]:
        return encode_collated_breasts(
            batch,
            self.backbone,
            device=self.device,
            micro_batch_size=self.micro_batch_size,
            mean=self.data_mean,
            std=self.data_std,
            return_pyramid=self.return_pyramid,
            feature_cache=self.feature_cache,
            deterministic_eval=deterministic_eval,
        )

    def run_epoch(
        self,
        loader: Iterable[Mapping[str, Any]],
        *,
        training: bool,
    ) -> dict[str, float]:
        self.model.train(training)
        self.backbone.eval()
        if self.backbone.training or any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        ):
            raise TrainingError("Frozen Mammo-CLIP invariant was violated")
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        loss_sums: defaultdict[str, float] = defaultdict(float)
        targets: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        breasts_seen = 0
        batch_count = 0
        pending_accumulation = 0
        run_started = time.perf_counter()
        for batch_count, batch in enumerate(loader, start=1):
            labels = torch.as_tensor(batch["labels"])
            batch_size = int(labels.shape[0])
            with torch.set_grad_enabled(training):
                with _amp_context(self.device, self.mixed_precision):
                    encoded = self._encode(
                        batch, deterministic_eval=not training
                    )
                    output = self.model(encoded)
                    loss = output["losses"]["total"]
                    if not bool(torch.isfinite(loss)):
                        raise TrainingError(
                            f"Non-finite total loss at loader batch {batch_count}"
                        )
                    scaled_loss = loss / self.accumulation
                if training:
                    self.scaler.scale(scaled_loss).backward()
                    pending_accumulation += 1
                    if pending_accumulation == self.accumulation:
                        self._optimizer_step(
                            context=f"at loader batch {batch_count}"
                        )
                        pending_accumulation = 0
            for name, value in output["losses"].items():
                loss_sums[name] += float(value.detach().float().cpu()) * batch_size
            targets.append(labels.detach().cpu().numpy())
            probabilities.append(
                output["breast_probabilities"].detach().float().cpu().numpy()
            )
            breasts_seen += batch_size
            if batch_count % 100 == 0:
                elapsed = time.perf_counter() - run_started
                print(
                    json.dumps(
                        {
                            "event": "batch_progress",
                            "phase": "train" if training else "validation",
                            "batches": batch_count,
                            "breasts": breasts_seen,
                            "elapsed_seconds": elapsed,
                            "breasts_per_second": breasts_seen / max(elapsed, 1e-9),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if not batch_count or not breasts_seen:
            raise TrainingError("DataLoader yielded no breast samples")
        if training and pending_accumulation:
            # Formal 6400/4/2 is divisible, but a smoke subset need not be.
            correction = self.accumulation / pending_accumulation
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            self._optimizer_step(context="in the final partial accumulation")

        result = {
            f"loss_{name}": total / breasts_seen for name, total in loss_sums.items()
        }
        result.update(
            _classification_summary(
                np.concatenate(targets, axis=0),
                np.concatenate(probabilities, axis=0),
            )
        )
        result["breasts"] = float(breasts_seen)
        return result

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        validation_loader: Iterable[Mapping[str, Any]],
        *,
        output_dir: str | Path,
        provenance: Mapping[str, Any],
        resume: str | Path | None = None,
    ) -> dict[str, Any]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_cache = build_evaluation_feature_cache(
            self.config,
            backbone_sha256=str(provenance["backbone_sha256"]),
            manifest_sha256=str(provenance["manifest_sha256"]),
            device=self.device,
        )
        training = self.config["training"]
        maximum_epochs = int(training["epochs"])
        patience = int(training["early_stopping_patience"])
        minimum_delta = float(training.get("early_stopping_min_delta", 0.0))
        start_epoch = 0
        best_metric = -math.inf
        without_improvement = 0
        history: list[dict[str, Any]] = []
        if resume is not None:
            payload = load_training_checkpoint(
                resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=self.device,
                expected_provenance=provenance,
                restore_rng=True,
            )
            start_epoch = int(payload["epoch"]) + 1
            self.global_step = int(payload["global_step"])
            best_metric = float(payload["best_metric"])
            without_improvement = int(payload["epochs_without_improvement"])
            history = [dict(row) for row in payload.get("history", [])]
            self.skipped_optimizer_steps = _restored_skipped_optimizer_steps(
                payload,
                train_loader,
                start_epoch=start_epoch,
                accumulation=self.accumulation,
            )

        stopped_early = False
        for epoch in range(start_epoch, maximum_epochs):
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            epoch_started = time.perf_counter()
            train_metrics = self.run_epoch(train_loader, training=True)
            validation_metrics = self.run_epoch(validation_loader, training=False)
            epoch_seconds = time.perf_counter() - epoch_started
            current = float(validation_metrics["mean_roc_auc"])
            improved = current > best_metric + minimum_delta
            if improved:
                best_metric = current
                without_improvement = 0
            else:
                without_improvement += 1
            record = {
                "epoch": epoch,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "global_step": self.global_step,
                "skipped_optimizer_steps": self.skipped_optimizer_steps,
                "train": train_metrics,
                "validation": validation_metrics,
                "is_best": improved,
                "epoch_seconds": epoch_seconds,
            }
            history.append(record)
            self.scheduler.step()
            common = dict(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metric=best_metric,
                epochs_without_improvement=without_improvement,
                config=self.config,
                provenance=provenance,
                history=history,
                skipped_optimizer_steps=self.skipped_optimizer_steps,
            )
            save_training_checkpoint(directory / "last.pt", **common)
            if improved:
                save_training_checkpoint(directory / "best.pt", **common)
            _atomic_json(directory / "history.json", history)
            print(
                json.dumps(
                    {
                        "event": "epoch_complete",
                        "epoch": epoch + 1,
                        "maximum_epochs": maximum_epochs,
                        "epoch_seconds": epoch_seconds,
                        "train_mean_roc_auc": train_metrics["mean_roc_auc"],
                        "validation_mean_roc_auc": current,
                        "best_mean_roc_auc": best_metric,
                        "is_best": improved,
                        "global_step": self.global_step,
                        "skipped_optimizer_steps": self.skipped_optimizer_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if without_improvement >= patience:
                stopped_early = True
                break
        summary = {
            "completion_status": "completed",
            "provenance_sha256": sha256_mapping(provenance),
            "best_mean_validation_auc": best_metric,
            "epochs_completed": len(history),
            "global_steps": self.global_step,
            "skipped_optimizer_steps": self.skipped_optimizer_steps,
            "stopped_early": stopped_early,
            "best_checkpoint": str(directory / "best.pt"),
            "last_checkpoint": str(directory / "last.pt"),
        }
        if self.feature_cache is not None:
            summary["evaluation_feature_cache"] = self.feature_cache.report()
        _atomic_json(directory / "training_summary.json", summary)
        return summary
