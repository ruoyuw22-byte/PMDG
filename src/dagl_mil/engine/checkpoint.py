from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dagl_mil.evaluation.artifacts import sha256_file
from dagl_mil.reproducibility import sha256_mapping


CHECKPOINT_FORMAT = "dagl-mil-training-v1"
REQUIRED_PROVENANCE = {
    "manifest_sha256",
    "backbone_sha256",
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or belongs to another protocol."""


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    keys = (
        torch.as_tensor(numpy_state["keys"], device="cpu")
        .numpy()
        .astype(np.uint32, copy=False)
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cpu_state = torch.as_tensor(
        state["torch_cpu"], device="cpu", dtype=torch.uint8
    ).contiguous()
    torch.set_rng_state(cpu_state)
    cuda_states = [
        torch.as_tensor(value, device="cpu", dtype=torch.uint8).contiguous()
        for value in state.get("torch_cuda", [])
    ]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointError(
                "CUDA device count differs from the saved checkpoint: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_PROVENANCE - set(provenance))
    if missing:
        raise CheckpointError(f"Checkpoint provenance is missing: {missing}")
def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    epochs_without_improvement: int,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    history: list[Mapping[str, Any]] | None = None,
    skipped_optimizer_steps: int = 0,
) -> dict[str, Any]:
    _validate_provenance(provenance)
    destination = Path(path)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "skipped_optimizer_steps": int(skipped_optimizer_steps),
        "best_metric": float(best_metric),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": dict(config),
        "provenance": dict(provenance),
        "rng_state": capture_rng_state(),
        "history": [dict(row) for row in (history or [])],
    }
    _atomic_torch_save(payload, destination)
    digest = sha256_file(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256.json")
    sidecar_payload = {
        "path": destination.name,
        "sha256": digest,
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "skipped_optimizer_steps": int(skipped_optimizer_steps),
        "provenance_sha256": sha256_mapping(provenance),
    }
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_sidecar, sidecar)
    return {**sidecar_payload, "path": str(destination), "sha256": digest}


def _load_and_verify(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
    if not sidecar_path.is_file():
        raise CheckpointError(f"Checkpoint SHA256 sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if actual != sidecar.get("sha256"):
        raise CheckpointError(f"Checkpoint hash mismatch: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"Unsupported checkpoint format: {path}")
    _validate_provenance(payload.get("provenance", {}))
    return payload


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    expected_provenance: Mapping[str, Any] | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = _load_and_verify(Path(path), map_location)
    if expected_provenance is not None:
        for key, expected in expected_provenance.items():
            actual = payload["provenance"].get(key)
            if actual != expected:
                raise CheckpointError(
                    f"Checkpoint provenance mismatch for {key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise CheckpointError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise CheckpointError("Checkpoint has no AMP scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
