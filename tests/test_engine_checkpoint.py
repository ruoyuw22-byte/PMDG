import random

import numpy as np
import pytest
import torch

from dagl_mil.engine.checkpoint import (
    CheckpointError,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)


def _provenance():
    return {
        "manifest_sha256": "manifest",
        "backbone_sha256": "backbone",
        "manuscript_sha256": {"paper": "p", "supplement": "s"},
    }


def test_rng_state_round_trip():
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_restore_rng_converts_cuda_states_to_cpu_bytes(monkeypatch):
    state = capture_rng_state()
    state["torch_cuda"] = [torch.tensor([1, 2, 3], dtype=torch.int64)]
    observed: list[torch.Tensor] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: observed.extend(values),
    )

    restore_rng_state(state)

    assert len(observed) == 1
    assert observed[0].device.type == "cpu"
    assert observed[0].dtype == torch.uint8
    assert observed[0].is_contiguous()


def test_checkpoint_strict_round_trip_and_provenance(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    path = tmp_path / "last.pt"
    original = {key: value.clone() for key, value in model.state_dict().items()}
    summary = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=2,
        global_step=7,
        best_metric=0.75,
        epochs_without_improvement=1,
        config={"name": "unit"},
        provenance=_provenance(),
        skipped_optimizer_steps=3,
    )
    assert len(summary["sha256"]) == 64
    assert summary["skipped_optimizer_steps"] == 3
    with torch.no_grad():
        model.weight.zero_()
    payload = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_provenance=_provenance(),
    )
    assert payload["epoch"] == 2
    assert payload["skipped_optimizer_steps"] == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    with pytest.raises(CheckpointError, match="provenance mismatch"):
        load_training_checkpoint(
            path,
            model=model,
            expected_provenance={"manifest_sha256": "wrong"},
        )
