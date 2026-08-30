import torch

from dagl_mil.reproducibility import (
    EpochSeededRandomSampler,
    environment_record,
    seed_everything,
    sha256_mapping,
)


def test_epoch_sampler_resume_order_depends_on_epoch_not_prior_iteration():
    uninterrupted = EpochSeededRandomSampler(range(17), seed=2026)
    orders = []
    for epoch in range(3):
        uninterrupted.set_epoch(epoch)
        orders.append(list(uninterrupted))

    resumed = EpochSeededRandomSampler(range(17), seed=2026)
    resumed.set_epoch(2)
    assert list(resumed) == orders[2]
    assert orders[0] != orders[1]
    assert all(tag == 2 for tag, _ in orders[2])


def test_mapping_hash_is_order_independent():
    assert sha256_mapping({"a": 1, "b": {"c": 2}}) == sha256_mapping(
        {"b": {"c": 2}, "a": 1}
    )


def test_environment_record_locks_preprocessing_package_versions():
    record = environment_record()
    packages = record["packages"]
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "pydicom",
        "PyYAML",
        "Pillow",
        "torchvision",
    ):
        assert isinstance(packages[name], str) and packages[name]
    for optional_decoder in ("pylibjpeg", "pylibjpeg-libjpeg", "python-gdcm"):
        assert optional_decoder in packages
    assert record["nvidia_smi"]["status"] in {
        "available",
        "unavailable",
        "invalid_output",
        "no_gpus",
    }


def test_deterministic_seed_is_fail_closed():
    seed_everything(2026, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
