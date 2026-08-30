import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sized
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

TRAINING_ORDER_PROTOCOL = "epoch-seeded-order-stateless-augmentation-v1"


class EpochSeededRandomSampler(Sampler[tuple[int, int]]):
    """Yield ``(epoch, index)`` pairs from an epoch-local permutation.

    The epoch tag reaches persistent Windows workers as part of each index, so
    augmentation need not depend on uncheckpointed worker RNG state.
    """

    def __init__(self, data_source: Sized, *, seed: int) -> None:
        if int(seed) < 0:
            raise ValueError("sampler seed must be non-negative")
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        for index in torch.randperm(
            len(self.data_source), generator=generator
        ).tolist():
            yield self.epoch, int(index)

    def __len__(self) -> int:
        return len(self.data_source)


def seed_everything(seed: int = 2026, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without enabling conflicting cuDNN modes."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # This must be set before the first cuBLAS workspace is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Flash/memory-efficient SDPA backward is non-deterministic on the
        # current Windows CUDA build.  The math kernel is slower but honors
        # the locked seed rather than merely emitting a warning and proceeding.
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 2026) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_provenance(
    *,
    manifest_path: str | Path,
    backbone_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from .data.preprocessing import _CACHE_PROTOCOL_VERSION

    return {
        "manifest_sha256": sha256_file(manifest_path),
        "backbone_sha256": sha256_file(backbone_path),
        "preprocessing_protocol": _CACHE_PROTOCOL_VERSION,
        "training_order_protocol": TRAINING_ORDER_PROTOCOL,
        "config_sha256": sha256_mapping(config),
    }


def environment_record() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    package_names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "pydicom",
        "PyYAML",
        "Pillow",
        "torchvision",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
        "python-gdcm",
    )
    package_versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "packages": package_versions,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": _nvidia_smi_record(),
    }
    if cuda:
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def _nvidia_smi_record() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version,name,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
        }
    gpus = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",", maxsplit=2)]
        if len(values) != 3:
            return {"status": "invalid_output"}
        driver_version, name, uuid = values
        gpus.append(
            {
                "driver_version": driver_version,
                "name": name,
                "uuid": uuid,
            }
        )
    if not gpus:
        return {"status": "no_gpus"}
    return {
        "status": "available",
        "driver_versions": sorted({gpu["driver_version"] for gpu in gpus}),
        "gpus": gpus,
    }
