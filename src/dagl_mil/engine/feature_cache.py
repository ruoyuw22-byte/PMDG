from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


FEATURE_CACHE_PROTOCOL = "mammoclip-b2-gap-ragged-v1"
_ALLOWED_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    # Viewing as bytes also supports bfloat16, which NumPy cannot represent on
    # every version used by the project.
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


@dataclass
class FeatureCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalid_entries: int = 0
    bypassed: int = 0
    write_failures: int = 0


@dataclass(frozen=True)
class CacheEntryIdentity:
    item_digest: str
    order_digest: str
    expected_instances: int
    path: Path


class FrozenGAPFeatureCache:
    """Validated, evaluation-only cache for frozen 352-D backbone outputs.

    The caller must still construct the current ragged bags.  Cache identity
    includes their source image and ordered candidate geometry, so a hit can
    never reorder candidates or detach features from the anatomy/coordinate
    records used downstream.  Training augmentation and FPN maps deliberately
    have no API here.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        provenance: Mapping[str, Any],
        storage_dtype: torch.dtype,
        feature_dim: int = 352,
    ) -> None:
        if storage_dtype not in _ALLOWED_DTYPES:
            raise ValueError("Feature cache dtype must be float16, bfloat16, or float32")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        backbone_sha256 = str(provenance.get("backbone_sha256", "")).lower()
        if len(backbone_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in backbone_sha256
        ):
            raise ValueError("Cache provenance requires a hexadecimal backbone_sha256")
        required = {
            "preprocessing_protocol",
            "mean",
            "std",
            "patch_size",
            "eval_stride",
            "minimum_breast_coverage",
            "amp_dtype",
        }
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"Cache provenance is missing: {missing}")
        if str(provenance["amp_dtype"]) != _ALLOWED_DTYPES[storage_dtype]:
            raise ValueError("amp_dtype does not match storage_dtype")

        self.root = Path(root)
        self.storage_dtype = storage_dtype
        self.feature_dim = int(feature_dim)
        self.provenance = {
            "cache_protocol": FEATURE_CACHE_PROTOCOL,
            **dict(provenance),
            "feature_dim": self.feature_dim,
            "storage_dtype": _ALLOWED_DTYPES[self.storage_dtype],
        }
        self.namespace_digest = _canonical_digest(self.provenance)
        self.stats = FeatureCacheStats()

    @property
    def namespace(self) -> Path:
        return self.root / self.namespace_digest

    @staticmethod
    def _source_identity(view: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(view.get("source_path", ""))
        source = Path(raw_path) if raw_path else None
        size: int | None = None
        mtime_ns: int | None = None
        resolved = raw_path
        if source is not None:
            try:
                resolved = str(source.resolve())
                stat = source.stat()
                size = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
            except OSError:
                # Synthetic/unit inputs can lack a source file.  Their other
                # identity fields still isolate entries; formal DICOM inputs
                # always provide a stat-able source.
                resolved = str(source.absolute())
        return {"path": resolved, "size": size, "mtime_ns": mtime_ns}

    def identify(
        self,
        *,
        exam_id: str,
        breast_id: str,
        laterality: str,
        view_key: str,
        view: Mapping[str, Any],
    ) -> CacheEntryIdentity | None:
        patches = view.get("patches")
        coordinates = view.get("coords")
        anatomy = view.get("anatomy_base", view.get("anatomy"))
        coverage = view.get("coverage")
        if not all(
            isinstance(value, Tensor)
            for value in (patches, coordinates, anatomy, coverage)
        ):
            self.stats.bypassed += 1
            return None
        assert isinstance(patches, Tensor)
        assert isinstance(coordinates, Tensor)
        assert isinstance(anatomy, Tensor)
        assert isinstance(coverage, Tensor)
        if (
            patches.ndim != 4
            or patches.shape[0] <= 0
            or coordinates.shape != (patches.shape[0], 4)
            or anatomy.ndim != 2
            or anatomy.shape[0] != patches.shape[0]
            or coverage.shape != (patches.shape[0],)
        ):
            self.stats.bypassed += 1
            return None

        order = {
            "coords_sha256": _tensor_digest(coordinates),
            "anatomy_sha256": _tensor_digest(anatomy[:, :5]),
            "coverage_sha256": _tensor_digest(coverage),
        }
        order_digest = _canonical_digest(order)
        item = {
            "namespace_digest": self.namespace_digest,
            "exam_id": str(exam_id),
            "breast_id": str(breast_id),
            "laterality": str(laterality).upper(),
            "view_key": str(view_key),
            "view_name": str(view.get("view_name", view_key)).upper(),
            "image_id": str(view.get("image_id", "")),
            "source": self._source_identity(view),
            "image_shape": [int(value) for value in view.get("image_shape", ())],
            "patch_shape": [int(value) for value in patches.shape],
            "patch_dtype": str(patches.dtype),
            "order_digest": order_digest,
        }
        item_digest = _canonical_digest(item)
        path = self.namespace / item_digest[:2] / f"{item_digest}.pt"
        return CacheEntryIdentity(
            item_digest=item_digest,
            order_digest=order_digest,
            expected_instances=int(patches.shape[0]),
            path=path,
        )

    def load(
        self,
        identity: CacheEntryIdentity | None,
        *,
        device: torch.device,
    ) -> Tensor | None:
        if identity is None:
            return None
        path = identity.path
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError("payload is not a mapping")
            features = payload.get("features")
            expected_shape = (identity.expected_instances, self.feature_dim)
            if (
                payload.get("protocol") != FEATURE_CACHE_PROTOCOL
                or payload.get("namespace_digest") != self.namespace_digest
                or payload.get("item_digest") != identity.item_digest
                or payload.get("order_digest") != identity.order_digest
                or tuple(payload.get("feature_shape", ())) != expected_shape
                or payload.get("feature_dtype") != _ALLOWED_DTYPES[self.storage_dtype]
                or not isinstance(features, Tensor)
                or tuple(features.shape) != expected_shape
                or features.dtype != self.storage_dtype
                or payload.get("feature_sha256") != _tensor_digest(features)
                or not bool(torch.isfinite(features).all())
            ):
                raise ValueError("cache metadata, shape, order, or hash mismatch")
        except Exception:
            self.stats.invalid_entries += 1
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return features.to(device=device, non_blocking=True)

    def store(
        self,
        identity: CacheEntryIdentity | None,
        features: Tensor,
    ) -> bool:
        if identity is None:
            return False
        if (
            features.shape != (identity.expected_instances, self.feature_dim)
            or features.dtype != self.storage_dtype
            or not bool(torch.isfinite(features).all())
        ):
            # A dtype mismatch means the current execution does not match the
            # AMP provenance.  Returning the online value is safe; caching it
            # under a false identity is not.
            self.stats.bypassed += 1
            return False
        cpu = features.detach().to(device="cpu").contiguous()
        payload = {
            "protocol": FEATURE_CACHE_PROTOCOL,
            "namespace_digest": self.namespace_digest,
            "item_digest": identity.item_digest,
            "order_digest": identity.order_digest,
            "feature_shape": list(cpu.shape),
            "feature_dtype": _ALLOWED_DTYPES[self.storage_dtype],
            "feature_sha256": _tensor_digest(cpu),
            "features": cpu,
        }
        path = identity.path
        temporary_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".pt",
                prefix=f".{path.stem}.",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                torch.save(payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
            self.stats.writes += 1
            return True
        except (OSError, RuntimeError, ValueError):
            self.stats.write_failures += 1
            return False
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass

    def report(self) -> dict[str, Any]:
        return {
            **asdict(self.stats),
            "root": str(self.root),
            "namespace_digest": self.namespace_digest,
            "provenance": dict(self.provenance),
        }
