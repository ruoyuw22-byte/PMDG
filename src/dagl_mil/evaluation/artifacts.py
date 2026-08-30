from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "dagl-mil-predictions-v1"


@dataclass(frozen=True)
class PredictionArtifactInfo:
    path: str
    sha256: str | None
    expected_records: int | None
    schema_version: str


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def encode_binary_mask(mask) -> dict[str, object]:
    array = np.asarray(mask, dtype=bool)
    flat = array.reshape(-1).astype(np.uint8)
    padded = np.concatenate(([0], flat, [0]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    starts = transitions[0::2]
    ends = transitions[1::2]
    return {
        "shape": list(array.shape),
        "runs": [
            [int(start), int(end - start)]
            for start, end in zip(starts, ends, strict=True)
        ],
    }


def decode_binary_mask(encoded: Mapping[str, object]) -> np.ndarray:
    shape = tuple(int(value) for value in encoded["shape"])
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    for start, length in encoded["runs"]:
        flat[int(start) : int(start) + int(length)] = True
    return flat.reshape(shape)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_prediction_artifact(
    path: str | Path,
    records: Iterable[Mapping[str, object]],
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": SCHEMA_VERSION, "metadata": dict(metadata)}
    count = 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    header, ensure_ascii=False, sort_keys=True, default=_json_default
                )
                + "\n"
            )
            for record in records:
                required = {"exam_id", "laterality", "labels", "probabilities", "views"}
                missing = required - set(record)
                if missing:
                    raise ValueError(f"Prediction record is missing {sorted(missing)}")
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_default,
                    )
                    + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    sidecar = destination.with_suffix(destination.suffix + ".sha256.json")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(
        json.dumps(
            {
                "path": destination.name,
                "sha256": digest,
                "records": count,
                "schema_version": SCHEMA_VERSION,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_sidecar, sidecar)
    return {"path": str(destination), "sha256": digest, "records": count}


def read_prediction_artifact(
    path: str | Path,
    *,
    verify_hash: bool = True,
) -> tuple[dict[str, object], Iterator[dict[str, object]]]:
    source = Path(path)
    if verify_hash:
        sidecar_path = source.with_suffix(source.suffix + ".sha256.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        actual = sha256_file(source)
        if actual != sidecar["sha256"]:
            raise RuntimeError(f"Prediction artifact hash mismatch: {source}")
    handle = source.open("r", encoding="utf-8")
    header = json.loads(handle.readline())
    if header.get("schema_version") != SCHEMA_VERSION:
        handle.close()
        raise ValueError("Unsupported prediction artifact schema")

    def iterator() -> Iterator[dict[str, object]]:
        try:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        finally:
            handle.close()

    return header, iterator()


@contextmanager
def open_prediction_artifact(
    path: str | Path,
    *,
    verify_hash: bool = True,
) -> Iterator[
    tuple[dict[str, object], Iterator[dict[str, object]], PredictionArtifactInfo]
]:
    """Open one prediction artifact with a bounded-lifetime streaming iterator.

    Hash verification and JSONL parsing use the same open file description.  In
    addition to avoiding a second full-file hash in report generation, this
    prevents a path replacement between verification and parsing.  The context
    manager also guarantees prompt handle closure when record processing raises,
    which is required before a file can be replaced or removed on Windows.
    """

    source = Path(path).resolve()
    expected_records: int | None = None
    expected_sha256: str | None = None
    if verify_hash:
        sidecar_path = source.with_suffix(source.suffix + ".sha256.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, Mapping):
            raise ValueError("Prediction artifact sidecar must be a JSON object")
        if sidecar.get("path") != source.name:
            raise ValueError("Prediction artifact sidecar path does not match")
        if sidecar.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Prediction artifact sidecar schema does not match")
        records_value = sidecar.get("records")
        if (
            isinstance(records_value, bool)
            or not isinstance(records_value, int)
            or records_value < 0
        ):
            raise ValueError("Prediction artifact sidecar records must be non-negative")
        expected_records = records_value
        expected_sha256 = sidecar.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("Prediction artifact sidecar SHA-256 is invalid")

    binary_handle = source.open("rb")
    text_handle: io.TextIOWrapper | None = None
    try:
        actual_sha256: str | None = None
        if verify_hash:
            digest = hashlib.sha256()
            while chunk := binary_handle.read(8 * 1024 * 1024):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"Prediction artifact hash mismatch: {source}")
            binary_handle.seek(0)

        text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
        header_line = text_handle.readline()
        if not header_line:
            raise ValueError("Prediction artifact is missing its header")
        header = json.loads(header_line)
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported prediction artifact schema")

        def iterator() -> Iterator[dict[str, object]]:
            count = 0
            for line in text_handle:
                if line.strip():
                    count += 1
                    yield json.loads(line)
            if expected_records is not None and count != expected_records:
                raise RuntimeError(
                    "Prediction artifact record count mismatch: "
                    f"expected {expected_records}, read {count}"
                )

        info = PredictionArtifactInfo(
            path=str(source),
            sha256=actual_sha256,
            expected_records=expected_records,
            schema_version=SCHEMA_VERSION,
        )
        yield header, iterator(), info
    finally:
        if text_handle is not None:
            text_handle.close()
        else:
            binary_handle.close()
