from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

from dagl_mil.config import load_config
from dagl_mil.experiment import build_dataset, build_loader
from dagl_mil.data import load_manifest
from dagl_mil.reproducibility import seed_everything


def _patch_count(batch: dict) -> int:
    return sum(
        int(view["patches"].shape[0])
        for sample in batch["views"]
        for view in sample.values()
    )


def benchmark(
    config_path: Path,
    workers: int,
    *,
    split: str,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, float | int | str]:
    config = copy.deepcopy(load_config(config_path))
    config["training"]["num_workers"] = workers
    seed_everything(int(config["training"]["seed"]), deterministic=True)
    manifest = load_manifest(config["paths"]["split_manifest"], validate=True)
    training = split == "train"
    dataset = build_dataset(config, manifest, split=split, include_bboxes=False)
    loader = build_loader(dataset, config, training=training)
    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)
    started = time.perf_counter()
    patches = 0
    breasts = 0
    for _ in range(measured_batches):
        batch = next(iterator)
        patches += _patch_count(batch)
        breasts += len(batch["views"])
    elapsed = time.perf_counter() - started
    # Explicitly release persistent Windows workers before benchmarking the
    # next setting in the same process.
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    del iterator, loader, dataset
    gc.collect()
    return {
        "workers": workers,
        "split": split,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "elapsed_seconds": elapsed,
        "batches_per_second": measured_batches / elapsed,
        "breasts_per_second": breasts / elapsed,
        "patches_per_second": patches / elapsed,
        "patches": patches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark cached VinDr train DataLoader worker counts."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dagl.yaml"))
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/dataloader_worker_benchmark.json")
    )
    args = parser.parse_args()
    if any(value < 0 for value in args.workers):
        raise ValueError("worker counts must be non-negative")
    if args.warmup_batches < 0 or args.measured_batches <= 0:
        raise ValueError("invalid batch counts")
    records = []
    for workers in args.workers:
        record = benchmark(
            args.config,
            workers,
            split=args.split,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    payload = {"config": str(args.config.resolve()), "results": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
