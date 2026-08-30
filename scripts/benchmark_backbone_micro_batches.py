"""Read-only Mammo-CLIP micro-batch throughput probe for the local GPU."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from dagl_mil.config import load_config
from dagl_mil.evaluation.artifacts import sha256_file
from dagl_mil.evaluation.report import write_evaluation_report
from dagl_mil.models import build_backbone
from dagl_mil.reproducibility import environment_record, seed_everything


def benchmark(
    config_path: str | Path,
    *,
    batch_sizes: tuple[int, ...] = (16, 32, 64),
    warmup: int = 5,
    repetitions: int = 20,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if warmup < 0 or repetitions <= 0 or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch sizes/repetitions must be positive and warmup non-negative")
    config = load_config(config_path)
    seed_everything(
        int(config["training"]["seed"]),
        deterministic=bool(config["training"].get("deterministic", True)),
    )
    device = torch.device("cuda")
    backbone = build_backbone(config).to(device).eval()
    rows = []
    for batch_size in batch_sizes:
        patches = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            # Zero is a valid already-standardized (mean-centered) input value.
            patches = torch.zeros(
                (batch_size, 3, 512, 512),
                device=device,
                dtype=torch.float32,
            )
            elapsed_ms: list[float] = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                for _ in range(warmup):
                    backbone.encode_patches(
                        patches, micro_batch_size=batch_size
                    )
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline_allocated = int(torch.cuda.memory_allocated(device))
                baseline_reserved = int(torch.cuda.memory_reserved(device))
                for _ in range(repetitions):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter_ns()
                    backbone.encode_patches(
                        patches, micro_batch_size=batch_size
                    )
                    torch.cuda.synchronize(device)
                    elapsed_ms.append(
                        (time.perf_counter_ns() - start) / 1_000_000.0
                    )
            mean_ms = float(statistics.fmean(elapsed_ms))
            rows.append(
                {
                    "micro_batch_size": batch_size,
                    "status": "completed",
                    "latency_mean_ms": mean_ms,
                    "latency_median_ms": float(statistics.median(elapsed_ms)),
                    "latency_std_ms": float(statistics.pstdev(elapsed_ms)),
                    "throughput_patches_per_second": batch_size * 1000.0 / mean_ms,
                    "baseline_allocated_bytes": baseline_allocated,
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "baseline_reserved_bytes": baseline_reserved,
                    "peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            rows.append(
                {
                    "micro_batch_size": batch_size,
                    "status": "out_of_memory",
                    "error": str(exc),
                }
            )
        finally:
            del patches
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "status": "completed",
        "scope": "frozen Mammo-CLIP EfficientNet-B2 only",
        "input": "GPU-resident standardized float32 [B,3,512,512]",
        "precision": "fp16 autocast",
        "warmup": warmup,
        "repetitions": repetitions,
        "synchronize_each_repetition": True,
        "batch_sizes": list(batch_sizes),
        "checkpoint_sha256": sha256_file(config["paths"]["checkpoint"]),
        "environment": environment_record(),
        "results": rows,
        "formal_config_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dagl.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/backbone_micro_batch_benchmark.json"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    result = benchmark(
        args.config,
        batch_sizes=tuple(args.batch_sizes),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    write_evaluation_report(args.output, result)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
