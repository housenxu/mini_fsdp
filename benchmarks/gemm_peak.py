"""Measure this GPU/software stack's practical PyTorch/cuBLAS GEMM ceiling."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be M,N,K with positive integers")
    return parts


def benchmark_gemm(m: int, n: int, k: int, dtype: torch.dtype, warmup: int, iterations: int, trials: int):
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    trial_tflops = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = torch.mm(a, b)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
        trial_tflops.append((2.0 * m * n * k) / (elapsed_ms / 1e3) / 1e12)
    return {
        "m": m,
        "n": n,
        "k": k,
        "median_tflops": statistics.median(trial_tflops),
        "best_tflops": max(trial_tflops),
        "trial_tflops": trial_tflops,
        "output_checksum": float(output.float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--sizes", type=int, nargs="*", default=[2048, 4096, 8192])
    parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vendor-peak-tflops", type=float, default=None)
    parser.add_argument("--output", default="benchmark_results/gemm_peak.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GEMM peak benchmark")
    if min(args.warmup, args.iterations, args.trials) < 1:
        raise ValueError("warmup, iterations, and trials must all be positive")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = DTYPES[args.dtype]
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support bfloat16")

    shapes = [(size, size, size) for size in args.sizes]
    shapes.extend(args.shape)
    if not shapes:
        raise ValueError("at least one square size or model-like shape is required")

    results = []
    for m, n, k in shapes:
        result = benchmark_gemm(m, n, k, dtype, args.warmup, args.iterations, args.trials)
        results.append(result)
        print(
            f"GEMM [{m}, {k}] x [{k}, {n}]: "
            f"median={result['median_tflops']:.2f} TFLOP/s, "
            f"best={result['best_tflops']:.2f} TFLOP/s"
        )

    props = torch.cuda.get_device_properties(args.device)
    best_result = max(results, key=lambda item: item["median_tflops"])
    report = {
        "gpu_name": torch.cuda.get_device_name(args.device),
        "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "best_tflops": best_result["median_tflops"],
        "best_shape": [best_result["m"], best_result["n"], best_result["k"]],
        "vendor_peak_tflops": args.vendor_peak_tflops,
        "gemm_efficiency_vs_vendor_percent": (
            best_result["median_tflops"] / args.vendor_peak_tflops * 100
            if args.vendor_peak_tflops
            else None
        ),
        "results": results,
        "interpretation": "practical cuBLAS ceiling, not vendor peak and not end-to-end model throughput",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Practical peak report written to {output_path}")


if __name__ == "__main__":
    main()