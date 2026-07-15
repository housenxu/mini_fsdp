from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


@dataclass
class StepStats:
    steps: int
    elapsed_s: float
    tokens_or_samples: int
    peak_memory_bytes: int

    @property
    def throughput(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return self.tokens_or_samples / self.elapsed_s


class WallTimer:
    def __enter__(self) -> "WallTimer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self.start


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated()


@contextmanager
def maybe_profile(
    enabled: bool,
    trace_dir: str,
    rank: int,
    *,
    vendor_peak_tflops: float | None = None,
    practical_peak_tflops: float | None = None,
    benchmark_gpu_name: str | None = None,
) -> Iterator[object | None]:
    """Collect a trace and a clearly-labelled operator-FLOPs estimate.

    ``with_flops`` uses PyTorch formulas for supported operators. It does not
    read executed FLOPs from hardware counters, and fused/custom kernels may be
    missing. Standard end-to-end MFU remains the formula result printed by the
    benchmark; this report is a separate diagnostic view.
    """
    if not enabled:
        yield None
        return

    out_dir = Path(trace_dir) / f"rank{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    previous_profiler_flag = os.environ.get("MINIFSDP_TORCH_PROFILER")
    os.environ["MINIFSDP_TORCH_PROFILER"] = "1"
    started = time.perf_counter()
    elapsed_seconds = 0.0
    try:
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        ) as prof:
            yield prof
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started
    finally:
        if previous_profiler_flag is None:
            os.environ.pop("MINIFSDP_TORCH_PROFILER", None)
        else:
            os.environ["MINIFSDP_TORCH_PROFILER"] = previous_profiler_flag

    estimated_flops = sum(
        float(getattr(event, "flops", 0) or 0)
        for event in prof.key_averages()
    )
    estimated_tflops = (
        estimated_flops / elapsed_seconds / 1e12
        if elapsed_seconds > 0
        else None
    )
    vendor_percent = (
        estimated_tflops / vendor_peak_tflops * 100
        if estimated_tflops is not None and vendor_peak_tflops
        else None
    )
    practical_percent = (
        estimated_tflops / practical_peak_tflops * 100
        if estimated_tflops is not None and practical_peak_tflops
        else None
    )
    current_gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    limitations = [
        "with_flops uses formulas for supported operators rather than hardware FLOP counters",
        "fused/custom kernels such as scaled-dot-product attention may be missing",
        "the profiler changes timing, so this value is diagnostic rather than the clean benchmark MFU",
        "one rank does not reveal distributed load imbalance; compare all rank reports",
    ]
    if benchmark_gpu_name and current_gpu and benchmark_gpu_name != current_gpu:
        limitations.append("the practical-peak benchmark GPU does not match this profiler GPU")

    report = {
        "rank": rank,
        "elapsed_seconds": elapsed_seconds,
        "profiler_estimated_flops": estimated_flops,
        "profiler_estimated_tflops_per_gpu": estimated_tflops,
        "vendor_peak_tflops": vendor_peak_tflops,
        "practical_peak_tflops": practical_peak_tflops,
        "profiler_estimated_mfu_vs_vendor_percent": vendor_percent,
        "profiler_estimated_efficiency_vs_practical_peak_percent": practical_percent,
        "flops_source": "torch.profiler with_flops operator formulas",
        "gpu_name": current_gpu,
        "benchmark_gpu_name": benchmark_gpu_name,
        "limitations": limitations,
    }
    output_path = out_dir / "profiler_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def format_bytes(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"
    mib = num_bytes / 1024 / 1024
    return f"{mib:.2f} MiB"