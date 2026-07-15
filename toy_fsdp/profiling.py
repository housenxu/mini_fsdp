from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.profiler import record_function


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@contextmanager
def trace_range(message: str) -> Iterator[None]:
    """Emit the same logical range to PyTorch Profiler and NVIDIA Nsight."""
    use_torch_profiler = _env_flag("MINIFSDP_TORCH_PROFILER")
    use_nvtx = _env_flag("MINIFSDP_NVTX") and torch.cuda.is_available()
    if not use_torch_profiler and not use_nvtx:
        yield
        return

    torch_context = record_function(message) if use_torch_profiler else nullcontext()
    if use_nvtx:
        torch.cuda.nvtx.range_push(message)
    try:
        with torch_context:
            yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


class NsightCapture:
    """Bound an Nsight Systems capture with the CUDA Profiler API.

    ``step`` is one-based and relative to the measured loop, after warmup.
    Every rank controls its own CUDA context so an nsys process-tree capture can
    retain the same window from all distributed workers.
    """

    def __init__(self, enabled: bool, start_step: int, num_steps: int) -> None:
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.start_step = int(start_step)
        self.num_steps = int(num_steps)
        self.stop_step = self.start_step + self.num_steps - 1
        self.active = False
        if self.start_step < 1:
            raise ValueError("nsight_start_step must be at least 1")
        if self.num_steps < 1:
            raise ValueError("nsight_num_steps must be at least 1")
        if self.enabled:
            os.environ["MINIFSDP_NVTX"] = "1"

    def on_step_start(self, step: int) -> None:
        if self.enabled and not self.active and step == self.start_step:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self.active = True

    def on_step_end(self, step: int) -> None:
        if self.enabled and self.active and step == self.stop_step:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            self.active = False

    def close(self) -> None:
        if self.enabled and self.active:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            self.active = False


@dataclass(frozen=True)
class TransformerMFU:
    parameter_count: int
    flops_per_token: float
    local_tokens_per_second: float
    model_tflops_per_gpu: float
    vendor_peak_tflops: float | None
    practical_peak_tflops: float | None
    formula_mfu_vs_vendor_percent: float | None
    efficiency_vs_practical_peak_percent: float | None


def transformer_flops_per_token(
    parameter_count: int,
    num_layers: int,
    hidden_dim: int,
    sequence_length: int,
) -> float:
    """Approximate dense Transformer training FLOPs for one token.

    The 6P term covers forward plus backward parameterized operations. The
    attention term restores the sequence-length-dependent quadratic attention
    work that is not represented well by 6P alone.
    """
    if min(parameter_count, num_layers, hidden_dim, sequence_length) <= 0:
        raise ValueError("Transformer dimensions and parameter_count must be positive")
    return float(
        6 * parameter_count
        + 12 * num_layers * hidden_dim * sequence_length
    )


def calculate_transformer_mfu(
    *,
    parameter_count: int,
    num_layers: int,
    hidden_dim: int,
    sequence_length: int,
    local_tokens: int,
    elapsed_seconds: float,
    vendor_peak_tflops: float | None,
    practical_peak_tflops: float | None,
) -> TransformerMFU:
    if local_tokens <= 0 or elapsed_seconds <= 0:
        raise ValueError("local_tokens and elapsed_seconds must be positive")
    if vendor_peak_tflops is not None and vendor_peak_tflops <= 0:
        raise ValueError("vendor_peak_tflops must be positive")
    if practical_peak_tflops is not None and practical_peak_tflops <= 0:
        raise ValueError("practical_peak_tflops must be positive")

    flops_per_token = transformer_flops_per_token(
        parameter_count,
        num_layers,
        hidden_dim,
        sequence_length,
    )
    local_tokens_per_second = local_tokens / elapsed_seconds
    model_tflops = local_tokens_per_second * flops_per_token / 1e12
    return TransformerMFU(
        parameter_count=parameter_count,
        flops_per_token=flops_per_token,
        local_tokens_per_second=local_tokens_per_second,
        model_tflops_per_gpu=model_tflops,
        vendor_peak_tflops=vendor_peak_tflops,
        practical_peak_tflops=practical_peak_tflops,
        formula_mfu_vs_vendor_percent=(
            model_tflops / vendor_peak_tflops * 100
            if vendor_peak_tflops is not None
            else None
        ),
        efficiency_vs_practical_peak_percent=(
            model_tflops / practical_peak_tflops * 100
            if practical_peak_tflops is not None
            else None
        ),
    )


def load_practical_peak(path: str | None) -> tuple[float | None, str | None]:
    if not path:
        return None, None
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return None, None
    with benchmark_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    peak = report.get("best_tflops", report.get("practical_peak_tflops"))
    return (float(peak) if peak is not None else None, report.get("gpu_name"))


def write_formula_mfu(path: str | Path, result: TransformerMFU, **metadata: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = asdict(result)
    report.update(metadata)
    report["flops_source"] = "6P + 12*L*H*S dense Transformer training approximation"
    report["limitations"] = [
        "formula MFU is an analytical model-work estimate, not a hardware counter",
        "the vendor peak must match GPU, dtype, sparsity mode, and clock assumptions",
        "the practical GEMM ceiling is a tuning denominator and is not standard MFU",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)