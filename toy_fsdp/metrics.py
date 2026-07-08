from __future__ import annotations

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
def maybe_profile(enabled: bool, trace_dir: str, rank: int) -> Iterator[object | None]:
    if not enabled:
        yield None
        return

    out_dir = Path(trace_dir) / f"rank{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
    ) as prof:
        yield prof


def format_bytes(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 B"
    mib = num_bytes / 1024 / 1024
    return f"{mib:.2f} MiB"
