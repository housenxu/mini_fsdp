from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommunicationEstimate:
    strategy: str
    forward_collective: str
    backward_collective: str
    bytes_per_step: int


def estimate_ddp_bytes(param_numel: int, bytes_per_element: int = 4) -> CommunicationEstimate:
    # Ring all-reduce has roughly 2 * (N - 1) / N * P bytes per rank.
    # Without knowing N here, use the common algorithmic shorthand: O(2P).
    return CommunicationEstimate(
        strategy="ddp",
        forward_collective="none",
        backward_collective="all_reduce(grads)",
        bytes_per_step=2 * param_numel * bytes_per_element,
    )


def estimate_minifsdp_bytes(param_numel: int, bytes_per_element: int = 4) -> CommunicationEstimate:
    # FSDP has all-gather(params) and reduce-scatter(grads).
    return CommunicationEstimate(
        strategy="minifsdp",
        forward_collective="all_gather(params)",
        backward_collective="reduce_scatter(grads)",
        bytes_per_step=2 * param_numel * bytes_per_element,
    )
