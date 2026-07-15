from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def get_full_model_state_dict(
    model: nn.Module,
    *,
    cpu_offload: bool = True,
) -> dict[str, torch.Tensor]:
    """Materialize a normal full state dict from an FSDP2/DTensor model.

    This function is collective and must be called by every rank.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload,
        ),
    )


def save_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> None:
    """Save reshardable model and optimizer state without rank-0 gathering."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
        },
        checkpoint_id=str(checkpoint_dir),
    )


def load_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> None:
    """Load a DCP checkpoint, allowing a different data-parallel world size."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
    }
    dcp.load(state, checkpoint_id=str(checkpoint_dir))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
