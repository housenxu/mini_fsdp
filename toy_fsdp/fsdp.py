from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn
from .profiling import trace_range


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: torch.Size
    numel: int
    start: int
    end: int


class MiniFSDP(nn.Module):
    """A small educational FSDP wrapper.

    This implementation favors readability over performance. It gathers one
    full flat parameter before forward and manually scatters gradients after
    backward. Production FSDP uses per-module wrapping, prefetching, stream-aware
    memory management, mixed precision, and many more details.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: dist.ProcessGroup | None = None,
        *,
        prefetch: bool = False,
    ):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.prefetch = prefetch

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank(process_group)
            self.world_size = dist.get_world_size(process_group)
        else:
            self.rank = 0
            self.world_size = 1

        named_params = list(module.named_parameters(recurse=True))
        if not named_params:
            raise ValueError("ToyFSDP requires a module with parameters")

        self.param_infos: list[ParamInfo] = []
        flat_parts = []
        offset = 0
        for name, param in named_params:
            numel = param.numel()
            self.param_infos.append(
                ParamInfo(
                    name=name,
                    shape=param.shape,
                    numel=numel,
                    start=offset,
                    end=offset + numel,
                )
            )
            flat_parts.append(param.detach().reshape(-1))
            offset += numel

        flat_param = torch.cat(flat_parts)
        self.total_numel = flat_param.numel()
        self.shard_numel = (self.total_numel + self.world_size - 1) // self.world_size
        self.padded_numel = self.shard_numel * self.world_size

        if self.padded_numel != self.total_numel:
            pad = torch.zeros(
                self.padded_numel - self.total_numel,
                dtype=flat_param.dtype,
                device=flat_param.device,
            )
            flat_param = torch.cat([flat_param, pad])

        shard_start = self.rank * self.shard_numel
        shard_end = shard_start + self.shard_numel
        local_shard = flat_param[shard_start:shard_end].clone()
        self.flat_param_shard = nn.Parameter(local_shard)

        self._original_params = [param for _, param in named_params]
        self._full_flat_param: torch.Tensor | None = None
        self._prefetched_full_param: torch.Tensor | None = None
        self._prefetch_work: dist.Work | None = None
        self._prefetch_parts: list[torch.Tensor] | None = None
        self.num_all_gathers = 0
        self.num_reduce_scatters = 0

    def sharded_parameters(self) -> Iterator[nn.Parameter]:
        yield self.flat_param_shard

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield self.flat_param_shard

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        name = f"{prefix}.flat_param_shard" if prefix else "flat_param_shard"
        yield name, self.flat_param_shard

    def forward(self, *args, **kwargs):
        if self.prefetch and self._prefetch_work is None and self._prefetched_full_param is None:
            self.prefetch_full_params()
        self.unshard()
        with trace_range("minifsdp::compute_forward"):
            return self.module(*args, **kwargs)

    @torch.no_grad()
    def prefetch_full_params(self) -> None:
        """Start an asynchronous all-gather for the next forward.

        This is a small experiment for explaining communication/computation
        overlap. It is intentionally conservative: the next forward waits for
        the work before using the gathered parameters.
        """
        if self.world_size == 1 or self._prefetch_work is not None:
            return

        local = self.flat_param_shard.detach()
        self._prefetch_parts = [torch.empty_like(local) for _ in range(self.world_size)]
        with trace_range("minifsdp::prefetch_all_gather_params"):
            self._prefetch_work = dist.all_gather(
                self._prefetch_parts,
                local,
                group=self.process_group,
                async_op=True,
            )

    @torch.no_grad()
    def unshard(self) -> None:
        """All-gather local shards and rebuild full module parameters."""
        local = self.flat_param_shard.detach()
        with trace_range("minifsdp::all_gather_params"):
            if self._prefetch_work is not None:
                self._prefetch_work.wait()
                assert self._prefetch_parts is not None
                gathered = torch.cat(self._prefetch_parts, dim=0)
                self._prefetch_work = None
                self._prefetch_parts = None
            elif self._prefetched_full_param is not None:
                gathered = self._prefetched_full_param
                self._prefetched_full_param = None
            elif self.world_size == 1:
                gathered = local
            else:
                gathered_parts = [torch.empty_like(local) for _ in range(self.world_size)]
                dist.all_gather(gathered_parts, local, group=self.process_group)
                gathered = torch.cat(gathered_parts, dim=0)
            self.num_all_gathers += int(self.world_size > 1)

        full_flat = gathered[: self.total_numel].contiguous()
        self._full_flat_param = full_flat

        for info, param in zip(self.param_infos, self._original_params):
            view = full_flat[info.start : info.end].view(info.shape)
            param.data = view
            param.grad = None

    @torch.no_grad()
    def reduce_scatter_grad(self, average: bool = True) -> None:
        """Reduce full gradients and keep only this rank's gradient shard."""
        with trace_range("minifsdp::flatten_full_grads"):
            grad_parts = []
            for info, param in zip(self.param_infos, self._original_params):
                if param.grad is None:
                    grad = torch.zeros(
                        info.numel,
                        dtype=self.flat_param_shard.dtype,
                        device=self.flat_param_shard.device,
                    )
                else:
                    grad = param.grad.detach().reshape(-1)
                grad_parts.append(grad)

            full_grad = torch.cat(grad_parts)
            if self.padded_numel != self.total_numel:
                pad = torch.zeros(
                    self.padded_numel - self.total_numel,
                    dtype=full_grad.dtype,
                    device=full_grad.device,
                )
                full_grad = torch.cat([full_grad, pad])

        with trace_range("minifsdp::reduce_scatter_grads"):
            if self.world_size == 1:
                shard_grad = full_grad
            else:
                shard_grad = torch.empty_like(self.flat_param_shard)
                try:
                    dist.reduce_scatter_tensor(
                        shard_grad,
                        full_grad.contiguous(),
                        op=dist.ReduceOp.SUM,
                        group=self.process_group,
                    )
                except RuntimeError:
                    # Some CPU/Gloo builds do not support reduce_scatter_tensor.
                    # This fallback is mathematically equivalent but less memory efficient.
                    dist.all_reduce(full_grad, op=dist.ReduceOp.SUM, group=self.process_group)
                    start = self.rank * self.shard_numel
                    end = start + self.shard_numel
                    shard_grad.copy_(full_grad[start:end])
                self.num_reduce_scatters += 1

        if average and self.world_size > 1:
            shard_grad.div_(self.world_size)

        self.flat_param_shard.grad = shard_grad
        for param in self._original_params:
            param.grad = None

    @torch.no_grad()
    def reshard(self) -> None:
        """Drop references to full parameters after optimizer step."""
        self._full_flat_param = None

    @torch.no_grad()
    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Materialize a normal state dict for debugging or saving."""
        self.unshard()
        return {
            info.name: param.detach().clone()
            for info, param in zip(self.param_infos, self._original_params)
        }

    def memory_breakdown(self, optimizer_state_multiplier: int = 2) -> dict[str, int]:
        """Return element counts for the sharded training state."""
        return {
            "param_shard": self.shard_numel,
            "grad_shard": self.shard_numel,
            "optimizer_state_shard": optimizer_state_multiplier * self.shard_numel,
            "temporary_full_param": self.total_numel,
        }


ToyFSDP = MiniFSDP
