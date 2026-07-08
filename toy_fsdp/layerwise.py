from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from .fsdp import MiniFSDP


class LayerWiseMiniFSDP(nn.Module):
    """Wrap selected leaf modules with MiniFSDP.

    The first version of this project wraps the whole model as one FSDP unit.
    This class is closer to production FSDP's mental model: each compute-heavy
    block owns an independent parameter shard, all-gathers before its forward,
    and reduce-scatters its gradients after backward.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        wrap_cls: tuple[type[nn.Module], ...] = (nn.Linear,),
        prefetch: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.wrap_cls = wrap_cls
        self.prefetch = prefetch
        self._wrapped: list[MiniFSDP] = []
        self._wrap_children(self.module)

        if not self._wrapped:
            raise ValueError("LayerWiseMiniFSDP did not find any modules to wrap")

    def _wrap_children(self, parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, self.wrap_cls):
                wrapped = MiniFSDP(child, prefetch=self.prefetch)
                setattr(parent, name, wrapped)
                self._wrapped.append(wrapped)
            else:
                self._wrap_children(child)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for wrapper in self._wrapped:
            yield from wrapper.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        for idx, wrapper in enumerate(self._wrapped):
            name_prefix = f"{prefix}.wrapped_{idx}" if prefix else f"wrapped_{idx}"
            yield from wrapper.named_parameters(
                prefix=name_prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            )

    @torch.no_grad()
    def reduce_scatter_grad(self, average: bool = True) -> None:
        for wrapper in self._wrapped:
            wrapper.reduce_scatter_grad(average=average)

    @torch.no_grad()
    def reshard(self) -> None:
        for wrapper in self._wrapped:
            wrapper.reshard()

    @property
    def total_numel(self) -> int:
        return sum(wrapper.total_numel for wrapper in self._wrapped)

    @property
    def num_all_gathers(self) -> int:
        return sum(wrapper.num_all_gathers for wrapper in self._wrapped)

    @property
    def num_reduce_scatters(self) -> int:
        return sum(wrapper.num_reduce_scatters for wrapper in self._wrapped)

    @property
    def num_wrapped_modules(self) -> int:
        return len(self._wrapped)
