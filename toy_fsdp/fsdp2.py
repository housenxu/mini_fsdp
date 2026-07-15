from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


def _load_fsdp2_api():
    """Load the public FSDP2 API with a compatibility fallback for PyTorch 2.4."""
    try:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            FSDPModule,
            MixedPrecisionPolicy,
            fully_shard,
        )
    except ImportError:
        try:
            from torch.distributed._composable.fsdp import (  # type: ignore[no-redef]
                CPUOffloadPolicy,
                FSDPModule,
                MixedPrecisionPolicy,
                fully_shard,
            )
        except ImportError as error:
            raise RuntimeError(
                "FSDP2 is unavailable. Install PyTorch 2.4 or newer; a recent "
                "stable PyTorch release is recommended."
            ) from error
    return fully_shard, FSDPModule, MixedPrecisionPolicy, CPUOffloadPolicy


def _dtensor_type():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    return DTensor


@dataclass(frozen=True)
class FSDP2Config:
    """Configuration for the FSDP2 comparison path.

    ``reshard_after_forward=None`` keeps FSDP2's recommended defaults: child
    groups reshard after forward while the root group does not. A positive
    prefetch depth enables explicit forward and backward all-gather schedules.
    """

    reshard_after_forward: bool | int | None = None
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cpu_offload: bool = False
    prefetch_depth: int = 0

    def __post_init__(self) -> None:
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")


@dataclass
class FSDP2Runtime:
    """Runtime metadata and controls for a model transformed by FSDP2."""

    model: nn.Module
    mesh: Any
    fsdp_modules: tuple[nn.Module, ...]
    root_has_parameter_group: bool
    config: FSDP2Config
    _pending_first_unshard: Any = field(default=None, init=False, repr=False)

    @property
    def dtensor_parameter_count(self) -> int:
        DTensor = _dtensor_type()
        return sum(isinstance(param, DTensor) for param in self.model.parameters())

    @property
    def global_parameter_numel(self) -> int:
        # DTensor.numel() follows the logical/global tensor shape.
        return sum(param.numel() for param in self.model.parameters())

    @property
    def communication_group_count(self) -> int:
        return len(self.fsdp_modules) + int(self.root_has_parameter_group)

    @property
    def local_parameter_numel(self) -> int:
        DTensor = _dtensor_type()
        return sum(
            param.to_local().numel() if isinstance(param, DTensor) else param.numel()
            for param in self.model.parameters()
        )

    def parameter_layouts(self) -> dict[str, dict[str, object]]:
        """Return inspectable global/local shapes and DTensor placements."""
        DTensor = _dtensor_type()
        layouts: dict[str, dict[str, object]] = {}
        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                layouts[name] = {
                    "global_shape": tuple(param.shape),
                    "local_shape": tuple(param.to_local().shape),
                    "placements": tuple(str(p) for p in param.placements),
                }
            else:
                layouts[name] = {
                    "global_shape": tuple(param.shape),
                    "local_shape": tuple(param.shape),
                    "placements": ("Replicate",),
                }
        return layouts

    def prefetch_first_module(self) -> None:
        """Issue the first layer all-gather before its pre-forward hook."""
        if not self.fsdp_modules or self.config.prefetch_depth == 0:
            return
        self._pending_first_unshard = self.fsdp_modules[0].unshard(async_op=True)

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        """FSDP2 equivalent of no_sync(), useful for gradient accumulation."""
        self.model.set_requires_gradient_sync(enabled, recurse=True)

    def set_reshard_after_backward(self, enabled: bool) -> None:
        self.model.set_reshard_after_backward(enabled, recurse=True)


def _fully_shard_kwargs(config: FSDP2Config, mesh: Any) -> dict[str, object]:
    fully_shard, _, MixedPrecisionPolicy, CPUOffloadPolicy = _load_fsdp2_api()
    del fully_shard
    kwargs: dict[str, object] = {"mesh": mesh}
    if config.reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = config.reshard_after_forward
    if any(
        dtype is not None
        for dtype in (config.param_dtype, config.reduce_dtype, config.output_dtype)
    ):
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=config.param_dtype,
            reduce_dtype=config.reduce_dtype,
            output_dtype=config.output_dtype,
        )
    if config.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    return kwargs


def apply_fsdp2(
    model: nn.Module,
    *,
    device_type: str,
    config: FSDP2Config | None = None,
    wrap_cls: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> FSDP2Runtime:
    """Apply official FSDP2 bottom-up and expose its DTensor state.

    Each selected child becomes one communication group. The root is sharded
    last so that it manages only parameters not already claimed by children.
    This preserves original parameter names and makes the result directly
    comparable with the hand-written flat-parameter ``MiniFSDP``.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "FSDP2 requires an initialized process group. Launch with torchrun, "
            "including --nproc_per_node=1 for a single-process smoke test."
        )
    config = config or FSDP2Config()
    fully_shard, FSDPModule, _, _ = _load_fsdp2_api()
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        from torch.distributed import init_device_mesh

    world_size = dist.get_world_size()
    mesh = init_device_mesh(
        device_type,
        (world_size,),
        mesh_dim_names=("fsdp",),
    )
    kwargs = _fully_shard_kwargs(config, mesh)

    # model.modules() is pre-order. Apply in reverse to honor FSDP2's bottom-up
    # contract even when wrap_cls contains nested module types.
    selected = [
        module
        for module in model.modules()
        if module is not model and isinstance(module, wrap_cls)
    ]
    child_parameter_ids = {
        id(parameter)
        for module in selected
        for parameter in module.parameters()
    }
    root_has_parameter_group = any(
        id(parameter) not in child_parameter_ids
        for parameter in model.parameters()
    )
    for module in reversed(selected):
        fully_shard(module, **kwargs)
    fully_shard(model, **kwargs)

    if not isinstance(model, FSDPModule):
        raise RuntimeError("fully_shard did not convert the root module to FSDPModule")

    if config.prefetch_depth > 0:
        for index, module in enumerate(selected):
            forward_modules = selected[
                index + 1 : index + 1 + config.prefetch_depth
            ]
            if forward_modules:
                module.set_modules_to_forward_prefetch(forward_modules)

            backward_modules = list(
                reversed(selected[max(0, index - config.prefetch_depth) : index])
            )
            if backward_modules:
                module.set_modules_to_backward_prefetch(backward_modules)

    runtime = FSDP2Runtime(
        model=model,
        mesh=mesh,
        fsdp_modules=tuple(selected),
        root_has_parameter_group=root_has_parameter_group,
        config=config,
    )
    if runtime.dtensor_parameter_count != sum(1 for _ in model.parameters()):
        raise RuntimeError("expected every FSDP2-managed parameter to be a DTensor")
    return runtime
