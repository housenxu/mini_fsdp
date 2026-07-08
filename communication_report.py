from __future__ import annotations

from toy_fsdp import TinyMLP
from toy_fsdp.communication import estimate_ddp_bytes, estimate_minifsdp_bytes
from toy_fsdp.metrics import format_bytes


def count_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    model = TinyMLP(input_dim=128, hidden_dim=4096, num_classes=10)
    params = count_params(model)
    estimates = [estimate_ddp_bytes(params), estimate_minifsdp_bytes(params)]

    print(f"model_params={params:,}")
    print(f"{'strategy':<10} {'forward':<22} {'backward':<24} {'approx bytes/step'}")
    for item in estimates:
        print(
            f"{item.strategy:<10} "
            f"{item.forward_collective:<22} "
            f"{item.backward_collective:<24} "
            f"{format_bytes(item.bytes_per_step)}"
        )


if __name__ == "__main__":
    main()
