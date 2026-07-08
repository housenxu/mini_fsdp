from toy_fsdp import TinyMLP


def count_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


def mib(num_elements: int, bytes_per_element: int = 4) -> float:
    return num_elements * bytes_per_element / 1024 / 1024


def main() -> None:
    model = TinyMLP(input_dim=128, hidden_dim=4096, num_classes=10)
    params = count_params(model)

    # AdamW has two fp32 state tensors per parameter: exp_avg and exp_avg_sq.
    ddp_elements = params + params + 2 * params

    print(f"model parameters: {params:,} ({mib(params):.2f} MiB fp32)")
    print()
    print("Approximate steady-state training memory for params/grads/Adam states")
    print("Activation memory and temporary all-gather buffers are not included.")
    print()
    print(f"{'world':>5} {'DDP MiB/rank':>14} {'FSDP MiB/rank':>15} {'saving':>10}")
    for world_size in [1, 2, 4, 8]:
        ddp_mib = mib(ddp_elements)
        fsdp_mib = mib(ddp_elements // world_size)
        saving = 1.0 - fsdp_mib / ddp_mib
        print(f"{world_size:>5} {ddp_mib:>14.2f} {fsdp_mib:>15.2f} {saving:>9.1%}")


if __name__ == "__main__":
    main()
