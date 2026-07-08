import os

import torch
import torch.distributed as dist

from toy_fsdp import MiniFSDP, TinyMLP, make_batch


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def main() -> None:
    rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    torch.manual_seed(1234)
    input_dim = 128
    num_classes = 10
    batch_size = 32

    base_model = TinyMLP(input_dim=input_dim, hidden_dim=512, num_classes=num_classes).to(device)
    model = MiniFSDP(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(20):
        x, y = make_batch(
            batch_size=batch_size,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            seed=10_000 + rank * 1_000 + step,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        model.reduce_scatter_grad()
        optimizer.step()
        model.reshard()

        if rank == 0 and step % 5 == 0:
            print(f"step={step:02d} world_size={world_size} loss={loss.item():.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
