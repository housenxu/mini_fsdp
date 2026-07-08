from __future__ import annotations

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import record_function

from toy_fsdp import MiniFSDP, TinyMLP, make_batch
from toy_fsdp.communication import estimate_ddp_bytes, estimate_minifsdp_bytes
from toy_fsdp.distributed import cleanup_distributed, setup_distributed, synchronize
from toy_fsdp.metrics import WallTimer, format_bytes, maybe_profile, peak_memory_bytes, reset_peak_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DDP and MiniFSDP.")
    parser.add_argument("--strategy", choices=["ddp", "minifsdp"], default="minifsdp")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--prefetch", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device, local_rank: int, world_size: int):
    model = TinyMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    if args.strategy == "ddp":
        if world_size > 1:
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            model = DistributedDataParallel(model, **ddp_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return model, optimizer

    model = MiniFSDP(model, prefetch=args.prefetch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return model, optimizer


def train_step(model, optimizer, args, device, rank: int, step: int) -> float:
    x, y = make_batch(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        device=device,
        seed=10_000 + rank * 1_000 + step,
    )

    optimizer.zero_grad(set_to_none=True)
    with record_function(f"{args.strategy}::forward_backward"):
        loss = model(x, y)
        loss.backward()

    if args.strategy == "minifsdp":
        model.reduce_scatter_grad()

    with record_function(f"{args.strategy}::optimizer_step"):
        optimizer.step()

    if args.strategy == "minifsdp":
        model.reshard()

    return float(loss.detach().item())


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    torch.manual_seed(1234)

    model, optimizer = build_model(args, ctx.device, ctx.local_rank, ctx.world_size)
    param_numel = sum(p.numel() for p in model.parameters())
    if args.strategy == "minifsdp":
        param_numel = model.total_numel

    for step in range(args.warmup):
        train_step(model, optimizer, args, ctx.device, ctx.rank, step)
    synchronize()

    reset_peak_memory()
    losses: list[float] = []
    with maybe_profile(args.profile, args.trace_dir, ctx.rank) as prof:
        with WallTimer() as timer:
            for step in range(args.steps):
                loss = train_step(model, optimizer, args, ctx.device, ctx.rank, args.warmup + step)
                losses.append(loss)
                if prof is not None:
                    prof.step()
        synchronize()

    if ctx.rank == 0:
        samples = args.steps * args.batch_size * ctx.world_size
        comm = (
            estimate_ddp_bytes(param_numel)
            if args.strategy == "ddp"
            else estimate_minifsdp_bytes(param_numel)
        )
        print(f"strategy={args.strategy}")
        print(f"backend={ctx.backend} world_size={ctx.world_size}")
        print(f"steps={args.steps} batch_size={args.batch_size}")
        print(f"last_loss={losses[-1]:.4f}")
        print(f"throughput={samples / timer.elapsed_s:.2f} samples/s")
        print(f"peak_cuda_memory={format_bytes(peak_memory_bytes())}")
        print(f"forward_collective={comm.forward_collective}")
        print(f"backward_collective={comm.backward_collective}")
        print(f"approx_comm_bytes_per_step={format_bytes(comm.bytes_per_step)}")
        if args.strategy == "minifsdp":
            print(f"all_gathers={model.num_all_gathers}")
            print(f"reduce_scatters={model.num_reduce_scatters}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
