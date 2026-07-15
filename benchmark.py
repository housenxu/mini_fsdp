from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from toy_fsdp import (
    FSDP2Config,
    LayerWiseMiniFSDP,
    MiniFSDP,
    TinyMLP,
    TinyTransformerLM,
    TransformerBlock,
    TransformerConfig,
    apply_fsdp2,
    make_batch,
    make_lm_batch,
)
from toy_fsdp.communication import estimate_ddp_bytes, estimate_minifsdp_bytes
from toy_fsdp.distributed import cleanup_distributed, setup_distributed, synchronize
from toy_fsdp.metrics import WallTimer, format_bytes, maybe_profile, peak_memory_bytes, reset_peak_memory
from toy_fsdp.profiling import (
    NsightCapture,
    calculate_transformer_mfu,
    load_practical_peak,
    trace_range,
    write_formula_mfu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DDP, MiniFSDP, and FSDP2.")
    parser.add_argument("--model", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument(
        "--strategy",
        choices=["ddp", "minifsdp", "minifsdp-layerwise", "fsdp2"],
        default="minifsdp",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--intermediate-dim", type=int, default=2048)
    parser.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--freeze-token-embedding", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="Vendor dense peak TFLOP/s per GPU for the exact training dtype.",
    )
    parser.add_argument(
        "--practical-peak-json",
        default="benchmark_results/gemm_peak.json",
        help="Output from benchmarks/gemm_peak.py; missing files produce N/A.",
    )
    parser.add_argument("--metrics-dir", default="benchmark_results")
    parser.add_argument("--nsight", action="store_true")
    parser.add_argument("--nsight-start-step", type=int, default=1)
    parser.add_argument("--nsight-num-steps", type=int, default=2)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument(
        "--fsdp2-param-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="FSDP2 unsharded parameter compute dtype.",
    )
    parser.add_argument(
        "--fsdp2-reduce-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="FSDP2 reduce-scatter dtype.",
    )
    parser.add_argument(
        "--fsdp2-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override FSDP2 child/root defaults.",
    )
    parser.add_argument("--fsdp2-prefetch-depth", type=int, default=0)
    parser.add_argument("--fsdp2-cpu-offload", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def build_model(args: argparse.Namespace, device: torch.device, local_rank: int, world_size: int):
    if args.model == "transformer":
        model = TinyTransformerLM(
            TransformerConfig(
                vocab_size=args.vocab_size,
                max_seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                intermediate_dim=args.intermediate_dim,
                tie_embeddings=args.tie_embeddings,
                freeze_token_embedding=args.freeze_token_embedding,
            )
        ).to(device)
        fsdp2_wrap_cls = (TransformerBlock,)
    else:
        model = TinyMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
        ).to(device)
        fsdp2_wrap_cls = None

    if args.model == "transformer" and args.strategy == "minifsdp-layerwise":
        raise ValueError(
            "minifsdp-layerwise only manages wrapped leaf parameters and is not "
            "correct for Transformer root embeddings/norms. Use minifsdp or fsdp2."
        )
    if args.freeze_token_embedding and args.strategy.startswith("minifsdp"):
        raise ValueError(
            "flat-parameter MiniFSDP does not preserve per-parameter frozen-state "
            "optimizer semantics; use fsdp2 for this experiment."
        )

    if args.strategy == "ddp":
        if world_size > 1:
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            model = DistributedDataParallel(model, **ddp_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return model, optimizer, None

    if args.strategy == "fsdp2":
        runtime = apply_fsdp2(
            model,
            device_type=device.type,
            config=FSDP2Config(
                reshard_after_forward=args.fsdp2_reshard_after_forward,
                param_dtype=parse_dtype(args.fsdp2_param_dtype),
                reduce_dtype=parse_dtype(args.fsdp2_reduce_dtype),
                cpu_offload=args.fsdp2_cpu_offload,
                prefetch_depth=args.fsdp2_prefetch_depth,
            ),
            **({"wrap_cls": fsdp2_wrap_cls} if fsdp2_wrap_cls is not None else {}),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        return model, optimizer, runtime

    if args.strategy == "minifsdp-layerwise":
        model = LayerWiseMiniFSDP(model, prefetch=args.prefetch)
    else:
        model = MiniFSDP(model, prefetch=args.prefetch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return model, optimizer, None


def train_step(model, optimizer, runtime, args, device, rank: int, step: int) -> float:
    optimizer.zero_grad(set_to_none=True)
    if runtime is not None:
        runtime.prefetch_first_module()
    if args.model == "transformer":
        x, y = make_lm_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
            seed=10_000 + rank * 1_000 + step,
        )
    else:
        x, y = make_batch(
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            device=device,
            seed=10_000 + rank * 1_000 + step,
        )

    with trace_range(f"{args.strategy}::forward_backward"):
        loss = model(x, y)
        loss.backward()

    if args.strategy.startswith("minifsdp"):
        model.reduce_scatter_grad()

    if args.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

    with trace_range(f"{args.strategy}::optimizer_step"):
        optimizer.step()

    if args.strategy.startswith("minifsdp"):
        model.reshard()

    return float(loss.detach().item())


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size) <= 0 or args.warmup < 0:
        raise ValueError("steps and batch_size must be positive; warmup cannot be negative")
    if args.nsight and args.nsight_start_step + args.nsight_num_steps - 1 > args.steps:
        raise ValueError("the requested Nsight capture window exceeds measured steps")

    ctx = setup_distributed()
    torch.manual_seed(1234)
    practical_peak, benchmark_gpu_name = load_practical_peak(args.practical_peak_json)
    nsight = NsightCapture(args.nsight, args.nsight_start_step, args.nsight_num_steps)

    model, optimizer, runtime = build_model(
        args,
        ctx.device,
        ctx.local_rank,
        ctx.world_size,
    )
    param_numel = sum(p.numel() for p in model.parameters())
    if args.strategy.startswith("minifsdp"):
        param_numel = model.total_numel
    elif runtime is not None:
        param_numel = runtime.global_parameter_numel

    for step in range(args.warmup):
        train_step(model, optimizer, runtime, args, ctx.device, ctx.rank, step)
    synchronize()

    reset_peak_memory()
    losses: list[float] = []
    with maybe_profile(
        args.profile,
        args.trace_dir,
        ctx.rank,
        vendor_peak_tflops=args.peak_tflops,
        practical_peak_tflops=practical_peak,
        benchmark_gpu_name=benchmark_gpu_name,
    ) as prof:
        with WallTimer() as timer:
            for measured_step in range(1, args.steps + 1):
                nsight.on_step_start(measured_step)
                loss = train_step(
                    model,
                    optimizer,
                    runtime,
                    args,
                    ctx.device,
                    ctx.rank,
                    args.warmup + measured_step - 1,
                )
                nsight.on_step_end(measured_step)
                losses.append(loss)
                if prof is not None:
                    prof.step()
        synchronize()
    nsight.close()

    if ctx.rank == 0:
        global_samples = args.steps * args.batch_size * ctx.world_size
        comm = (
            estimate_ddp_bytes(param_numel)
            if args.strategy == "ddp"
            else estimate_minifsdp_bytes(param_numel)
        )
        print(f"strategy={args.strategy}")
        print(f"model={args.model}")
        print(f"backend={ctx.backend} world_size={ctx.world_size}")
        print(f"steps={args.steps} batch_size={args.batch_size}")
        print(f"last_loss={losses[-1]:.4f}")
        if args.model == "transformer":
            global_tokens = global_samples * args.seq_len
            local_tokens = args.steps * args.batch_size * args.seq_len
            result = calculate_transformer_mfu(
                parameter_count=param_numel,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                sequence_length=args.seq_len,
                local_tokens=local_tokens,
                elapsed_seconds=timer.elapsed_s,
                vendor_peak_tflops=args.peak_tflops,
                practical_peak_tflops=practical_peak,
            )
            print(f"throughput_global={global_tokens / timer.elapsed_s:.2f} tokens/s")
            print(f"throughput_per_gpu={result.local_tokens_per_second:.2f} tokens/s")
            print(f"formula_flops_per_token={result.flops_per_token:.0f}")
            print(f"model_tflops_per_gpu={result.model_tflops_per_gpu:.4f}")
            print(
                "formula_mfu_vs_vendor_peak="
                f"{_format_percent(result.formula_mfu_vs_vendor_percent)}"
            )
            print(
                "efficiency_vs_practical_gemm_peak="
                f"{_format_percent(result.efficiency_vs_practical_peak_percent)}"
            )
            report_path = Path(args.metrics_dir) / f"formula_mfu_{args.strategy}.json"
            write_formula_mfu(
                report_path,
                result,
                strategy=args.strategy,
                model=args.model,
                world_size=ctx.world_size,
                elapsed_seconds=timer.elapsed_s,
                global_tokens_per_second=global_tokens / timer.elapsed_s,
                gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                benchmark_gpu_name=benchmark_gpu_name,
            )
            print(f"formula_mfu_report={report_path}")
        else:
            print(f"throughput={global_samples / timer.elapsed_s:.2f} samples/s")
            print("formula_mfu=N/A (dense Transformer FLOPs model is not valid for TinyMLP)")
        print(f"peak_cuda_memory={format_bytes(peak_memory_bytes())}")
        print(f"forward_collective={comm.forward_collective}")
        print(f"backward_collective={comm.backward_collective}")
        print(f"approx_comm_bytes_per_step={format_bytes(comm.bytes_per_step)}")
        if args.strategy.startswith("minifsdp"):
            print(f"all_gathers={model.num_all_gathers}")
            print(f"reduce_scatters={model.num_reduce_scatters}")
        if args.strategy == "minifsdp-layerwise":
            print(f"wrapped_modules={model.num_wrapped_modules}")
        if runtime is not None:
            print(f"dtensor_parameters={runtime.dtensor_parameter_count}")
            print(f"global_parameter_elements={runtime.global_parameter_numel}")
            print(f"local_parameter_elements={runtime.local_parameter_numel}")
            print(f"fsdp2_block_groups={len(runtime.fsdp_modules)}")
            print(f"fsdp2_communication_groups={runtime.communication_group_count}")

    cleanup_distributed()


if __name__ == "__main__":
    main()