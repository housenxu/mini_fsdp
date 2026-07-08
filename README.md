# MiniFSDP

A lightweight Fully Sharded Data Parallel training framework built with
PyTorch Distributed. This repo is designed to explain the core mechanics behind
FSDP/ZeRO-3 rather than to replace production PyTorch FSDP.

## Resume Summary

MiniFSDP implements parameter sharding, forward all-gather, backward
reduce-scatter, and sharded optimizer-state management. The first strategy wraps
the whole model as one FSDP unit. The second strategy, `minifsdp-layerwise`,
wraps each `nn.Linear` block independently to better reproduce FSDP's
layer-level parameter lifecycle. The repo also includes DDP baselines,
communication analysis, CUDA memory measurement, throughput benchmarking, and
PyTorch Profiler traces.

## Core Idea

DDP keeps a full copy of parameters, gradients, and optimizer states on every
rank. FSDP shards all three:

- parameters are stored as local shards outside computation
- full parameters are materialized with `all_gather` before forward/backward
- gradients are reduced and scattered with `reduce_scatter`
- the optimizer only updates local parameter shards

This means DDP's gradient `all_reduce` can be viewed as:

```text
all_reduce = reduce_scatter + all_gather
```

FSDP keeps the reduce-scatter result and avoids all-gathering gradients for the
optimizer step.

## Project Layout

```text
MiniFSDP/
  toy_fsdp/
    fsdp.py              # MiniFSDP wrapper
    distributed.py       # torchrun/NCCL/Gloo setup
    metrics.py           # memory, throughput, profiler helpers
    communication.py     # communication cost estimates
    model.py             # tiny MLP workload
  benchmark.py           # DDP vs MiniFSDP benchmark
  run_layerwise_experiments.sh
  train_toy_fsdp.py      # simple MiniFSDP training script
  train_ddp.py           # DDP baseline
  benchmark_memory.py    # theoretical memory comparison
  communication_report.py
  profile_minifsdp.py
```

## Run

Install PyTorch first:

```bash
pip install torch
```

Single-process MiniFSDP smoke test:

```bash
python train_toy_fsdp.py
```

Two-process MiniFSDP:

```bash
torchrun --nproc_per_node=2 benchmark.py --strategy minifsdp
```

Layer-wise MiniFSDP:

```bash
torchrun --nproc_per_node=2 benchmark.py --strategy minifsdp-layerwise
```

Two-process DDP baseline:

```bash
torchrun --nproc_per_node=2 benchmark.py --strategy ddp
```

Profile MiniFSDP:

```bash
torchrun --nproc_per_node=2 benchmark.py --strategy minifsdp --profile --trace-dir traces
tensorboard --logdir traces
```

Run the main comparison used in the README:

```bash
bash run_layerwise_experiments.sh
```

Experimental communication prefetch:

```bash
torchrun --nproc_per_node=2 benchmark.py --strategy minifsdp --prefetch
```

Memory and communication reports:

```bash
python benchmark_memory.py
python communication_report.py
```

## Implementation Details

`MiniFSDP` in `toy_fsdp/fsdp.py` does the following:

1. Flattens wrapped module parameters into one flat tensor.
2. Pads and splits the flat tensor into `world_size` shards.
3. Exposes only `flat_param_shard` to the optimizer.
4. Calls `all_gather` before forward to rebuild full parameters.
5. Flattens local full gradients after backward.
6. Calls `reduce_scatter_tensor` to produce the local gradient shard.
7. Falls back to `all_reduce + local slice` when a CPU/Gloo build does not
   support reduce-scatter.
8. Records profiler ranges such as `minifsdp::all_gather_params` and
   `minifsdp::reduce_scatter_grads`.

`LayerWiseMiniFSDP` in `toy_fsdp/layerwise.py` recursively wraps selected leaf
modules, currently `nn.Linear`, so the training step contains multiple smaller
FSDP units instead of one whole-model unit. This creates a more realistic
all-gather/compute/reduce-scatter pattern in profiler traces.

## What to Look for in Profiler

Important trace ranges:

- `minifsdp::all_gather_params`
- `minifsdp::compute_forward`
- `minifsdp::flatten_full_grads`
- `minifsdp::reduce_scatter_grads`
- `minifsdp::optimizer_step`

On CUDA with multiple GPUs, NCCL collectives should appear under these ranges.
The `--prefetch` flag launches an asynchronous all-gather before the next
forward to demonstrate the idea behind communication/computation overlap.

## Expected Analysis

For a model with `P` parameters and `N` ranks:

| Strategy | Parameters / rank | Gradients / rank | Adam states / rank | Backward communication |
| --- | ---: | ---: | ---: | --- |
| DDP | `P` | `P` | `2P` | `all_reduce(P)` |
| MiniFSDP | `P/N` | `P/N` | `2P/N` | `reduce_scatter(P)` |

MiniFSDP saves steady-state training memory but introduces parameter
all-gather before computation. Whole-model MiniFSDP materializes all parameters
at once, while layer-wise MiniFSDP materializes smaller blocks independently,
which is closer to PyTorch FSDP's design.

Example result on 2x RTX 4090, PyTorch 2.1.2+cu121, hidden_dim=8192:

| Strategy | Peak CUDA memory | Throughput | Forward collective | Backward collective |
| --- | ---: | ---: | --- | --- |
| DDP | 1578.52 MiB | 2235.95 samples/s | none | all_reduce(grads) |
| Whole-model MiniFSDP | 1318.15 MiB | 1892.61 samples/s | all_gather(params) | reduce_scatter(grads) |

The result shows the expected tradeoff: MiniFSDP reduces peak memory but loses
throughput because this educational implementation does not fully overlap
communication and computation.

## Limitations

This repo intentionally keeps the implementation small. It does not implement
production features such as DTensor, mixed precision policies, CPU offload,
sharded checkpointing, nested auto-wrap policies, or CUDA stream memory
management.

## GitHub Description

```text
MiniFSDP: an educational PyTorch Distributed implementation of parameter sharding, forward all-gather, backward reduce-scatter, and DDP/FSDP profiling.
```

## References

- PyTorch FSDP docs: https://docs.pytorch.org/docs/stable/fsdp.html
- PyTorch FSDP2 tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- PyTorch FSDP source: https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp
