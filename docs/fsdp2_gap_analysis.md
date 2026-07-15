# MiniFSDP versus production FSDP2

This document separates implemented functionality from proposed work. It is an
engineering backlog, not a list of completed performance claims.

## Implemented in this project

The original MiniFSDP exposes flat parameter sharding, AllGather,
ReduceScatter, and local optimizer state. The `fsdp2` benchmark path applies
official per-parameter FSDP2 to the same training harness using a one-dimensional
DeviceMesh, DTensor parameters, optional mixed precision and CPU offload,
explicit prefetch controls, DTensor-aware gradient clipping, and DCP helpers.

The workload is now a causal Transformer with learned embeddings, SDPA
attention, SwiGLU MLPs, normalization, tied token/LM-head weights, and an
optional frozen shared embedding. FSDP2 is applied bottom-up at
`TransformerBlock` granularity, and the root group owns embeddings, final norm,
and the tied LM head. The two-rank test covers forward and optimizer-step parity,
DTensor gradients and AdamW state, tied-weight preservation, frozen parameters,
and block/root communication-group accounting.

## Priority 0: expand correctness coverage

The current numerical test uses world size two, FP32, two Transformer blocks,
and one optimizer step. Expand the matrix to world sizes one, two, and four;
non-divisible first dimensions; tied and untied heads; frozen and fully
trainable embeddings; BF16 compute with FP32 reduction; gradient clipping;
dropout with deterministic seeds; and multiple optimizer steps. Compare loss,
full gradients, parameters, and optimizer state against a replicated reference.

Add gradient accumulation using `set_requires_gradient_sync(False)` on nonfinal
microbatches and restore synchronization on the final microbatch. Compare both
`set_reshard_after_backward` choices: retaining full parameters removes the next
AllGather but increases memory, while resharding minimizes memory.

Validate DCP save/load across different world sizes. Save at world size two,
resume at world size one or four, and verify the next optimizer step. Add
checkpoint metadata and atomic publication so a failed partial save is never
selected for recovery.

## Priority 1: memory and throughput

Initialize the Transformer on the meta device, apply FSDP2, then materialize
only local shards with `to_empty`. The current benchmark constructs the full
model on every device before sharding, which is not viable when the unsharded
model itself exceeds one GPU.

Sweep communication grouping policies instead of assuming one block is always
optimal. Compare half-block, one-block, and multiple-block groups. Record peak
allocated and reserved memory, step time, AllGather/ReduceScatter duration,
overlap percentage, and NCCL message-size distribution. Small groups can become
latency-bound; large groups increase live unsharded memory.

Treat explicit prefetch depth as a memory-budgeted scheduling decision. Use
Nsight Systems to verify that NCCL kernels overlap the `transformer::block_N`
ranges and do not merely increase reserved memory. Compare implicit prefetch,
depth one, and depth two with identical warmup and profiler windows.

Combine selective activation checkpointing with FSDP2. FSDP shards model state
but does not eliminate attention and MLP activations. Sequence length can make
the Transformer activation-memory bound even when parameters are fully sharded.

Start with BF16 compute and FP32 ReduceScatter. FP16 requires loss scaling.
FP8 parameters or quantized AllGather should only follow a stable BF16 baseline
and hardware-specific accuracy measurements.

## Priority 2: topology and parallel composition

Add a two-dimensional DeviceMesh for Hybrid Sharded Data Parallel. Shard within
a fast domain and replicate across the other dimension. The parameter placement
becomes `(Replicate(), Shard(0))`, and gradients require ReduceScatter plus a
replica AllReduce. Compare intra-node sharding/cross-node replication with
global FSDP on real multi-node hardware.

Compose Tensor Parallel with FSDP2 on named mesh dimensions. Apply TP to the TP
submesh and FSDP2 to the data-parallel submesh. Verify DTensor placements for
parameters and activations because mixed plain Tensor/DTensor operations have
ambiguous distributed semantics.

Add symmetric-memory or custom collective backends only behind capability
checks. GPU topology, PyTorch and NCCL versions, message size, and network
fabric can change which implementation wins.

## Priority 3: reliability and operations

Add distributed timeout handling, rank-tagged structured logs, monitored
barriers, NCCL flight-recorder capture, and a reproducible worker-failure test.
Checkpoint data-loader state alongside model and optimizer state so elastic
recovery does not repeat or skip samples.

Track allocated memory, reserved memory, inactive split blocks, host-pinned
memory, and checkpoint staging memory. Peak `memory_allocated` alone does not
explain fragmentation or CPU-offload pressure.

Run long stability tests with periodic finite-loss, finite-gradient,
gradient-norm, and parameter-checksum validation. A short profiler benchmark
cannot establish numerical stability or recovery correctness.

## Interview boundary

The defensible claim is that the project now integrates official FSDP2 and
DTensor into the MiniFSDP benchmark and uses a Transformer-block policy. It
tests embeddings, attention, SwiGLU, norms, tied weights, frozen parameters,
sharded gradients, and optimizer state. Meta initialization, gradient
accumulation, multi-node HSDP, TP composition, elastic recovery, and measured
GPU speedups remain proposed work until their implementations and experiment
evidence exist.
