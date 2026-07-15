# FSDP2 and DTensor integration

The project keeps the hand-written flat-buffer `MiniFSDP` for explaining the
mechanics of AllGather, ReduceScatter, padding, and local optimizer state. The
`fsdp2` strategy applies PyTorch's official `fully_shard` implementation to the
same Transformer, language-model batch, optimizer, profiler, and benchmark.
This is an A/B path inside one training harness rather than a disconnected API
example.

## Transformer workload

`TinyTransformerLM` contains learned token and position embeddings, causal
scaled-dot-product attention, a SwiGLU MLP, pre-norm residual blocks, final
normalization, and a language-model head. The LM head can share the token
embedding parameter, and that shared parameter can be frozen. Those cases are
important because a realistic sharding implementation must preserve aliases,
per-parameter `requires_grad`, original fully-qualified names, and optimizer
state semantics.

```text
input IDs
  -> token embedding + position embedding       root FSDP2 group
  -> TransformerBlock 0                         block group 0
  -> TransformerBlock 1                         block group 1
  -> ...
  -> final norm + tied LM head                  root FSDP2 group
  -> next-token cross entropy
```

Each `TransformerBlock` groups its two norms, QKV/output projections, and
SwiGLU projections into one FSDP2 communication group. `fully_shard` is applied
to blocks first and the root model last. The root therefore claims only the
parameters not already managed by a block: embeddings, final norm, and LM
head. With tied embeddings, the token table and LM head refer to one parameter
and are sharded once.

This block boundary is more realistic than wrapping every `nn.Linear`. A group
that is too small pays collective launch latency repeatedly, while a group that
is too large keeps more unsharded parameter memory live. Transformer blocks are
a useful starting point because their compute can overlap the next block's
parameter AllGather.

## MiniFSDP versus FSDP2 state

The hand-written implementation concatenates every managed parameter into one
flat parameter. Its optimizer sees only `flat_param_shard`, and its training
loop explicitly calls `reduce_scatter_grad()` and `reshard()`.

The FSDP2 path keeps original parameter boundaries. Outside forward and
backward, each managed parameter is a `DTensor` with `Shard(0)` placement.
FSDP2 hooks automatically AllGather parameters, free unsharded storage
according to `reshard_after_forward`, and ReduceScatter gradients. The normal
optimizer consumes sharded DTensor parameters and creates sharded optimizer
state.

```text
MiniFSDP:  original params -> one flat buffer -> one local flat shard
FSDP2:     each original param -> DTensor(global shape, local Shard(0))
```

`FSDP2Runtime.parameter_layouts()` reports each unique parameter's global
shape, local shape, and placement. `communication_group_count` includes every
Transformer block group plus the root remainder group when it owns parameters.

## Correctness test

The two-rank CPU/Gloo test builds a two-block Transformer with tied and frozen
token embeddings. It verifies original state-dict keys, alias preservation,
`Shard(0)` placements, block/root group counts, local/global parameter counts,
forward parity, averaged gradient semantics, absence of gradients on the frozen
shared parameter, AdamW DTensor optimizer state, and full-parameter parity
against a replicated reference after one optimizer step.

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  tests/test_fsdp2_integration.py
```

## Benchmark and profiler

The default benchmark model is now the Transformer. Compare whole-model
MiniFSDP with block-wise FSDP2 using identical dimensions and batches:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  benchmark.py --model transformer --strategy minifsdp \
  --hidden-dim 512 --intermediate-dim 2048 --num-layers 4 \
  --num-heads 8 --seq-len 128 --vocab-size 8192

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  benchmark.py --model transformer --strategy fsdp2 \
  --hidden-dim 512 --intermediate-dim 2048 --num-layers 4 \
  --num-heads 8 --seq-len 128 --vocab-size 8192
```

The result reports tokens/s, logical/global parameter elements, local parameter
elements, DTensor parameter count, block groups, total communication groups,
peak CUDA memory, and an approximate communication volume. Add
`--profile --trace-dir traces/fsdp2` to inspect block compute and NCCL kernels.
The model emits `transformer::block_N` profiler ranges, so the timeline can show
whether the next AllGather overlaps attention or MLP compute in the current
block.

Mixed precision keeps sharded master parameters and optimizer state in FP32,
uses BF16 for unsharded computation, and uses FP32 for ReduceScatter:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  benchmark.py --model transformer --strategy fsdp2 \
  --fsdp2-param-dtype bf16 --fsdp2-reduce-dtype fp32 \
  --clip-grad-norm 1.0
```

The frozen/tied-root-group case can be exercised directly:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  benchmark.py --model transformer --strategy fsdp2 \
  --tie-embeddings --freeze-token-embedding
```

Explicit prefetching trades additional live unsharded memory for a longer
communication lead:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
  benchmark.py --model transformer --strategy fsdp2 \
  --fsdp2-prefetch-depth 1 --profile \
  --trace-dir traces/fsdp2-prefetch
```

Do not claim a speedup until the GPU benchmark and profiler show one. The
important evidence is not only total tokens/s but peak allocated/reserved
memory, AllGather and ReduceScatter duration, NCCL message sizes, and the
percentage of communication hidden under block compute.

## Unsupported comparison kept explicit

`minifsdp-layerwise` currently returns parameters only from its wrapped leaf
modules. It is correct for the original MLP where every parameter belongs to a
wrapped Linear, but it would omit Transformer root embeddings and norms if it
wrapped only blocks. The benchmark rejects that combination instead of showing
an invalid performance number. Whole-model `minifsdp` remains available as the
hand-written comparison.

The flat MiniFSDP also does not preserve independent optimizer semantics for a
frozen slice inside its trainable flat parameter. The benchmark therefore
requires `fsdp2` for the frozen-embedding experiment. This is a concrete reason
why per-parameter DTensor sharding is easier to compose than one exposed flat
parameter.

## Checkpointing

`toy_fsdp/checkpoint.py` exposes a normal full state dict for interoperability
and DCP save/load helpers for sharded model and optimizer state. All ranks must
participate. A production validation still needs to save with one world size,
load with another, and verify the next optimizer step.

## Remaining production gaps

The realistic Transformer and block policy are implemented. Remaining work is
meta-device initialization, gradient accumulation with selective gradient
synchronization, activation checkpointing, 2D HSDP, TP plus FSDP composition,
async DCP checkpointing, failure recovery, and topology-aware multi-node
experiments.
