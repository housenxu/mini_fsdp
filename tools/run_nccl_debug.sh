#!/usr/bin/env bash
set -euo pipefail

mkdir -p evidence

STRATEGY="${STRATEGY:-minifsdp-layerwise}"
MODEL="${MODEL:-mlp}"
HIDDEN_DIM="${HIDDEN_DIM:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-20}"
WARMUP="${WARMUP:-5}"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nproc_per_node=2 benchmark.py \
  --model "${MODEL}" \
  --strategy "${STRATEGY}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  2>&1 | tee "evidence/nccl_${STRATEGY}.log"
