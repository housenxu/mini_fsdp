#!/usr/bin/env bash
set -euo pipefail

HIDDEN_DIM="${HIDDEN_DIM:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-20}"
WARMUP="${WARMUP:-5}"

echo "== DDP =="
torchrun --nproc_per_node=2 benchmark.py \
  --model mlp \
  --strategy ddp \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}"

echo
echo "== Whole-model MiniFSDP =="
torchrun --nproc_per_node=2 benchmark.py \
  --model mlp \
  --strategy minifsdp \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}"

echo
echo "== Layer-wise MiniFSDP =="
torchrun --nproc_per_node=2 benchmark.py \
  --model mlp \
  --strategy minifsdp-layerwise \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}"

echo
echo "== Layer-wise MiniFSDP profiler =="
torchrun --nproc_per_node=2 benchmark.py \
  --model mlp \
  --strategy minifsdp-layerwise \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps 10 \
  --warmup 2 \
  --profile \
  --trace-dir traces_layerwise
