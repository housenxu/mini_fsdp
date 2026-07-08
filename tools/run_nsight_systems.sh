#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. Install Nsight Systems or use PyTorch Profiler instead."
  exit 1
fi

mkdir -p evidence

STRATEGY="${STRATEGY:-minifsdp-layerwise}"
HIDDEN_DIM="${HIDDEN_DIM:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10}"
WARMUP="${WARMUP:-2}"
OUT="evidence/nsys_${STRATEGY}"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=none \
  --force-overwrite=true \
  --output="${OUT}" \
  torchrun --nproc_per_node=2 benchmark.py \
    --strategy "${STRATEGY}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --batch-size "${BATCH_SIZE}" \
    --steps "${STEPS}" \
    --warmup "${WARMUP}"

echo "Nsight Systems report written to ${OUT}.nsys-rep"
