#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${STRATEGY:-minifsdp-layerwise}"
HIDDEN_DIM="${HIDDEN_DIM:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10}"
WARMUP="${WARMUP:-2}"
TRACE_DIR="${TRACE_DIR:-traces_${STRATEGY}}"

torchrun --nproc_per_node=2 benchmark.py \
  --strategy "${STRATEGY}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --profile \
  --trace-dir "${TRACE_DIR}"

echo "Profiler traces written to ${TRACE_DIR}"
echo "Open with: tensorboard --logdir ${TRACE_DIR}"
