#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${STRATEGY:-fsdp2}"
MODEL="${MODEL:-transformer}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE_DIM="${INTERMEDIATE_DIM:-2048}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-6}"
WARMUP="${WARMUP:-3}"
NPROC="${NPROC:-2}"
TRACE_DIR="${TRACE_DIR:-traces_${STRATEGY}}"
PEAK_TFLOPS="${PEAK_TFLOPS:-}"
PRACTICAL_PEAK_JSON="${PRACTICAL_PEAK_JSON:-benchmark_results/gemm_peak.json}"

EXTRA_ARGS=(--practical-peak-json "${PRACTICAL_PEAK_JSON}")
if [[ -n "${PEAK_TFLOPS}" ]]; then
  EXTRA_ARGS+=(--peak-tflops "${PEAK_TFLOPS}")
fi

torchrun --standalone --nproc_per_node="${NPROC}" benchmark.py \
  --model "${MODEL}" \
  --strategy "${STRATEGY}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --intermediate-dim "${INTERMEDIATE_DIM}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --warmup "${WARMUP}" \
  --profile \
  --trace-dir "${TRACE_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "Profiler traces and per-rank profiler_metrics.json written to ${TRACE_DIR}"
echo "Open traces with: tensorboard --logdir ${TRACE_DIR}"