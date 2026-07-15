#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. Install Nsight Systems or use PyTorch Profiler instead."
  exit 1
fi

mkdir -p evidence
export MINIFSDP_NVTX=1

STRATEGY="${STRATEGY:-fsdp2}"
MODEL="${MODEL:-transformer}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE_DIM="${INTERMEDIATE_DIM:-2048}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5}"
WARMUP="${WARMUP:-3}"
NPROC="${NPROC:-2}"
CAPTURE_START="${CAPTURE_START:-1}"
CAPTURE_STEPS="${CAPTURE_STEPS:-2}"
GPU_METRICS="${GPU_METRICS:-0}"
OUT="${OUT:-evidence/nsys_${STRATEGY}}"

GPU_METRICS_ARGS=()
if [[ "${GPU_METRICS}" == "1" ]]; then
  GPU_METRICS_ARGS+=(--gpu-metrics-device=all --gpu-metrics-frequency=10000)
fi

nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${OUT}" \
  "${GPU_METRICS_ARGS[@]}" \
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
    --nsight \
    --nsight-start-step "${CAPTURE_START}" \
    --nsight-num-steps "${CAPTURE_STEPS}"

echo "Nsight Systems report written to ${OUT}.nsys-rep"
echo "Set GPU_METRICS=1 when permissions allow system-wide SM/Tensor/DRAM samples."