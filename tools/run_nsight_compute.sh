#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. Install Nsight Compute on the GPU machine."
  exit 1
fi

mkdir -p evidence
export MINIFSDP_NVTX=1

STRATEGY="${STRATEGY:-fsdp2}"
MODEL="${MODEL:-transformer}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE_DIM="${INTERMEDIATE_DIM:-2048}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NPROC="${NPROC:-2}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-20}"
OUT="${OUT:-evidence/ncu_${STRATEGY}}"

ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --section SpeedOfLight \
  --section SpeedOfLight_RooflineChart \
  --section LaunchStats \
  --section Occupancy \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --force-overwrite \
  --export "${OUT}" \
  torchrun --standalone --nproc_per_node="${NPROC}" benchmark.py \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --num-heads "${NUM_HEADS}" \
    --intermediate-dim "${INTERMEDIATE_DIM}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --steps 1 \
    --warmup 1

echo "Nsight Compute report written to ${OUT}.ncu-rep"
echo "Increase LAUNCH_SKIP to target steady-state GEMM/SDPA kernels; full replay is intentionally avoided."