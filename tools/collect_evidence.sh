#!/usr/bin/env bash
set -euo pipefail

mkdir -p evidence

echo "== Environment =="
{
  date
  nvidia-smi
  python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count())"
} | tee evidence/environment.log

echo
echo "== DDP / whole-model MiniFSDP / layer-wise MiniFSDP benchmark =="
bash run_layerwise_experiments.sh 2>&1 | tee evidence/benchmark_summary.log

echo
echo "== NCCL collective log for layer-wise MiniFSDP =="
STRATEGY=minifsdp-layerwise bash tools/run_nccl_debug.sh

echo
echo "Evidence files:"
find evidence -maxdepth 2 -type f -print
