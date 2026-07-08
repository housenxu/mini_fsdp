#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-evidence/nvidia_smi_dmon.log}"
mkdir -p "$(dirname "${OUT}")"

echo "Writing GPU utilization samples to ${OUT}"
echo "Press Ctrl+C to stop this monitor after the benchmark finishes."
nvidia-smi dmon -s pucm -d 1 -o DT > "${OUT}"
