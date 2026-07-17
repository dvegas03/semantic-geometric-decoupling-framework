#!/usr/bin/env bash
# Simulates Vista's 48h walltime chain at 30 min scale, restarting the runner until drained.
set -uo pipefail

PROFILE="${1:-local}"
QUEUE="${2:-experiments/queue.yaml}"
SLICE="${SLICE:-30m}"
EXIT_DRAINED=3
GPU_LOG="${GPU_LOG:-data/pilot/gpu.csv}"
EXTRA_ARGS=()
if [ "${EVOLVE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--evolve)
fi

mkdir -p "$(dirname "${GPU_LOG}")"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -l 60 >> "${GPU_LOG}" &
GPU_SAMPLER_PID=$!
trap 'kill "${GPU_SAMPLER_PID}" 2>/dev/null' EXIT

while true; do
    timeout --signal=SIGTERM --kill-after=90 "${SLICE}" \
        python -m training.runner --profile "${PROFILE}" --queue "${QUEUE}" "${EXTRA_ARGS[@]}"
    code=$?
    if [ "${code}" -eq "${EXIT_DRAINED}" ]; then
        echo "[chain] queue drained — stopping."
        break
    fi
    echo "[chain] runner exited ${code} (124=walltime SIGTERM) — restarting in 5 s."
    sleep 5
done
