#!/usr/bin/env bash
#SBATCH -J sgdf-chain
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -t 47:30:00
#SBATCH -o %x.%j.out
# Self-resubmitting Slurm chain: launches its own successor, then drains the queue until walltime or drained.
set -uo pipefail

sbatch --dependency=afterany:"${SLURM_JOB_ID}" "$0"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -l 60 >> "${SCRATCH}/sgdf/gpu.${SLURM_JOB_ID}.csv" &
trap 'kill %% 2>/dev/null' EXIT

export HF_HOME="${SCRATCH}/hf" HF_HUB_OFFLINE=1 PILOT_ROOT="${SCRATCH}/sgdf"
EXTRA_ARGS=()
if [ "${EVOLVE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--evolve)
fi
apptainer exec --nv "${SCRATCH}/containers/pytorch-arm64.sif" \
    python -m training.runner --profile vista --queue experiments/queue.yaml "${EXTRA_ARGS[@]}"
code=$?
if [ "${code}" -eq 3 ]; then
    scancel --state=PENDING --name=sgdf-chain -u "${USER}"
fi
