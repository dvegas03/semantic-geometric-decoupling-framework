#!/usr/bin/env bash
# Resubmits the Vista chain job if it isn't already queued/running (install: crontab -e -> */30 * * * * bash slurm/watchdog.sh).
set -uo pipefail
if ! squeue -u "${USER}" --name=sgdf-chain -h | grep -q .; then
    job_id=$(sbatch --parsable "$HOME/sgdf/slurm/chain_job.sh")
    printf '%s resubmitted chain as %s\n' "$(date -Is)" "${job_id}" \
        >> "$HOME/sgdf/incidents.log"
fi
