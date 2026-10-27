#!/usr/bin/env bash
# No training can start before the Python controller revalidates every gate.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  printf 'usage: wrapper PYTHON MANIFEST APPROVAL MANIFEST_SHA APPROVAL_SHA\n' >&2
  exit 2
fi
: "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
# sbatch --chdir binds this to the reviewed checkout/main.
test -f audits/probe_discrepancy/launch.py
exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=2 --gpu-bind=per_task:2 \
  "$1" -m audits.probe_discrepancy.launch worker \
  --checkout "$(git rev-parse --show-toplevel)" \
  --manifest "$2" --approval "$3" --manifest-sha "$4" --approval-sha "$5"
