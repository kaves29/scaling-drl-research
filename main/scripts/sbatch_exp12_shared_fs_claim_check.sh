#!/usr/bin/env bash
# CPU-only two-node shared-filesystem claim check; replaces only the mount part of the
# two-node GPU preflight. No GPU, no JAX, no training. Account, partition, nodes, CPUs,
# memory and time are supplied explicitly to sbatch (a CPU partition); this file never submits.
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}" "${EXP12_CLAIM_CHECK_ROOT:?}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/exp12_shared_fs_claim_check.py \
  scripts/sbatch_exp12_shared_fs_claim_check.sh scripts/exp12_dynamic_scheduler.py
test "${SLURM_JOB_NUM_NODES:?}" = 2
[[ "$EXP12_CLAIM_CHECK_ROOT" = /* ]] && [ ! -e "$EXP12_CLAIM_CHECK_ROOT" ] || {
  echo 'claim-check root must be new and absolute' >&2; exit 2; }
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$EXP12_CLAIM_CHECK_ROOT")/" in "$repo/"*)
  echo 'claim-check output must be outside the checkout' >&2; exit 2;; esac
mkdir "$EXP12_CLAIM_CHECK_ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$EXP12_CLAIM_CHECK_ROOT/exit_status.txt"' EXIT
printf '{"commit":"%s","job_id":"%s"}\n' "$EXPECTED_COMMIT" "$SLURM_JOB_ID" \
  > "$EXP12_CLAIM_CHECK_ROOT/submission.json"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export PYTHONDONTWRITEBYTECODE=1
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  python -I scripts/exp12_shared_fs_claim_check.py worker --root "$EXP12_CLAIM_CHECK_ROOT"
python -I scripts/exp12_shared_fs_claim_check.py verify --root "$EXP12_CLAIM_CHECK_ROOT"
