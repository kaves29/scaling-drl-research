#!/usr/bin/env bash
# Separate owner-approved Exp2 arm allocation. No parent training starts here.
set -euo pipefail
cd "${EXP2_SCHEDULER_MAIN:?}"
test -n "${EXP2_QUEUE:-}" && test -n "${EXP2_APPROVAL:-}"
test -n "${EXP2_APPROVAL_SHA256:-}" && test -n "${EXP2_SCHEDULER_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$EXP2_SCHEDULER_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(sha256sum "$EXP2_APPROVAL" | cut -d' ' -f1)" = "$EXP2_APPROVAL_SHA256"
python -m scripts.exp96_exp2_arms discover --queue "$EXP2_QUEUE" --approval "$EXP2_APPROVAL"
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" \
  python -u -m scripts.exp96_exp2_arms worker --queue "$EXP2_QUEUE" --approval "$EXP2_APPROVAL"
