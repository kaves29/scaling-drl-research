#!/bin/bash
#SBATCH --job-name=echocritic-exp2-heldout
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=echocritic-exp2-heldout-%j.out
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: sbatch heldout_pilot_2h.sbatch REPO_ROOT PYTHON PARENT_MANIFEST RESULTS fresh|resume" >&2
    exit 2
fi

repo_root=$1
python=$2
parent_manifest=$3
results=$4
mode=$5
if [[ "$mode" != fresh && "$mode" != resume ]]; then
    echo "Mode must be fresh or resume" >&2
    exit 2
fi

cd "$repo_root/main"
flags=(--heldout --workers-per-gpu 6 --checkpoint-count 15)
if [[ "$mode" == fresh ]]; then
    flags+=(--fresh)
fi
exec "$python" -m infra.delta.replay_pilot \
    --manifest "$parent_manifest" --results "$results" "${flags[@]}"
