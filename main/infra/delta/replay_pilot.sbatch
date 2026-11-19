#!/bin/bash
#SBATCH --job-name=echocritic-exp2-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: sbatch [site flags] replay_pilot.sbatch REPO_ROOT PYTHON PARENT_MANIFEST RESULTS" >&2
    exit 2
fi
cd "$1/main"
exec "$2" -m infra.delta.replay_pilot --manifest "$3" --results "$4"
