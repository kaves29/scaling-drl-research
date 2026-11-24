#!/bin/bash
#SBATCH --job-name=echocritic-exp3-pilots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%3
set -euo pipefail
if [[ $# != 5 && $# != 7 && ( $# != 8 || ${8:-} != both ) ]]; then
    echo "Usage: sbatch [site flags] exp3_pilot.sbatch REPO PYTHON MANIFEST RESULTS SOURCE_SHA [EXP2_RESULTS EXP2_SHA [both]]" >&2
    exit 2
fi
repo=$1 python=$2 manifest=$3 results=$4 source_sha=$5
cd "$repo/main"
if [[ "$(git rev-parse HEAD)" != "$source_sha" ]] || ! git diff --quiet HEAD --; then
    echo "Unexpected or dirty source checkout" >&2
    exit 2
fi
gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
echo "GPU inventory: $gpu"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Expected one allocated device" >&2
    exit 2
fi
index=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
if (( index < 0 || index > 11 )); then
    echo "Task index must be 0 through 11" >&2
    exit 2
fi
envs=(dog-run humanoid-run)
refs=(small large)
environment=${envs[$((index / 6))]}
seed=$(((index / 2) % 3 + 1))
reference=${refs[$((index % 2))]}

export JAX_PLATFORMS=cuda MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
mkdir -p "$results"
run_phase() (
    phase=$1
    shift
    output="$results/${environment}_seed${seed}_${reference}_${phase}.json"
    exec 9>"$output.lock"
    flock -n 9
    if [[ -e "$output" ]]; then
        echo "Result already exists: $output" >&2
        exit 2
    fi
    telemetry="$output.$SLURM_JOB_ID.gpu.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
        --format=csv,noheader,nounits -l 2 > "$telemetry" 2>&1 &
    monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' EXIT
    /usr/bin/time -v "$python" -B -m experiments.exp12.exp3_pilot \
        --manifest "$manifest" --env "$environment" --seed "$seed" \
        --reference "$reference" --output "$output" --require-a100-40 "$@"
)
if [[ $# -ge 7 ]]; then
    exp2_case="$6/${environment}_seed${seed}_reference_${reference}"
    if [[ ! -f "$exp2_case/state/LATEST" ]]; then
        echo "Exp2 checkpoint missing: $exp2_case/state/LATEST" >&2
        exit 2
    fi
fi
if [[ $# == 5 || $# == 8 ]]; then
    run_phase exp1-screen
fi
if [[ $# -ge 7 ]]; then
    run_phase exp2-feedback --exp2-root "$exp2_case" --exp2-commit "$7" --pilots 4
fi
