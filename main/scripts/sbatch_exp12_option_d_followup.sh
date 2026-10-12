#!/usr/bin/env bash
# Separate, UNAUTHORIZED until specifically approved: five extra null targets.
#SBATCH --job-name=exp12_option_d_extra_targets
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${OPTION_D_ROOT:?}" "${FIRST_FIVE_JOB_ROOT:?}" "${SLURM_SUBMIT_DIR:?}"
cd "$SLURM_SUBMIT_DIR"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout scripts/sci_investigation/paired_probe_budget.py \
  scripts/sci_investigation/combine_option_d.py scripts/sbatch_exp12_option_d_followup.sh
test -z "$(git status --porcelain --untracked-files=all)"
[[ "$OPTION_D_ROOT" = /* ]] && [ ! -e "$OPTION_D_ROOT" ]
repo=$(git rev-parse --show-toplevel)
case "$(realpath -m "$OPTION_D_ROOT")/" in "$repo/"*)
  echo 'Option D output must be outside the checkout' >&2; exit 2;; esac
test "$(cat "$FIRST_FIVE_JOB_ROOT/exit_status.txt")" = 0
test "$(cat "$FIRST_FIVE_JOB_ROOT/null_991.exit_status.txt")" = 0
sha256sum -c "$FIRST_FIVE_JOB_ROOT/SHA256SUMS" > /dev/null
python - "$FIRST_FIVE_JOB_ROOT/null_991" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
assert json.loads((root / 'summary.json').read_text())['complete'] is True
assert len(list(root.glob('pair_*.json'))) == 100
assert json.loads((root / 'provenance.json').read_text())['code'] == {
    'commit': '18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af', 'dirty': False}
PY
mkdir "$OPTION_D_ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$OPTION_D_ROOT/exit_status.txt"; find "$OPTION_D_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > "$OPTION_D_ROOT/SHA256SUMS"' EXIT
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
timeout --signal=TERM --kill-after=30s 600 \
  python -u scripts/sci_investigation/paired_probe_budget.py \
    --mode null --seed 991 --pairs 1 --round-start 0 --rounds 5 \
    --pooled-threshold 0.0009080017606417335 --max-elapsed-seconds 540 \
    --expected-commit "$EXPECTED_COMMIT" --out-dir "$OPTION_D_ROOT/reference_pair" \
    > "$OPTION_D_ROOT/reference_pair.log" 2>&1
python - "$FIRST_FIVE_JOB_ROOT/null_991" "$OPTION_D_ROOT/reference_pair" <<'PY'
import json, sys
import numpy as np
from pathlib import Path
old, new = (Path(p) for p in sys.argv[1:])
a, b = (json.loads((p / 'pair_000.json').read_text()) for p in (old, new))
for name in ('fresh', 'current'):
    for rnd in range(5):
        for field in ('losses_1000', 'losses_2000', 'mse_1000', 'mse_2000'):
            x = np.asarray(a['rounds'][rnd][name][field], np.float32)
            y = np.asarray(b['rounds'][rnd][name][field], np.float32)
            if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
                raise ValueError(f'first-five reference pair differs: {name} round {rnd} {field}')
for field in ('null_1000', 'null_2000'):
    if a[field]['ci_low'] != b[field]['ci_low'] or a[field]['ci_high'] != b[field]['ci_high']:
        raise ValueError(f'first-five reference pair differs: {field} bootstrap')
old_runtime = json.loads((old / 'provenance.json').read_text())['runtime']
new_runtime = json.loads((new / 'provenance.json').read_text())['runtime']
for key in ('jax', 'jaxlib', 'platform', 'device_kind', 'backend_platform_version',
            'jax_cuda12_plugin', 'jax_cuda12_pjrt', 'matmul_precision'):
    if old_runtime.get(key) != new_runtime.get(key):
        raise ValueError(f'first-five reference runtime differs: {key}')
PY
timeout --signal=TERM --kill-after=30s 12000 \
  python -u scripts/sci_investigation/paired_probe_budget.py \
    --mode null --seed 991 --pairs 100 --round-start 5 --rounds 5 \
    --max-elapsed-seconds 11940 --expected-commit "$EXPECTED_COMMIT" \
    --out-dir "$OPTION_D_ROOT/extra_five" \
    > "$OPTION_D_ROOT/extra_five.log" 2>&1
python scripts/sci_investigation/combine_option_d.py \
  --first-five-root "$FIRST_FIVE_JOB_ROOT/null_991" \
  --extra-five-root "$OPTION_D_ROOT/extra_five" \
  --first-commit 18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af \
  --extra-commit "$EXPECTED_COMMIT" --out "$OPTION_D_ROOT/option_d_summary.json"
