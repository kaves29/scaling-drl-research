#!/bin/bash
# Installs HumanoidBench for Exp 1/2 into the active (pinned) environment without
# touching any pinned dependency: the pinned commit, --no-deps, editable (its
# dmc_deps/ and assets are not packaged, so a non-editable install is incomplete).
# experiments/exp12/envs.py imports it against the pinned mujoco/dm_control.
# Needs MUJOCO_GL=egl on GPU nodes (osmesa on CPU-only machines) at run time.
set -euo pipefail
DEST="${1:?usage: install_humanoid_bench.sh <absolute clone dir>}"
COMMIT=cb1189039151c8aadaaa987b442da54383c87fab
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/carlosferrazza/humanoid-bench.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
pip install --no-deps -e "$DEST"
cd "$(dirname "$0")/.."
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
from experiments.exp12.envs import make_humanoid_bench_env
for name in ("h1-reach-v0", "h1-run-v0"):
    env = make_humanoid_bench_env(name, 0)
    print(name, env.observation_space.shape, env.action_space.shape)
PY
