# Reproducing CPU validation, including real HumanoidBench

No production environment or dependency lock was changed. The original `/workspace/scaling-drl-research/.venv` remains unchanged: Python3.12.14, JAX/JAXLIB0.4.34, NumPy1.26.4, Flax0.8.4, Optax0.2.3, Orbax0.5.3, MuJoCo3.6.0, dm-control1.0.38 and Gymnasium1.0.0a2. `uv --no-cache pip check` reports all117 installed packages compatible. This is not Delta's Python3.12.13/CUDA environment.

The machine has Mesa EGL **llvmpipe (LLVM19.1.7,256 bits)** software rendering. An isolated overlay adds only editable `humanoid-bench@cb1189039151c8aadaaa987b442da54383c87fab`, the exact pin already specified by `scripts/install_humanoid_bench.sh`. A `.pth` file makes the unchanged base environment's packages visible. No shared packages are upgraded/downgraded. This avoids mutating a running validation environment or a historical checkout.

HumanoidBench's declared upstream lock differs from this project's reviewed stack: e.g. upstream MuJoCo3.1.6/dm-control1.0.20/Gymnasium0.29.1 versus the versions above. The existing `experiments/exp12/envs.py` adapter uses the current dm-control index and stubs unused torch-only hierarchical policy wrappers. The upstream package is intentionally installed `--no-deps`, as the repository installer specifies. An overlay-only `uv pip check` lists thirteen unresolved upstream requirements because it does not treat `.pth` inheritance as an installed overlay distribution; an independent Python metadata audit also records the actual inherited versions and upstream lock differences. **Do not describe this as a fully upstream-dependency-compatible environment.** The successful coverage concerns the actual H1 reach/run paths and project compatibility adapter, not unrelated hierarchical/MJX tasks or CUDA.

Reproduce in a fresh workspace location with the pinned base environment already available:

```bash
set -euo pipefail
cpu_python=/absolute/path/to/pinned/.venv/bin/python
cpu_overlay=/absolute/new/path/hb-venv
hb_source=/absolute/new/path/humanoid-bench
test_output=/absolute/new/path/cpu-validation
git clone --no-checkout --filter=blob:none https://github.com/carlosferrazza/humanoid-bench.git "$hb_source"
git -C "$hb_source" checkout --detach cb1189039151c8aadaaa987b442da54383c87fab
uv --no-cache venv --python "$cpu_python" "$cpu_overlay"
reference_site=$("$cpu_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
overlay_site=$("$cpu_overlay/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$reference_site" > "$overlay_site/research_reference.pth"
uv --no-cache pip install --python "$cpu_overlay/bin/python" --no-deps --no-build-isolation --editable "$hb_source"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME=/absolute/writable/path/cpu-cache
export MPLCONFIGDIR=/absolute/writable/path/matplotlib
cd /absolute/path/to/reviewed/candidate/main
"$cpu_overlay/bin/python" -B scripts/run_exp12_cpu_validation.py --jobs 2 --out-dir "$test_output"
"$cpu_overlay/bin/python" -B -u -m tests.exp12_break_checks
"$cpu_overlay/bin/python" -B scripts/check_exp12_cpu_environments.py
"$cpu_overlay/bin/python" -B scripts/methodology_cpu_reference.py
```

EGL software rendering must actually work on the target CPU machine; do not infer that from an installed library. If it is unavailable, the repository's CPU OSMesa route requires the corresponding system renderer. That is an environment prerequisite, not a scientific setting change or a reason to count skipped tasks as validated.

If interrupted, rerun the **same** module-validation command with `--resume` and the same output path. Completed modules are reused only when the source/configuration bytes, commit, interpreter, selected dependency versions, HumanoidBench source pin, CPU environment and intact owned log match. Failed/incomplete attempts are preserved and rerun. Do not reuse receipts across a changed tree; use a new directory. Module logs, receipts, caches, installed packages, the dependency checkout and generated evidence remain outside Git.

`check_exp12_cpu_environments.py` checks all13 tasks×five confirmatory seeds=65 cells using canonical environment wrappers/budgets. It constructs/reset/steps training and evaluation envs without building or training an agent; it verifies action-space RNG restoration and an exact next transition against a copied reference observation/reward/termination/truncation tuple. This is simulator engineering coverage, not full-budget learning or scientific qualification. The full foundations module is also rerun when HB is present, since it conditionally adds HB subtests even beyond the two explicit dependency skips.

Local files for this verification are under `/workspace/scratch/exp12-integrated/`; the overlay is `hb-venv`, the pinned dependency clone is `humanoid-bench`, and `isolated-cpu/provenance.json`, `modules.json`, per-attempt receipts/logs and `summary.json` carry the complete test provenance. The interrupted monolithic `full-cpu.log` has no completion summary and is **not** counted as a completed suite. No OOM event is recorded in the inspected cgroup; the interruption's exact cause is unverified.
