# Block B A100 Exp1 test failures: root causes and the minimal pilot gate (2026-10-10)

The failures come from Block B job 22706349 at `b4a90cb` ([report.txt]: `tests_default` and `tests_deterministic`,
identical in both modes). Code is compared at `b4a90cb`, the approved baseline `9d6a82d` and `dcb7491`. The raw A100
tracebacks were analysed by Codex in `docs/exp12_engineering_followup.md` @ `9d6a82d` (table rows at lines 32–34);
that is the only archived source of the exception text. No production code is changed here and nothing is submitted.

## Summary

| Test group (A100 result at `b4a90cb`) | Cause | Class | Fixed in | Blocks the pilot? |
|---|---|---|---|---|
| `KillMatrixTest` (7 subtests ERROR), `KillAndResumeEntryPointTest` (3 subtests ERROR) | **Test-harness comparison defect.** Training runs in CPU subprocesses (`_subprocess` sets `JAX_PLATFORMS=cpu`: `tests/test_exp12_foundations.py:49-51`, `tests/test_exp12_fork.py:494-495`). The GPU test process then compared their checkpoints with `state.load_agent_tree`, which at `b4a90cb` restored Orbax arrays onto their saved device (`TFRT_CPU_0`). That raised `SingleDeviceSharding ... TFRT_CPU_0 ... not found` before any assertion, hence ERROR rather than FAIL. | Test isolation / environment | `781dc6c` (in `9d6a82d`): `load_agent_tree` restores every array as NumPy (`experiments/exp12/state.py:111-128`). Training restore (`agent.load_checkpoint`) is untouched. | **No** |
| `ForkEndToEndTest.test_check1` (FAIL) | **Test stricter than the approved rule.** The assertion `injected pre_vs_after max_abs_dq == 0.0` (`tests/test_exp12_fork.py:123` @ `b4a90cb`) failed with 2.98e-7, which is inside amendment (m)'s 64-eps injected tolerance. Every assertion before it passed on the A100: identity- and injected-arm Check 1 pass, and all identity pairs (pre/after/control) have `max_abs_dq = max_abs_d_dq_da = 0.0` (lines 120–122). | Hardware-dependent numerics in an over-strict test | `9d6a82d`: the test now asserts the existing 64-eps rule (lead-approved) and adds an explicit check of the control's restore. Production threshold unchanged. | **No** |
| `Angle1ParityTest` (ERROR) | `GlobalHydra is already initialized` left over from an earlier test | Test isolation | `67d420f` (in `9d6a82d`) | No |

**No production correctness defect** is implicated by any of the four. `dcb7491` adds strict bitwise comparison (dtype
and bytes; NaN not equal to NaN) in `state_differences`, `fork.check1` and `compare_identity_fork.py`. That tightens
these tests and does not relax them.

## CPU reproduction (done here)
The comparison defect and its fix were reproduced on CPU with Codex's regression
`tests.test_exp12_runtime.RestoreRuntimeTest.test_comparison_does_not_require_the_saved_device`, which mocks the
saved device as unavailable:
- with the `9d6a82d` loader the test **passes**;
- with `b4a90cb`'s `load_agent_tree` swapped in verbatim it **errors** with "saved device is unavailable".

That is the same failure path as the A100 errors.

## What none of these tests show: GPU resume across processes
- **Kill tests run training on CPU.** Both kill tests force CPU in their training subprocesses, so even a passing A100
  rerun does not qualify kill-and-resume on the GPU.
- **Block B's A100 Check 1 evidence is in-process only.** `ForkEndToEndTest` runs `exp1.run`/`exp2_arm.run`
  in-process, so its bit-exact control/identity pairs come from one process, at tiny scale, with a forced trigger.
  Production's control restart after a fork is also in-process (`exp1.py`: `ForkNow` → `build(fork state)`), so
  that evidence matches the pilot's path.
- **The real gap:** cross-process resume (a job restarted after a timeout or preemption) and cross-process identity
  (Exp2 arms) have never run on an A100.

`scripts/sci_investigation/gpu_resume_probe.py` (added here) closes the first gap with no code change:
- it runs the kill-test scenario (tiny hopper-hop, crash at interaction steps 5, 60 and 95, relaunch) without
  overriding the backend;
- it includes a repeat of the uninterrupted run, which separates GPU run-to-run nondeterminism from a resume defect.

On CPU it passes: identical repeat reference, three crash/resume cases with `state_differences == []`, about 3 min.

## Pilot gate (dog-run D4W1536, seed 102, `run_role=dev`)

| Group | Needed before the pilot? | Path exercised by the pilot? | Minimal A100 test | Pass condition | Source fix? |
|---|---|---|---|---|---|
| ForkEndToEndTest | Recommended, not strictly needed. The pilot's own Check 1 control guard stops a bad restore safely (`fork.validate_control_restore` → `Check1Failed`). | Yes, if it triggers: in-process fork, control restart, Check 1 | `python -m unittest -v tests.test_exp12_fork.ForkEndToEndTest` | `OK`; `check1_control` and `check1_identity` pairs 0.0/0.0/0.0; injected ≤ 64 eps units; `matmul_precision == "highest"` | None |
| KillMatrixTest | No (harness fix only; training is on CPU) | No | `python -m unittest -v tests.test_exp12_fork.KillMatrixTest` (closes the historical record) | `OK` | None |
| KillAndResumeEntryPointTest | No (same reason) | No | `python -m unittest -v tests.test_exp12_foundations.KillAndResumeEntryPointTest` | `OK` | None |
| GPU cross-process resume (new probe) | Recommended. It matters only if the pilot is interrupted; required before the grid. | Only on a timeout or restart | `python scripts/sci_investigation/gpu_resume_probe.py --out <new dir>` | exit 0; `reference_repeat_differences == []`; each case: crash exit 3, no DONE, resume exit 0, DONE, `differences == []` | None expected |

**Reading the probe.** If `reference_repeat_differences` is non-empty, default-ops GPU training is not run-to-run
deterministic. Resume can then not be bit-exact (an Exp1 resume would still be a valid but different trajectory),
and cross-process identity forks for Exp2 would face the same limit. That would be a finding for your decision, not
something to fix in code.

## Minimal A100 validation job: NOT AUTHORIZED FOR EXECUTION
Run it on the pilot commit (the `integration/exp12` HEAD once published, otherwise `9d6a82d` plus this branch's probe
script), from a clean checkout. One A100, 1 h. Expected about 30 min: the whole Block B suite took 25 min per mode.
```bash
cd /work/hdd/biqc/skaveti1/exp12_pilot/main && mkdir -p logs && export EXPECTED_COMMIT=<pilot commit>
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=exp12_a100_exp1_gate
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
cd "$SLURM_SUBMIT_DIR"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout tests/test_exp12_fork.py tests/test_exp12_foundations.py
OUT="$PWD/logs/a100_exp1_gate_$SLURM_JOB_ID"; mkdir -p "$OUT"
module reset; source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh; conda activate scaling-drl-py31213
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 WANDB_MODE=disabled MUJOCO_GL=disable
export EXP12_JAX_CACHE_DIR="$OUT/jax_cache"
python -c "import jax; d=jax.devices(); print(d); assert d[0].device_kind=='NVIDIA A100-SXM4-40GB'" | tee "$OUT/devices.txt"
set +e
python -m unittest -v tests.test_exp12_fork.ForkEndToEndTest tests.test_exp12_fork.KillMatrixTest \
  tests.test_exp12_foundations.KillAndResumeEntryPointTest > "$OUT/tests.log" 2>&1; t=$?
python scripts/sci_investigation/gpu_resume_probe.py --out "$OUT/resume_probe" > "$OUT/resume_probe.log" 2>&1; p=$?
printf "tests\t%s\nresume_probe\t%s\n" "$t" "$p" | tee "$OUT/status.tsv"
exit $(( t || p ))
EOF
```
**Return:**
- `logs/a100_exp1_gate_<jobid>/` containing `status.tsv`, `tests.log`, `resume_probe/gpu_resume_probe.json`,
  `resume_probe.log` and `devices.txt`;
- `logs/exp12_a100_exp1_gate_<jobid>.out`.

**Gate decision:**
- both statuses 0: the historical A100 red is closed and GPU resume is qualified, so the pilot can proceed;
- any test failure: send `tests.log` before the pilot;
- `reference_repeat_differences` non-empty: a nondeterminism finding for your decision. The pilot can still run, but
  resume and Exp2 identity expectations change.

## For Codex
No source correction is needed for these failures. One suggestion only: the kill tests' docstrings claim
GPU-relevant resume coverage while forcing CPU. A one-line note, or a GPU-backend variant like this probe, would stop
a future A100 pass being over-read.

[report.txt]: logs/blockB_22706349/report.txt
