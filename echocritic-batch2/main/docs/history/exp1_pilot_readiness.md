# Exp1: readiness for one full-length A100 pilot (2026-10-10)

Read-only audit of the approved baseline `9d6a82d4aef81299901177865e383640439cde2a` (source paths below are at that
commit) plus the scientific investigation (`claude/scientific-validation-investigation` @ `cb33733`). No setting,
budget, seed rule or methodology is changed. **No command here is authorized for execution.**

## 1. What one Exp1 parent actually depends on (verified in source)
- **Entry point** `experiments/exp1.py:run`:
  - composes `configs/base_exp12.yaml` with the run's overrides;
  - trains to `num_interaction_steps`;
  - runs the 20 probe checks plus the fresh check at the first update (`RunProbes`);
  - saves the routine state at every `checkpoint_interval` (the grid uses N/20, `generate_manifest.py:add_exp12_grid`)
    and writes the ledger (`ledger.write_run`);
  - writes `DONE` at the end;
  - resumes from `state/LATEST`.
- **Scaled parents also fork** (`fork.architectures` = D4W1024 and D4W1536). At `f*_run` they write the complete fork
  state (`fork.write_fork`), restart from it as the CONTROL arm, check bit-exact restore of Q and dQ/da on the panel
  (`fork.validate_control_restore`, which raises `Check1Failed` on any difference), then continue to
  max(N, fork + 0.25N) with post-fork evaluations.
- **Not read by an Exp1 parent:**
  - `injection.m` (read only by `experiments/exp2_arm.py:98,128`);
  - the positive control (`scripts/positive_control.py` reads finished runs);
  - the Exp1/Exp2 confirmatory certification (`analysis/exp12_validation.py`, analysis time only);
  - the single-seed reporting rule (`analysis/exp2_analysis.py`);
  - the GPU diagnostics-isolation tolerances (tests only).
- **The fork state is m-agnostic.** The injection is applied only in the arm job, so a parent's checkpoints are
  compatible with Exp2 whatever m is later frozen (`exp2_arm.py` reads `fork.source`).
- **The trigger** uses `trigger.null_threshold: 0.0` (approved). Every per-round probe loss is stored
  (`probe_checks.csv`, ledger `checks.csv`), so f*_run under any later threshold can be recomputed offline. The
  forked state itself exists only at the f*_run actually used, since only LATEST plus the fork state are kept
  (amendment (t)).

## 2. Blocker classification
**Key:** A = required before one pilot; B = before the 195-run grid; C = Exp2-only; D = needs your decision;
E = Codex (engineering).

| # | Blocker | Class | Why, and evidence |
|---|---|---|---|
| 1 | Unexplained A100 failures from Block B's GPU suite (`b4a90cb`, job 22706349): `ForkEndToEndTest.test_check1` FAIL; `KillMatrixTest` (7 subtests) and `KillAndResumeEntryPointTest` (3) ERROR, in both default and deterministic modes | **A, E** | Exactly the Exp1 paths a pilot exercises: resume bit-exactness and the control-restore Check 1, which stops a forking parent. They pass on CPU (`exp12_coverage_audit.md`); no A100 rerun at `9d6a82d` and no logged root cause. (The Angle1 parity ERROR was Hydra test isolation, fixed.) |
| 2 | No full-length A100 run at the baseline. Block B's dev run (same configuration as the pilot) hit its 6 h timeout at `b4a90cb`. | **A** (this is what the pilot measures) | `a100_runtime_root_cause_investigation.md`: job 22740708 timeout, blocking stage unresolved. The metric-transfer fix is in `9d6a82d`, and job 22765261 (profile) completed. Only a 60-interaction warm window is measured. |
| 3 | D4W1536 hopper range gate FAIL (P/b 0.602) | **B, D** | Affects only probe validity for D4W1536 on hopper-hop. Pilot (dog-run D4W1536 passes at 0.993 on CPU; Block A's D6W1536 0.993) is unaffected. |
| 4 | D4W1024 fresh-null FAIL (13/100); seed-991 replication unverified | **B, D** | Concerns the trigger's false-fork rate and so where scaled parents fork. The pilot is D4W1536 (5/100, passes). The parent's probe data are threshold-independent (stored per round). It matters for treating the pilot's fork as the positive control's natural trigger (row 5). |
| 5 | Positive control and m not obtained | **B** for the Exp1 grid (Methodology l.144: "before the main runs of Experiment 1"); **C** for m | Needs a natural trigger in a D4W1536 dog-run dev run (amendment (e)). The recommended pilot *is* that run. m is used only by `exp2_arm`. |
| 6 | A100 identity-fork validation (`exp2_arm` with `fork.arm=identity`) | **C**, plus **B** via row 1 | The identity arm is an Exp2 validation (amendment (b): before the Exp1 *grid*). The Exp1-critical part, control restore, is covered by row 1 and exercised by the pilot if it forks. Block B's identity forks all TIMED OUT; the D4W1536 warm identity step is projected (~329 s) to exceed its 300 s limit (`codex/a100-exact-identity-audit` @ `dcb7491`). |
| 7 | Four GPU numerical tolerances unset | **B, D** | Diagnostics-isolation test bounds (`tests/exp12_helpers.check_gpu_tolerance`). They do not change training. |
| 8 | Entropy-target wording (`abs(A)/2` vs the implemented `−|A|/2`) | **D**, not blocking | G1 and its recorded approval explicitly approve `temp_target_entropy_coef=-0.5` (`entropy_temperature_reconciliation.md`). Only the wording is open. A change would invalidate all runs, the pilot included, so confirm before the grid. |
| 9 | Single-seed reporting convention | **C, D** | Exp2 confirmatory plotting only (`analysis/exp2_analysis.py`). |
| 10 | Full-run checkpoint compatibility with Exp2 | **C**; verified by the pilot if it forks | Fork state, panel and `check1_pre` written by `write_fork`; m-agnostic. Arms must run on the same GPU model (amendment (p)). |
| 11 | Deployment: one GPU model, per-suite runtime (MyoSuite, HumanoidBench twins at 2M steps), packing, about 631 GB storage, frozen study manifest and commit | **B, D, E** | Pilot covers one DMC D4W1536 run only. HumanoidBench has never run on a GPU (Block HB not run). |
| 12 | Compilation-cache race ("Error -5") | **B, E** | Fixed on `claude/precision-cache-fix` @ `d23191b`. The pilot avoids it with a job-private cache directory. |
| 13 | GPU test failure in twin Check 1 eager reference (CPU-red) | **C, E** | HumanoidBench twin Exp2 only. |

**Before one pilot:** rows 1 and 2 (row 2 is the pilot itself) and the decisions in section 6.

## 3. Recommended pilot
**Use the positive control's own development run.** It is the approved configuration that is already required
before the Exp1 grid (amendment (e)), so the pilot doubles as the positive-control attempt and costs no extra GPU.
It is excluded from confirmatory analyses (`run_role=dev`; ledger drops dev by default).

| Setting | Value (approved source) |
|---|---|
| Command | grid command (`generate_manifest.exp12_command`) plus `run_role=dev` |
| Architecture | D4W1536 critic (`critic_num_blocks=4 critic_hidden_dim=1536`), D1W128 actor |
| Environment | dog-run, `env=dmc_hard`, 1M raw steps = N = 500,000 interaction steps, action repeat 2, UTD 2 |
| Seed | **102**, Block B's approved dev seed (`exp12_blockB.sh` `DEV_OVERRIDES`); outside 1–5 per amendment (e) |
| Checkpoints | `--checkpoint_interval 25000 --checkpoint_start_frac 0.0` (one save per check, as in the grid) |
| Probe, trigger, fork | config unchanged: 5 rounds × 1,000 steps, threshold 0, two consecutive checks, fork at f*_run, control to max(N, fork + 0.25N) |
| Precision | TF32 via `set_matmul_precision`; Check 1 at "highest" |
| GPU | 1 × A100-SXM4-40GB (the grid's and Block B's model) |
| Compilation cache | job-private directory |

**What it tests:**
- end-to-end A100 runtime at full length;
- 21+ probe checks, routine saves and the LATEST pointer;
- complete metrics, ledger and run metadata;
- numerical stability (finite probes, losses and returns);
- if it triggers, the full fork path: fork state, control restart, bit-exact control restore on the A100, post-fork
  evaluations, and a real fork state for later Exp2 identity and positive-control jobs.

## 4. Resource estimate (measured vs extrapolated)

| Quantity | Value | Basis |
|---|---|---|
| Warm training rate | **42.44 interaction steps/s** | *Measured*, job 22765261 @ `9d6a82d`, D4W1536 dog-run, 60-interaction window only |
| Training to N = 500k | 3.27 h | *Extrapolated* from that rate |
| 21 probe checks | about 14 min | *Measured* fit cost (~3.5 s × 10 fits per check) × 21; extrapolated |
| Evaluations: 10 × 5 episodes and a final 50 | about 4 min | *Measured* 22.8 s per 10 episodes; extrapolated |
| 20 routine saves | about 2–4 min | *Measured* 6.1 s save / 2.5 s restore (synthetic 475k transitions) |
| Startup, compilation, structural metrics | about 5–10 min | *Measured* 30 s compilation and 15.5 s first structural call; per-window cost uncertain |
| **Total, no fork** | **about 3.7 h** | *Extrapolated* |
| **Total with a late fork** (control to 1.2N, +4 checks, fork save and restore, 26 × 10-episode post-fork evaluations) | **about 4.6 h** | *Extrapolated* |
| Block B dev run, same configuration, at `b4a90cb` | > 6 h (TIMEOUT) | *Measured*; before the metric-transfer fix |
| Peak GPU memory | 3.5 GiB | *Measured* (job 22765261) |
| Host memory | well under 64 GB; replay about 2 GB | *Estimate* |
| Disk | about 5 GB routine state; about 10 GB with a fork state; plus the job-private cache | *Estimate* (`exp12_master_summary.md` §5) |

**Request:** 1 GPU, 16 CPUs, 64 GB, `--time=08:00:00`. That is 1.7× the worst extrapolation, and the training rate
rests on a 60-step window. If the job overruns, the same command resumes from the last routine save, losing at most
25k steps (about 10 min).

## 5. Prerequisite checklist
- [ ] **(E) A100 rerun of the Exp1-relevant tests at the pilot commit, with full logs archived:**
      `tests.test_exp12_fork.ForkEndToEndTest`, `tests.test_exp12_fork.KillMatrixTest`,
      `tests.test_exp12_foundations.KillAndResumeEntryPointTest`. A root cause for any failure is required before
      the pilot. One A100, well under 1 h (Block B's whole suite took 25 min per mode).
- [ ] **(E) Pilot commit fixed:** the `integration/exp12` HEAD once published and checked, otherwise `9d6a82d`.
      Clean, fresh checkout.
- [ ] **(D)** Approve the pilot doubling as the positive-control development run (D4W1536, dog-run, seed 102,
      `run_role=dev`).
- [ ] **(D)** Approve that its natural trigger uses the current `null_threshold: 0.0`. If you might adopt the p95
      option before the positive control, decide now. A different threshold would move the fork, and only the
      fork-time state is kept.
- [ ] **(D)** Approve the 8 h, 1-GPU allocation.
- [ ] Fresh checkpoint and results directories outside the checkout. GPU model A100-SXM4-40GB.

## 6. Decisions needed before submission
1. The pilot doubles as the positive-control development run (seed 102, dev).
2. The trigger threshold stays 0.0 for this run, so its fork is usable as the positive control's natural trigger.
3. The allocation (8 h, 1 × A100).
4. If the pilot does not trigger by check 19: per amendment (e) that is "stop and consult". The pilot's engineering
   evidence still stands.

## 7. Pilot commands: NOT AUTHORIZED FOR EXECUTION
```bash
# Fresh checkout at the approved pilot commit (replace PILOT_COMMIT; default 9d6a82d4aef81299901177865e383640439cde2a)
git clone https://github.com/kaves29/scaling-drl-research.git /work/hdd/biqc/skaveti1/exp12_pilot
cd /work/hdd/biqc/skaveti1/exp12_pilot && git checkout --detach "$PILOT_COMMIT" && cd main && mkdir -p logs
export EXPECTED_COMMIT="$PILOT_COMMIT" PILOT_ROOT=/work/hdd/biqc/skaveti1/exp12_pilot_runs
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=exp12_pilot_exp1
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e
cd "$SLURM_SUBMIT_DIR"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout run.py experiments/exp1.py configs/base_exp12.yaml
RUN="$PILOT_ROOT/exp1/D4W1536/dog-run/seed_102"; RES="$PILOT_ROOT/results"
[ ! -e "$RUN/DONE" ] || { echo "$RUN already complete" >&2; exit 2; }
mkdir -p "$PILOT_ROOT/logs" "$RES"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
unset JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled MUJOCO_GL=disable EXP12_JAX_CACHE_DIR="$PILOT_ROOT/jax_cache"
git rev-parse HEAD; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
python -c "import jax; d=jax.devices(); print(d); assert len(d)==1 and d[0].device_kind=='NVIDIA A100-SXM4-40GB'"
set +e
python -u run.py --experiment exp1 --config_name base_exp12 \
  --overrides critic_num_blocks=4 --overrides critic_hidden_dim=1536 --overrides env_name=dog-run \
  --overrides env=dmc_hard --overrides seed=102 --overrides results_root="$RES" --overrides run_role=dev \
  --checkpoint_dir "$RUN" --checkpoint_interval 25000 --checkpoint_start_frac 0.0 \
  > "$PILOT_ROOT/logs/exp1_D4W1536_dog-run_seed102_$SLURM_JOB_ID.log" 2>&1
status=$?; echo "exp1 exit $status at $(date)"; exit $status
EOF
# If it times out: resubmit the identical sbatch; exp1 resumes from $RUN/state/LATEST (or the fork state).
```

**Expected artifacts:**
- `$RUN/DONE`
- `$RUN/run_metadata.json` (launches: commit, device, `matmul_precision=tensorfloat32`)
- `$RUN/state/LATEST` and its state directory (agent checkpoint, `buffer.npz`, `buffer_meta.pkl`, `obs_rms.pkl`,
  `meta.pkl`)
- `$RUN/fresh_critic/`
- `$RUN/probes/probe_checks.csv` and `check_00.npz` to `check_20.npz` (more after a fork)
- the evaluation-episodes CSV
- `$RES/exp12/exp1/runs/exp1_D4W1536_dog-run_seed102/{run.csv,checks.csv,metrics.csv,probe_curves.npz,source.json}`
- **If it forks:**
  - `$RUN/fork/{state/,panel.npz,check1_pre.npz,check1_control.npz,fork.json,FORK_READY}`;
  - `check1_control.json` PASS and the `control` arm files under `$RES/exp12/exp2/`;
  - routine saves continuing to max(N, fork + 0.25N).

**Post-pilot acceptance (engineering; criteria from the existing code, no new thresholds):**
- exit 0 and `DONE` present;
- check indices 0–20 complete, every round finite;
- metadata launches all at `EXPECTED_COMMIT` with a clean tree;
- if it forked, `check1_control.json` pass = true;
- record the measured wall time per phase;
- any `Check1Failed` or non-finite probe is a blocker for the grid.
