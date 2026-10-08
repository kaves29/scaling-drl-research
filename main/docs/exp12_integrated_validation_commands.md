# Integrated candidate: review, transport and next validation

These are manual commands. No push, Delta access, job submission, numerical-bound selection, m selection or production launch has been performed. GPU sections require approval. The candidate is a different source revision from the frozen runtime comparisons; never silently substitute it for f6.

## Local review and transport into the normal repository

The final local branch is `codex/exp12-integrated-candidate`. Obtain its exact final hash from the delivered verification results. The portable bundle is `/workspace/scratch/exp12-integrated/exp12-integrated-candidate.bundle`; copy that file and verify the delivered SHA-256 before importing. It retains merge/correction provenance and requires the published f6 commit already present in the receiving repository.

```bash
set -euo pipefail
cd /absolute/path/to/normal/scaling-drl-research
test -z "$(git status --porcelain --untracked-files=all)"
git fetch origin codex/a100-runtime-debug
test "$(git rev-parse FETCH_HEAD)" = f6b7f73b018eac365819ce64c90d22942b211f42
candidate_bundle=/absolute/path/to/exp12-integrated-candidate.bundle
candidate_rev=FULL_FINAL_HASH_FROM_DELIVERED_VERIFICATION
git bundle verify "$candidate_bundle"
git fetch "$candidate_bundle" codex/exp12-integrated-candidate:refs/remotes/candidate/exp12-integrated-candidate
test "$(git rev-parse refs/remotes/candidate/exp12-integrated-candidate)" = "$candidate_rev"
git merge-base --is-ancestor a3f44aa56455e3aae4b36459a78c97237bad62a8 "$candidate_rev"
git merge-base --is-ancestor f6b7f73b018eac365819ce64c90d22942b211f42 "$candidate_rev"
git log --graph --oneline f6b7f73b018eac365819ce64c90d22942b211f42.."$candidate_rev"
git diff --stat f6b7f73b018eac365819ce64c90d22942b211f42 "$candidate_rev"
git diff --check f6b7f73b018eac365819ce64c90d22942b211f42 "$candidate_rev"
git branch codex/exp12-integrated-review "$candidate_rev"
```

This creates a separate review ref without modifying the checked-out branch or merging into main/launcher. Review the literal twin-loss correction separately from subsequent engineering guards. For CPU reproduction, use [exp12_cpu_validation_setup.md](exp12_cpu_validation_setup.md), a new output directory and the reviewed candidate source. Do not label another Python/dependency stack as this Cloud run.

## Next same-budget runtime comparison: published f6

The exact new detached checkout and same-budget submission commands are in [exp12_launch_validation_commands.md, next same-budget A100 runtime stage](exp12_launch_validation_commands.md#next-same-budget-a100-runtime-stage). Use **f6b7f73b018eac365819ce64c90d22942b211f42**, `DIAGNOSTIC_MODE=profile`, `DIAGNOSTIC_ARCH=D4W1536`, the existing 300-second internal timeout and seven-minute Slurm limit. That section is still the next runtime comparison; its later statement that no integrated hash exists is a historical snapshot superseded by this report.

The required new evidence is full trace/profile/backend/exit/Slurm output, repeated Python stacks, stage begin/end and metrics-flush boundaries, and any synchronization exception. Absence of a terminal profile remains incomplete. Do not increase the timeout or infer that the six-hour failure has the same cause.

## Fresh detached candidate checkout for approved final-source validation

After copying the bundle onto Delta through an independently authorized transfer, the following commands prepare a **new** checkout; they do not submit a job. The final hash must be explicitly reviewed first. The existing frozen checkout and all earlier diagnostic directories remain intact.

```bash
set -euo pipefail
candidate_rev=FULL_FINAL_HASH_FROM_DELIVERED_VERIFICATION
candidate_bundle=/absolute/path/to/exp12-integrated-candidate.bundle
stamp=$(date -u +%Y%m%dT%H%M%SZ)
candidate_checkout=/work/hdd/biqc/skaveti1/exp12_checkouts/${candidate_rev}_${stamp}
mkdir -p "$(dirname "$candidate_checkout")"
test ! -e "$candidate_checkout"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$candidate_checkout"
git -C "$candidate_checkout" bundle verify "$candidate_bundle"
git -C "$candidate_checkout" fetch "$candidate_bundle" codex/exp12-integrated-candidate:refs/remotes/candidate/exp12-integrated-candidate
test "$(git -C "$candidate_checkout" rev-parse refs/remotes/candidate/exp12-integrated-candidate)" = "$candidate_rev"
git -C "$candidate_checkout" checkout --detach "$candidate_rev"
test "$(git -C "$candidate_checkout" rev-parse HEAD)" = "$candidate_rev"
test -z "$(git -C "$candidate_checkout" status --porcelain --untracked-files=all)"
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
cd "$candidate_checkout/main"
export EXPECTED_COMMIT="$candidate_rev"
export JAX_PLATFORMS=cuda,cpu
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
export WANDB_MODE=disabled MUJOCO_GL=disable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout experiments/exp12/probe.py scripts/capture_exp12_range_evidence.py scripts/positive_control.py
```

The Delta environment must be independently checked for its real A100/HumanoidBench/runtime dependencies. The isolated CPU overlay is not a reviewed CUDA environment. The preflight script validates source/files, not GPU scientific qualification. No script introduced in this sprint automatically calls sbatch.

## Minimum range evidence, at unchanged scientific settings

After approval, inside the above clean checkout on an approved A100 allocation:

```bash
range_output=/absolute/new/evidence/directory
test ! -e "$range_output"
timeout 1800 python -u scripts/capture_exp12_range_evidence.py \
  --arch D4W1536 --expected-commit "$candidate_rev" --out-dir "$range_output"
```

This uses the historical hopper-hop seed990/check0, pool25,600, five rounds, 1,000 fitting steps, original AdamW/teacher/target scale and original centralized 0.9 range criterion. The existing range timeout is not enlarged. It captures the actual teacher inputs and post-fit curves after the same delegated computation; it does not refit or regenerate the target. Inspect checksum-covered `capture_metadata.json`, `range_D4W1536.json`, all five teacher-pool NPZs, replay/normalizer and curve NPZ. A one-architecture capture is not all-size qualification. See the scientific report for estimable resource limits and the distinction between optimization failure and trained-critic degradation.

## Minimum positive-control evidence

First recover/validate any existing full-setting natural D4W1536 dog-run development fork. The historical development-seed102 timeout did not supply such a fork. A forced/reduced/testing fork cannot qualify m. If a valid natural fork exists, the canonical command is:

```bash
python scripts/positive_control.py --run_dir "$approved_natural_dev_run" --out_dir "$new_pc_evidence_directory"
```

Do not use `--allow_any_setting` for qualification. This performs the mandatory four-offset noise sensitivity plus last/half/all comparisons: 25 fits for five critics across five rounds, plus 30 fits for two critics across three m candidates, totaling 55,000 fitting updates. It preserves the approved pool/budget/sampling. GPU duration/VRAM are not defensibly estimable from current CPU evidence. If no eligible natural fork exists, unchanged full-budget development training and an eligible natural trigger must precede this command. A lack of trigger is evidence, not authorization for a forced qualification fork.

The command's recommendation is descriptive until lead review freezes m. All-negative recovery can still produce a recommendation under the existing near-best rule; the lead must decide whether successful-rescue or reproduction criteria beyond the existing procedure are required. No new cutoff is selected here.

Recover the raw original null-pair JSONLs before repeating calibration. A single failed-pair replay can distinguish mechanisms, but cannot replace the required 100-pair/all-size population. Sequential-trigger qualification, different bootstrap procedures, changed round counts or altered convergence budgets require methodological approval. The current procedures and four unset CUDA numerical bounds remain unchanged.

## Production preparation, only after qualification and approval

Use the production preparation commands in [exp12_launch_validation_commands.md](exp12_launch_validation_commands.md#production-preparation--blocked-until-every-checklist-gate-is-approved), with the final reviewed source, lead-frozen m, runtime-by-suite, GPU model and evidence-based allocation. Start with `set -euo pipefail`; the mandatory read-only inventory must succeed before manifest generation. Never treat thin DONE/FORK_READY scheduling markers as result certification.

```bash
set -euo pipefail
cd "$fresh_manifest_directory"
python "$approved_checkout/main/scripts/check_exp12_campaign_inventory.py" \
  --checkpoint-root "$campaign_checkpoint_root" --expected-commit "$approved_revision" \
  --out "$new_inventory_report"
python "$approved_checkout/main/generate_manifest.py" --grid exp12 \
  --ckpt-root "$campaign_checkpoint_root" --results-root "$campaign_results_root" \
  --injection-m "$approved_m"
```

Freeze the immutable master195 population and provenance before any training; preserve it across recovery queues. Verify source/config/runtime/complete-state integrity before every resume and arm fork. Run strict frozen-manifest analyses, retaining failed Check2 arms and refusing missing/duplicate/mixed/incomplete results. Resource declarations, packing/concurrency, source publication, launch authorization and all scientific decisions remain pending; these commands do not supply them.

For collecting completed local diagnostic evidence, use `package_exp12_diagnostic.py` with the **job's actual full revision** and explicit actual job ID, run directory, Slurm stdout/stderr and a new ZIP path. It performs no Slurm queries or submissions. The detailed packaging command and missing-provenance rules remain in the earlier command document.
