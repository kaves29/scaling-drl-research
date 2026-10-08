# Exp1/Exp2 next validation and deployment commands

Companion to `exp12_launch_readiness.md`. These are manual instructions, not executed submissions. No command below grants approval to merge, push, submit jobs, select m, choose CUDA bounds, or launch production. Run GPU sections only after explicit lead authorization. Never reuse or alter frozen checkouts or diagnostic outputs.

## Reproduce CPU validation

Run the corrected reference independently from its dedicated checkout:

```bash
cd /workspace/scaling-drl-methodology/main
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MUJOCO_GL=disable
export EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/workspace/scratch/mpl_methodology
export XDG_CACHE_HOME=/workspace/scratch/xdg_methodology
/workspace/scaling-drl-research/.venv/bin/python -B -m unittest discover -s tests -t . -p 'test_exp12_*.py' -v
/workspace/scaling-drl-research/.venv/bin/python -B -u -m tests.exp12_break_checks
/workspace/scaling-drl-research/.venv/bin/python -B scripts/methodology_cpu_reference.py
```

Run the new engineering regressions from the engineering branch:

```bash
cd /workspace/scaling-drl-launch-readiness/main
/workspace/scaling-drl-research/.venv/bin/python -B -m unittest tests.test_exp12_campaign_inventory tests.test_exp12_artifacts tests.test_exp12_validation_checkout tests.test_runtime_observability tests.test_runtime_diagnostic_launcher tests.test_exp12_manifest -v
bash -n scripts/check_exp12_validation_checkout.sh scripts/sbatch_exp12_blockA.sh scripts/sbatch_exp12_blockB.sh
git diff --check
```

Full other-module coverage assembles every `tests/test_*.py` module except `test_exp12_*`, with an `if __name__ == '__main__'` guard for multiprocessing. This sprint's runner is `/workspace/scratch/methodology_older_runner.py`; its source is described here for reproduction without that temporary file:

```python
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
if __name__ == '__main__':
    suite = unittest.TestSuite()
    for path in sorted(Path('tests').glob('test_*.py')):
        if not path.name.startswith('test_exp12_'):
            suite.addTests(unittest.TestLoader().discover('tests', pattern=path.name, top_level_dir='.'))
    sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
```

## Next same-budget A100 runtime stage

The already-published f6b7f73 revision preserves the312bb01 workload and adds boundary/flush/stack/error instrumentation. It is the next comparison candidate, **not** the corrected integrated production tree. Existing profile mode, D4W1536,300-second internal timeout,7-minute Slurm limit and explicit CPU initialization are unchanged. Backend initialization must expose cuda,cpu; the launcher sets this and unsets conflicting platform/precision overrides itself.

After explicit submission approval, these exact commands create a new detached checkout and submit that same reviewed diagnostic. The first command's external prefix is the existing Delta project location, not the frozen checkout. No timings/tolerances are selected:

```bash
set -euo pipefail
runtime_rev=f6b7f73b018eac365819ce64c90d22942b211f42
stamp=$(date -u +%Y%m%dT%H%M%SZ)
checkout=/work/hdd/biqc/skaveti1/exp12_checkouts/${runtime_rev}_${stamp}
output_root=/work/hdd/biqc/skaveti1/exp12_diagnostics
mkdir -p "$(dirname "$checkout")" "$output_root"
test ! -e "$checkout"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$checkout"
git -C "$checkout" checkout --detach "$runtime_rev"
test "$(git -C "$checkout" rev-parse HEAD)" = "$runtime_rev"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=all)"
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
cd "$checkout/main"
export EXPECTED_COMMIT="$runtime_rev"
export EXPECTED_GPU_MODEL='NVIDIA A100-SXM4-40GB'
export DIAGNOSTIC_MODE=profile DIAGNOSTIC_ARCH=D4W1536
export OUT="$output_root/${runtime_rev}_${stamp}"
test ! -e "$OUT"
sbatch --export=ALL --output="$OUT.slurm-%j.out" --error="$OUT.slurm-%j.err" scripts/sbatch_exp12_runtime_diagnostic.sh
```

Interpretation: completed stage/error/boundary timestamps and repeat60s Python stacks can locate the last entered operation. Exit124 remains a timeout; profile/validation/terminal-summary absence remains incomplete. Existing finite-output/work-count and identity checks apply unchanged. No result from this future job is inferred. The four diagnostics numerical bounds remain unapproved; no observer comparison or speed estimate qualifies production by itself.

If a new checkout is intentionally based on this unpublished engineering branch, transport its reviewed local commits first and use its full approved hash as EXPECTED_COMMIT; do not replace f6 in the comparison command silently. This branch adds provenance fields and wrapper guards and therefore has its own source identity.

## Package completed local artifacts

The small incremental Git bundle `exp12-launch-readiness.bundle` can transfer this local engineering branch into an ordinary repository that already has published f6. No push is needed. After copying the bundle to that machine:

```bash
git bundle verify /path/to/exp12-launch-readiness.bundle
git fetch /path/to/exp12-launch-readiness.bundle codex/exp12-launch-readiness:refs/heads/codex/exp12-launch-readiness
```

Do not check out over an existing dirty tree. Invoke the packager from a clean separate engineering checkout, and point --repo to any local repository containing the **job's actual commit**, not merely the current branch head. It runs locally, with no Slurm query:

```bash
python /path/to/engineering-checkout/main/scripts/package_exp12_diagnostic.py \
  --run-dir /path/to/completed/diagnostic_output \
  --job-id ACTUAL_JOB_ID \
  --expected-commit ACTUAL_FULL_JOB_COMMIT \
  --repo /path/to/repository-containing-that-commit \
  --slurm-stdout /path/to/actual.slurm-JOB_ID.out \
  --slurm-stderr /path/to/actual.slurm-JOB_ID.err \
  --out /path/to/new/job_JOB_ID.zip
cd /path/to/new
sha256sum -c job_JOB_ID.zip.sha256
```

Replace explicitly labelled actual values, not numerical criteria. Output parent must exist and ZIP/sidecar must be new. Stop the job/writers first. `MANIFEST.json` distinguishes caller job ID, recorded commits, frozen source context, runtime argv/resolved-config availability, missing files, partial trace lines and exit status. Full raw bytes remain in the archive. Absent actual resolved configuration/argv must be supplied as local `resolved_config.json`/`command.json` when recorded by the job; do not retrospectively invent them. This short-runtime profile packager deliberately excludes cache executables and full training/checkpoint evidence; retain those originals separately when required for a scientific diagnosis.

## Integration and final-source qualification — approval required

After explicit approval, create a new integration branch from the engineering branch and perform a normal merge of a3, preserving its provenance. Do not squash/cherry-pick hidden subsets or merge into the runtime-comparison branch. Review merge diff, rerun all current CPU checks, commit a new source identity, and obtain publication/deployment approval. No combined-source hash currently exists.

On that approved final revision, submit from its clean main directory with full EXPECTED_COMMIT. Block A/B now require the explicit submission checkout; `mkdir -p logs` must precede sbatch because Slurm opens output files before preflight. Existing manual qualification entry points:

```bash
# Only after explicit validation-stage approval, in the approved clean main/:
export JAX_PLATFORMS=cuda,cpu
unset JAX_PLATFORM_NAME
mkdir -p logs
EXPECTED_COMMIT="$approved_revision" sbatch --export=ALL scripts/sbatch_exp12_blockA.sh
# Block B includes natural dev/PC, calibration, packing and identity work.
# It needs separate approval; default ARM_M=half is diagnostic, not a production freeze.
EXPECTED_COMMIT="$approved_revision" sbatch --export=ALL scripts/sbatch_exp12_blockB.sh
# Actual HB runtime must already be reviewed/installed; do not install ad hoc.
EXPECTED_COMMIT="$approved_revision" EXPECTED_GPU_MODEL="$approved_gpu_model" HB_ENV="$approved_hb_environment" \
  sbatch --export=ALL scripts/sbatch_exp12_hb.sh
```

These retain existing resources/time limits and scientific settings. The Block B reduced/forced identity/preflight fixtures test plumbing; they cannot qualify its natural full-budget PC. Any repeated calibration, precision contrast, convergence instrumentation design or resource/timeout change requires review before submission.

## Production preparation — blocked until every checklist gate is approved

Freeze m/GPU/runtime/source first. From a fresh external manifest directory, use the approved integrated source explicitly (never the generator's default Angle grid):

```bash
# Variables must come from lead-approved evidence; do not guess them.
cd "$fresh_manifest_directory"
# Read-only inventory first; nonzero blocks queue generation pending review.
python "$approved_checkout/main/scripts/check_exp12_campaign_inventory.py" \
  --checkpoint-root "$campaign_checkpoint_root" --expected-commit "$approved_revision" \
  --out "$new_inventory_report"
python "$approved_checkout/main/generate_manifest.py" --grid exp12 \
  --ckpt-root "$campaign_checkpoint_root" --results-root "$campaign_results_root" \
  --injection-m "$approved_m"
# Before any training, freeze immutable expected population/runtime/config provenance.
python "$approved_checkout/main/scripts/freeze_exp12_analysis_manifest.py" \
  --checkpoint-root "$campaign_checkpoint_root" --expected-commit "$approved_revision" \
  --gpu-model "$approved_gpu_model" --injection-m "$approved_m" \
  --runtime-json "$approved_runtime_by_suite_json" --out "$new_study_manifest"
```

Verify the initial195 unique parent master commands and39 canonical configuration hashes against the independent census; preserve this master even when recovery queues filter DONE. A later fresh arm queue requires all eligible parent's complete saved fork and matching device provenance; selected m must be frozen. Launch allocation resources/concurrency are deliberately unspecified pending measured packing/storage/runtime approval. Explicit claim-launcher selection, once the allocation is approved, is:

```bash
cd "$approved_checkout/main"
python scripts/claim_launcher.py --concurrency "$approved_concurrency" --num-gpus "$approved_gpu_count" \
  --phase-files "$parent_manifest_absolute_path"
# Later generate a fresh arm manifest and use only its matching GPU-model queue.
python scripts/claim_launcher.py --concurrency "$approved_concurrency" --num-gpus "$approved_gpu_count" \
  --phase-files "$eligible_arm_manifest_absolute_path"
```

These launch actual training; they are documentation only and require explicit campaign approval. Do not use claim --dry-run as a read-only census: it creates/releases claims. Do not resume across commits, infer success from DONE presence, drop failed Check2 arms, or accept missing HB as a completed suite. After completion, strict corrected analysis requires the frozen manifest:

```bash
python -m analysis.exp1_analysis --results-root "$campaign_results_root" --study-manifest "$new_study_manifest" --out "$new_exp1_report"
python -m analysis.exp2_analysis --results-root "$campaign_results_root" --study-manifest "$new_study_manifest" --out "$new_exp2_report"
```

Do not use --exploratory/--include-dev as confirmatory certification. One-seed ambiguity, missing/incomplete/mixed revisions or failed integrity checks must remain explicit blockers, not dropped runs.

## Current-run CPU evidence

All commands below terminated successfully in this sprint. Counts are per revision/run; overlapping reruns are not additional independent tests. Full integration remains uncreated/unverified.

| Revision / check | Result |
|---|---|
| a3f44aa, full Exp12 discovery |223 methods,221 pass+2 HB dependency skips;1659.063s, exit0 |
| Engineering branch, all non-Exp12 test modules |302/302 pass;816.377s, exit0 |
| Engineering branch, final checkout/artifact/inventory/observer/launcher/manifest group |83/83 pass;90.948s, exit0 |
| Final inventory guard/assertion rerun |13/13 pass after the final dirty-boolean/model-header guards; exit0 |
| a3f44aa, expanded mutation harness |73/73 OK; all72 original mutations plus approved twin arithmetic; exit0 |
| a3f44aa, independent configuration/statistical reference |195 unique parent configs,39 cells,130 potential scaled parents; exact paired IQM percentile reference, temperature-gradient sign illustration |
| a3f44aa, extended independent composition audit |All195 gamma, single environment, probe checks/chunk/scale, Check1 existing64eps, diagnostic cadence/evaluation and absent identity-stop hooks verified |
| Existing isolated plasticity reference on a3 |68 IQM/10000-resample CI fixtures,4 exact-bootstrap crosschecks, pairing/trigger/eligibility,9 BlockB range rows; default12000 exact toy checks and1000 finite-bootstrap controls completed |
| Original job22740708 packaged with final packager |33 ZIP members,82825 bytes, every member checksum/CRC verified; all1833 trace records and core raw bytes preserved; exit124; actual argv/resolved config explicitly absent |
| Read-only fresh inventory fixture |195 unique fresh parents; no payload or scientific qualification inferred; no checkpoint root created |
| Formatting/source preservation/syntax/dependency checks |Black/pyflakes on new Python, modified shell bash-n, Git diff whitespace; unchanged scientific/precision/config/SAC/runtime-timeout paths;117 compatible installed packages |

The two HB skips are `tests.test_exp12_fork.HumanoidBenchReachEvalSeedingTest.test_reach_eval_seeding_saves_and_restores_the_training_rng` and `tests.test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench`. An initial new-test invocation from the repository top instead of main/ produced a loader import error; it was corrected by using the documented main/ command, not a source or assertion change. Initial pip-module/cache-check commands were unavailable in this uv-managed sandbox; `uv --no-cache pip check` completed successfully. No failing assertion was weakened.

Raw logs/generated packages remain outside Git under `/workspace/scratch/launch-readiness/`. Completed log fingerprints:

| File | SHA-256 |
|---|---|
| `methodology-full-exp12.log` | `616f22d7d2192e9c79d59e7e7a1e61eec3b18ce791d813375421e574e28fc0cc` |
| `engineering-other-full.log` | `ee6ae5c108d36a3dec1d309e178cef87b13868d2df16887d767f26c6a9f9f4f9` |
| `final-engineering-tests.log` | `1d0a5a5cfea259ab722a9f2e6b6a1604e55d899ea8650202c1bb454052a71bb5` |
| `inventory-final-tests.log` | `aba7b32beec617224038c95b38df328526f737383bacd6791892b3ad8a891f57` |
| `methodology-break.log` | `cda5cb054dabdc487f7670237d1c9400be341549cb73133892a80f9340b06dff` |
| `methodology-reference.json` | `6e2c35fc1702f774598ba6479ce67f8892ba6ef69029094e335926efdf755a9d` |
| `plasticity-reference.json` | `8018efc95c2aaa1745f95a7c973531726f07ac91fd1903d362da12ad689f3be0` |
| `job22740708-final-package.zip` | `a088b3d5871c6428c2ab9ef110b68de835873660015658d570ccb5630844c391` |

The old187+2 expectation and72-check harness are superseded by the explicitly explained regression additions, not lower standards. Refer to current counts above rather than combining duplicate tests or attributing CPU outcomes to CUDA. Existing BlockB/job22740708 artifact hashes and source identities are preserved in the readiness report and original investigation documents.
