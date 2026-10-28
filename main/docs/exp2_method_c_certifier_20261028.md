# Method C curve certification follow-up — 28 October 2026

**Yes: this was a stale validator requirement from the earlier methodology.**
It was a real confirmatory-analysis blocker, not truncation of the saved curves
or evidence that the approved producer fitted only 1,000 updates. PR #19 now
corrects the validator; the scientific Method C remains 2,000 continuous updates.

## Provenance and root cause

Reviewed PR #19 began clean at `716916b29613c6bde4ce1d264b2d1953209340e3`,
based on approved integration `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`.
Remote integration and the draft PR head were checked before editing. The
certifier literals were introduced at `2dc8734f` (6 October); Method C's
producer/configuration changed at `c7fc4ea` (12 October), but the two literals
and synthetic fixture's 1,000-entry arrays remained unchanged. Full SHAs,
file hashes and preserved-source comparisons are in external `source_proof.json`.

Implementation/test SHA on PR #19:
`dbf9eb85ab25fedf67f2e560b5f118831fa696c6`.
To verify the actual approved Method C path without importing it into integration,
a separate **local CPU-only** checkout starts from published approved Method C
`087feab341b9be1f377697668f2d1c750ec0467f` and applies only the same certifier
and test changes: `aff6b72282582c3ebee4e957d4557613003d5d09`. Its producer,
configuration, methodology and optimizer implementation remain byte-identical
to 087. Neither branch is merged or retargeted; no active source/run is changed.

## Complete call and data path

Producer citations below are pinned to approved Method C source 087; corrected
validator citations refer to dbf9eb8. No producer file was edited.

| Stage | Exact source location | Behavior |
|---|---|---|
| Required budget | 087 `configs/base_exp12.yaml:56`; methodology amendment (aa), line565 | Five independently keyed target rounds, 2,000 continuous updates per target. |
| Construct paired minibatches | 087 `experiments/exp12/probe.py:112`, especially129–132 | First1,000 indices use original `idx_key`; next1,000 use `fold_in(idx_key,1)`; concatenate. This intentional 1,000 constant is **not** obsolete. |
| Execute fit | 087 `probe.py:74`, scan line94 | One `_fit` call, one optimizer initialization and one scan carrying params/moments/count across all2,000 updates; returns2,000 minibatch losses and a separate final full-pool error. |
| Stack/save longitudinal probes | 087 `probe.py:155`; `exp12/run_probes.py:131` | Stack five loss arrays as `(5,2000)`; save every field unchanged in `probes/check_XX.npz`. |
| Save immediate arm Check2 | 087 `experiments/exp2_arm.py:55` (PR19 line61), archive at89 in PR19 | Calls the same `run_probe_networks` and saves all arrays to `probes/check2_curves.npz`; fresh/control/injected and per-Q fields are retained. |
| Consolidate ledger | 087 `exp12/ledger.py:55`, archive at97 | Read every saved NPZ field and write it under `check_XX/<field>` in `probe_curves.npz`; no slicing, resampling or truncation. A sentinel-array regression retains all10,000 entries, including the last update of every round. |
| Enter confirmatory Exp2 analysis | `analysis/exp2_analysis.py:312`, call at336 | Default non-exploratory path calls `publish(...,"exp2", study_manifest, False)`. Training does not call this offline certifier. |
| Certify before publication | dbf `analysis/exp12_validation.py:1115` → `validate:987` | Validate exact frozen study/config/source/population, then each parent and eligible fork. Exploratory mode remains explicitly uncertified. |
| Parent curves | dbf `_parent:428` → `_curves:401`, forwarded config at524 | Previously required `(5,1000)` at old line395; now requires the exact frozen configuration shape. |
| Arm Check2 curves | dbf `_fork:707` → `_check2:888` | Previously required `(5,1000)` at old line879; now uses the same declared shape, including each twin network. |

There is no alternate obsolete-format conversion or hidden curve truncation on
this path. Plotting also consumes the full saved loss arrays (`exp1_analysis.py:156`).
The defect blocks confirmatory certification/publication; it does not shorten
training/probe execution or explain a historical learning-performance pattern.

## Minimal correction and fail-closed behavior

The only production-code change in this follow-up is
`main/analysis/exp12_validation.py`: `_probe_curve_shape` binds expected rows and
columns to the **frozen configuration already checked against canonical settings**.
It supports the five-target historical1,000-step and approved Method C2,000-step
contracts, but never accepts both lengths for one declaration. A Method C
configuration declaring2,000 rejects1,000,1,999,2,001, transposed/wrong-round or
missing curves. Invalid/ambiguous step declarations and duplicate NPZ field names
also fail closed. It does not infer a protocol from the observed array length.
No scores, acceptance thresholds, bootstrap, eligibility, Check2 inclusion policy
or nonfinite-check rules were changed. No new finite-loss cutoff was introduced.

The synthetic complete-study fixture now produces the step count in its own
configuration rather than silently manufacturing1,000-entry curves. New tests
exercise single/twin parent and injected-arm evidence, expected rejection cases,
raw→ledger full-array retention, and read-only certification. A failed Check2
still remains valid primary evidence. These are engineering fixtures, not
natural positive-control or scientific campaign results.

Before the fix, the preserved two-method reproduction had one assertion failure
and two subtest errors: shortened1,000-entry parent curves were accepted under a
2,000-step declaration, while complete2,000-entry curves were rejected.
`before-fix.log` is retained; this expected exit1 is not counted as a passed test.

## CPU verification

| Scope / exact tested SHA | Actual result |
|---|---|
| PR source `dbf9eb85ab25fedf67f2e560b5f118831fa696c6` | 80 tests across seven relevant modules; zero failures/errors/skips; every module exit0 |
| Approved Method C + same patch `aff6b72282582c3ebee4e957d4557613003d5d09` | 75 tests across seven modules; zero failures/errors/skips; includes top-level195/130 census certification and three existing Method C producer tests; every module exit0 |
| PR source required maintenance | 74/74 break/restore mutations (70 assertion + four intentional runtime-error detections, every restored case passes), 65/65 simulator cells and195-parent legacy census; each command exit0 |
| Static checks | Runtime/new regression Pyflakes, Black on changed tests, diff check: exit0. Shared fixture Pyflakes exit1 reproduces the same three unused imports (`io`, `Path`, `pandas`) at716916b; retained as pre-existing lint findings, not hidden or counted as a pass |

Counts overlap between sources and mutation/regression scopes; they are not independent scientific samples. Logs retain dependency/Orbax/plotting/simulator warnings; exact occurrence counts are in `receipt_audit.json`. The duplicate-key fixture locally filters only its deliberately generated ZIP duplicate-name warning.


Exact argv, interpreter, tested SHA, directory, test counts, skips, failures,
warnings and log SHA256s are under
`/workspace/artifacts/exp2-certifier-followup-20261028/`:
`validation.json`, `maintenance.json`, `static.json`, per-module JSON/logs,
`source_proof.json` and preserved before-fix output. Fresh directories were used.
Tests and source scopes are not transferred between checkouts or commits.
Python3.12.14/JAX0.4.34/Flax0.8.4/Optax0.2.3 pinned CPU/HumanoidBench environment
and software EGL match the documented setup; no dependencies changed.

Reproduce the focused modules from either clean checkout's `main/` using the
documented CPU environment. The approved Method C checkout additionally runs
the existing three `tests.test_exp12_method_c` regressions: unchanged first1,000
draws, paired target/index streams, and uninterrupted versus two-stage fitting
with live AdamW state/count1,000→2,000. No scientific probe experiment is launched.
The worker command used for each module is recorded literally in `validation.json`:

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=''
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 EXP12_JAX_CACHE_DIR=off
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu_python=/workspace/venvs/scaling-drl-hb/bin/python
"$cpu_python" -B -m unittest tests.test_exp2_method_c_certifier tests.test_exp12_completeness tests.test_exp12_exp2_analysis -v
# On the approved Method C source only:
"$cpu_python" -B -m unittest tests.test_exp12_method_c -v
```

## What remains blocked

This resolves the stale1,000-step curve checks on the actual five-target Method C
analysis path; the CPU source-matched qualification demonstrates that correction.
It does **not** promote source into approved integration. That ref still has the
older producer/configuration and lacks amendments (aa)/(ab); review the proper
source integration before treating its default configuration as Method C.
The separate ten-target/96-parent certifier, with its distinct population,
schedule and IQM10, is outside this fix and remains separate unfinished work.

Real natural eligible forks, same-source positive-control evidence, owner-frozen
`m`, source-bound GPU/resource/durability qualification and launch approval remain
Exp2 requirements. No `m`, methodology, training condition, pilot, active artifact,
optimizer, RNG stream, job or threshold was changed. No training campaign, GPU
execution, Delta preflight/submission, merge or force-push occurred. PR #19 stays draft.
