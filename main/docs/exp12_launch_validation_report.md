# Launch-readiness CPU validation receipt

2026-10-10. Task branch `codex/exp12-launch-readiness`, approved base
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. Resolve the delivery commit from
Git. Engineering implementation commit: `a0b7125a82c0705b8f69d5a198199a5cd735b2e9`. Subsequent
documentation commits preserve its executable tree. No main/integration/Exp3
modification, GPU submission or scientific change.

## Completed verification

| Verification | Result / scope |
|---|---|
| Fresh broad suite at approved base | 68 modules, 621 tests run: **620 passed, one skipped**, zero failed modules; source unchanged |
| Final focused suite on readiness worktree | 23 new readiness tests plus 23 existing GPU-resume-probe unit tests and five null-follow-up unit tests; see final receipt below |
| Mutation checks | **74/74** detect the intended defects; no mutation persists on disk |
| Environment restoration | **65/65**, all 13 canonical tasks × five confirmatory seeds; real HumanoidBench H1 paths available |
| Configuration isolation | **195/195** fully resolved hashes equal the approved base; 298 protected source/configuration/authority files byte-for-byte identical |
| Pinned cache guards | Eight existing CPU tests passed; eviction optional dependency remains the broad-suite skip |
| Actual JAX observer call | One real CPU JIT compilation observed, exact independent NumPy values/shape/float32 dtype preserved; no GPU claim |
| Static checks | Black on all five new Python files; AST/JSON parsing; bash syntax on both new Slurm scripts; git diff whitespace checks passed |
| Historical profile integrity | Uploaded job22765261 ZIP SHA-256 matched; all 36 members intact; profile/validation/backend/trace extracted bytes equal original archive |

The skipped test is
`tests.test_exp12_compilation_cache.AtomicCacheWriteTest.test_eviction_cache_keeps_jax_put`:
JAX requires optional `filelock` for an evicting cache, absent here. The existing
unlimited production cache path and atomic-write regressions passed. No package
or cache policy was changed to remove the skip.

The broad suite ran freshly in the unchanged base checkout, while focused new
and existing investigation tests ran in the isolated readiness worktree. Do not
represent these as one monolithic final-tree invocation. The 298-file/195-config
comparison establishes unchanged production behavior for the broad regression
coverage; the new harness/launch interfaces have their separate CPU coverage.
The broad runner does not discover the investigation tests under `scripts/`, so
those are explicitly included in the focused command.

CPU interpreter: `/workspace/scratch/a100-audit-venv/bin/python`, Python3.12.14,
JAX/JAXLIB0.4.34, NumPy1.26.4, Flax0.8.4, Optax0.2.3, Orbax0.5.3. Real HB source
`cb1189039151c8aadaaa987b442da54383c87fab`, Mesa/EGL software rendering. This is
not Delta Python3.12.13/CUDA. See [CPU setup](exp12_cpu_validation_setup.md) for
upstream dependency differences and rendering requirements.

CPU tests exercise fail-closed scientific/source/approval/resource gates,
195-cell population/configs, role-separated seed102 pilot, actual shell source
preflight and retained error evidence, array enumeration, dtype/backend/moment
placement rejection, cached executable preservation, incomplete DONE output,
normalization/replay payload completeness, stale/scope-incompatible certificates,
and approved-resource-only printed commands. Synthetic GPU metadata is never
counted as GPU execution or qualification.

No original source/config/test files were edited. Existing shared handoff/status
files only received dated additions. All other new files are optional diagnostic,
launch-preparation, investigation, test or documentation interfaces. Original
trainer/SAC/fork/restore/Check1/comparator/precision/RNG/probe/bootstrap/trigger
code and the full grid remain untouched. Diagnostic panel/compiled-call observers
use the existing implementations and return their literal results.

## Reproduce (CPU only, from main/)

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu EXP12_JAX_CACHE_DIR=off
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cpu_python=/absolute/path/to/pinned/HB/CPU/python
"$cpu_python" -B scripts/run_exp12_cpu_validation.py --jobs 2 --out-dir /absolute/new/broad-evidence
"$cpu_python" -B -m unittest tests.test_exp12_launch_preparation \
  scripts.sci_investigation.test_gpu_resume_probe scripts.sci_investigation.test_null_followup -v
"$cpu_python" -B -u -m tests.exp12_break_checks
"$cpu_python" -B scripts/check_exp12_cpu_environments.py
"$cpu_python" -B scripts/check_exp12_readiness_isolation.py \
  --base 844e3c0cc24998b25c3d1e667bcbfe056b1b336c --out /absolute/new/isolation.json
```

One attempted focused invocation used nonexistent `tests.test_exp12_gpu_resume_probe`
and `tests.test_exp12_null_followup` import paths: the 23 readiness cases passed,
but the two loader errors made that invocation fail. The corrected package paths
above are the completed receipt. Earlier cwd/EGL setup mistakes were corrected;
no failed attempt is reused as validation evidence. Raw attempts remain outside
Git.

## Remaining unverified paths

- No new full-width GPU cold/warm/crash/restore execution, A100 peak-memory or
  timing measurement, native CUDA identity-certificate success path, full-budget
  training or actual Slurm allocation/submission was executed.
- Existing reported tiny-resume22783981 and null22779974 files remain unavailable;
  these outcomes are explicitly owner-reported. No Delta access was attempted.
- Full-width Check1 observes actual online/target/twin dispatch when executed,
  but no current-source GPU Check1 proof is implied by CPU wrappers/tests.
- Pilot, 195 parents and Exp2 remain **NO-GO** until their separate prerequisites
  in [the launch report](exp12_full_grid_readiness.md) are evidenced/approved.
  Fresh-null/range failures and positive-control/m/numeric decisions are not waived.

## External evidence receipts

Generated logs/results stay outside Git. The following local names are under
`/workspace/scratch/`; broad summary under
`exp12-launch-readiness/baseline-full/summary.json`. Their SHA-256 receipts bind
this verification, not future runs:

- `exp12-launch-readiness/baseline-full/summary.json`: `983664e8b541b4bb5ff5649a6b33d2d9fef31df4fd9dc7e638d1bad16c2a0b2b`.
- `exp12-readiness-mutations.log`: `db08ec55bfbfba20c0a7d258ae850f0754f296a7c945b12d8b0e6e1b246119cd`.
- `exp12-readiness-restoration.log`: `6dbcec9cfa9c2c1dd0b1bfa2752cfd6ee5e2c0f044d7b5a00cb49b60b1563db4`.
- `exp12-readiness-isolation.json`: `da243cb3d7888b26a03dd7a335d62c4e07bb4b4d56047afb467d14e187b93b28`.
- `exp12-readiness-cache-guards-2.log`: `81b8ab50f754d720a73796aaba6473d8a43fa688fd58abd0e3433dd2b29b2166`.
- `exp12-readiness-profile-integrity.json`: `5d9b1984aeff2fbc525546d38dcbb5839f3b105eccaf8dae912f40661dab122e`.
- `exp12-null-count-reference-final.json`: `0a2c9ccfb43ccbd2495f5a62a3ef0eb3fdf217c10708d08d3ae760a3c23b754d`.
- `exp12-readiness-real-cache-observer.json`: `8b3a51b8d46fe5ac0dbb576c2d08a77ecb326536404def67aa62f25cb1c97517`.

Final focused receipt: **51/51 passed** (23 new + 23 probe + five null), zero skips/errors.
`exp12-readiness-final-corrected-tests.log`: `c8bdb3e77452ad3d7376e5b2030a6c900dc718efbe71d28fa6e6516240d7674e`.
