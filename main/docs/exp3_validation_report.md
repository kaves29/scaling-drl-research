# Exp3 CPU engineering validation

Approved unchanged base: `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`.
Task branch: `codex/exp3-pilot-implementation`. This report qualifies isolated
engineering infrastructure, not scientific procedures, real pilot data or CUDA.

## Results and scope

| Check | Observed result | Scope |
|---|---|---|
| Complete Exp3 run before final guards | 37/37 pass, 391.360 s | All three CPU end-to-end pilots, Stage B, recording parity, restart, single/twin targets and guidance, mismatch rejection |
| Final ordinary-gradient rerun | 9/9 pass, 102.245 s | Action/parameter sanity/full controls and all-three-pilot smoke |
| Final compatibility rerun | 8/8 pass, 78.901 s | Added update-counter rejection, atomic array failure test, all-three-pilot smoke and stream suite |
| Atomic storage focused run | 6/6 pass, 0.034 s | Stream integrity and simulated interrupted publication |
| Existing selected regression suites | 38/38 pass, 307.502 s | Fork units, runtime, bitwise comparator, manifest, twin defect tests and SAC checkpoint |
| Existing full fork/exact restoration | 15/15 pass, 530.112 s | ForkEndToEndTest and ResumeExactnessTest |
| Baseline isolation | 337/337 files unchanged | Every tracked approved-base file compared byte-for-byte |
| Configuration equivalence | 195/195 match | Complete resolved production parent config hashes, not names alone |
| Independent methodology reference | pass | 195 parents, 39 cells, 130 scaled candidates; entropy/paired statistical references |

All39 current new tests were exercised across the complete37-test run and the
affected reruns; these are overlapping runs, not extra independent tests. The
entire repository suite, full mutation matrix and65-environment restoration
matrix were not rerun. Previously reported validation is not counted as new
evidence. Final guards introduced no production changes. New Python files pass
Black and syntax checks; `git diff --check` passes.

CPU environment: Python3.12.14, JAX/JAXLIB0.4.34 in
`/workspace/scratch/a100-audit-venv`. Delta's Python3.12.13/CUDA environment was
not exercised. Existing dependency files remain unchanged. Tiny synthetic test
sizes and reference-oracle tolerances are engineering fixtures, not owner-approved
Exp3 panel sizes, budgets or scientific acceptance rules.

## Reproduction

Use the approved pinned environment, from `main/`, with external output:

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu EXP12_JAX_CACHE_DIR=off
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest tests.test_exp3_pilots tests.test_exp3_streams -v
python -m unittest tests.test_exp12_fork.ForkUnitTest \
  tests.test_exp12_runtime tests.test_exp12_bitwise_comparison \
  tests.test_exp12_manifest tests.test_exp12_twin_critic.TwinCriticDefectTest \
  tests.test_sac_agent_checkpoint -v
python -m unittest tests.test_exp12_fork.ForkEndToEndTest \
  tests.test_exp12_foundations.ResumeExactnessTest -v
python scripts/check_exp3_isolation.py \
  --base 844e3c0cc24998b25c3d1e667bcbfe056b1b336c \
  --out /absolute/new/external/isolation.json
```

## Review conclusions

No existing SAC math, optimizer, checkpoint format, entry point, methodology,
configuration, timeout, tolerance or resource declaration was edited. The source
recorder is process-local and opt-in; it calls the existing training loop without
extra actions, random draws, saves or updates. CPU recording/restart parity is
exact-state evidence; representative GPU recording parity remains unverified.

Restoration rejects incomplete/mismatched artifacts rather than reconstructing
optimizer, RNG or chronology. Guidance retains per-state measurements and uses
matched actions/actor optimizer; ordinary controls retain literal existing SAC
gradients. Passive arms share deliveries/sampling and maintain separate ordinary
SAC states. Frozen target assignment retains latent tanh preimages to avoid
artificial inverse-tanh failures at rounded action bounds. Output artifacts are
immutable on replay and published atomically. These are diagnostic interfaces,
not newly approved scientific interventions.

Scientific/execution blockers remain: the requested two MyoSuite environments
are ambiguous among four registered tasks; select the pair before generating the
20-cell manifest. Approve explicit null panel/budget/cadence/normalization/alpha,
intervention-space/zero-signal and target-evaluator choices. Supply real compatible
fork/U/I artifacts and ordered streams; replay snapshots cannot substitute for
unrecorded chronology. No new source continuation or GPU execution is authorized
by this implementation. GPU time, peak device memory, passive cost and full-width
qualification remain unmeasured. See `exp3_pilots.md` for the reuse matrix, cost
equations and staged minimal GPU plan.

## Evidence integrity

Raw evidence remains outside Git under `/workspace/scratch/`; no generated logs,
results, caches or scratch checkpoints are committed.

| File | SHA-256 |
|---|---|
| exp3-reviewed-tests.log | 4790ff16f40545f45c28fcca9f83c4ab2122c5348e4008fd44473919bbe26ac3 |
| exp3-final-update-regression.log | c55245f0fd24d21d5a4660737a221e8e79ab62bf489c14359b60cf4cb50cf6ef |
| exp3-final-compatibility-tests.log | 3ba83b103688abc79f42c75978d1125e5ae128daa23a2fed59a087a6d6da60f1 |
| exp3-atomic-output-tests.log | c3b88c8b4cb1ec0728fadd2b60a4dcca524ae33d2e15d26e96c82a0d41f205bf |
| exp3-existing-regression.log | 54a9692e0ae8cee656914a53b7c376ea3e759ad71eefa4a26cff534be0738640 |
| exp3-fork-restoration-regression.log | efa87dac6fca7f14db9bef9a37728c2c4b60d26c27ae9e3092df3669d805e6a4 |
| exp3-final-isolation.json | c8177ceac3d1889d25e5c05248e089b1bf4a2b7516a97e6d5793b9fa601838c7 |
| exp3-methodology-reference.log | 6e2c35fc1702f774598ba6479ce67f8892ba6ef69029094e335926efdf755a9d |
