# Reviewed Exp1 development gate (not a launch authorization)

This review targets `integration/exp12`, starting at
`2d007663548be315b5bf4f438497058e6eb0f429`. Use the exact final published SHA
in the delivery report, not a moving branch tip, for every command below.
No job was submitted. No scientific settings or resource allocations were changed.

## Reviewed changes and limits

- Scientific handoff `cb33733aee96e882b10417316227a64bf732f38d` adds one document.
- Pilot readiness `3734215e7c546e53871a7da87d6871b627bafbbe` adds one document
  above that handoff. Both retain their original historical evidence and proposals.
- Cache fix `d23191befd1d6a3254c61ac1edbad363b9b744e5` changes only persistent
  cache publication and its tests/mutation. Review follow-up preserves JAX's
  empty-key exception, cleans ordinary failed writes and strengthens byte-exact,
  publication-boundary and concurrent-writer tests. No compiled payload, cache key,
  optimizer, RNG, math, precision or configuration is changed.
- The pilot branch subsequently advanced to `2879b64499490e93bc6374465a6b7b2bcbd2724e`.
  Its two extra files are **not integrated** under the pinned-review authorization.

Atomicity is visibility of complete `-cache` entries through same-directory
POSIX rename, not power-loss durability. Readers can miss an unpublished entry
and compile normally. Simultaneous writers can publish different complete values
for the same key; the patch does not provide a concurrent first-writer lock
(the original unlimited cache also lacked one). Sequential existing entries are
retained. Published files use mkstemp's owner-only mode (0600), rather than
write_bytes' umask-derived mode; this review assumes the same Delta user across
workers. Cross-user shared-cache reuse is not qualified. The separate eight-byte access timestamp retains JAX's existing writes;
unlimited caches do not use it for eviction. Evicting and non-filesystem-URI paths
still delegate to JAX. SIGKILL before rename can leave an ignored `.tmp` file,
not a truncated published entry. Previously corrupt entries are not repaired.
No shared cache is deleted. Delta filesystem rename behavior and CUDA cold/warm
identity still need actual GPU evidence. This private-JAX adapter is reviewed for
pinned JAX/JAXLIB 0.4.34; do not silently upgrade that dependency.

Reports are evidence, not amendments. In particular, the handoff's categorical
"no implementation defect" and "GPU numerics excluded" wording is its author's
interpretation: CPU reproduction establishes that CUDA is not necessary for the
hopper failure, not that every CUDA effect is excluded. Its proposed larger
identity budget is **not authorized** by merging documentation. The pilot report's
16-CPU/64-GB/eight-hour allocation remains a proposal. The unsigned entropy wording,
positive-control success ambiguities and scientific gate dispositions remain
owner decisions. No alternative is adopted by this review.

## Minimal coverage and what it proves

| Stage | Existing checks | Meaning |
|---|---|---|
| CPU prerequisite | ForkEndToEndTest, ForkUnitTest, KillMatrixTest, IdentityValidationTest, KillAndResumeEntryPointTest; restore/comparator tests and mutations | Regression coverage, including interrupted saves/forks, exact state comparison and rejection of a corrupted control restore. Not CUDA qualification. |
| A100 cache | AtomicCacheWriteTest | Concurrent-reader safety and CPU-process cache I/O; the JIT cold/warm equality test runs on the selected GPU. This is not full continuation identity. |
| A100 fork | Three selected ForkEndToEndTest methods below | Actual-backend tiny single-critic fork/control/identity/injected continuations, Check 1 highest precision and existing exact/64-eps rules. Forced tiny validation fixture, not a natural positive control or D4W1536 qualification. |
| A100 restore | RestoreRuntimeTest | Saved-device-independent comparison and single/twin parameter, optimizer, RNG/reference-batch restoration followed by updates; the wrapper checks shape/dtype/bytes. Tiny critics, not full pilot-size checkpoints. |
| A100 pilot-size smoke | Existing preflight_checkpoint_check.py, D4W1536 dog-run seed102 with fork | Real-width production entry point and saved state, in-process control Check 1 plus injected arm. Existing shortened dev smoke/probe settings; not full scientific probe qualification. |
| Historical closure (optional GPU rerun) | Two KillMatrix methods and KillAndResumeEntryPointTest | Training subprocesses explicitly use CPU. On an A100 the comparison process runs on GPU. Do not call this GPU kill/resume qualification. |

Historical Block B job22706349 errors are documented with exception text in
[exp12_engineering_followup.md](exp12_engineering_followup.md): the kill tests
saved CPU arrays then attempted device-specific comparison restoration;
`load_agent_tree` now reads NumPy arrays. The injected Check 1 test demanded
zero rather than the already approved 64-eps rule. The current checks retain the
stronger dtype/byte comparator from dcb7491; no assertion is relaxed here.

**Remaining gap:** GPU cross-process restart of actual training and full-width
cold/warm identity. Neither CPU-forced kill tests nor in-process Check 1 proves
this. The newer, excluded Claude revision proposes a GPU resume probe; review
and authorization to incorporate that revision remain separate. A passing
uninterrupted development pilot does not retrospectively qualify resume or Exp2.
Default-mode GPU nondeterminism, if observed, must be reported and reviewed rather
than cured by selecting flags/tolerances. No unset CUDA diagnostic bound is filled.

## Exact future Delta commands

These are prepared commands for owner-authorized execution. Run nothing on a
login node that initializes JAX. The wrapper below retains the existing diagnostic
allocation **one A100, four CPUs, 32 GB, seven minutes**, with a **300-second total
workload timeout and 15-second kill grace**. One stage per allocation; no longer
budget is implied. Feasibility of the fork/smoke stages is unmeasured. A timeout is
an incomplete gate, not a pass or proof of a correctness failure. Do not extend
it automatically. The historical kill methods may exceed this cap even though
their CPU regressions pass; their optional rerun is not the first GPU priority.

Fresh checkout and setup (replace the first literal with the full delivered SHA):

```bash
set -euo pipefail
export EXPECTED_COMMIT=REPLACE_WITH_DELIVERED_FULL_SHA
export VALIDATION_ROOT=/work/hdd/biqc/skaveti1/exp12_validation/${EXPECTED_COMMIT}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$VALIDATION_ROOT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$VALIDATION_ROOT/checkout"
git -C "$VALIDATION_ROOT/checkout" fetch origin integration/exp12
git -C "$VALIDATION_ROOT/checkout" checkout --detach "$EXPECTED_COMMIT"
test "$(git -C "$VALIDATION_ROOT/checkout" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$VALIDATION_ROOT/checkout" status --porcelain --untracked-files=all)"
cd "$VALIDATION_ROOT/checkout/main"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
# Reuse the reviewed Delta environment. Do not install/upgrade dependencies here.
```

Save the following wrapper **outside the checkout** as
`$VALIDATION_ROOT/a100_gate.sh`. Production execution and the checks' precision contexts are unchanged.
The restore shard strengthens its existing value-equality test helper to the
approved shape/dtype/byte comparator for checkpoint state. Its existing
metric oracle retains the intentional unscheduled actor-gradient NaN marker;
no tolerance is introduced. The workload is saved as a file with a main guard, so multiprocessing spawn
does not attempt to import `<stdin>` or rerun the parent gate. Workload logs,
source/config identities and complete fork fixtures
are retained outside Git. The fork fixture copy occurs only in test teardown.

```bash
#!/bin/bash
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${VALIDATION_ROOT:?}" "${STAGE:?}" "${SLURM_JOB_ID:?}"
cd "$VALIDATION_ROOT/checkout/main"
source scripts/check_exp12_validation_checkout.sh
exp12_check_checkout tests/test_exp12_fork.py tests/test_exp12_foundations.py \
  tests/test_exp12_compilation_cache.py tests/test_exp12_runtime.py \
  scripts/preflight_checkpoint_check.py configs/base_exp12.yaml
export OUT="$VALIDATION_ROOT/${STAGE}_${SLURM_JOB_ID}"
mkdir "$OUT"
mkdir "$OUT/tmp" "$OUT/jax_cache"
export TMPDIR="$OUT/tmp" EXP12_JAX_CACHE_DIR="$OUT/jax_cache"
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_FLAGS
# Production selects TF32 itself; Check 1 keeps its highest/FP32 context.
git rev-parse HEAD > "$OUT/commit.txt"
git status --porcelain --untracked-files=all > "$OUT/git-status.txt"
cp configs/base_exp12.yaml "$OUT/base_exp12.yaml"
python -m pip freeze > "$OUT/dependencies.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$OUT/nvidia-smi.csv"
cat > "$OUT/workload.py" <<'PY'
def main():
    import json, os, shutil, sys, unittest
    from pathlib import Path
    import jax, jaxlib
    sys.path.insert(0, os.getcwd())
    out = Path(os.environ['OUT'])
    devices = jax.devices()
    assert sys.version_info[:2] == (3, 12)
    assert jax.__version__ == jaxlib.__version__ == '0.4.34'
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    assert devices[0].device_kind == 'NVIDIA A100-SXM4-40GB'
    assert jax.local_devices(backend='cpu')
    from experiments.exp12.precision import set_matmul_precision, configure_compilation_cache
    set_matmul_precision()
    configure_compilation_cache()
    assert jax.config.jax_compilation_cache_max_size == -1, "this review qualifies the existing unlimited-cache policy only"
    (out / 'backend.json').write_text(json.dumps(dict(
        python=sys.version, jax=jax.__version__, jaxlib=jaxlib.__version__,
        devices=[str(d) for d in devices], cpu_devices=[str(d) for d in jax.local_devices(backend='cpu')],
        device_kind=devices[0].device_kind, x64=jax.config.jax_enable_x64,
        matmul_precision=jax.config.jax_default_matmul_precision,
        cache_max_size=jax.config.jax_compilation_cache_max_size,
        platforms=os.environ['JAX_PLATFORMS'], stage=sys.argv[1]), indent=2))
    stage = sys.argv[1]
    if stage == 'pilot_smoke':
        import subprocess
        sys.exit(subprocess.call([sys.executable, '-u', 'scripts/preflight_checkpoint_check.py',
            '--experiment', 'exp1', '--with-fork', '--checkpoint_dir', str(out / 'checkpoint_smoke'),
            '--override', 'critic_num_blocks=4', '--override', 'critic_hidden_dim=1536',
            '--override', 'env_name=dog-run', '--override', 'env=dmc_hard', '--override', 'seed=102']))
    names = {
        'cache': ['tests.test_exp12_compilation_cache.AtomicCacheWriteTest'],
        'restore': ['tests.test_exp12_runtime.RestoreRuntimeTest'],
        'fork': [f'tests.test_exp12_fork.ForkEndToEndTest.{name}' for name in
                 ('test_fork_plan_and_files', 'test_identity_fork_is_bit_identical', 'test_check1')],
        'kill_before': ['tests.test_exp12_fork.KillMatrixTest.test_exp1_before_the_fork'],
        'kill_after': ['tests.test_exp12_fork.KillMatrixTest.test_at_and_after_the_fork'],
        'kill_entry': ['tests.test_exp12_foundations.KillAndResumeEntryPointTest.test_kill_points'],
    }[stage]
    if stage == 'restore':
        from tests.test_exp12_runtime import RestoreRuntimeTest
        from experiments.exp12.state import bitwise_equal
        original_tree = RestoreRuntimeTest.assert_tree_equal
        def exact_tree(self, left, right):
            # This metric deliberately contains NaN at unscheduled diagnostic steps.
            # Preserve the existing metric oracle; strengthen checkpoint state only.
            if isinstance(left, dict) and 'train/actor_grad_cosine' in left:
                return original_tree(self, left, right)
            a, ta = jax.tree_util.tree_flatten(left)
            b, tb = jax.tree_util.tree_flatten(right)
            self.assertEqual(ta, tb)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertTrue(bitwise_equal(x, y), f'restored leaf {i}: shape/dtype/bytes differ')
        RestoreRuntimeTest.assert_tree_equal = exact_tree
    if stage == 'fork':
        from tests.test_exp12_fork import ForkEndToEndTest
        original = ForkEndToEndTest.tearDownClass
        @classmethod
        def preserve(cls):
            shutil.copytree(cls.tmp, out / 'fork_fixture')
            original()
        ForkEndToEndTest.tearDownClass = preserve
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (out / 'test-result.json').write_text(json.dumps(dict(
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skips=[(test.id(), reason) for test, reason in result.skipped], pass_=result.wasSuccessful()), indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
PY
set +e
timeout -k 15 300 python -u "$OUT/workload.py" "$STAGE" > "$OUT/tests.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$OUT/exit_status.txt"
# Do not treat expected crash-hook exits inside kill tests as outer failures.
# Preserve raw logs; warnings/errors still require review even when status is zero.
(cd "$OUT"; find . -type f ! -path './jax_cache/*' \
  ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
exit "$status"
```

After approval, submit **sequentially**, review each artifact before the next:

```bash
# Do not execute these submission commands without owner authorization.
# Start cache, then restore, then fork, then pilot_smoke; one command per stage.
export STAGE=cache  # then restore; then fork; then pilot_smoke
sbatch --job-name="exp12_gate_${STAGE}" \
  --output="$VALIDATION_ROOT/${STAGE}_%j.out" --error="$VALIDATION_ROOT/${STAGE}_%j.err" \
  --export=ALL,EXPECTED_COMMIT,VALIDATION_ROOT,STAGE "$VALIDATION_ROOT/a100_gate.sh"
# Optional historical-record closure: STAGE=kill_before, kill_after, kill_entry.
```

After completion, preserve outputs including Slurm stdout/stderr (no JAX required):

```bash
cd "$VALIDATION_ROOT"
find . -type f ! -path '*/jax_cache/*' ! -path './checkout/*' \
  ! -name evidence.sha256 -print0 | sort -z | xargs -0 sha256sum > evidence.sha256
tar --exclude=checkout --exclude='*/jax_cache' \
  -czf "${VALIDATION_ROOT}.tar.gz" .
sha256sum "${VALIDATION_ROOT}.tar.gz" > "${VALIDATION_ROOT}.tar.gz.sha256"
```

## Qualification and pilot decision

Require exact commit, clean source, expected A100 and both backends; outer status0;
expected test counts cache8, restore3, fork3 (kill_before1, kill_after1, kill_entry1
with all subtests passing). No unexpected skips. Eviction-only filelock skip,
if present, must be explicitly reported; it does not cover/qualify bounded caches.
Cache warm hit must occur and output dtype/shape/bytes must match. No truncated
cache reads or hidden numerical/Check1 errors. Fork state/identity continuation
must compare with no differences; all Check1 JSON passes, exact control/identity
representation and highest precision; injected arm keeps the approved64-eps rule.
For pilot_smoke retain both logs, DONE, LATEST/meta/buffer/agent/RMS, ledgers,
FORK_READY and all Check1 JSONs; exit0 alone is not the evidence review.
Its smoke m=`last` is a pre-existing plumbing fixture, not an m selection.

The proposed full-length pilot remains dog-run/D4W1536/seed102/run_role=dev,
1M raw steps, 500k interactions, UTD2/action repeat2, actorD1W128, full approved
probe and natural zero-threshold trigger. No confirmatory result is created by
these forced/tiny tests. After the gates pass it is an engineering candidate for
an **explicitly authorized uninterrupted development run**, with measured runtime
and storage reviewed and its allocation approved. The seven-minute allocation
cannot run that full-length pilot. The historical eight-hour pilot proposal is
not approval to allocate it. If restart capability is required before launch,
review/qualify a genuine GPU cross-process resume gate first; do not assume it
from the CPU-forced tests. A lack of natural trigger is a result, not permission
to force one. The 195-run campaign and Exp2 remain blocked by scientific gates,
positive-control/m and broader final-source GPU qualification. No criteria are
changed to remove these blockers.
