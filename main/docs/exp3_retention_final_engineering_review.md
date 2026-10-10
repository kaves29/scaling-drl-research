# Final independent Exp3 retention review

2026-10-10. **Verdict: FAIL against the requested integrity requirements.** The
logical-resave correction solves the legitimate Orbax serialization conflict,
but does not yet qualify corrupted checkpoints or finalization correctly. No
production code, scientific setting, resource allocation, or other agent branch
was changed by this review. This branch adds tests and this report only.

## Exact revisions and integration recommendation

- Reviewed Claude branch: `claude/exp3-capture-fixes-review`, exact commit
  `91e370a0c983452f3a5c1ce82b166ebd51d78ef6`, verified against the live remote.
- Its only parent is `853b6a523799132cf96a2e889150c10e3ea1c2ac`.
- Approved Exp1/Exp2 baseline: `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`.
- Findings/regressions: `codex/exp3-retention-final-review`; the enclosing full
  report commit is delivered separately, not self-referenced here.

**There is no recommended passing integration commit yet. Do not approve 91e370a
as an integrity-qualified candidate.** It can be staged on an isolated debug-only
development branch for remediation, with the failures below explicitly preserved;
that is not clearance for A100 qualification, production capture, or pilot runs.
Repair the guards, pass these regressions and peer review the resulting exact
descendant before naming a development-validation candidate.

Technically, both 844e3c0 and 853b6a5 are ancestors of 91e370a: staging on a new
branch from either exact baseline is a fast-forward, with **no merge conflicts**.
No merge was performed. Compatibility/conflicts with a different future branch
tip are not inferred. Claude's unrelated later documentation on its other branch
is not merged by this review.

## Confirmed behavior and remaining defects

| Requirement | Independent evidence | Verdict |
|---|---|---|
| Different serialized bytes, identical retained logical values | Re-encode metadata with a different pickle protocol and replay with compressed NPZ; compare values independently through the existing exact state comparator; original snapshot remains unchanged | PASS for both replay modes |
| Actual resumed final save | Claude's real Exp1/Exp2 test dies after final save, before DONE, then resumes; retained endpoint and stream remain usable | PASS, rerun independently |
| Real state differences | Same-step update-counter mutation, injection-provenance mutation, normalization dtype difference, signed-zero metadata difference; later trained state in Claude's tests | PASS: conflict markers, downstream refusal |
| Replay semantics | Retained replay reward changes conflict; omitted replay is deliberately outside the equivalence scope | PASS; omitted mode does not establish full replay equality |
| Concurrent conflicting writers | Controlled thread race and separate interpreter-process race on first publication; simultaneous conflict at an existing slot | PASS locally: no double publication/overwrite; existing-slot conflict prevents use |
| Bad source agent payload | Truncated Orbax checkpoint is rejected rather than labelled equivalent | PASS |
| Corrupt existing snapshot | Corrupt retained `_sharding`, then resave against both acceptance paths | FAIL: `retain_state` returns it without validating the record's file hashes |
| Unrestorable source metadata | Corrupt source `_sharding`; NumPy comparison reports no difference, but actual `Exp12Trainer.restore` fails | FAIL: source is labelled `EQUIVALENT` |
| DONE finalization with invalid retention | Actual Exp1 fork/Exp2 arm, death after DONE before seal, copied retention with corrupt endpoint or a conflict marker | FAIL: finalizer reports `COMPLETE` |
| Healthy DONE finalization / provenance / locking | Real source finalizes idempotently without train/save calls; altered source identity and active writer are rejected | PASS |

### E1: existing snapshot integrity is not checked before acceptance

`retention.retain_state` reads `snapshot.json` directly at its existing-target
branch. Its byte-equality shortcut compares the **recorded old hashes** to incoming
source hashes, not to current snapshot bytes. Its logical fallback likewise never
calls `read_snapshot` before loading values.

Reproduction: retain a valid source, corrupt the retained `_sharding` file, and
call retention with either an independent intact copy or a logically identical
resave. Both calls return the corrupted target. `read_snapshot` independently
raises a checksum error. The first path is inherited; the new logical path adds
another acceptance route. No original snapshot is repaired or replaced.

### E2: NumPy-value equality is weaker than actual checkpoint restorability

`logical_differences` uses `load_agent_tree`, whose explicit NumPy restore arguments
allow it to read values without the saved device/sharding. This is appropriate for
value comparison across machines, but insufficient as an integrity validator.

Reproduction: copy a real saved state, replace `agent_ckpt/_sharding` with invalid
JSON. `state_differences(original, copy)` returns `[]`; actual
`Exp12Trainer.restore(copy)` raises. Retention then writes `EQUIVALENT`, rather
than rejecting this invalid incoming checkpoint. This is a regression from using
raw byte differences to mark every mismatch as conflicting, although the old rule
also falsely rejected legitimate reserialization.

Do not restore the old byte-equality requirement. Keep exact logical comparison,
but separately validate required Orbax metadata and checkpoint integrity and
compatibility before accepting equivalence. Valid serialization/layout differences
must remain admissible. A device-independent comparison must not be represented
as proof of valid device-backed restoration.

### E3: finalization can hide an invalid retained endpoint

Using actual entrypoints, simulate death in `StreamWriter.seal` after the injected
arm has written DONE. The stream remains unsealed at the matching source cursor.
Healthy finalization passes and does not alter the source state or invoke
`Exp12Trainer.train`/`save`.

Now corrupt the selected retained endpoint or add a conflict marker. Its
`read_snapshot` call correctly fails, but `finalize_capture` calls retention without
checking the resulting snapshot and reports `capture_verdict: COMPLETE`.

This is an **availability/integrity verdict gap**, not proof that corrupted data
has been consumed scientifically: the result correctly retains
`scientific_qualification: false`, and `Artifact`/`read_snapshot` still refuse the
invalid data. Nevertheless, COMPLETE must not serve as acceptance of requested
retention. Either fail finalization for required invalid retention or explicitly
report separate stream and retention verdicts that cannot be mistaken for a usable
combined capture. Merely repairing E1 may also make these boundary tests pass;
rerun them rather than assuming so.

## Scope of provenance and concurrency conclusions

Fork/injection fields in saved metadata remain exact-comparison inputs; source
metadata identity/fingerprint is additionally guarded by the recorder/finalizer.
The standalone retention API takes no `run_metadata.json`, so its comparison
alone is not certification of every configuration, source revision or static
optimizer closure. Preserve those outer provenance guards.

Initial-publication races produce one winner and one rejected writer (an OS
exception). Conflicting writes at an existing slot produce a CONFLICT marker and
prevent downstream use. These are **local Linux filesystem results**, including
independent processes, not qualification of Delta's distributed filesystem or
power-loss durability. No cross-node test was executed.

## Recovered-request reconciliation

The recovered request is unchanged, full SHA-256:
`3a444073cbbeae02e810256121ad218115060f17f32e019b276432d55e10ab10`.
I compared its actual text against `exp3_methodology_reconciled.md`.

**Substantively faithful:** three required actors; matched post-fork comparisons
and fork-time assertions; per-state/per-action preservation; complete measurement
chain; intervention set with unfrozen normalization; matched passive keys and
Stage A first; optional Stage B and Pilot 3; preserved existing precision as the
default; explicit unresolved Myo pair/widths and numerical settings. These are
development pilots, not authorization for final scientific experiments. The
request does not require inventing final-experiment hypotheses/statistical rules
before implementing their infrastructure.

Two wording qualifications, without choosing alternatives:

1. The request requires all three actors, but does not designate formal
   **primary statistical endpoints**. Read "all three actors are primary" as
   "all three are required," not a new outcome hierarchy.
2. It requires fork-time equivalence assertions, but does not explicitly prohibit
   additional fork-time diagnostic measurements. "Fork appears only as an
   equivalence assertion" describes the mandatory minimum, not an approved ban.

The reconciliation should also keep the previously documented Pilot 3 saturation
issue visible: latent-based density and production's inverse-tanh round trip
agree mathematically away from saturation but need not have identical numerical
behavior at saturated actions. The recovered request does not authorize silently
resolving this. It is relevant if optional Pilot 3 is enabled, not a reason to
change Exp1/Exp2 or choose new capture thresholds.

No new scientific decision is made here. Source-cell scope, retention/replay
authorization, m freeze, unfrozen pilot inputs and any diagnostic deviations
remain owner matters. Seed-102 development is not automatically a five-seed Exp3
source. The preservation envelope remains 60 potentially needed cells until scope
is resolved, not 60 selected scientific sources.

## CPU verification and reproductions

Environment reused unchanged: Python 3.12.14; JAX/JAXLIB 0.4.34; NumPy 1.26.4,
Flax 0.8.4, Optax 0.2.3, Orbax 0.5.3; CPU backend and EGL. Delta's reported
Python 3.12.13 and CUDA behavior are not qualified by these runs.

| Run | Independently observed result |
|---|---|
| Claude resave/final-save tests + real capture + publication/finalization boundaries | **26/26 pass**, zero skips, 238.598 s, exit 0 |
| Exp1/Exp2 exact comparator regressions | **8/8 pass**, zero skips, exit 0 |
| New adversarial release run | **18 tests: 13 pass, 5 fail, zero errors/skips**, 72.530 s, exit 1 |
| Baseline isolation/configuration audit | **337 files byte-unchanged; 195/195 configurations match**, exit 0 |

The five failures are ordinary safety assertions, not expected-failure decorators
or weakened checks. Three are E1/E2 and two are actual E3 finalization boundaries.
No production implementation was changed to make them pass. Prior 84-test Claude
and 74-mutation/65-restoration receipts are historical; this review does not
relabel them as independently rerun current checks.

From `main/`, activate the pinned CPU environment and set:

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu EXP12_JAX_CACHE_DIR=off
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B -m unittest -v tests.test_exp3_capture_review \
  tests.test_exp3_capture tests.test_exp3_capture_boundaries
python -B -m unittest -v tests.test_exp12_bitwise_comparison
# Currently exits 1 with exactly five safety failures:
python -B -m unittest -v tests.test_exp3_retention_adversarial
```

Each failure can be reproduced individually with `python -B -m unittest -v`
and the following full test IDs:

```text
tests.test_exp3_retention_adversarial.RetentionSafetyTest.test_corrupt_retained_bytes_are_not_accepted_by_the_fast_path
tests.test_exp3_retention_adversarial.RetentionSafetyTest.test_corrupt_retained_bytes_are_not_accepted_by_logical_fallback
tests.test_exp3_retention_adversarial.RetentionSafetyTest.test_unrestorable_sharding_is_not_logically_equivalent
tests.test_exp3_retention_adversarial.RealFinalizationSafetyTest.test_real_done_finalization_does_not_claim_complete_with_corrupt_retention
tests.test_exp3_retention_adversarial.RealFinalizationSafetyTest.test_real_done_finalization_does_not_claim_complete_with_conflicted_retention
```

Evidence remains outside Git under `/workspace/scratch/`:

| File | SHA-256 |
|---|---|
| `exp3-retention-final-baseline.log` | `c7281812fd59b23e256e66c4c508310794bb2e62109b76123f1710807a1c4ce1` |
| `exp3-retention-final-release-adversarial.log` | `83221e8c375dafc13287885d635a2bd067e165acc6bbbcec10d3fc1f70120f2b` |
| `exp3-retention-final-bitwise.log` | `f5bfa5b9503edb9d1407e122af2112834903fe10a5940301a6fb9c92b659c655` |
| `exp3-retention-final-isolation.json` | `c8177ceac3d1889d25e5c05248e089b1bf4a2b7516a97e6d5793b9fa601838c7` |

## Minimal A100 qualification after remediation

**STOP now:** current CPU integrity failures are a submission blocker. No job was
submitted and no allocation is increased. The commands below are conditional
execution instructions for an independently reviewed, CPU-passing descendant,
not authorization to run this failing source.

### A: small actual-GPU retention/restart/finalization gate

Use one A100, four CPUs, 32 GB RAM, seven minutes wall time, and a 300-second
workload cap, following the existing reviewed allocation envelope. This is a
proposal requiring authorization for Exp3, not reuse of Exp1 job authorization.
The corresponding 44-test CPU coverage took about 311 s in two runs; actual A100
runtime is **unmeasured**. A 300 s completion promise is unjustified. A timeout
means incomplete coverage, not a resume defect. Stage only additional separately
approved suites if this cannot finish; do not enlarge the cap automatically.

Before obtaining any allocation, rerun all CPU checks above at the exact repaired
SHA, inspect the diff, and verify the live remote SHA and clean detached checkout.
Do not substitute 91e370a or a moving branch name for a qualified revision.

```bash
# Fresh detached checkout; run only after a passing descendant has been reviewed.
: "${QUALIFIED_SHA:?full reviewed and CPU-passing descendant SHA}"
export QUALIFIED_SHA
: "${CHECKOUT:?absolute fresh checkout path}"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$QUALIFIED_SHA"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$QUALIFIED_SHA"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
```

On an already authorized allocation, activate the existing pinned environment:

```bash
module reset
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false WANDB_MODE=disabled
unset JAX_PLATFORM_NAME
export MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CHECKOUT/main"
: "${GPU_OUT:?absolute new evidence directory outside checkout}"
mkdir -p "$(dirname "$GPU_OUT")"
set +e
/usr/bin/time -v -o "${GPU_OUT}.resource.log" \
  timeout --signal=TERM --kill-after=5s 300s python -B -u - "$GPU_OUT" <<'PY' \
  > "${GPU_OUT}.launcher.log" 2>&1
import json, sys, unittest
from pathlib import Path
import jax, jaxlib
from scripts.exp3_gpu_validation import check_source
from experiments.exp12.precision import configure_compilation_cache, runtime_info, set_matmul_precision
import os
root = Path.cwd().parent
head = check_source(root, os.environ['QUALIFIED_SHA'])
assert jax.__version__ == jaxlib.__version__ == '0.4.34'
assert jax.default_backend() == 'gpu'
assert len(jax.devices()) == 1
assert jax.devices()[0].device_kind == 'NVIDIA A100-SXM4-40GB'
assert jax.local_devices(backend='cpu')
out = Path(sys.argv[1]).resolve()
assert out != root and root not in out.parents
out.mkdir(parents=True, exist_ok=False)
set_matmul_precision()  # Existing production policy, not a new precision choice.
configure_compilation_cache()
(out / 'backend.json').write_text(json.dumps({'commit': head, 'python': sys.version, 'runtime': runtime_info()}, indent=2))
names = ['tests.test_exp3_retention_adversarial', 'tests.test_exp3_capture_review',
         'tests.test_exp3_capture', 'tests.test_exp3_capture_boundaries']
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
expected = suite.countTestCases()
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
passed = expected > 0 and result.testsRun == expected and result.wasSuccessful() and not result.skipped
report = {'commit': head, 'expected_tests': expected, 'tests_run': result.testsRun,
          'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
          'verdict': 'ENGINEERING_PASS' if passed else 'FAIL_OR_INCOMPLETE',
          'scientific_qualification': False}
(out / 'validation.json').write_text(json.dumps(report, indent=2))
sys.exit(0 if passed else 1)
PY
status=$?
set -e
printf '%s\n' "$status" > "${GPU_OUT}.exit_status.txt"
test "$status" -eq 0
```

Export `QUALIFIED_SHA` before this command so the child sees the exact revision.
Store launcher/resource/exit-status files alongside the evidence directory, and
archive/checksum all of them. Expected outputs: `backend.json`, `validation.json`,
all verbose test traces, source commit, workload exit, host RSS/wall time. The real
capture tests independently verify actual saved/restored floating-state placement
and FP32 dtype; a GPU device listing alone is insufficient.

**PASS:** all selected tests executed, zero failures/errors/skips, GPU backend and
saved/restored state verified, exit 0 within cap. **FAIL:** stop and inspect complete
artifacts; a bitwise difference is not attributed to a resume defect until
uninterrupted references are independently deterministic. **TIMEOUT 124, kill,
missing outputs, backend fallback, source mismatch:** INCOMPLETE/preflight failure;
preserve artifacts and stop. No resubmission, allocation extension or new numerical
tolerance is authorized by this plan.

This gate explicitly adds Claude's four tests and this review's adversarial tests.
The current generic `exp3_gpu_validation.py` EXACTNESS list omits them; running it
alone would not qualify this retention fix.

### B: scaled checkpoint re-save on actual Delta storage

After A passes, reuse one **owner-selected existing GPU-trained scaled checkpoint**
and its matching metadata. Do not generate a new parent, natural fork, source cell
or m just for this test. Missing compatible artifacts are an evidence blocker.

Minimum experiment: restore through `Exp12Trainer`, verify actual GPU parameter
and optimizer placement and FP32, save to an external owned directory, close/drop
that trainer, restore the new save into a fresh trainer, save again, and compare
all state exactly with the existing comparator (only its existing telemetry
omission). Retain both saves at the actual step with explicit replay mode; require
no CONFLICT and a verified usable original snapshot despite serialization noise.
This performs **zero train calls, environment interactions or additional updates**.
Repeat controlled publication/retention failures on Delta storage; never damage
the supplied source or reuse its writable root.

After A passes at the repaired SHA, the exact command on a separately authorized
allocation is below. Reuse A's pinned environment/backend variables, export
`QUALIFIED_SHA`, and supply the four owner inputs; no source cell or replay policy
is selected here. This command has not been executed on an A100.

```bash
: "${SOURCE_STATE:?absolute owner-approved complete GPU-trained checkpoint}"
: "${SOURCE_METADATA:?its matching run_metadata.json}"
: "${GPU_OUT_B:?absolute new external evidence directory}"
: "${REPLAY_MODE:?explicit retain or omit}"
export QUALIFIED_SHA
cd "$CHECKOUT/main"
mkdir -p "$(dirname "$GPU_OUT_B")"
set +e
/usr/bin/time -v -o "${GPU_OUT_B}.resource.log" \
  timeout --signal=TERM --kill-after=5s 300s python -B -u - \
  "$SOURCE_STATE" "$SOURCE_METADATA" "$GPU_OUT_B" "$REPLAY_MODE" <<'PY' \
  > "${GPU_OUT_B}.launcher.log" 2>&1
import gc, json, os, sys
from pathlib import Path
import jax, jaxlib, numpy as np
from omegaconf import OmegaConf
from scripts.exp3_gpu_validation import check_source
from experiments.exp12.precision import set_matmul_precision, configure_compilation_cache, runtime_info
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp12.state import state_differences
from experiments.exp3.artifacts import inspect_artifact, digest, state_fingerprint, core
from experiments.exp3.retention import retain_state, read_snapshot
root = Path.cwd().parent
head = check_source(root, os.environ['QUALIFIED_SHA'])
assert jax.__version__ == jaxlib.__version__ == '0.4.34'
assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
assert jax.devices()[0].device_kind == 'NVIDIA A100-SXM4-40GB'
assert jax.local_devices(backend='cpu')
source, metadata, out = [Path(p).resolve() for p in sys.argv[1:4]]
mode = sys.argv[4]
assert mode in ('retain', 'omit')
assert out != root and root not in out.parents
assert out != source and source not in out.parents
out.mkdir(parents=True, exist_ok=False)
(out / 'wandb').mkdir()
os.environ['WANDB_DIR'] = str(out / 'wandb')
assert os.environ.get('WANDB_MODE') == 'disabled'
set_matmul_precision()
configure_compilation_cache()
info, meta, _, _ = inspect_artifact(source, metadata)
assert info['identity']['architecture'] in ('D4W1024', 'D4W1536')
source_hash, metadata_hash = state_fingerprint(source), digest(metadata)
(out / 'backend.json').write_text(json.dumps({'commit': head, 'runtime': runtime_info(), 'python': sys.version}, indent=2))
def placed_on_gpu(trainer):
    agent = core(trainer.agent)
    for name in ('_actor', '_critic', '_target_critic', '_temperature'):
        network = getattr(agent, name)
        for field in ('params', 'opt_state'):
            for leaf in jax.tree_util.tree_leaves(getattr(network, field)):
                if hasattr(leaf, 'dtype') and np.issubdtype(leaf.dtype, np.floating):
                    assert str(leaf.dtype) == 'float32'
                    assert {device.platform for device in leaf.devices()} == {'gpu'}
def resave(input_state, name):
    # Scientific config unchanged; only trainer/output directory differs.
    trainer = Exp12Trainer(OmegaConf.create(info['resolved_config']), str(out / (name + '_run')))
    try:
        trainer.restore(input_state)
        placed_on_gpu(trainer)
        assert trainer.interaction_step == meta['interaction_step']
        assert trainer.update_step == meta['update_step']
        # No start(), train(), sample_actions(), or evaluation call.
        return trainer.save(out / name)
    finally:
        trainer.close()
first = resave(source, 'first_state')
gc.collect()
second = resave(first, 'second_state')
gc.collect()
assert not state_differences(source, first)
assert not state_differences(first, second)
target = retain_state(first, out / 'retained', meta['interaction_step'], mode)
retain_state(second, out / 'retained', meta['interaction_step'], mode)
read_snapshot(target)
assert not list((out / 'retained').glob('CONFLICT_*.json'))
assert state_fingerprint(source) == source_hash and digest(metadata) == metadata_hash
(out / 'validation.json').write_text(json.dumps({'commit': head, 'source_code': info['code'],
    'source_identity': info['identity'], 'source_state_sha256': source_hash,
    'source_metadata_sha256': metadata_hash, 'replay_mode': mode,
    'interaction_step': meta['interaction_step'], 'update_step': meta['update_step'],
    'verdict': 'ENGINEERING_PASS', 'scientific_qualification': False}, indent=2))
PY
status=$?
set -e
printf '%s\n' "$status" > "${GPU_OUT_B}.exit_status.txt"
test "$status" -eq 0
```

Keep the two new complete states, retained snapshot/audit records, backend and
validation JSON, launcher/resource logs and exit status. Hash/archive them together.
A source/input error or exception stops the command; a missing final JSON or timeout
is INCOMPLETE. The source commit and provenance are recorded separately from the
executing candidate. The command does not certify that every different source
revision is compatible merely because it loads. Use an independently approved
compatible source, and retain the provenance review with the receipt.

Controlled publication/retention failure tests run in A on the same Delta storage;
B supplies scale and a real owner-selected checkpoint. It does not generate a new
large-source training trajectory. If real-width continuation parity itself is
still outstanding, it remains a separate authorized gate; a no-training resave is
not a replacement.

Use the same one-A100/four-CPU/32-GB/seven-minute envelope and 300-second cap only
if separately authorized. An older ordinary D4W1536 source measured approximately
6.1 s per save and 2.5 s per restore, but cold CPU initialization, replay, logical
comparison, injected-head size and filesystem I/O are additional unmeasured costs.
No completion guarantee is inferred from that receipt. This isolated re-save
avoids the known expensive identity_warm training/probe workload; it does not
replace that scientific gate.

Storage planning: D4W1536 ordinary agent payload is approximately 1.22 GB; an I
state adds about 0.30–1.21 GB depending on the already chosen head. Two temporary
complete states plus replay, retained links/copies and logs need additional owned
space; measured allocation and actual m/replay determine the exact quota. The
earlier whole dog-run cell estimate is 19.2–24.6 GB before extra U tails/pilot
outputs. Peak host/device memory is unmeasured for this probe; 32-GB host and
40-GB device are a proposed envelope, not a demonstrated requirement. Each
seven-minute allocation costs at most about 0.117 A100-hours; no jobs are booked.

B passes only with valid compatible source, no source modification, both exact
state comparisons, actual GPU save/restore, verified retention, correct counters,
and completion within the approved cap. Any mismatch, conflict, corruption,
missing artifact or timeout stops qualification. A+B validate capture engineering
only; source-scope decisions, scientific gates, distributed-filesystem limitations
and production authorization remain separate.

## Handoff

Codex owns this isolated final-review branch; Claude owns the implementation
under review. The Exp1/Exp2 launch-readiness checkout remains clean at
`7d21eaa106a336168464cd5418a59136b6936a64`; integration/exp12 remains at 844e3c0.
Next engineering actions: guard existing snapshot integrity, validate incoming
restorability metadata separately from value equality, and make retention failure
visible at finalization. Preserve legitimate logical resaves and strict comparisons.
Then rerun the failing IDs, the real-entrypoint suite and configuration audit and
publish a new exact candidate for independent review. No scientific alternative
has been adopted, no source branch merged, and no GPU job submitted.
