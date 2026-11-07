# Block B A100 Exp1 test failures: root causes and the minimal pilot gate (2026-10-10)

The failures come from Block B job 22706349 at `b4a90cb` ([report.txt]: `tests_default` and `tests_deterministic`,
identical in both modes). Code is compared at `b4a90cb`, the approved baseline `9d6a82d` and `dcb7491`. The raw A100
tracebacks were analysed by Codex in `docs/exp12_engineering_followup.md` @ `9d6a82d` (table rows at lines 32–34);
that is the only archived source of the exception text. No production code is changed here and nothing is submitted.

Rebased for review onto `integration/exp12` @ `2d007663548be315b5bf4f438497058e6eb0f429` (originally
`claude/exp1-pilot-readiness` @ `2879b64`). Line references to `state.py` below are to `9d6a82d`; the integration tip
adds `dcb7491`'s bitwise comparison and does not change the root causes.

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

Re-checked on the integration tip (CPU): see "CPU validation on the integration tip" below.

## What none of these tests show: GPU resume across processes
- **Kill tests run training on CPU.** Both kill tests force CPU in their training subprocesses, so even a passing A100
  rerun does not qualify kill-and-resume on the GPU.
- **Block B's A100 Check 1 evidence is in-process only.** `ForkEndToEndTest` runs `exp1.run`/`exp2_arm.run`
  in-process, so its reported zero numerical-difference control/identity pairs come from one process, at tiny scale, with a forced trigger. Those old norms do not independently qualify the newer dtype/signed-zero byte checks.
  Production's control restart after a fork is also in-process (`exp1.py`: `ForkNow` → `build(fork state)`), so
  that evidence matches the pilot's path.
- **The real gap:** cross-process resume (a job restarted after a timeout or preemption) and cross-process identity
  (Exp2 arms) have never run on an A100.

`scripts/sci_investigation/gpu_resume_probe.py` (added here) provides a test of the first gap without changing production code; actual GPU qualification remains pending:
- it runs the kill-test scenario (tiny hopper-hop, crash at interaction steps 5, 60 and 95, relaunch) through the
  existing `tests/exp12_subprocess_runner.py` entry and crash hook, without overriding the backend;
- every child process records the JAX backend and device kinds it actually initialised; with `--require-backend gpu`
  any child on another backend (for example a silent CPU fallback) makes the result `INCOMPLETE`, never `PASS`;
- it runs the uninterrupted reference twice. Resumed states are compared with both references using the strict
  bitwise `state_differences` (agent, optimiser, obs normaliser, replay buffer, RNG and metadata);
- it gives one verdict with a distinct exit status:

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | `PASS` | two references identical; selected crashes exit 3 without DONE; relaunches exit 0 with DONE and complete bit-identical states; actual backend/restore evidence required |
| 1 | `RESUME_DEFECT` | references identical in these two samples, but a relaunch failed or differs: suspected resume-path problem |
| 2 | `NONDETERMINISTIC_BACKEND` | the two uninterrupted references differ, so bit-exact resume cannot be judged; resume diffs are still reported against both references |
| 3 | `INCOMPLETE` | harness/environment failure: reference failed, crash hook did not fire, state missing, or wrong backend |

  Harness failures take precedence, then nondeterminism, so a nondeterministic backend is never reported as a resume
  defect and a CPU fallback is never reported as GPU evidence.
- it keeps every child's spec, backend record, exit status, wall time, combined stdout/stderr log and run directory
  (checkpoints, results) under `--out`, plus provenance (commit, dirty paths, Python/JAX/jaxlib versions, host, Slurm
  job id and the precision/backend environment variables) in `gpu_resume_probe.json`.

The verdict logic is unit-tested on synthetic reports (`scripts/sci_investigation/test_gpu_resume_probe.py`).

## Pilot gate (dog-run D4W1536, seed 102, `run_role=dev`)

| Group | Needed before the pilot? | Path exercised by the pilot? | Minimal A100 test | Pass condition | Source fix? |
|---|---|---|---|---|---|
| ForkEndToEndTest | Recommended, not strictly needed. The pilot's own Check 1 control guard stops a bad restore safely (`fork.validate_control_restore` → `Check1Failed`). | Yes, if it triggers: in-process fork, control restart, Check 1 | `python -m unittest -v tests.test_exp12_fork.ForkEndToEndTest` | `OK`; `check1_control` and `check1_identity` pairs 0.0/0.0/0.0; injected ≤ 64 eps units; `matmul_precision == "highest"` | None |
| KillMatrixTest | No (harness fix only; training is on CPU) | No | `python -m unittest -v tests.test_exp12_fork.KillMatrixTest` (closes the historical record) | `OK` | None |
| KillAndResumeEntryPointTest | No (same reason) | No | `python -m unittest -v tests.test_exp12_foundations.KillAndResumeEntryPointTest` | `OK` | None |
| GPU cross-process resume (new probe) | Recommended. It matters only if the pilot is interrupted; required before the grid. | Only on a timeout or restart | `python scripts/sci_investigation/gpu_resume_probe.py --out <new dir> --require-backend gpu` | exit 0, verdict `PASS` | None expected |

**Reading the probe.** If the verdict is `NONDETERMINISTIC_BACKEND` (`reference_repeat_differences` non-empty), these two reference executions differ; the cause is not established. This does
not by itself show a GPU operation caused the difference, nor justify changed
resume or cross-process identity expectations. That would be a finding for your decision, not
something to fix in code.

## Reviewed first A100 gate (not authorized for execution)

The original PR proposed a one-hour/16-CPU/64GB job running the whole relevant
suite plus the eight-child probe. That allocation is **not approved**. Use
[exp1_a100_minimal_gate.md](exp1_a100_minimal_gate.md), `STAGE=gpu_resume_95`,
with the final delivered integration SHA: oneA100/fourCPUs/32GB/seven minutes,
300-second total workload limit. Completion within that cap is unmeasured.
The explicit crash95 subset includes two references, a crash, and a relaunch;
it does not claim all three scenarios or D4W1536-scale qualification.

Codex review corrected diagnostic-only gaps in the original head4afc772:
actual trained/saved/restored parameter placement is observed, restore path and
counters are checked, missing/unreadable state is INCOMPLETE, all declared
scenarios and both comparisons are required, and child logs/provenance are
streamed/persisted before completion. Step5 starts fresh (no checkpoint); steps60
and95 must demonstrably restore a trained step60 checkpoint. Copies retain the
crash checkpoint when later saves replace it. Production code is unchanged.

The verdict names are retained for compatibility, not as causal findings.
`NONDETERMINISTIC_BACKEND` establishes only a difference between two references;
CUDA, initialization, environment or harness causes remain unproven.
`RESUME_DEFECT` indicates an error/mismatch conditional on those two references
agreeing, not an independently established save/restore root cause. Two samples
do not establish backend determinism. Missing/corrupt evidence and fallback must
not qualify. Any changed scientific interpretation requires owner approval.

The old paragraph suggesting that a different resumed trajectory is valid
merely because references differ is withdrawn: it cannot authorize relaxing
approved exact equality. Tiny-scale PASS is not full-width or Exp2 qualification.

## For Codex
No source correction is needed for these failures. One suggestion only: the kill tests' docstrings claim
GPU-relevant resume coverage while forcing CPU. A one-line note, or a GPU-backend variant like this probe, would stop
a future A100 pass being over-read.

[report.txt]: logs/blockB_22706349/report.txt
