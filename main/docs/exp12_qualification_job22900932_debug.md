# Job 22900932: CPU debugging of recording-isolation qualification

The failure is a harness precision expectation error, confirmed against Delta's
original log and source. The `production_tf32` triple checks for `highest` at
the first completed interaction, although the Exp1 GPU entry point selects
`tensorfloat32`. No TF32 fingerprint was written before the exception. This
failure does not measure training divergence.

The recovered debugging branch already contained commit
`723da0d1c0c6f95910468a4aae9c94f1c40f60ed`, which fixes the precision assertion
and excludes only the three declared recording-only meters from checkpoint
comparison. This continuation adds final-checkpoint step validation, explicit
failed-arm receipts, accurate backend/interaction scope, and stronger CPU tests.
The script and tests were formatted with Black. Production training code,
scientific configuration, tolerances and the frozen baseline are unchanged.

## Sources and historical evidence

- Frozen baseline: `087feab341b9be1f377697668f2d1c750ec0467f`.
- Failed job source/harness: `12839f656e53b00a3327384f2cb4fc1be47e73fe`.
- Recovered branch/base of this continuation:
  `723da0d1c0c6f95910468a4aae9c94f1c40f60ed`.
- Authorized branch: `codex/exp12-qualification-debug`; resolve its final SHA
  from GitHub. No merge or GPU job was performed.
- Delta original artifacts:
  `/work/hdd/biqc/skaveti1/exp12_recording_qualification_20261014T182709Z/run`.
- The original `evidence.sha256` manifest verified completely on Delta.
- Original harness SHA256:
  `80e1e4ab06530e6b2c031423d6817c4075c336b712d863381d5a36939b6c0ed4`.
- `sacct`: FAILED, exit `1:0`, elapsed `00:06:55`.
- Historical highest triple: PASS at that source/configuration, 400 interactions,
  782 updates per arm, identical fingerprint hashes and complete checkpoints.
- Historical TF32 triple: INCOMPLETE, `off_a` marked RUNNING, zero fingerprints;
  exception at `observed_step`, line 226. The new failed-receipt handling prevents
  subsequent failures from retaining that misleading arm status.

## CPU verification

Evidence stays outside Git at
`/Users/shouryakaveti/VS_Projects/exp12-debug-evidence`.
The isolated `cpu-venv` inherits the existing Python 3.12 research packages via
a `.pth` file, with local NumPy 1.26.4, SciPy 1.13.1 and pandas 2.1.4 overrides.
JAX/JAXLIB 0.4.34, Flax 0.8.4 and MuJoCo 3.6.0 are used. This is CPU engineering
coverage on macOS, not a reproduction of Delta's CUDA environment or its complete
dependency lock. Initial attempts with NumPy 2 failed in TensorFlow Probability;
those logs are retained and are not counted as passing coverage.

The extended driver bypasses only the A100 backend gate and emulates the GPU
entry point's precision setter on CPU. It retains the real Exp1 entry point,
training loop, checkpoint implementation and per-interaction fingerprints.
Both configured precision settings pass three 400-interaction arms with probes
enabled and 782 updates per arm. CPU arithmetic does not establish TF32 GPU
equivalence. `extended-qualification/summary.json` binds the run receipts to
the harness SHA256 and the uncommitted patch digest used during validation.

Both original defects were reintroduced only in memory: the old precision
expectation raises the original exception, and the old meter comparison blocks
a 41-interaction triple on `meta:meters`. Restoring the fixes passes. The shorter
regression triples now cross two logging windows and save a partial third
window. Negative tests reject precision drift and stale final checkpoints;
failure handling restores the training hook and leaves the verdict INCOMPLETE.
The production backend guard still rejects CPU execution.

Both single performance-reference modes also pass CPU smoke checks. They retain
Exp1's ordinary checkpoint callback and do not add the qualification fingerprint
hook or produce fingerprint files.

A clean, detached CPU checkout of the frozen baseline was created solely for
read-only source comparison, with all outputs outside it. Separate unhooked
400-interaction runs of frozen ON, frozen genuine OFF and candidate ON completed
with 782 updates each. Candidate ON equals frozen ON's complete checkpoint
(excluding only the run id). Candidate ON versus genuine frozen OFF has no
training/ancillary differences after the declared recording exclusions and zero
agent-leaf deviation. The original and new frozen checkouts remain clean at
`087feab341b9be1f377697668f2d1c750ec0467f`. Receipts and comparison are in
`cpu-source-comparison.json`. This CPU result does not replace the historical
GPU divergence or establish computational non-invasiveness on A100.

The broader CPU cycle found test portability defects, which were fixed without
changing production logic: the Linux affinity mock can be created on macOS;
shared-filesystem fixtures explicitly preserve hard links and supply both mount
descriptions; the fake `sbatch` uses a real POSIX `fcntl` lock instead of requiring
the Linux `flock` CLI; fsync ordering records opened descriptor paths without
`/proc`. A twin-critic oracle had demanded bitwise equality between eager and
compiled neural-network arithmetic (observed maximum absolute Q difference
`8.940697e-7`). Its independent real-critic comparison now uses eager execution
on both sides, with the same assertions and tolerances. A separate compiled
linear-twin known-answer test checks Q layout and which critic supplies each
action gradient exactly. The original wrong-Q-gradient mutation fails and the
restored test passes. No scientific tolerance was relaxed.

Additional checks include the repository's isolated CPU module suite,
`tests.exp12_break_checks`, `check_exp12_cpu_environments.py`,
`methodology_cpu_reference.py`, shell syntax and diff/format checks. Their exact
counts, skips, exit codes and log checksums are retained in the external receipts.
The final complete run is `final-clean-cpu`, with stable executable/configuration
inputs and CPU environment: **79 modules, 765 tests run, 760 passed, five
HumanoidBench tests skipped, zero failed modules, exit 0**. The runner confirms
`source_unchanged=true`; its source fingerprint is
`427ac810143706c21cde9966a51d62940d777afd96197166c657d661be5830a6`.
The validated harness SHA256 is
`0880e7be8fa83a5e42249391f21bd7a5e4ac82004128528b94b040800fbe0c42`.
Earlier failed runs and targeted reruns remain
available rather than being overwritten. The 74-case break/restoration run
initially had one restored-oracle failure; its targeted rerun passed after the
oracle repair. The two qualification-harness mutations are additional checks.
GNU coreutils are installed locally for the Linux launcher fixtures, and
`filelock` is installed in the isolated CPU environment for eviction-cache
coverage. These are test-environment prerequisites, not production changes.
Simulator construction and exact next-transition/action-RNG restoration pass
55 available cells (11 tasks by five seeds). Ten HumanoidBench cells are
UNAVAILABLE because the dependency is absent in the local environment.

## Remaining external limits

GitHub `ls-remote`, authenticated Git dry-run push and Delta BatchMode SSH work.
Delta's `accounts` output exposes only `biqc-delta-gpu`. A CPU-only allocation
attempt was rejected: “Jobs requesting no GPU resources must be submitted under
CPU type accounts.” No GPU allocation or job was requested. This prevents using
Delta's installed HumanoidBench CPU environment without a CPU account.

The repaired TF32 triple and final combined qualification still require a
source-bound A100 rerun. That work is outside this CPU-only authorization.
Historical highest evidence does not qualify the updated source. Computational
non-invasiveness versus genuine no-diagnostic computation remains a separate,
unresolved scientific/GPU gate. Neither Exp1 nor Exp2 gains launch approval.
