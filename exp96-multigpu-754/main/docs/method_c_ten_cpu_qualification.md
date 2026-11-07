# Ten-function Method C CPU qualification (15 October 2026)

This is a versioned candidate. `configs/base_exp12_ten.yaml` selects ten
targets, 50 checks, and a separate W&B project; the historical five-target
`configs/base_exp12.yaml` is unchanged. The old pilot commit
`f8619dd5f494399d9febc79fc9092010234c714d` and its evidence remain
historical. This report is CPU engineering evidence, **not** A100, bootstrap
coverage, construct-validity, or official-study qualification.

The affected combined CPU suite passed **51 tests** across the ten-function,
historical probe/Method C, twin-reporting and dynamic-scheduler modules.
The full-budget synthetic ten-function probe was included. The slowest tiny
ten-function save/restore smoke took 23.31 seconds; the forced-fork smoke took
9.44 seconds on this cached CPU environment. These small-model timings do
not estimate full-width checkpoint-control runtime.

Reproduction from the repository root, with the previously uploaded pilot
archive extracted to `PILOT_ARCHIVE_ROOT`:

```bash
PILOT_ARCHIVE_ROOT=/workspace/scratch/exp12-pilot-audit PYTHONPATH=main \
  /workspace/scratch/a100-audit-venv/bin/python -m pytest -q \
  main/tests/test_exp12_ten_cpu_qualification.py \
  main/tests/test_exp12_twin_reporting.py \
  main/tests/test_exp12_dynamic_scheduler.py \
  main/tests/test_exp12_probe.py \
  main/tests/test_exp12_method_c.py \
  main/tests/test_method_c_ten_reference.py
```

The scratch Python path is specific to this Codex workspace. On another host,
use the equivalent pinned environment and supply the original archive path.

## Focused CPU coverage

| Check | Result / limit |
|---|---|
| Ten target seeds and pairing | Distinct deterministic teacher keys and numerical sine draws; current/fresh share normalized inputs, actions, base target and minibatch indices within a round. The first five keys preserve the old sequence. |
| Ten-point IQM and bootstrap | Production functions match the isolated empirical-quantile reference and independent CPU oracle. Tests cover all-zero, all-negative, all-positive, tied and extreme-outlier scores, exact 10,000 resamples, type-7 linear percentiles, and strict lower bound `>0`. An extreme negative draw can keep the bootstrap lower bound negative when the point IQM is positive; this is expected uncertainty, not a test failure. |
| Schedule and fork arithmetic | 50 check steps are 10k…500k agent interactions; with repeat 2 they are 20k…1M raw frames. Last eligible check is 47 (470k). The 25% horizon remains 125k interactions; a check-47 fork ends at 595k, under the existing 600k cap. Missing/noncontiguous checks cannot form a two-check trigger. |
| Full-budget probe | Small synthetic CPU critic executed ten independently keyed, continuously optimized 2,000-step paired fits. This verifies the code path and optimizer continuity, not full-width fit quality or time. |
| Tiny end-to-end path | Small SAC training, ten-round checks, source-bound ledger, checkpoint save/restore, W&B mock, local CSV and NPZ artifacts, and a forced **development-only** fork with saved trigger state ran on CPU. The forced fork tests plumbing, not natural sensitivity. |
| Fresh/fresh on CPU | Four pairs on a synthetic tiny pool fired 1/4; four pairs on replay saved by the tiny SAC smoke fired 2/4. The two pools, model size, fitting steps and pair count differ sharply from the proposed A100 experiment. Neither rate estimates population calibration or shows a systematic production effect. |
| W&B and local artifacts | Separate project `echocritic_method_c_ten_96`; existing environment/architecture/UTD group and seed name; `probe/interaction_step` custom x-axis, raw `probe/env_step`, check index, IQM, bounds and decision. Tiny smoke checks all ten loss columns, all check indices, `probe_curves.npz`, `probe_checks.csv`, training CSV, and checkpoint pointer. Networking was mocked. |
| Dynamic admission | Simulated mixed sizes, measured-throughput prerequisite, GPU/CPU/RAM/headroom limits, six-small ceiling, completion and immediate compatible replacement. Existing contention/recovery regression tests cover atomic claims, failed attempts and duplicate prevention for the historical scheduler. This does **not** yet qualify the scheduler's hard-coded 195-parent queue for the new 96-parent campaign. |
| Pilot preservation | Original five-function Git configuration at `f8619dd5` still has 5 rounds, 20 checks and 2,000 steps. Local archived Myo Key Turn probe-curve SHA-256 values match the previously recorded D2/D4 medium/D4 large hashes. No pilot file was edited. |

The changes found and addressed by CPU checks were: ten input observations
initially followed the five-point SciPy trim convention; the percentile
bootstrap used that same trim; and the Exp1 ledger silently omitted round
columns 5–9. The versioned path now uses the specified ten-point empirical
IQM and paired bootstrap, retains all ten rounds, and rejects missing
function-level values. The five-point path and schema remain unchanged.
Probe W&B metrics are now recorded only for the ten-target version, with an
explicit interaction axis. A versioned packing mode adds 6/4/2 admission
weights/ceilings without changing the historical 7/4/2 mode.

## Scientific controls unavailable from the uploaded pilot archive

The seven-run upload has five-function curves and check summaries but no
complete 50k/500k Myo Key Turn D4W1024/D4W1536 critic checkpoints, replay
buffers, normalizers, `LATEST` pointers, or `probes/check_02.npz` and
`check_20.npz`. Consequently **none** of the pilot-state fresh/fresh,
fixed-pool crossover, shared-target, initial-error/gradient-conditioning, or
4,000-step extension comparisons can be run in this workspace. The saved
Delta originals, if still present, can be read without training; their exact
paths and fail-closed wrapper are in
`method_c_v2_ten_target_design.md`. CPU execution is logically possible
after transferring the originals, but a reliable full-width CPU runtime
cannot be quoted without the checkpoint files and a timed fit. The D4W1536
network has tens of millions of parameters, so the A100 wrapper's maximum
eight GPU-hours should not be misrepresented as a CPU runtime estimate.

## Remaining before a 96-parent official launch

1. Review the ten-point IQM convention, bootstrap calibration, shared-target
   construct evidence, and predefined cross-seed/missing-run analysis.
   Passing arithmetic tests does not establish 95% CI coverage or trained
   critic sensitivity.
2. Build and validate a **source-bound 96-parent** manifest, approval,
   completion validator and scheduler queue. The current launch gate and
   validator require 195 parents, five round fields and 20 checks. Their
   historical checks must not be bypassed or relabeled as 96-run evidence.
3. Measure full-width ten-target memory, probe time, training throughput and
   6/4/2 mixed packing on an A100. Approve per-cell resources from those
   measurements. The tiny CPU smoke cannot establish packing capacity.
4. Freeze the eight seed IDs, four verified environment IDs, source SHA,
   results/W&B namespaces and separate Experiment 2 policy. The 96 parents
   do not include intervention arms; the 125k-interaction fork horizon is
   unchanged.

## Shortest A100 engineering smoke after a reviewed source and approval

Use one A100, 16 CPUs, 64 GiB RAM, at most **90 minutes (1.5 GPU-hours)**.
Run one D2W512, one D4W1024 and one D4W1536 development instance of the
versioned config together on the same device, with a shortened 20k-interaction
budget but unchanged 10k check spacing, ten functions and full 2,000-step
fits. Record per-process and combined peak device/host memory, actual
interactions per second, per-check probe duration, compilation time, W&B
receipts, full ten-round local output, one checkpoint/restart and a completion
event that admits a waiting compatible task. Start these only if host
CPU/RAM admission passes; otherwise the smoke should fail closed and report
the measured limit. Expected elapsed time is approximately 30–60 minutes,
an estimate from older shared-GPU pilot rates and measured D4 fitting costs,
not a ten-target measurement. This is the minimum engineering smoke for
memory/throughput/concurrency, not a statistical calibration experiment.

No Slurm job or official parent was submitted by this CPU qualification.
