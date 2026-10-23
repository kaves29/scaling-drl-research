# Retrospective fixed-task investigation

The original campaign remains frozen at
`f597ee9898e4d401912092c336c3e5330cf2fda1`. This separate development path never
updates SAC, claims a parent, changes the official detector, or writes into an
original checkpoint. Its results are diagnostic measurements, not new official
onsets or source-matched campaign approvals.

## Authorization and controls

The owner authorized at most **4 physical A100 GPU-hours** for this investigation.
The first proposed allocation reserves only one A100 for two hours, with16CPUs
and60GiB. The workload cap is6600seconds, followed by a6900-second process limit
and Slurm's7200-second limit. Any later allocation must count actual previous
allocation elapsed time, including compilation and idle time, against4GPU-hours.
Never automatically retry an ambiguous submission. No full SAC training and no
learning-rate, optimizer, weight-decay, reward, entropy or architecture-specific
hyperparameter search is permitted.

The live Delta balance, billing weights and allocation must be verified before
submission. First synchronized original-fit measurements determine conservative
remaining-fit estimates. A comparison is skipped when its estimated cost exceeds
the remaining workload cap. Partial results are retained and never called PASS.

## Predictions recorded before GPU execution

1. An identical original fit repeated with the same numeric inputs must have the
   same losses. A discrepancy stops new fitting pending investigation.
2. Historical check50 reconstruction on completed small pen seed5 and large
   humanoid seed1 should reproduce the observed scale/order of errors. Archived
   targets were not retained, so agreement is reconstruction consistency, not
   independent proof of bitwise historical target identity. Record every mismatch.
3. With the same target tensors and optimizer state, two continuous1000-step
   segments should reproduce the frozen2000-step fit closely. Full curves and final
   pool errors qualify this instrumentation on the actual A100. No optimizer reset
   occurs at the segment boundaries.
4. On a fixed late replay and normalizer, limiting sampled support to5000 distinct
   candidate transitions should reduce fresh error if repeated support explains
   the easy baseline. Absence of a reproducible effect weakens that explanation.
5. Early-versus-late replay with matched support isolates a distribution effect
   conditional on the chosen normalizer and keyed teacher. It does not identify a
   universal distribution-independent effect.
6. On identical early raw inputs, switch actual early-fork and latest normalizers.
   Evaluate both regenerated targets and fixed numeric targets from the other
   normalizer. The latter isolates a learner-input contrast; the former includes
   the teacher pathway. Nonadditive differences are interactions, not causal shares.
7. Fresh, early-fork and latest critics fit a common zero-offset task with identical
   inputs, targets and minibatches. This is distinct from the operational own-mean
   metric. Initial mean mismatch is measured, not ignored.
8. Selected fits continue through2k,4k,8k with the same Adam state. Catch-up weakens
   an interpretation of2k error as inability to fit; persistence does not prove a
   global capacity limit.

The first target bank covers two seeds in key-turn and pen-twirl before later
target rounds. Starting with three targets is a mechanism screen, not calibrated
IQM10 validation. Pool/target/index arrays and per-target full curves are saved.
Report seed and target replication separately. Comparisons across architectures
use the same large-parent input/teacher task, retaining all original architectures.

## Implementation and provenance

`fixed_task.py` preserves the frozen scalar MSE, AdamW settings and first2000
minibatch draws. Further1000-step blocks use new folded keys; this extension is a
separately specified diagnostic, not a relabeling of the original2k detector.
Checkpoint access reads only the required critic leaves; replay is read-only mmap.
Record parameter-leaf hashes, metadata hashes, normalized pools, targets, indices,
normalizer identity and source/runtime versions. Reconstructed5k statistics are
labeled retrospective and are not substituted into official results.

Compilation and synchronized execution are timed separately for original fits.
Segment timings explicitly include first-use compilation. Training throughput is
reused from existing campaign logs because this task does not authorize another
SAC campaign. Performance candidates require their own commit, numerical checks
and synchronized before/after measurements before being adopted.

Seven small real-JAX CPU contracts passed on Delta's pinned stack in3.426seconds
at initial diagnostic commit `f297dd49088a3ef7184fe13c884fa3f7ce8d58b0`.
This is CPU engineering evidence only; no GPU results are asserted here.

## Short diagnostic scheduling disposition, before GPU execution

Normal-queue job23120674 remained pending with unknown start time. Delta's
published policy explicitly permits short-duration tests of a new workflow in
the interactive partition. If and only if the pending job remains unallocated,
cancel it with a pending-state filter, verify zero elapsed/no allocation, and
replace it with a single one-hourA100 diagnostic in gpuA100x4-interactive.
Set DIAGNOSTIC_CAP_SECONDS=3000 so work ends at50minutes, with a55-minute process
limit. The live interactive billing rate is2weightedSU/hour: maximum charge2SU
and maximum physical GPU use1hour. Never leave both allocations eligible.
All original scientific controls and the4physicalGPU-hour total ceiling remain.

## Immutable task-bank continuation (23 October UTC)

The four task-bank tests at `5d86e23a3ad5c009f31bd67149e15c11e0ccd16b`
passed on Delta's CPU stack in0.014seconds. The queued job23120792 remains
bound to its earlier `bcf6cf3` checkout, which is not modified by development.
The successor runner persists each exact numerical task through `FrozenTask`,
reads it back with checksum verification, and records its identity in fit
receipts. Provenance includes source, optimizer configuration and target recipe;
learner names are excluded so identical tasks retain identity across critics.
Own-offset tasks need not have the same identity as shared-offset tasks.

New integration coverage checks that the runner's persisted task reproduces the
same CPU scan curve and final parameter/optimizer state, and that changed numeric
payloads are rejected. This is numerical persistence coverage, not A100 execution
or scientific calibration. No additional GPU allocation is submitted on this basis.
