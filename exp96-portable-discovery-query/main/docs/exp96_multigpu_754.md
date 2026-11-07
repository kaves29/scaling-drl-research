# Exp1 multi-GPU 7/5/4 successor

Owner authorization: 18 October 2026, full96-parent Exp1, operational detector,
600 weighted Delta SUs cumulative, including at most2 physical A100 GPU-hours
of targeted qualification. This implementation does not alter scientific code,
configurations, data validation or the existing frozen launchers.

## Reviewed request

Delta live checks18October: gpuA100x8 has six single nodes, each128 allocated
CPU cores (8 sockets ×16, ThreadsPerCore1), eight A10040GB GPUs, RealMemory
2063952MiB, MemSpecLimit8450MiB, and max48h. Request one node, eight GPUs,
128CPUs and2007GiB RAM. MAX_TRES weights CPU93.75, Mem6/GiB and GPU1500,
normalized by1000, yield12.042SU/hour:578.016SU maximum for48h. Verify live
weights, topology, account balance, actual allocation and billing on every use.
Node CfgTRES billing is not a substitute for the allocation's actual billing.

## Qualification and compatibility

New supported schema `exp96-multigpu-754-v1` requires a **real final-source**
qualification receipt. It never edits the originalbe1ffae smoke. It hash-binds
that report and revalidates all four real probe outputs using the corrected
validator. Git fingerprints cover run.py, experiments, agents, scientific
configs, metadata/utility code and dependency files against the original smoke
source. The only excluded added file is the unrelated seven-parent operational
resource draft at configs/launch_readiness/exp1_seven_parent_resource_estimates.draft.json.
Other scientific-byte changes fail compatibility rather than inherit the smoke.

A new8GPU/128CPU/2007GiB,15minute Slurm qualification is capped at2 allocated
physical GPU-hours. Its workload cap is12minutes, leaving cleanup/evidence time.
It directly exercises the production durable child-launch path on all eight
physical GPU UUIDs, CPU affinities and CUDA contexts, then races twelve processes
for one isolated development claim. It does not claim these probes are official
parents. All outcomes and failed evidence remain outside Git.

Packing qualification uses7small,5medium,4large and five bounded mixed patterns:
(1,0,3),(0,1,3),(0,2,2),(2,1,2),(3,1,1). Only these compositions and their subsets
are admissible. GPU lanes have16 disjoint allocated CPUs; workers receive2,3,4
cores by size, respectively. This obeys128 physical CPU capacity, including the
fully packed homogeneous cases. Taskset and CUDA_VISIBLE_DEVICES=physical UUID
isolate every process. Child PYTHONPATH is the frozen checkout; each attempt
has its own compilation cache/logs. The coordinator and validators remain CPU-only.

Packed development processes restore retained500k dog states, touch the full1M
replay capacity by explicitly repeating transitions, warm the actual SAC update,
and synchronize before one full paired ten-target ×2000-step check. These are
engineering memory inputs, not fresh-null or scientific calibration evidence.
Dog's223+38 input width bounds the other three observed smoke-task critic input
widths; architecture, normalization, single-Q implementation and probe budgets
are unchanged. Host projections include the earlier Myo measurements and added
reserves; GPU rendering is disabled exactly as in the original smoke.

One representative per lane saves, restores and saves complete state; bitwise
state comparison must pass. NVIDIA per-PID and whole-device sampling plus JAX
allocator peak statistics set conservative reservations (15% device margin and
2GiB device free headroom;30% host margin plus2GiB/process). Both actual peak
and sum-of-reservations checks must support7/5/4 and all permitted mixtures.
Missing telemetry, OOM, shape/nonfinite errors, incomplete checkpoints, incorrect
GPU/CPU mapping, source mutation or insufficient headroom yield INCOMPLETE,
never PASS. Successful Slurm COMPLETED0:0 accounting, the15min bound, immutable
child receipts/curves and current source fingerprints are rechecked at startup.

## Scientific and allocation gates

External source-bound study approval still needs all five owner decisions:
ten_function_method, scientific_calibration, gpu_smoke, allocation_and_storage,
cross_seed_analysis. The operational-detector limitation is an explicit scientific
disposition, not evidence of calibrated false-positive control. The GPU gate now
references the new real qualification, whose scope explicitly inherits unchanged
training evidence and newly measures orchestration/packing. The old gate remains
unchanged and does not accept this new schema accidentally.

Allocation approval binds full source, queue digest, study digest, exact resources,
qualification profiles/compositions and safety margins. It cannot replace measured
profiles with historical projections. Preserve all96 source-hashed identities and
500k/50+baseline/10×2000 settings. The new scheduler interleaves environment/size
admission for utilization without changing immutable rows or seed identities.
Original result validation remains the sole completion authority.

## Budget and recovery

Create a fresh external budget once, with the owner's decision reference and any
verified prior Exp96 charges. Atomic budget transitions reserve each allocation's
maximum bill **before** sbatch; cumulative actual+reserved charges cannot exceed
600SU, nor can qualification actual+reserved allocation time exceed2GPUh. The
qualification reservation is3.0105SU, campaign maximum578.016SU. An ambiguous
submission retains its reservation and must be reconciled, never blindly retried.
Only terminal sacct accounting releases unused reservations. Gate startup checks
actual billing against the reserved rate; pending jobs count at their full limits.

Parent claims and completion receipts reuse the historical atomic Ledger.
Failures are item-local and retain existing blocked-review behavior. A planned
allocation boundary stops each process before retiring its claim, validates its
complete checkpoint (or deterministic pre-first-save restart), and records a
planned pause without declaring completion. A new approved allocation can resume
the same source/queue. Admission estimates time to a safe checkpoint, not an entire
parent, so long parents can span48h allocations. Unexpected failures never become
successful pauses. No duplicate ownership is permitted.

When no work fits, release the allocation instead of holding GPUs idle. Further
allocations require intact approvals and a fresh maximum-cost reservation within
the remaining600SU; neither dependencies nor an unchecked sbatch chain bypass
this constraint. Scientific scope and the global spending ceiling remain fixed.

## Targeted checks

`python -B tests/test_exp96_multigpu.py` runs infrastructure-only checks for
billing/shape validation, physical GPU visibility, packing/CPU constraints,
concurrent reservations, ambiguous submissions, checkpoint pause behavior,
intact source/evidence bindings and terminal accounting. The existing corrected
completion tests and actual-artifact receipt are reused, not rerun broadly.
No GPU PASS is recorded in source; inspect the external actual qualification.
