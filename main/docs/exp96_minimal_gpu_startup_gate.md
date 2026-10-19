# Minimal required qualification with a real GPU startup gate

Owner's latest direction: run only unfinished mandatory checks, after extensive
relevant preparation, and actually verify GPU execution before costly fitting.
This authorizes one renewed minimal qualification within the existing one-GPU,
2physicalGPUh targeted-qualification and600weightedSU cumulative limits.
Earlier failed attempts remain immutable; no engineering PASS is assumed.

## Scope and order

One A10040GB on gpuA100x4,16CPU,60GiB,30minute maximum,28minute workload cap.
First execute a small real CUDA/JAX startup worker inside this SAME allocation:
CUDA-visible UUID matching, exact CPU affinity, one JAX GPU, synchronized8x8
FP32 matmul with exact finite result, scientific TF32 mode, and the same NVIDIA
PID/GPU/RSS telemetry used by the fitting stages. Parent checks the actual worker
PID on the assigned physical UUID. Discovery uses only the four NVIDIA fields
already executed successfully on Delta; no optional PCI/minor query columns.
No fitting process launches until this gate passes. On startup failure/timeout,
terminate the child and exit with durable error evidence. Startup has90s cap,
uses two allocated CPUs, and exits before packing to free its CUDA context.

Then run only the unfinished homogeneous7/5/4 packing phases sequentially on
that one GPU: full paired ten-target2000-update fitting, finite saved curves,
warmed SAC throughput and GPU/host memory, one full checkpoint save/restore/save
bitwise comparison per architecture. Keep15% per-process memory margin and2GiB
GPU headroom. Failure stops remaining phases; no automatic retry. Optional mixed
packing stays disabled. The original four-case smoke, eight-device assignment
and single-claim race are reused through existing source/hash compatibility;
none is repeated. The new one-GPU startup tests the changed partial-GPU binding,
not the already-passed eight-device orchestration.

## Preparation evidence

Three new focused startup tests cover success/admission with actual-telemetry
interface, wrong GPU rejection and timeout cleanup. Three affected controller
and gate checks verify no fitting after startup failure, sequential7/5/4, exact
source/artifact/accounting requirements and bounded request accounting. Existing
portable-query/CUDA ABI regressions and real CPU fitting/checkpoint contracts
are retained; their unchanged code need not be rerun. A new pinned-stack CPU
numerical contract checks the startup worker's synchronized kernel and receipt
path with hardware mocked and explicitly labeled NOT GPU evidence. The actual
GPU stage is the required runtime proof; CPU results cannot certify packing.

No scientific training/configuration/dependency bytes are changed. Startup PASS
is mandatory in the source-bound qualification gate and its artifacts are hashed.
A startup-only pass cannot admit the campaign; all packing/checkpoint/source
checks plus COMPLETED0:0 Slurm accounting are still required.

## Spending and old campaign

OneGPU/16CPU/60GiB charges1SU/hour under the verified gpuA100x4 MAX_TRES weights.
New maximum is0.5SU/0.5physicalGPUh. Earlier targeted actual GPU usage totals
0.248611111GPUh; renewed cumulative maximum0.748611111GPUh, below2. Use the SAME
atomic600SU budget with historical actual charges and old campaign reservation.
Refresh live accounting, quota, topology and duplicate-job state before submission.

Oldcampaign22996973 remains frozen/blocked at its original source. This
qualification cannot release that job automatically; campaign source/queue/
approval bindings must be legitimately reconciled after actual qualification.
No official parents, extra diagnostics, mixed packing or Exp2 GPU work are part
of this qualification.

Pinned-stack startup numerical unit contract passed in8.662s. Evidence:
`/projects/biqc/skaveti1/exp96/startup_cpu_v2_20261019T215623Z/receipt.json`.
Hardware was mocked and the actual kernel ran on CPU; this is not GPU PASS.
The preceding CPU fixture import failure is preserved separately and was fixed
before any renewed GPU submission. No additional GPU allocation was used for it.
