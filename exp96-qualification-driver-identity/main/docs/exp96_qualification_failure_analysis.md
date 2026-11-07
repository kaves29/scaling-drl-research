# Qualification harness failure analysis and final authorized retry

19 October2026. This report is engineering analysis, not GPU PASS or scientific approval.
Base ec7e8d918d76e1bf6da00edcda5c74d6c70b8eed; resolve the successor from Git.

## Why the jobs failed

22990946 (a5c376d) ran108s on8GPUs. Its children restored real retained states
and warmed SAC, but the harness read shape[0] of vector-environment spaces as
observation/action widths:1/1 instead of223/38. Production8GPU assignment and
single-owner claim race had passed. Packed fitting/memory/checkpoints had not.
The geometry correction uses single-environment spaces and cross-checks replay.

23020460 (ea98652) never started. Owner rejected the8GPU qualification request;
it was cancelled while pending, elapsed0/noAllocTRES. It consumed no GPU time.

23020573 (ec7e8d9) started21:10:54UTC, ended21:11:11UTC, FAILED3:0,17s on one
A100/16CPU/60GiB. Source/runtime/allocation/reuse gates passed; no worker launched.
The exact failing branch was `qualification_device`: no NVIDIA inventory index
or UUID equalled `SLURM_JOB_GPUS`' token. The harness failed closed before fitting.
Raw runtime identifiers were not saved, so their exact values are unrecoverable.
The stored submission environment lacks those runtime-generated Slurm variables.

The implementation incorrectly treated Slurm's global GPU identifier as an
NVML/nvidia-smi ordinal, and treated an absent optional identifier as an empty
single token. Slurm documents global/device-minor/NVML PCI-order and cgroup-local
numbering differences. The CPU fixture reproduced only a successful numeric
match, so it did not detect this deployment assumption. These were test-harness
defects, not observed training, OOM or scientific-detector failures.

## Corrected discovery and diagnostics

Preserve Slurm's CUDA_VISIBLE_DEVICES. Query CUDA Driver API for exactly one
visible device's UUID and PCI bus address, with explicit ctypes ABI signatures;
no CUDA context, tensor or workload is created in the CPU controller. Join that
identity to NVIDIA inventory by UUID and normalized PCI address. Numeric Slurm,
NVML and device-minor identifiers are recorded, never equated. Slurm actual
allocation/CPU/RAM/billing checks still require one GPU; optional JOB/STEP GPU
identifiers may be absent, but multiple identifiers or a conflicting UUID fail.
Absent CUDA visibility, CUDA error, multiple devices, mismatched PCI, duplicate
UUID, wrong model/capacity or controller context creation also fail closed.

Persist allowlisted runtime GPU variables, raw NVIDIA inventory, CUDA identity,
selected physical UUID/model/capacity and any discovery error. Persist each
child's actual progress stage and coordinator traceback. No credentials or
unrelated environment are recorded. Existing historical evidence stays immutable.

## Review of the remaining path

| Required path | Evidence/disposition |
|---|---|
| Source and scientific compatibility | Same training/probe/config/dependency Git fingerprint; clean immutable successor required. Original smoke is not relabeled. |
| GPU/CPU isolation and unique claim | Independently passed22990946 components remain hash-bound; their function ASTs and scheduler/validator blobs unchanged. No repeat. |
| Partial-node identity | New tests cover mismatched ordinal/minor namespaces, multiple NVML devices, absent optional IDs, UUID/PCI conflicts, multi-GPU visibility and driver errors. |
| Child initialization/replay | Real dog vector geometry223/38 already verified. Retained500k headers/capacity reviewed; full1M stress replay is engineering data only. |
| Probe contract | Real pinned-stack CPU run:10targets×2000updates, current/fresh, real JAX/Optax fitting; finite output and saved-array reload; NaN/truncated curve mutations rejected. Synthetic narrow network/pool only. |
| Checkpoint recovery | Real pinned-stack trainer CPU fixture: train, save/restore/save bitwise equality, then successful next update. Original GPU medium restart remains reused. New packed full-size roundtrips still required. |
| Phase orchestration | Affected controller test proves7/5/4 phases do not overlap, all16 children finish, all3 barriers are recorded. |
| Completion gate | Scientific result validator bytes unchanged; qualification PASS still requires complete finite full-width curves, real checkpoint equality, measured packing/headroom and successful source-matched Slurm accounting. |
| Memory/throughput | Still UNVERIFIED at7/5/4. Only actual one-GPU measurements may qualify; keep15% per-process margin and2GiB GPU headroom. No mixed packing. |
| Failure/cleanup | Any missing/nonfinite/failed child fails the job; bounded process-group cleanup and durable partial evidence. No automatic retry. |
| Spending | Atomic shared600SU ledger; old17s settles0.004722222SU/GPUh. New30min oneGPU max0.5SU/GPUh; cumulative qualification max0.744722222GPUh, below2. |

CPU evidence: `/projects/biqc/skaveti1/exp96/driver_contract_cpu_20261019T213541Z/`.
Two real API contracts passed in50.374s. These are small CPU unit fixtures, not
production/GPU qualification. Five new/affected local identity/controller tests
pass. Existing broad suites and original four-case GPU smoke are not repeated.

## One final authorized GPU retry

Owner expressly authorized one last corrected retry after analysis. Request one
A100 on gpuA100x4,16CPUs,60GiB,30min wall/28min workload cap. Test only unfinished
homogeneous7/5/4 fitting/memory and three architecture-specific full checkpoint
roundtrips. Stop early if required headroom fails. Preserve every result. Never
submit a further retry automatically. The96-parent job22996973 remains frozen
and blocked; a new source's receipt cannot release that old source automatically.

Sources: [Slurm GPU management](https://slurm.schedmd.com/gres.html#GPU_Management),
[CUDA device identity API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).
