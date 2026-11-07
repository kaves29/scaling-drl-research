# One-GPU required-only packing qualification

Owner correction19October: at most ONE GPU for the qualification allocation.
Eight-GPU test23020460 was CANCELLED while pending, zero elapsed and no AllocTRES;
no GPU work ran. Its frozen source/evidence are retained. Historical failed
qualification22990946 consumed0.24physicalGPUh and0.36126weightedSU.

This successor uses one A10040GB on gpuA100x4,16CPUs,60GiB,30minutes maximum,
28minute workload cap and2minute cleanup allowance. At live MAX_TRES weights
CPU62.5,Mem16/GiB,GPU1000, the normalized rate is1SU/h. New worst-case cost is
0.5SU/0.5physicalGPUh; cumulative qualification actual+reserved is0.74GPUh.
The existing cumulative600SU ledger also counts the held campaign reservation.
Only verified zero-runtime CANCELLED records with empty AllocTRES settle at zero;
missing billing for a job with runtime remains a hard error.

Three sequential phases measure7D2W512,5D4W1024,4D4W1536 workers on the same
allocated physical GPU. Each phase finishes all processes before the next starts.
Use separate directories, barriers, CUDA contexts and disjoint2/3/4CPU worker
masks inside16allocated CPUs. GPU UUID comes from Slurm's physical ownership,
not a guessed CUDA-local ordinal on a partially allocated node. The controller
stays CPU-only. NVIDIA driver/PID telemetry and allocator peaks must both support
measured packing with15% memory margin and2GiB device headroom. Each phase fails
early on insufficient headroom; no remaining phases run after failure.

Only required unfinished checks run: full ten-target2000-update paired fitting,
finite curves, warmed SAC updates, memory and one complete checkpoint roundtrip
per architecture. Passed eight-device production assignment/CPU isolation and
unique-claim race are reused from22990946 with original hashes and unchanged
implementation verification. Existing one-GPU smoke scientific-byte compatibility
and corrected actual-artifact validation are retained; no broad suite is repeated.
The parameterized srun GPU count is one for qualification/eight for the separately
gated campaign; this receipt explicitly does not claim new simultaneous8GPU load.

Optional mixed packing is DISABLED. Only homogeneous7/5/4 compositions and their
subsets can be admitted, so unmeasured mixtures cannot slip through accounting.
No scientific settings, frozen checkpoints, old queues or existing campaign job
are changed. Full campaign source-bound approvals/queue still require the actual
successor SHA, real PASS receipt and successful Slurm accounting before launch.

Seven focused CPU checks cover new partial-node mapping, sequential phase
orchestration/no overlap, cancelled-before-allocation settlement, short request
bounds, source gate accounting and cumulative reservations. They are engineering
unit tests with explicit synthetic fixtures, not GPU evidence or approval files.
