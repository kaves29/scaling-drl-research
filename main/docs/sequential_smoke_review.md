# Historical five-job engineering review

**Superseded for the current request by [six-run review](sequential_six_review.md).**
The producer-first gate below is retained historical material, not the current launch plan.

**approved=false; executable=false; submitted=false. NO-GO for execution.**
This packet and companion JSON are review material, not runnable training requests,
scientific authorization, a launch manifest, or a queue-ready plan. No GPU/Delta
job was submitted. The 10-run pilot is untouched; no status is inferred for it.

## Source and scope

Approved base: integration/exp12 at
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c` (remote verified27 October2026).
Original tested candidate: `00ed566e4c7f00c784b61d98afc9768e89906665`.
Named readiness branch remotely points to
`46ad72e1aa8863cc8400aa4b8399e14af5a054f6`: one later documentation-only commit
(five paths); its eight Python candidate files are unchanged. Inspected this
mismatch rather than silently resetting the published branch. This successor
starts at the exact00ed566 commit. It preserves the eight files byte-for-byte,
adds only this review material, a read-only staging gate and its CPU tests, and
updates handoff/status. No approved production, config or methodology file changes.
PR17 remains at `a6958337f411c30f6ebe1a1b3d5c5a0403287780` with its original
PR16 base; this is a separate draft PR, not a retarget, merge or history rewrite.

## Diagnostic meaning and invocation

This opt-in diagnostic is distinct from the approved independent probe and is
not claimed to reproduce a paper-validated sequential method. K=1..8;2000 updates
per target/arm. A starts from the current SAC critic and carries weights through
phases; B starts from its exact saved pre-SAC initialization and carries weights;
C starts from that fresh initialization for each target. All reset optimizer
state each phase and share the literal stored targets and minibatch indices.
A per-architecture offset, computed once from the fresh critic on the frozen
bank, is explicit; cross-architecture literal targets can differ by this offset.
An A deficit may include mean adjustment rather than uniquely identify plasticity.

The enabled dedicated route never constructs RunProbes. Ordinary SAC training
and actor/environment evaluation continue. The default delegates to the approved
exp1 entrypoint unchanged. After owner choices, review a fully explicit request:

```
python -m diagnostics.sequential_runner --request /absolute/reviewed-request.json
```

The request has `args` (config_path, config_name, overrides, absolute checkpoint_dir,
checkpoint_interval) and `alternative` (enabled, bank, reference, output, teacher,
teacher_architecture, architecture, environment, seed, source, bank_producer).
There are no scientific defaults in this packet. Dedicated runs require explicit
fork.enabled=false and steps2000; no fork-trigger extension is implemented.
Unsupported twin/sparse/mixed-precision/vector configurations fail closed.
The companion review JSON is deliberately NOT this request schema.

## Fixed five-job proposal

| ID | Environment | Critic | Seed | Role |
|---|---|---|---:|---|
| dog-small | dog-run | D2W512 |0| producer |
| dog-medium | dog-run | D4W1024 |0| consumer |
| dog-large | dog-run | D4W1536 |0| consumer |
| myo-small | myo-pen-twirl | D2W512 |0| producer |
| myo-large | myo-pen-twirl | D4W1536 |0| consumer |

**Missing: myo-pen-twirl D4W1024 seed0.** These are five engineering jobs, not a
balanced scaling experiment. One development training seed per cell; tasks,
phases, checkpoints and GPU replicas do not increase training-seed replication.
No task, seed, missing cell, teacher, target, cadence or resource decision is
silently changed. Uses: routing, phase artifacts, shared-bank consumption and
restoration engineering. Cannot establish balanced scaling, general DMC–Myo
mechanisms, seed uncertainty, hypothesis proof or a guaranteed improvement.

Unchanged documented full-cadence proposal: initial fresh capture after5000
warmup interactions, then50k,100k,...,500k, giving11 observations/job. Each costs
8 targets x3 arms x2000 =48000 probe updates. **528000 per job;2640000 across
five jobs**, in addition to SAC training. There is no shortened smoke trajectory
in this packet. These counts are not GPU-hours or walltime estimates.

## Shared-bank dependency and non-idling staging proposal

The current Observer creates the environment/seed bank at the producer's initial
warmup, on the producer backend using its actual replay/RMS. Non-producers reject
a missing bank. A CPU-generated replacement would change the execution contract;
a replay simulation or different pool producer would change the task. Neither is
used. Starting five jobs together is therefore NOT a safe startup strategy.

Proposed conservative ordering, requiring owner approval:

1. Run only the two D2W512 producers under separately approved resources. Each
   completes its entire prescribed training/probe trajectory. No stop-at-warmup,
   extra warmup, training restart, new RNG split or replay regeneration is added.
2. A CPU/login-node controller inspects producer exit status. Require successful
   completion of both producers, their exact-source/config identities and each
   ALTERNATIVE_DONE.json at the prescribed endpoint. If a producer fails, do not
   release consumers; preserve evidence and use the runner's reviewed restore
   route with identical identity and committed phase reuse. Do not create a
   substitute bank or automatically replace a failed producer.
3. Freeze expected bank/identity/DONE receipt hashes from those reviewed producer
   outputs into a separate resolved copy of the review plan. This is not approval.
   Run the following CPU-only gate BEFORE requesting any consumer GPU allocation:

```
python scripts/sequential_smoke_preflight.py --plan /absolute/resolved-plan.json
```

   It requires BOTH committed banks, pinned receipts, file/array hashes,
   environment/seed, teacher definition, source, warmup/pool and exact runtime
   identity; plus pinned producer identity and full-endpoint completion. Re-run
   after transfer to the consumer filesystem. Failure exits2. The checked-in
   unresolved plan intentionally fails. Success says BANKS_VERIFIED_ONLY and
   submission_allowed=false; it does not submit, wait for a GPU or approve a run.
4. Only after independent review, completed hardware qualification, all owner
   decisions and explicit GPU approval may the three consumers be submitted.
   They read the same immutable shared bank. If bytes change after preflight,
   runner receipt/source/runtime verification still fails at initialization.
   Protect shared banks from writes between gate and execution.

No GPU job waits on a file barrier. A scheduler dependency alone does not replace
bank verification. Do not allocate consumer GPUs while a controller waits for the
producers. Two producers may be serial or concurrent only after owner resource
approval; no packing assumption is made. Existing CPU lifecycle/resume tests
verify diagnostics preserve trainer state; synthetic CPU bank-gate tests do not
qualify these five full-size configurations. A real two-producer/scheduler staging
execution has NOT been run. The plan is not queue-ready today.

This complete-producers-first route needs no training/probe semantic change.
If early consumer release is desired, owner must authorize a separately reviewed
warmup staging/resume design; it is not implemented here. Do not silently replace
this conservative ordering with early-stop/resume or a new bank generator.

## Owner decisions (all pending)

- Confirm the listed tasks and development seed0 are eligible; explicitly accept
  the omitted Myo medium cell. No eligibility is inferred from old removed-task
  comments. Adding a cell requires a new proposal, not this packet.
- Approve the fixed teacher architecture/config (prior proposal D2W512), initial
  pool/RMS convention and the two designated D2W512 bank producers. Preserve sine
  targets and existing architecture-specific offsets; do not select by outcome.
- Approve the documented full cadence/endpoints/target settings above, and
  explicit fork.enabled=false solely for the alternative-only diagnostic.
- Approve complete-producers-first ordering; select producer/consumer concurrency
  without assuming GPU packing. Decide whether any separate hardware qualification
  runs are authorized; none are launched by this task.
- Select exact hardware/backend/precision/packages, scheduler/account/partition,
  GPUs and CPUs/job, host RAM, checkpoint interval, walltime and cost caps.
- Select immutable shared bank paths, distinct run/reference/output/checkpoint
  paths and ordinary W&B IDs/project; durable artifact destination, retrieval,
  retention and storage limits. Bind all requests to the final reviewed SHA and
  recorded implementation hashes; identical backend/precision contract is required
  for bank/reference reuse. Source and execution fields remain null until resolved.
- Independently review source and CPU evidence, then explicitly approve any GPU
  execution. No checked-in flag or CPU pass constitutes that approval.

## Resource and artifact checklist

All full-size GPU-hours, walltime, compilation/steady throughput, VRAM, CPU needs,
host RAM and storage figures are **unmeasured/unqualified**. Do not borrow
independent-probe/Kaggle timings for this sequential workload. Before execution,
record the actual GPU/device layout and JAX binding; qualify peak memory and
compile/steady throughput at each size, CPU/RAM and shared-filesystem load under
approved concurrency. Set explicit caps only by owner decision. Do not infer
feasibility from disabled preallocation or parameter counts alone.

Retain and independently hash/retrieve:

- Final source/base/request/runtime identities and package/precision/device metadata;
  two shared bank receipts, bank.npz and teachers.msgpack, raw and normalized
  observations/actions, replay indices, RMS state, teacher keys/base targets.
- Each run's alternative_identity, exact fresh.msgpack/reference targets and frozen
  offset, and full training checkpoints (parameters/target/optimizer/replay/RMS,
  counters, RNG, simulator state). Unique directories per job; never share a
  learner's fresh/reference directory across architectures.
- At each of11 observations: identity.json, trajectory.json,8 phase receipts;
  raw A/B/C errors/scores, all2000 minibatch losses/arm, full-pool milestone
  predictions/errors at0/100/500/1000/2000, literal targets and indices/hashes,
  A/B carried states, fresh-reset C evidence, initial/final parameter hashes.
- Ordinary SAC evaluation/training logs, ALTERNATIVE_DONE.json, failed/partial
  directories and errors, scheduler exit status, resource/timing telemetry,
  producer bank-gate evidence and a relative-path SHA256 inventory.

Alternative arrays are **local outputs**, not automatically published as W&B
artifacts. Ordinary training W&B logging is not an artifact backup. Use durable
shared storage at source; then read-only rsync/scp to an approved destination,
verify the full relative-path SHA256 inventory independently, preserve directory
structure and failed/partial files. Do not delete local copies or introduce
synchronous per-file W&B round-trip eviction. A transport/destination remains an
owner decision; no upload credentials or assumed backup are supplied here.

## CPU verification and limits

Fresh exact-source commands, stage exits, test counts/warnings and evidence hashes
are recorded outside Git under /workspace/artifacts/alternative-review-20261027/
and summarized in the successor PR description. This packet does not relabel the
prior00ed566 receipt as validation of added staging code. The rerun covers all
source-resident documented regressions plus staging gate tests, full mutation
checks twice,65 simulator restoration cells, tiny real-critic example with
independent archive reload, and static/diff checks. Missing later-branch test
modules are not imported or counted as passing. Source/base/final head SHAs and
precise results are reported with that receipt, without modifying the tested
head to insert its own hash.

Default/alternative paths, interruption at initial/later phase boundaries, reuse
of committed phases, complete-state isolation and artifact hashes are exercised
on tiny real CPU trainer configurations. Bank gate success/rejection uses actual
tiny CPU-created bank payloads plus synthetic producer identity/DONE fixtures;
it proves file/identity ordering checks, not full GPU producer completion. No
five-job execution, CUDA qualification, scheduler integration, durable transport
or scientific result is claimed. NO-GO remains until owner decisions and the
unmeasured qualification/evidence requirements are resolved.
