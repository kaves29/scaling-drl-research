# Six-run alternative engineering smoke — review packet

**approved=false; executable=false; submitted=false. NO-GO for cluster execution
from Cloud.** Replaces the five-job/producer-first proposal for this request only.
PR17, approved integration, the independent probe, production/config/methodology
files and the separate10-run pilot are unchanged. No live pilot status is inferred.

## Source

Approved base integration/exp12:
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c` (remote and PR18 base verified).
PR18 before this revision: `6fd8f478eccb5bc1c5f7c53f1306aa27f49b8af1`.
Final tested/reviewed SHA is published in the PR validation receipt and the exact
Delta handoff; never resolve a floating branch at launch. This revision modifies
only the optional alternative implementation/tests/example and review material.
Earlier00ed566/6fd8f47 CPU receipts do NOT qualify this changed schedule or gate.

## Exact six-cell mapping

| GPU index in allocation | ID | Environment | Learner | Seed | Bank role |
|---:|---|---|---|---:|---|
|0|dog-small|dog-run|D2W512|0|producer|
|0|myo-medium|myo-pen-twirl|D4W1024|0|consumer|
|0|dog-large|dog-run|D4W1536|0|consumer|
|1|myo-small|myo-pen-twirl|D2W512|0|producer|
|1|dog-medium|dog-run|D4W1024|0|consumer|
|1|myo-large|myo-pen-twirl|D4W1536|0|consumer|

Both GPUs host all three critic sizes. This is the complete2x3 condition grid at
one development seed, **not a powered or balanced-seed scaling study**. One task
per domain, no independent-seed replication, and no GPU replicas: GPU effects
cannot be independently estimated from condition effects. K, rounds, tasks and
checkpoints are not independent training seeds. No omitted cell remains.

One allocation/wave, one node, exactly two physical GPUs, six run processes,
three per GPU, walltime <=21600seconds: maximum12 aggregate reserved GPU-hours.
This counts bank waiting, initialization, compilation, training, probes and I/O.
Do not silently use extra GPUs, multiple waves or a longer cap. Scheduler billing
or exclusive allocation that actually reserves more than two GPUs must be checked
against the cap, not hidden by CUDA_VISIBLE_DEVICES. Resources are requests, not
proof of feasibility. Concrete hardware/account/partition, CPU/RAM, checkpoint
interval, gate timeout, VRAM, throughput and storage remain UNMEASURED/unqualified.

## Revised optional schedule (even_milestones_v2)

The eight-target bank and A/B sequence remain1..8. A starts at the checkpoint's
current SAC critic; B starts at the exact saved pre-first-SAC-update fresh critic.
Both carry weights through all eight targets, resetting probe optimizer state
for every target. C starts from that exact fresh critic and fits only target2,
4,6,8. Each fit is2000 updates. Scientific score summaries are produced only at
K=2,4,6,8; no C fit/file or comparison is produced at odd K. Odd A/B continuation
phases retain their raw engineering loss arrays and carry-state receipts for
integrity/resume; they are not reported scientific score milestones. Literal
phase targets and minibatch indices are identical across arms where compared.

A16000+B16000+C8000 =40000 probe updates/observation. Initial observation occurs
after5000 warmup interactions, before the first SAC update; later observations
are50k,100k,...,500k:11 total. **440000 updates/run;2640000 across six runs**, plus
SAC training. Equal aggregate update count to the former five-job proposal does
not imply equal runtime, memory or storage. No shorter cadence/horizon is adopted.

Legacy all_targets_v1 remains available for existing alternative API requests;
the default approved experiment still delegates unchanged. The six-run request
builder explicitly selects even_milestones_v2. The dedicated runner never
constructs RunProbes, even if the inherited probe.enabled flag is true. Ordinary
actor/environment evaluation and SAC updates continue; fork.enabled=false is
explicit for this alternative-only route and supplies no approved fork trigger.

## Exact teacher, data and normalization

Separate random **residual D2W512 SACCritic, float32**, not the trained producer's
parameters. Eight independently keyed initializations per environment/seed;
source create_bank persists the actual teacher trees and keys used. Targets are
sin(100000*teacher_Q(state,action)). No sign targets or alternative stabilizers.
A fixed25600-pair sample with replacement from the matching D2W512 producer's
5000-interaction warmup replay is used, with its observation RMS frozen in the
bank. Retain raw/normalized observations, actions, replay indices, RMS and hashes;
index uniqueness is not physical input diversity. Teacher chunk2560; learner
batch256; all current LR/decay, SAC parameters and training settings are inherited
from approved base_exp12 (Myo uses myosuite_simba, DMC dmc_hard).

Each learner separately freezes the mean Q of its exact fresh initialization on
the bank. Literal targets are base plus that architecture-specific fixed offset;
no later recentering. Cross-size targets can differ by this prescribed offset.
Initial versus final full-pool errors/trajectories must distinguish mean adjustment
from other fitting behavior. A favorable contrast is not proof of a plasticity
mechanism. Targets/normalizer/pool/offset functions are unchanged by this revision.

## Bounded in-wave bank gate

All six OS run processes start in the same scheduler wave. The four consumers
enter sequential_runner's gate BEFORE trainer construction, environment creation,
training seeding, warmup interactions or SAC updates. Producers do not wait.
Consequently waiting adds no transitions/updates or training random-key draws,
no reset/replay and no target substitution. It still consumes allocation walltime
and reserved resources. GPU backend initialization can occur for runtime identity;
three-process context/model memory must be qualified under this actual layout.

Delta infrastructure must supply each consumer a bank_wait object:
`timeout_seconds`, absolute `allocation_deadline_unix`, and absolute matching
`producer_status` JSON path. The deadline comes from the allocation's real start
and <=six-hour limit, not a fresh six hours for every child. Timeout must be finite,
positive and <=remaining allocation time. It is not invented by Cloud. Audit
attempts retain timeout/deadline, elapsed wait, expected identity and success/error.
Timeout/deadline/status are operational and excluded from the scientific resume
identity; the verified bank receipt SHA is included and rechecked on initialize
and restore. Gate configuration changes do not authorize new budgets/experiments.

Producer statuses must be atomically written and source/environment/seed bound:
STARTING, RUNNING, BANK_READY, COMPLETED or FAILED. Delta wrapper monitors real
child exit status and records failures, including crashes/signals, rather than
leaving RUNNING forever. Missing bank/status times out; failure refuses the
consumer; COMPLETED without a bank fails. Producer uses existing atomic directory
commit. A published but partial/invalid bank fails immediately rather than being
regenerated. Gate verifies both file hashes, numerical array hashes, source,
full runtime/precision/package identity, teacher definition/scale/chunk,
environment/seed, pool/index shape, warmup and stored RMS/normalization identity.
It does not recompute a new normalizer or teacher. Successful receipt SHA is
pinned into the consumer identity/reference. Observer still verifies before fits.

The launcher must monitor producer death for the allocation lifetime: fail and
stop affected consumers with retained partial artifacts if a producer dies after
startup release. Do not cancel unrelated jobs or the10-run pilot. Slurm owns the
hard <=six-hour limit. Qualification must demonstrate this watchdog, binding,
three-process peak memory, bank gate and remaining full-cadence runtime can
coexist. No indefinite file waits or separate producer wave are allowed.

This CPU-tested generic gate replaces the previous full-producers-first proposal;
it is not a tested Delta launcher. Delta Codex may implement only cluster-specific
Slurm allocation/binding/launch barrier/status watchdog pieces, without modifying
the diagnostic, targets, trajectories, settings or bank protocol. Preserve all
six sources/requests and infrastructure hashes separately. Any need to change
reviewed scientific code requires NO-GO and a new review, not a local patch.

## Source-bound request and preflight utilities

Review JSON: docs/sequential_six_review.json. Its flags remain false. Source and
cluster qualification fields are intentionally unresolved in the checked-in
review document. Validate the immutable design (not launch readiness):

```
python scripts/sequential_six_plan.py --plan docs/sequential_six_review.json
```

Output is DESIGN_VERIFIED_NOT_CLUSTER_QUALIFIED, submission_allowed=false.
The pure request_for function in that reviewed script builds the six literal
scientific requests from this plan with explicit final source SHA, absolute source
and durable output roots, project, checkpoint interval, finite gate timeout and
allocation deadline. Delta must verify supplied source SHA equals the reviewed
checkout and preserve the resolved configs/hashes. It must not rewrite overrides
to reduce cost. Per-run W&B IDs are distinct launcher environment bindings;
ordinary W&B logging is not an alternative-array publisher.

No sbatch script, account, memory guess or executable launch request is provided
by Cloud. Gate success alone does not establish cluster readiness. Cluster
preflight must fail closed without verified exact-source packing, capacity,
compile/steady/full-cadence timing, cap and artifact-retention evidence. Do not
submit this smoke simply to discover whether the layout fits. Use existing
source-matched qualification or separately authorized qualification resources;
missing evidence is NO-GO, not permission to spend an extra GPU allocation.

## Resource, evidence and retrieval checklist

- Live squeue/account/QOS/partition/time/GRES/CPU/host-memory information; actual
  node/GPU type/UUIDs and <=two reserved GPUs. Do not infer queue state from old
  comments or touch existing pilot allocations. No promise about queue delay.
- Same-source/package/precision three-process-per-GPU measurements for the actual
  small+medium+large mix: peak VRAM/RSS, CPU, compilation and steady throughput,
  checkpoint/retention I/O, bank-wait timing and conservative full-cadence
  completion within remaining21600seconds. No CPU test is GPU qualification.
- Distinct run/reference/alternative/checkpoint/log/W&B identities for all six;
  one immutable bank per environment/seed with shared read access. Preserve
  failed/partial files, producer status history and per-consumer gate audit.
- Source/base/launcher/request/config hashes; device/package/precision metadata;
  raw replay pairs/indices/RMS/normalized inputs and teacher trees/keys/base arrays;
  saved fresh initialization, frozen offset and literal targets; eight A/B phase
  receipts/states per observation,20 arm archives, four milestone summaries,
  all minibatch losses, full-pool predictions/errors at0/100/500/1000/2000,
  target/index/initial/final hashes; full SAC replay/RNG/simulator/optimizer
  checkpoints, ordinary evaluation logs, completion/error and resource receipts.
- Durable storage capacity and approved destination/transport. Alternative arrays
  are LOCAL ONLY. Preserve on durable Delta storage; copy with read-only rsync/scp
  or an approved equivalent, and independently verify a relative-path SHA256
  inventory. Do not delete local data or introduce per-file W&B round-trip eviction.
  Storage capacity, retention I/O and total bytes are unmeasured until qualified.

## CPU validation and remaining decisions

Fresh exact-source verification covers the six-cell map and resolved configs,
all-eight A/B carry with only even reported K, C's four fits, literal pairing,
optimizer resets, phase resume/corruption rejection, bounded bank gate success/
failure/partial/timeout/RMS/runtime contracts, no training RNG draws, default and
alternative-only routing, and tiny real consumer complete-state equivalence.
The real-critic example can select `--even-milestones`; independent reload checks
16 phase receipts/40 arm archives across two synthetic checkpoints. Full relevant
regressions, mutation suite twice,65 simulator restoration cells and static checks
are recorded in the new exact-head receipt, not relabeled from earlier PR18 work.

Pending Delta facts/decisions: exact source review, hardware/account/partition,
CPU/RAM and three-process binding/qualification, checkpoint interval, bounded gate
timeout, proven six-hour full-cadence feasibility, durable storage/root/retrieval,
project/run IDs, and approved qualification access if evidence is absent. The six
cells, teacher and revised schedule are fixed by this request; do not ask Delta
to choose alternative scientific settings. All review flags stay false. Only
Delta may submit after every hard check passes under the user's conditional
instruction; otherwise return NO-GO with no pilot submission.
