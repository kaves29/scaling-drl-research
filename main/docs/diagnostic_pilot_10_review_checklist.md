# Ten-run simultaneous diagnostic pilot review

Current verdict: **BLOCKED**. The adapter is unimplemented/unreviewed, source is
not frozen for launch, and no exact-source Delta smoke or five/five profile is
available. The draft flags remain false. This checklist is not an approval.

## Scientific plan and identities

- [x] Exactly ten executions: five matched condition/seed configurations, each
  once per GPU, with five children per device and one all-ten barrier release.
- [x] All four tasks and all three sizes represented. Device assignment has
  identical condition/seed marginals, rather than different domains per GPU.
- [x] Ten distinct run/output/checkpoint/W&B identities; no replacements or
  automatic retries. GPU replicas are not independent training seeds.
- [x] Original source/configuration hashes retained; original own-offset targets,
  teacher/base pairing, optimizer, fitting procedure, schedule, statistics,
  thresholds and hypothesis preserved. Shared base is identical within a pair;
  literal targets retain the intentional own-offset difference.
- [ ] Owner reviews the five-cell selection, missing factorial cells, source
  baseline, interpretation limits and unchanged scientific endpoint/extension.
- [ ] Global W&B identity check verifies new project/namespace IDs are unused;
  no archived run is resumed or relabeled. Assign a new canonical durable root.

## Adapter and source review — all still required

- [ ] Implement the observation-only adapter; review its exact Git entrypoint,
  protocol `echocritic-observation-only-v1`, baseline/adapter relationship and
  deployment-config reconstruction from the pinned source scientific config.
- [ ] All prescribed arrays/learning curves, scales, normalizer/replay/teacher/
  RNG state and dtype-aware hashes are retained at the specified checks/rounds.
- [ ] Establish initial/intermediate *full-pool* error capture with unchanged
  updates/optimizer/RNG. Keep pre-update minibatch curves separate. No refits.
- [ ] Validate diagnostics-on/off outputs, parameters, optimizer state, RNG,
  simulator restoration and continuous/segmented execution under existing
  criteria. Resolve the outstanding A100 equivalence failure without loosening
  tolerances or incorporating unrelated PR changes silently.
- [ ] Freeze a clean reviewed executable SHA containing the final adapter and
  launcher. Record Python/JAX/jaxlib/Optax/CUDA backend/driver identities.
- [ ] Adapter accepts only the documented manifest/run identity arguments;
  overrides only output/W&B deployment metadata, never scientific settings.
- [ ] Adapter writes per-run `diagnostic_completion.json` binding
  `expected_commit`, `plan_sha256`, `pilot_id`, `config_sha256`,
  `diagnostics_complete=true` and `scientific_endpoint_complete=true`, backed
  by actual hashes/arrays/counters. It preserves failed/partial evidence on signal.

## Actual Delta allocation and qualification

- [ ] Read-only `sinfo`/`scontrol` verify the actual node, gpuA100x4 partition,
  account, maximum walltime, two allocatable A100-SXM4-40GB devices and scheduler
  CPU/RAM/GRES/cgroup semantics. Capture raw scheduler responses and timestamps.
- [ ] The proposed one-node/two-GPU/one-task/32-CPU/128GiB/48h reservation is
  available and sufficient. It is a candidate request, not observed capacity.
- [ ] Same-source CUDA smoke succeeds on both intended physical UUIDs with the
  qualified runtime/driver and unchanged TF32 rules. No CPU fallback.
- [ ] Representative *simultaneous all-ten*, five/five profile uses exactly the
  pilot mixture and configs; no staged admission/waves/serial substitution.
- [ ] Profile covers cold compilation, steady training, a complete original
  scheduled ten-round paired2,000-step probe and overlapping checkpoint/storage
  work, with diagnostics enabled and existing equivalence checks passing.
- [ ] Record total/peak per-device VRAM, host/cgroup RAM, exact CPU affinity/NUMA
  topology, per-process RSS/CPU/I/O, storage throughput/free space/checkpoint
  latency, compile/probe timing and minimum steady throughput.
- [ ] JAX sees one intended GPU per child after inherited-token binding and
  before any training import. Match physical UUIDs/order, runtime and qualified
  CPU affinity in the real allocation. Qualify bootstrap-to-adapter exec too.
- [ ] Measured peaks plus reviewed GPU/host/storage headroom fit the reservation;
  projected full nominal/extended endpoints plus stop margin fit48h. Do not
  divide aggregate historical runtimes by five to claim capacity.
- [ ] Verify storage quota, bandwidth and retention costs on the actual durable
  filesystem under ten simultaneous writers; distinguish shared node counters
  from exclusive pilot usage. Assign reviewed telemetry/stop/barrier limits.
- [ ] Explicit source-bound allocation and budget approval; no old2/4h or
  96-parent approval is repurposed. Any qualification workload has a separately
  reviewed scope/budget; it is not a hidden extra scientific pilot arm.

## Receipt schema and sealing

External approval must satisfy `launch.preflight`; the unapproved template
under `audits/probe_discrepancy/` is a field inventory, not passing evidence.

Every named gate (`adapter_review`, `scientific_review`,
`cpu_restoration_equivalence`, `cuda_smoke`, `five_process_per_gpu`,
`storage_profile`, `live_delta_layout`, `wandb_identities`, `allocation_review`,
`owner_launch`) must have an absolute JSON path and SHA256 binding. Each receipt
needs real `verdict=PASS`, a decision reference, the same executable
`expected_commit`, canonical `plan_sha256`, exact `runtime` and `allocation`,
and nonempty underlying `raw_evidence` absolute paths/SHA256s. No synthetic CPU
fixture receipt may be installed as real approval/qualification evidence.

The packing receipt additionally requires `simultaneous_all_ten=true`,
`max_concurrent_processes_per_gpu=[5,5]`, `waves=1`, the sorted ten
`run_config_hashes`, `node_name`, ordered physical `gpu_uuids`, qualified
`cpu_affinity`, and measured `peak_gpu_memory_MiB`, `gpu_total_memory_MiB`,
`peak_host_memory_GiB`, `projected_endpoint_seconds`,
`minimum_interactions_per_second`, `paired_probe_seconds`, `cold_compile_seconds`.
The `cold_compile`, `steady_training`, `paired_full_probe`, `checkpoint_io`,
`binding_verified` and `on_off_equivalence` fields must all be true with raw
traces supporting them. The review must check late/extension-state projection
uncertainty rather than treating a prefix as an observed full completion.

Live layout evidence requires a current `valid_until_utc`, matching node and
true `scheduler_layout_verified`, `cpu_memory_cgroup_verified`, `storage_verified`.
The finite job memory envelope must also be measurable at actual startup.
Qualification/approval JSON is reviewed evidence, not a cryptographic signature;
its raw records and stated decision authority must be independently checked.

Before qualification, assign the future implementation SHA/root/runtime/layout
in a new external manifest. Only after unchanged-plan gates pass may its readiness
flags/status be sealed. `plan_hash` excludes only `launch_approved`, `executable`,
`diagnostic_status` and `resource_estimate.packing_qualified`; all scientific,
source, root, identity, resource and schedule fields remain bound. External
approval also hashes the complete sealed manifest and critical launcher files.
A substantive plan/source/configuration change invalidates prior gate receipts.

## Launch review and stop

- [ ] Record all real source/plan/runtime-bound gates, fresh live layout evidence,
  owner launch decision, absolute caps and unique identities. Only then seal the
  external flags; the checked-in draft remains unapproved/non-executable.
- [ ] Run fail-closed `preflight`, then obtain the exact command with `command`.
  Do not submit a manually assembled wrapper command to bypass preflight.
- [ ] Review the generated exact `sbatch` node/layout/paths/hash arguments; the
  wrapper revalidates them inside the allocation before spawning any child.
- [ ] Source-bound engineering completion checks each run's actual endpoint and
  diagnostic receipts. Partial/failure statuses and raw files remain intact.
- [ ] Stop after these ten attempts. No waves, free-slot replenishment, automatic
  retry, modified science setting, unreviewed fallback allocation or main launch.

No box requiring real Delta execution or owner approval was marked complete in
this task. A failed/unavailable five/five qualification leaves the requested
schedule blocked. A lower-concurrency wave plan or different GPU allocation
would require a new explicit owner decision; neither is automatically selected.
