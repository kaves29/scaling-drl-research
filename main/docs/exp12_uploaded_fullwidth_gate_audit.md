# Uploaded full-width gate metadata audit

2026-10-10. The small archive is readable and its supplied final PASS report is
internally consistent with the original stage metadata. This is an independent
metadata/receipt audit, not a rerun of bitwise comparisons on omitted tensors.
No code, methodology, evidence, jobs or scientific qualification was changed.

## Archive and provenance

Attachment: `exp12_gate_small.tar.gz`, 19,841 bytes; SHA-256
`899b018ba13dce49b7f6dd81b523ea7258807c606a76c5c688258bd25be8dc03`.
No independent external checksum sidecar was supplied. Both outer and nested gzip
CRC checks pass; all members are unique, readable and safely extracted without
links/path traversal. Outer archive: 5 members, 4 files. Nested
`stage_metadata.tar.gz`: 44 files. Original attachment is unchanged; extraction
is `/workspace/scratch/exp12-gate-small-audit/`.

Final `exp12_gate_small/validation.json` SHA-256:
`a963655cb5fd948e1afcd7afacf4d2e56a816639b41faabcbcfe39b0a29b634d`.
GPU source: `7d21eaa106a336168464cd5418a59136b6936a64`.
Comparator: `07721bf27055c2c407f635149bd9bbf54e97e989`.
Canonical spec SHA-256:
`a7b6c810f55253537740c0edeaafd2b5cc0e2da459311bf053c689aa9671423c`.
All four resolved configs match the canonical dev pilot's configuration hash:
`7efc91337a1b200b8c437dbea5f5f0f1d251df056cb8bbb5dc0e0fd790ee924a`.
They specify 1M **raw** steps, 500k interactions, seed102/dev, no forced trigger or
identity early stop. The diagnostic itself stops at6000/6001/6061; no full-budget
training is claimed.

All 26 JSON documents were parsed with duplicate keys/constants rejected and
recursive numeric-finiteness checks. Final report stage records exactly equal
the separately supplied receipts. Their byte hashes equal the accepted attempt
receipt hashes. All five process records, submitted specs, commits and exit files
agree with stage/job identities. Cache per-backend hit/miss totals independently
sum correctly. Expected interaction/update counts agree. The unchanged verdict
classifier returns PASS, and online/target Check 1 JSONs both report pass. Parameter
and floating optimizer placement records are GPU/float32; target optimizer is
correctly absent. All reported reference/resume/roundtrip differences are empty.

| Stage | Accepted job / exit | Endpoint / updates | Cache GPU hits/misses | Receipt binding |
|---|---|---|---|---|
| Cold | 22791700 / 0 | 6061 / 2124 | 0 / 172 | Verified |
| Warm | 22793638 / 0 | 6061 / 2124 | 172 / 0 | Verified |
| Crash | 22793941 / 3 | 6001 / 2004 | Not published at abrupt exit | Verified |
| Resume | 22795539 / 0 | 6061 / 2124 | 37 / 2 | Verified; restored6000/2002 |

Failed warm22791701/exit1 is preserved and explicitly listed, not replaced. Its
actual log says `ValueError: cold reference must execute first`, corresponding to
the cache directory not being present at preflight. It failed before the workload;
its exact scheduling/path cause is not established without scheduler history.
The stale `reference_warm_error.json` faithfully preserves that error. Nested
historical `validation.json` says INCOMPLETE with only warm marked incomplete;
this is the old singleton-attempt comparison, not an independent contradiction of
the final corrected report. Neither historical record was altered.

Comparator.log contains missing DISPLAY and AMD/ROCm detection warnings with
`CpuDevice(id=0)`. This is the **CPU-only comparator process**, not the GPU training
stages. Those separate receipts record A100 GPU training and GPU state placement.
The warnings do not establish fallback in the GPU jobs. Stage logs otherwise
contain Orbax deprecation/missing-RestoreArgs-sharding warnings; no additional
traceback appears in the successful attempts. These warnings alone are not proof
of invalid restoration; actual comparisons and placement remain separate checks.

## Actual runtime and memory measurements

| Quantity | Cold | Warm | Resume |
|---|---:|---:|---:|
| Initialization | 13.841 s | 21.768 s | 12.664 s |
| Prefix to6000 | 71.813 s | 96.672 s | Not applicable |
| Measured61-interaction continuation | 1.333 s | 1.349 s | 11.662 s |
| Continuation throughput | 45.748/s | 45.217/s | 5.231/s, restart-contaminated |
| Backend compilation total | 25.545 s | 0 s | 6.412 s, 2 calls |
| Checkpoint at6000 | 3.806 s | 4.520 s | Restore2.267 s |
| Final save | 3.565 s | 3.780 s | 3.871 s |
| Host process peak RSS | 11.878 GiB | 5.491 GiB | 5.294 GiB |
| Allocator peak in use | 3.510 GiB | 2.632 GiB | 2.656 GiB |
| Allocator pool peak | 4.006 GiB | 4.006 GiB | 4.006 GiB |

Cold diagnostic fork write19.969 s, fork restore1.511 s. Fork write includes
qualification work, not just a routine state-save measurement. Crash timings
contain prefix56.510 s and checkpoint4.197 s, but no final RSS/memory/compilation
summary because exit3 intentionally bypasses that publication.

The cold/warm windows project **3.036–3.072 training-only hours** for500k nominal
interactions, or **3.643–3.686 h** for the latest eligible600k control endpoint.
The resumed61-step window's reported26.553 h projection is arithmetically correct
but unsuitable as steady-state extrapolation: it includes6.412 s of measured
recompilation. Even subtracting that leaves5.250 s; its other costs are not
resolved by these logs, and must not all be called compilation. It is not evidence
of a stall or numerical resume defect, nor proof later training runs at45/s.
Monitor sustained checkpoint-to-checkpoint progress in the actual pilot.

Warm cache hits do not guarantee lower total prefix time: warm prefix96.672 s is
slower than cold71.813 s despite zero backend compilation. Cache loading, host/
filesystem contention and other work are hypotheses, not measured causes here.
Timed stage sums omit process/import/metadata publication overhead; they are not
Slurm elapsed times. Compilation is nested within stage times and must not be
added twice. The source still omits production probes from this engineering gate.

Combined with the older measured full-setting probe/evaluation/large-replay-save
costs, a roughly **3.5–4.7 h canonical pilot scenario**, rounded conservatively to
**3.7–5 h planning**, remains reasonable. It is not a full-run bound. One40-GB
A100,16 CPUs,64-GB host RAM and8h wall time have substantial observed short-run
memory/time headroom, but full probes, growing replay, late control and actual
Delta allocation/account constraints remain unmeasured/unverified. Resource
requests do not alter the training budget or create extra environments.

## Missing evidence and scope

This intentionally small package omits checkpoint tensors/replay payloads, the
attempt SHA256SUMS files whose hashes are referenced in the final report,
NVIDIA inventory files, Slurm accounting/stdout/stderr, full runtime traces and an
explicit comparator exit-status file. Thus the individual stage exits are
verified from supplied process/exit files, but comparator exit0 remains the
owner's report; the JSON verdict itself is verified. Full checkpoint comparison,
manifest integrity and scheduler allocation cannot be independently recomputed
from this package. If needed, preserve/provide those exact files from the original
Delta tree; no new GPU run is required to obtain existing evidence.

Receipts identify the actual stage evidence root as
`/work/hdd/biqc/skaveti1/exp12_validation/7d21eaa106a336168464cd5418a59136b6936a64_fullwidth_20261010T163647Z/evidence`
via its cache path. The previously supplied path without `/evidence` is the parent
container. Any future CPU recomparison must first locate that directory's
canonical `spec.json`, stages and attempts, rather than assume the parent contains
them. Do not try to run the complete comparator against this stripped archive.

No hidden failure is established in the successful stage logs/metadata. The
records support the reported engineering PASS and original source recommendation;
they do not prove primary scientific gates or production launch qualification.
The pilot's unchanged raw/interaction-unit discrepancy still needs owner
clarification. Its preparation/submission/recovery checklist remains in
[exp12_full_length_development_pilot.md](exp12_full_length_development_pilot.md).
