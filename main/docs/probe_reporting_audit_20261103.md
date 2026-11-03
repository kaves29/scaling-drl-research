# Probe reporting repair and implementation audit

This candidate fixes probe telemetry without changing the scientific calculation.
Its base is approved integration/exp12 at
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. It does not promote a newer probe
protocol, modify the completed pilots, or authorize execution. Resolve the
candidate's final tested SHA from the PR/external validation receipt.

## Confirmed issue and repair

The integration source persists probe records locally but does not transport
them to W&B. The later historical/PR16 path logs rounds and paired deficits but
omits the current/fresh aggregate IQMs at paired checks. This can leave aggregate
charts missing or summaries at their earlier logged values.

`experiments/exp12/run_probes.py` now logs each already-computed record, including
current/fresh IQMs, individual rounds, paired loss, validity and trigger fields,
with interaction steps and raw environment steps. The initial record contains
fresh only. A custom W&B step definition keeps probe charts on interaction time.
No active W&B run means local records still work and no upload is attempted.
Offline mode uses the SDK's normal local history; this is tested by reading the
actual W&B datastore after finish. Authenticated online delivery was not tested.

No sampling, normalization, model operation, score calculation or fit is added.
This fixes future transport; it does not edit historical W&B runs or backfill
their histories. SDK logging consumes wall time/I/O and can fail independently
of numerical correctness; no runtime or network qualification is claimed.

## Scientific impact and protocol boundaries

|Item|Disposition|Scientific effect|
|---|---|---|
|Missing aggregate telemetry|Repaired on isolated candidate|Same saved scores; additional correctly stepped W&B history|
|Probe kernels, optimizer, initialization and RNG|Byte-unchanged from base|No intentional numerical change|
|Training, schedules, budgets and thresholds|Byte-unchanged from base|No methodological change|
|Replay sampling/teacher/offset definitions|Preserved|Their interpretation limits remain; not established bugs|
|Completed results and active pilots|Untouched|No historical recalculation or runtime replacement|

The base retains its older five-round/1,000-step protocol. This PR deliberately
does **not** replace the approved newer ten-round/2,000-step/50-check source with
that base, implement a new protocol bridge, or qualify this candidate to run a
campaign. The reporting repair must be applied to the reviewed execution source
without reverting that source's scientific changes. In PR16's existing explicit
`wandb.log` path, the equivalent minimal transport repair is to add the two
already-computed `score_current_iqm`/`score_fresh_iqm` fields; do not retain both
the old explicit transport and this shared transport and log a check twice.
PR16 and other existing branches remain unchanged.

## Implementation sweep

Reviewed the available historical producer, current integration producer,
scheduler/capture, normalizer, twin expansion/combination, and diagnostic adapter.
PR16 remote HEAD was `5ecca42cc820f936782f92f708896abcfbe651a0`; its adapter
SHA256 is `0e2c430c3f908ced638c01a6e76a80ab25eb09dcbc3991c074fc6463d263f6dc`,
identical to the previously retained adapter inspected in the audit.

No new defect was established in current/fresh parameter routing, dedicated RNG
streams, single normalization, shared base targets/minibatches, fresh optimizer
state, continuous fitting, use of post-fit predictions, paired subtraction or
training-state isolation. Earlier independent audit controls and their scope
are recorded under `/workspace/artifacts/probe-code-sweep-20261103/REPORT.md`.
The new candidate's own test receipts are separate; earlier test counts are not
represented as tests of this candidate.

The ten-run compact export names execution source
`51dc78e04adf79257744a9d921a61d123443a962` in identity files and
`73e8fb6dcc2458fd34dad7607b2796f016a21b0c` in result tables. Those source trees
remain unavailable from the inspected local/GitHub records. The adapter
comparison above does not bind either execution SHA. Exact runtime/exporter
source and retained raw arrays are still required to certify that implementation
or explain this discrepancy. No speculative source hash rewrite was made.

## Confirmed measurement limits, not automatic fixes

- Uniform-with-replacement sampling produces about 4,970 distinct replay indices
  at check 0 versus about 24,960 at the final check. All 200 selected reconstructed
  index-array hashes match. This establishes index multiplicity, not physical
  diversity, and does not isolate why fitting difficulty changes.
- Teacher architecture changes with learner architecture. Equal target variance
  does not establish a common function or equal fitting difficulty across sizes.
- Own-mean offsets cancel a pure constant Q shift in the initial residual. Huge
  initial error must not simply be attributed to a huge Q mean. Frozen-offset
  sequential diagnostics have different semantics.
- Finite-budget endpoints mix starting burden and fitting trajectory. Slow
  improvement and a nearly flat trajectory should be distinguished, not fixed
  by automatically increasing steps.
- `sin(100000 * teacher_output)` is sensitive to small teacher-output changes.
  Replaying stored targets differs from regenerating them on a new backend.

A fixed-package control, shared teacher bank, different sampling/offset,
optimizer, precision, sine scale or fit budget would change the scientific
measurement and requires a separate explicit decision. None is introduced here.

## Validation and remaining scope

New regressions cover five- and ten-round record transport, exact stored values,
initial fresh-only capture without an extra fit, no-active-run behavior, actual
offline SDK readback and global Python/NumPy RNG preservation. The aggregate
transport tests fail on the unmodified base and pass with the fix. Existing
training on/off identity, twin reporting/restoration, runtime and mutation checks
provide additional regression coverage; exact commands/results are recorded in
the validation receipt rather than treating stale reports as authorization.

The focused candidate suite reports 48 passing tests, including five new
parameterized transport cases and existing probe, twin-reporting, twin-critic and
runtime checks. The two aggregate transport cases fail against the unmodified
base (five and ten rounds). Legacy simulator construction/next-transition and
action-RNG restoration reports 65 passing cells; that is infrastructure
compatibility coverage, not a change to the official environment/seed population.
Dependency deprecation warnings are retained in the logs. Final-head and
mutation results are recorded in the external receipt/PR description when
complete; an unfinished command is not counted as passed.

No GPU job, scientific probe fit, source promotion, merge or live pilot change
was performed. CPU validation is not CUDA/full-size adapter equivalence or
scientific qualification.
