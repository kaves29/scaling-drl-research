# PR #11 integration finalization

Scope: continue `b0617c60cfff4b33bc267773f888d6059e9cbc17` on its existing
`codex/echocritic-batch2-20261024` branch. Preserve the #8 → #9 → #10 ancestry;
PR #12 remains a sibling and is not merged. No GPU job, PR merge, scientific
setting, acceptance threshold or archived output is changed.

## Mutation investigation

The unchanged b061 suite passed all 74 mutation/restoration pairs in this Cloud
environment, including the twin metadata pair. Its mutated case raises the
expected missing-`critic_count` KeyError; restored setup and assertion pass.
Per-case baseline logs retain both outcomes. This does not reproduce the prior
Mac failure and is not evidence of its definitive root cause.

A confirmed fixture defect was reproduced using the original b061 twin fixture:
failed `setUpClass` leaves W&B patches, `wandb.run` and its temporary directory
behind because unittest does not call `tearDownClass` after setup fails. The
new failing-setup regression verifies cleanup of all three. Class cleanups now
register resources immediately, and the W&B helper restores `wandb.run` as well
as its patched functions. The mutation's original metadata builder is retained
in the harness rather than installed as an extra global in the production
Exp1 module.

The earlier task history also shows a commit while its full mutation run was
in progress. Exp2 deliberately rejects source/dirty state differing from its
parent, including during the twin restored-state fixture. A controlled test
reproduces that source mismatch failure. The mutation harness now fingerprints
HEAD and tracked file contents around each run and stops with an explicit
source-changed error. It retains restored-failure tracebacks rather than
silently discarding them in an unmanaged `/dev/null` stream. A persistent
file-descriptor leak was not established.

The original failure traceback was discarded by the old harness. Source
instability and failed-setup pollution are demonstrated hazards, but attribution
of the historical 73/74 result remains unverified. Do not present either as a
proven historical root cause. Deterministic stable-source reruns, rather than
tolerance changes or disabled checks, are required for current qualification.

## Instrumentation

See [runner usage and semantics](echocritic_batch2.md#runner-integration-and-finalization-24-october-2026).
Observers are attached only at runner/lifecycle and existing metric publication
boundaries. SAC kernels, RNG use, optimizer construction and numerical guards
remain unchanged. Fixed-task runner GPU admission gates remain in force;
entrypoint lifecycle tests simulate admission on CPU and do not execute jobs.
Original runner failures survive secondary terminal-sidecar I/O failures.

## Real campaign archive

BLOCKED: no complete W&B metadata/history ZIP is available in this Cloud task's
uploads or artifacts. The only accessible archives contain prior engineering
evidence. No 103-record ingestion, pilot/canonical attribution, seed independence,
crash/missing-metric inventory or raw-value comparison against real histories is
claimed. Existing synthetic tests remain synthetic.

Once supplied, run from `main/` in the pinned CPU environment:

```bash
python -m analysis.campaign.pipeline /absolute/path/real-export.zip \
  --out /absolute/new/real-analysis --bootstrap-reps 10000
```

Before accepting aggregation, review the validation report against raw metadata
for every ID, nominal condition, source, campaign declaration, seed and crash;
reconcile duplicates and compare every metric row directly to the ZIP. Unknown
membership/provenance stays excluded rather than inferred from names. Real
schema adaptation, if required, must preserve raw evidence and receive its own
fixtures. The command alone is not that validation.

## Validation

Final validation receipts are recorded after stable-source execution. Artifacts
are outside Git in `/workspace/artifacts/pr11-finalization-20261024`.
CPU scope does not qualify CUDA/A100 numerical equivalence. The historical A100
segmentation failure remains unresolved and no GPU work is authorized here.

### Additional integration guards

Diagnostic CLI arguments are parsed before entering its instrumented `execute`
function. The campaign path is therefore available for validation before even
`launch.json` is created. Unsafe output tests assert that no directory is
created. Argument-parser failures retain argparse's ordinary behavior and do
not create a sidecar session. The first final-validation attempt was stopped
for this correction; incomplete receipts are not counted as completed suites.

Two existing `test_exp96_multigpu` fixtures fail independently before correction:
one incorrectly assumes HEAD still has the frozen GPU-smoke training fingerprint;
the other omits the now-required reusable-component receipt before the mocked
submission-timeout path. The corrected tests pin the actual immutable smoke
fingerprint, verify rejection of changed source and supply/mock the required
receipt so the timeout/reservation assertion is reached. No scheduler or GPU
qualification guard changes. These two fixture corrections duplicate the
already-reviewed engineering changes in sibling PR #12; they do not import its
other checkpoint/audit work. Resolve identical hunks once when integrating the
siblings rather than overwriting either branch.


### Corrections from the first full CPU run

At `3f56f677fefd6dd93cb9ddba3ff19206c82c1032`, all103 modules completed
900 test executions with an unchanged source fingerprint: 899 passed and one
raised a test-fixture error. `test_a_double_claim_is_detected` assumed worker1
must lose at least one of150 claims; worker1 legitimately won every claim and
selection raised `StopIteration`. Positive ownership/handoff verification
passed. The corruption now first asserts one winner for row0, then marks row0
claimed in both worker reports and requires the same two-winner diagnostic.
Actual workers, ledger, verifier and acceptance criteria are unchanged.

A separate deterministic temporary-directory reproduction found that calling
`execute(args=namespace)` bypassed the new campaign-output guard, which examined
only positional arguments. Both forms now resolve the same argument before any
receipt is created. The actual-runner regression exercises positional and keyword
calls, requires rejection before preflight, and checks that the campaign directory
and historical sentinel are unchanged. No real campaign was used by that
reproduction. This correction affects only opt-in sidecar path admission.

Enabled-training tests also compare recorded parameter counts, checkpoint path,
replay occupancy, counters and source revision against the actual saved checkpoint.
State comparisons use the existing `state_differences` contract (only external
W&B run identity is excluded), not archive-container byte identity.


The combined focused run also reproduced import-order pollution in the planned-
pause test fixtures: replacing `sys.modules["scripts.exp12_dynamic_scheduler"]`
does not replace `scripts.exp12_dynamic_scheduler` already cached on the parent
package. Production pause code therefore called the real validator against the
minimal mock job and raised a missing-`row` error. The fixtures now import the
actual dependency and patch its methods, verify the validator arguments, and
require the corrupt-checkpoint path to report the intended `corrupt` error.
The scheduler remains unchanged. This test pollution is distinct from, and does
not establish the cause of, the historical twin-metadata failure.
