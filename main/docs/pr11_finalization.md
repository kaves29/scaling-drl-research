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

The owner supplied `echocritic-comprehensive-20261024.zip` (SHA256
`f7337130705a59622e1b639449e0dad452317fa2a45b479760a92aa3d9b12a44`).
The unchanged offline CLI processed this actual ZIP, separately from synthetic
fixtures. Independent checks establish:

- 103 unique record IDs; 96 campaign declarations at f597 and seven pilot
  declarations at f861. Source and probe configuration distinguish these cohorts;
  names are not identifiers.
- Complete four-environment × three-architecture × eight-seed campaign grid;
  12 groups of eight seeds. Across seeds, configurations differ only at `seed`,
  `agent.seed` and `env.seed`.
- Seven duplicate display-name pairs, no duplicate IDs. Pilot records are kept
  separate; observational identity is examined in the separate forensic audit.
- 95 finished records and eight crashed campaign records; no restart stitching,
  extrapolation or imputation. The crashed runs lack62 scheduled probe checks.
- 25,557 history rows; all103 counts match `attrs.historyLineCount`. Every raw
  history value, including nulls, survives the original pipeline unchanged.
- All697 raw-file manifest hashes/sizes verified. All52 supplied source snapshots
  match their manifest and immutable Git objects at the declared revision.
- Pilot W&B histories contain no probe measurements. Campaign nonzero checks
  retain round scores but do not directly log contemporary `score_fresh_iqm`;
  that field is initial-only. Per-record missing/null/observed coverage is retained.

**Native-export compatibility remains unqualified.** The current CLI expects
explicit normalized `code`, `campaign` and counter declarations; this export
records source under `attrs.commit` and uses `_step` plus explicit probe counters.
Consequently the current validator inventories all103 but labels every cohort/
source/counter unverified and emits zero qualified aggregate groups. Its generic
`partial_history` flags are not evidence that all103 raw exports are incomplete.
Independent accounting above does not make those failed integration gates pass.
Recorded revisions and matching source snapshots also do not establish that
launch worktrees were clean; that status is not silently invented.

A reviewed schema/counter/provenance adapter and seed-configuration grouping
checks are still required before qualifying real-data aggregates. Under the
owner's subsequent read-only forensic-audit instruction, newly discovered
analysis corrections are prepared separately as proposals, not silently adopted
in the production pipeline. The actual archive has been tested; synthetic103
success is not presented as compatibility evidence.

Reproduction command from `main/`:

```bash
python -m analysis.campaign.pipeline /absolute/path/echocritic-comprehensive-20261024.zip \
  --out /absolute/new/real-analysis --bootstrap-reps 10000
```

## Validation

The final tested implementation is
`f8ad82f064547c76e12539804c98be75b32656d8`. All runners exited successfully
with an unchanged source fingerprint. Artifacts are outside Git in
`/workspace/artifacts/pr11-finalization-20261024`.

| Check | Result |
| --- | --- |
| Isolated full CPU suite | 901 passed, 103 modules, no failures/errors/skips |
| Complete mutation suite | 74/74 pairs twice, including restored twin metadata |
| Combined focused integration/offline/qualification | 79 passed; overlaps full suite |
| Six supplementary pytest modules | 51 passed, one archive-dependent skip |
| Simulator restoration | 65/65 exact cells |
| Methodology/configuration reference | Passed; 195 parent commands / 39 configs checked |
| Both offline import CLIs | Passed on explicitly synthetic103 fixtures |
| Real ZIP | 103 records inventoried; original pipeline aggregate qualification blocked as above |
| Static | Scoped Black, compilation and production/new-module Pyflakes passed |

Six pytest-only modules reported zero unittest tests and were run explicitly;
they are not counted as unittest coverage. The skip requires absent
`PILOT_ARCHIVE_ROOT`; the supplied single campaign NPZ is not that pilot archive.
All-changed-file Pyflakes additionally reports four pre-existing test warnings,
reproduced at b061: two intentional helper re-exports and two unused imports.
No test is disabled and no numerical acceptance threshold is loosened.

Earlier failed/development and intentionally interrupted attempts are preserved
separately and are not counted as passing final suites. Counts overlap where
noted; do not sum focused tests with the full suite.
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


## Remaining qualification and scientific impact

The historical Mac restored-case failure cannot be definitively attributed from
its discarded traceback. Confirmed fixture leaks, import-order mocking and source
instability are separately identified; current repeated full mutation runs pass.
Real export mapping/grouping and unrecorded worktree provenance remain integration
limits, now backed by actual-data validation rather than an unavailable-file claim.

No historical parameter, result or score is changed. Fixture fixes change testing
reliability; sidecars change optional recording and I/O overhead. Small CPU enabled/
disabled Exp1 and injected-Exp2 learning states match. That does not establish
wall-time invariance, all-architecture instrumentation qualification or GPU identity.

GPU-dependent work remains the exact original A100 segmentation-equivalence
reproduction with fixed numeric inputs, optimizer state, streams and unchanged
acceptance guard, plus source-bound enabled/disabled recording checks under the
reviewed GPU precision modes. No such jobs are authorized or launched here.
PR #11 remains draft; nothing has been merged.
