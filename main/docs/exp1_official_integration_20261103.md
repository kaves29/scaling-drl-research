# Official Experiment 1 reconciliation candidate

Owner correction on 3 November 2026 supersedes the earlier pilot-population
assumption: the official population is **11 environments × 10 seeds × 3 critic
sizes = 330 parents**, with ten targets, 2,000 updates and 50 checks. The
four-environment/eight-seed dataset is pilot evidence, not the official population.

The owner clarified: use the legacy thirteen tasks without any HumanoidBench.
That removes h1-reach-v0 and h1-run-v0, leaving eleven, not twelve.
`experiments/exp12/official_exp1.py` freezes those identities and seeds 1–10.
The owner confirmed randomized sine targets, not a sign-function substitution.

The owner retained legacy swimmer/hopper budgets: 500,000 raw steps each.
The official profile explicitly permits fresh capture and scheduled check 1 at
interaction 5,000, ordered before/after the first SAC update respectively.
All fifty checks, warmup and budgets remain intact; legacy profiles keep their
previous guard. A tiny real-run lifecycle regression verifies that ordering.
This population is a review candidate, not a qualified/executable allocation.

## Source route

This isolated candidate begins at owner-selected PR21
`38e5abb7325a949e23440d6197b5d7d9ecd57c43`, which incorporates PR20 and PR19
once. Approved integration remains
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`; no merge or production promotion
has occurred. Original branches and active/completed pilots are unchanged.

Independent integration work so far:

- Explicit `base_exp1_official` profile: ten targets × 2,000 × 50. Legacy
  `base_exp12` and Experiment 2 settings remain unchanged.
- Historical ten-target fractional IQM and matching bootstrap, preserving the
  existing five-target behavior and continuous first/next-1,000 optimizer stream.
- Ten-round ledger schema support, preserving rounds 5–9 instead of silently
  truncating them to a five-round CSV.
- PR23's complete existing-record W&B transport, including both aggregate scores.
- Reference-backed diagnostics from PR14, including masks and W&B artifact/scalar
  transport. They retain the existing opt-in controls with optional expensive rank computation. The official profile enables
  the measurements and rank at the existing logging cadence; legacy profiles
  retain opt-in behavior.

## Complete source path and invocation

`base_exp1_official` is explicit; do not replace legacy `base_exp12` globally.
Generate the source-recorded 330-run planning manifest (never submits anything):

```bash
python scripts/generate_official_exp1_manifest.py --root /absolute/campaign --out /absolute/manifest.json
```

It includes distinct checkpoint paths, ten seeds, legacy budgets, per-check
checkpoint cadence and resolved configuration hashes. Approval/executable/submitted
flags remain false. The ordinary Exp1 runner consumes the generated argument lists.
No Delta resource layout or submission command is invented.

The frozen-analysis manifest tool accepts `--protocol-version no-hb-v1` without
requiring m for Exp1. The certifier selects the eleven-task/ten-seed population,
checks all ten target fields and exactly 2,000 updates, and recomputes ten-round
IQMs with their proper fractional weights. It does not certify Exp2 under the new
version. Legacy five-round certification and its population remain unchanged.
Learning-curve panels choose first/middle/final checks from the actual artifact.

Remaining: review/promotion of this source, actual CUDA and full-size numerical/throughput/
storage/recovery qualification and separate launch approval. Current tests include
actual tiny CPU-produced ten-target arrays accepted by the curve contract,
truncated/wrong-configuration rejection, all 330 resolved configurations and
fail-closed rejection of a missing 330-parent evidence set. They do not constitute
natural scientific evidence. Per-step losses and endpoint components remain
saved; full-pool input/target/prediction retention beyond existing checkpoints
requires its own source-matched capture/storage qualification, not an assumed
successful transfer from the diagnostic pilot.

Nominal workload: 666.6 million probe updates plus roughly 296.7 million SAC
updates across the 330 parents. Existing eligible-control continuations can add
work after the nominal endpoint. No reliable GPU-hour forecast is established.
Final exact commands, source SHAs and test counts are in the external receipt.

## Validation evidence and scientific effects

A complete **synthetic** 330-parent fixture passes official Exp1 certification;
changing source metadata makes certification fail. This is packaging/IO evidence,
not 330 trained agents. Actual tiny CPU ten-target, 2,000-step arrays pass the curve
contract; incomplete and wrong-config arrays fail. All 330 resolved identities and
legacy budgets are checked. The broader combined suite passed 85 tests before the
last entry-point/fixture additions; final exact-head counts are external rather
than added to that overlapping count. Reference diagnostics have known-answer
rank/dormancy tests and offline W&B artifact/history readback.

An initial mutation run was 72/74: the two restored-state arm tests were refused
because the working source was dirty. The exact clean-source guard was preserved;
clean-commit rerun results are recorded externally. No scientific gate was waived.

Intentional effects relative to the old official source: increased population,
ten-round fractional IQM and bootstrap, 50-check grid, same-step initial/check-1
ordering on the two short tasks, and automatic supplemental diagnostics. These
can change results relative to the old five-target/20-check study; they are the
owner-directed new protocol, not corrections to archived measurements. Optimizer,
teacher sine construction/scale, normalization, own offsets, first/next-1,000
streams, budgets, statistical rules and primary endpoint definitions are retained.
New diagnostics can add substantial SVD/storage/I/O cost and remain disableable;
full-size resource/throughput qualification is still needed.

Use `--experiment exp1` with the normal CLI defaults to select this official
profile; explicit legacy `--config_name base_exp12` remains available for old
fixtures/protocols. New source cannot resume old-source official runs through the
source-admission guard. No active/completed run or historical output was changed.

## Final integration regression found and repaired

The first configuration-driven diagnostics integration used one fixed output
folder. The collector correctly refuses existing folders, so reopening a run
would fail before restoration. The repair follows the existing environment-driven
observer pattern: each opening gets a distinct UUID session beneath the run's
`critic_diagnostics` directory. Previous masks/receipts are preserved. Persistence
statistics are session-scoped; do not interpret a restart as uninterrupted mask
history. No training RNG is used for the session identifier.

The reopening regression fails on the pre-fix trainer and passes with the repair.
This was an integration defect caught before release, not evidence of a cause of
the historical probe scores. Before this final repair, clean-source validation
passed 87 combined CPU tests and all 74 mutations; 110 official environment/seed
restoration checks passed. Final-head results are separately recorded externally
and must not be added to those overlapping counts.

## Final CPU result and remaining owner choice

At implementation `6d011afa7160c93b377b05452fb81ba1c1733c1e`, all 88 combined
CPU tests and all 74 mutation checks pass. There are 110 passing environment/seed
restoration cells. These counts overlap prior runs and are not additive.

The ten-seed **count** is settled; the current 1–10 IDs are provisional. Seed 9
was previously development calibration. Owner clarification was requested on
using IDs 1–8,10,11 to preserve its exclusion versus explicitly revisiting that
exclusion. Until resolved, do not freeze/promote or launch the provisional
population. The environment list, budgets, sine targets and all other reconciled
settings remain as described above. Final source freeze requires this answer;
this review draft is concrete and CPU-tested but not production-approved.
