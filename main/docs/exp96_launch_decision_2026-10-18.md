# Exp96 scientific launch decision — 18 October 2026

**Status: owner-directed limited-scope collection plan; final source-bound
study and allocation approvals remain pending.** The owner directed retention
of the existing detector and population without the proposed broad
300-pair/240-trial matrix before collection. This document records that
direction; it is not an approved study or GPU submission receipt.
The 96 parents are a separate ten-target version of Experiment 1. The approved
five-target, 195-parent protocol and seven-run pilot remain historical evidence,
not ten-target calibration. This note does not authorize Experiment 2 arms.

## Measurement and analysis for the source freeze

Keep the implemented four environments, three critic sizes, seeds 1–8,
500,000 agent interactions, action repeat 2, 50 checks at interactions
10,000–500,000, and ten independently keyed randomized targets per check.
Keep the same paired replay inputs, teacher draw and minibatch sequence for
fresh/trained fits, their existing network-specific offsets, and continuous
2,000-step AdamW fitting. For each target, loss is
`P(fresh) - P(trained)`; positive means poorer trained-critic fitting.

Sort the ten paired losses `x1 <= ... <= x10`. The implemented empirical
25%–75% IQM is `0.1*x3 + 0.2*(x4+x5+x6+x7) + 0.1*x8` (one-based ranks,
no value interpolation). Bootstrap **paired target-level losses** 10,000
times using the existing run-seed/check-index stream; recompute IQM10 and
take NumPy type-7 linear 2.5th and 97.5th percentiles. A valid check fires
only if its lower bound is strictly greater than zero. `f*_run` is the second
of two consecutive firing scheduled checks, if that second check is at most
47 (470,000 interactions); otherwise it is null. Check 0 is the fresh
baseline, not one of the 50 trigger checks. A late check cannot create a
retroactive eligible fork. The 125,000-interaction Exp2 horizon and separate
Exp2 approvals remain unchanged. These definitions are in
`experiments/exp12/probe.py`, `trigger.py`, `run_probes.py`, and
`configs/base_exp12_ten.yaml`.

**Selected primary Exp1 endpoint:** each completed parent's check-50
`loss_iqm`. For each scaled architecture, report the difference from D2W512
in IQM across the 32 parents (four environments × eight independent seeds).
Use an environment-stratified seed bootstrap, resampling whole parent
trajectories independently within each environment/architecture, with
50,000 draws and percentile 95% intervals, as in the original Exp1
architecture comparison. Preserve each environment
as a stratum; do not count ten targets, 50 checks, or repeated checkpoints
as independent training seeds. Report four environment-specific architecture
summaries and the complete trajectories descriptively. For onset, report
the number of parents with eligible `f*_run`, its interaction time among
those that trigger, and the number with no eligible trigger; the latter are
not zero-time events. Do not pool Exp2 fork outcomes into the parent endpoint.
No binary family-wide claim or unadjusted twelve-cell significance claims
are proposed. The owner requested this endpoint/uncertainty plan for the
96-parent version; its final source-bound approval record is still pending.
The original 195-parent analysis does not automatically authorize this version.

The prespecified, potentially confirmatory comparison is the **measured**
final-checkpoint IQM10 contrast between architectures, conditional on complete
source-bound runs. It is not, by itself, proof of intrinsic critic plasticity
loss. Onset counts/timing, environment-specific patterns, trajectories,
detector sensitivity, and causal or mechanistic interpretations are
descriptive/exploratory until the relevant full-width validity evidence is
reviewed. Nominal bootstrap confidence does not establish a 5% trajectory-wide
false-positive guarantee. Eight seeds per cell can yield wide uncertainty;
report the full intervals and null results.

All eight seeds in a cell must have complete source-bound artifacts before
that cell enters a confirmatory summary. List unfinished/missing runs and
recover them from valid checkpoints; do not silently replace, zero-fill or
classify them as non-triggers. Predefine any later multiplicity-adjusted or
causal claim separately. Raw check curves, intervals and trigger histories
must be retained regardless of their outcome.

## Calibration and pilot evidence

The earlier five-target Method C A100 result of 5/100 is a per-check boundary
result under another target count; it does not establish ten-target coverage,
the longitudinal two-check false-trigger rate, or trained-critic sensitivity.
The ten-target tiny CPU controls fired in 1/4 synthetic and 2/4 tiny-replay
fresh pairs, with short fits and small networks. They verify plumbing and
motivate caution, but cannot estimate full-width rates. The owner reports
the four-case A100 engineering smoke PASS at
`be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574`; this workspace has not
independently inspected its receipt. It verifies engineering only and its
source SHA differs from this branch. The source-bound smoke gate still
requires a hashed PASS receipt for the **exact final campaign SHA**.

The proposed 300 independent fresh/fresh pairs and 240 adjacent-check trials
are **not** a documented preregistered sample size. If divided evenly across
12 environment–architecture cells, they provide only 25 pairs and 20 trials
per cell. Even zero events have two-sided exact 95% upper bounds of about
13.7% and 16.8%, respectively; correlated repeated trials reduce the
effective sample further. Reversed orientations of a pair are not independent
replicates. The full matrix is therefore neither necessary for collecting
source-frozen parent trajectories nor sufficient to certify a <=5% rate in
each cell. The older five-target null data cannot substitute for ten-target
full-width calibration.

**Limited-scope disposition:** the owner has directed use of the ten-target
rule as an *operational detector with unestablished false-trigger rate and
sensitivity*. The final `scientific_calibration` approval must explicitly
record that limitation; the gate remains PENDING until the owner provides a
source-bound decision reference. Claims about calibrated onset or
plasticity-loss prevalence wait for independent full-width controls.
This is not a declaration that the gate is already passed. Under that scope,
training trajectories can be collected,
but their detector-based confirmatory interpretation remains conditional on
later validity evidence. If the intended confirmatory claim requires calibrated
<=5% per-check or longitudinal false-trigger control before collection,
additional independent full-width calibration is mandatory; neither the
existing evidence nor a small pooled matrix establishes it. Preselect cells,
replay positions, independent pair units and a decision criterion before
running that work. A bounded late-pool fresh/fresh and adjacent-check pilot
can screen for gross artifacts, but must not be reported as a 5% certificate.

The older Myo Key Turn pilot's fresh final-MSE/target-variance ratios rose
from early to late replay, while trained fits were often worse. Because the
replay pool, teacher draws and normalizer also changed, this does not isolate
critic plasticity from task difficulty or prove intrinsic degradation.
Saved-state fixed-pool crossover and shared-target checks would help separate
those mechanisms. They are needed before strong mechanistic interpretation,
not to preserve and collect the predefined training trajectories. Do not
change offsets, fitting steps or the trigger to improve these controls.

## Launch gates and timing

| Status | Item |
|---|---|
| Implemented, CPU checked | Ten-target arithmetic, 50-check schedule, durable 96-run identities, single-critic completion fix `d0bbde4352f6814c6ab8ae0736363469755fc540`. |
| Owner direction recorded; formal approval pending | Existing ten-target measurement, operational onset with unestablished calibration, selected cross-seed plan, and the four-task population including Myo Pen Twirl. Exp2 arms remain unauthorized. |
| Source-bound operational gate | Codex CLI must verify the final SHA's smoke receipt, resource/packing profiles, storage, queue identity, and both approval files against the existing gate. No old-SHA smoke may be relabeled. |
| After collection, before mechanistic/onset claims | Full-width fresh/fresh calibration with independent units, longitudinal interpretation, fixed-pool crossover/shared-target validity, and sensitivity evidence if a sensitivity claim is desired. |
| Optional analysis | Additional endpoints, multiplicity-adjusted family-wide claims, longer fitting, and Exp2 intervention analyses require separate prespecification/approval. |

No approval template is marked approved by this document. The staged Delta
queue, frozen checkouts and prior evidence are unchanged; Codex CLI owns
their live reconciliation and any submission.

## Source-bound handoff to Codex CLI

The existing `_approval()` in `scripts/exp96_campaign.py` requires an intact
hashed smoke `PASS` whose `expected_commit` equals the immutable queue plan's
commit. `startup_gate()` then requires the live clean checkout at that same
commit. There is **no** compatibility-disposition field or cross-commit
exception. The old `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` receipt
therefore cannot pass for this branch, even though the validator correction
was CPU-tested and the changes since that smoke are operational and
documentation changes. Repository tests model all four smoke output schemas;
the actual A100 output files were not independently inspected here. Do not
rewrite the old receipt or retag its source.

For the unchanged gate, the shortest legitimate route is one four-case A100
engineering smoke in a clean checkout of the **final published SHA** (existing
wrapper limit: one A100, 16 CPUs, 64 GiB, 90 minutes maximum), then review and
hash its real `PASS` receipt. Bind the final SHA, receipt hash, approved
scientific decision, queue hash and measured resource profiles in the external
approval files. A staged queue at an older SHA cannot be relabeled; Codex CLI
must reconcile its live state and prepare a new immutable queue for the final
source before any official parent starts. This handoff requests no job and
does not authorize an allocation.
