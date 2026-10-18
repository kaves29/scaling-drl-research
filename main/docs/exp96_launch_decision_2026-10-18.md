# Exp96 scientific launch decision — 18 October 2026

**Status: proposed freeze for owner review; no scientific approval is implied.**
The 96 parents are a separate ten-target version of Experiment 1. The approved
five-target, 195-parent protocol and seven-run pilot remain historical evidence,
not ten-target calibration. This note does not authorize Experiment 2 arms.

## Measurement and analysis proposed for freeze

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

**Proposed primary Exp1 endpoint:** each completed parent's check-50
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
are proposed. This endpoint/uncertainty plan needs explicit owner approval
**before** official data collection; the original 195-parent analysis does
not automatically authorize its use for the 96-parent population.

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

**Minimum defensible disposition before data collection:** the owner must
explicitly approve the ten-target rule as an *operational detector with
unestablished false-trigger rate and sensitivity*, record that limitation in
the `scientific_calibration` gate, and defer claims about calibrated onset or
plasticity-loss prevalence until independent full-width controls are reviewed.
This is a proposed limited scientific scope, not a declaration that the gate
is already passed. Under that scope, training trajectories can be collected,
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
| Owner decision before collection | Approve this versioned measurement, limited calibration scope, primary endpoint/cross-seed plan, four tasks including Myo Pen Twirl, and no automatic Exp2 authorization. |
| Source-bound operational gate | Codex CLI must verify the final SHA's smoke receipt, resource/packing profiles, storage, queue identity, and both approval files against the existing gate. No old-SHA smoke may be relabeled. |
| After collection, before mechanistic/onset claims | Full-width fresh/fresh calibration with independent units, longitudinal interpretation, fixed-pool crossover/shared-target validity, and sensitivity evidence if a sensitivity claim is desired. |
| Optional analysis | Additional endpoints, multiplicity-adjusted family-wide claims, longer fitting, and Exp2 intervention analyses require separate prespecification/approval. |

No approval template is marked approved by this document. The staged Delta
queue, frozen checkouts and prior evidence are unchanged; Codex CLI owns
their live reconciliation and any submission.
