# D4W1024 fresh-null: trigger-threshold decision framework (2026-10-10)

For the research owner. Analysis only: no threshold, rule or setting is changed, and no recommendation here is a
methodology amendment. Base: `integration/exp12` @ `2d007663548be315b5bf4f438497058e6eb0f429`. Evidence and readings
already fixed: `null_followup_original_results.md` (sections 2–7); not repeated here.

## Fixed points
- **Approved trigger:** IQM of 5 round losses, percentile bootstrap (10,000), lower bound > `null_threshold` = 0.0,
  two consecutive firing checks among checks 1–19 (`configs/base_exp12.yaml`, `trigger:`).
- **Approved NULL rule** (`exp12_decisions.md`, "Pre-specified rules for the CUDA calibration checks", logged before
  any result): if the per-check fire rate on ≥ 100 fresh pairs exceeds 5% at any size, *the lead decides* whether to
  adopt the 95th percentile of fresh-pair L at that size. Claude does not adopt it.
- **Result:** D4W1024 13/100 (exact 95% CI 7.1–21.2%) = **FAIL**; D2W512 3/100 and D4W1536 5/100 pass. This FAIL
  stands whatever the replication shows.
- **What the threshold touches:** only the fork decision (f*_run, hence where Exp2 and the positive control fork).
  Every per-round probe loss is stored, so Exp1's L curves and its primary endpoint (final-check L contrast) do not
  depend on it. A run keeps only LATEST plus the fork state at the f*_run actually used, so **a fork at a different
  check cannot be recovered from a finished run**; it needs a rerun.

## What seed-991 can tell us (readings fixed in advance, section 7)
Same pair keys (`PRNGKey(10000+pair)`), new warm-up replay and probe streams.

| Reading | Pre-specified condition | Scientific meaning |
|---|---|---|
| R1 Intrinsic initialisation effect | test X significant (cross-seed Spearman of `loss_iqm`, one-sided p < 0.01) | Some fresh-critic pairs differ systematically. The 5-round bootstrap resamples rounds only, so it cannot see between-initialisation variance and is anti-conservative at D4W1024 *(hypothesis for the mechanism, consistent with but not proven by R1)*. |
| R2 Replay-dependent pair effect | shuffle test significant on the replication, X not | A real pair-level excess that depends on the replay; same anti-conservatism, not tied to the keys. |
| R3 No reproducible effect | neither significant | The original excess did not reproduce (chance or replay-specific); the replication's fire rate is read against the unchanged 5% rule. |

Also reported: the replication's own fire rate at threshold 0, and the p95 option's **out-of-sample** fire rate (t
from seed 990, applied to seed 991; `null_followup.py --replication`, `p95_alternative_REPORT_ONLY`).

## The three options

### 1. Retain threshold 0.0
- **Consequences.** Under R3 with a replication rate ≤ 5%: the FAIL is documented as one calibration not reproduced;
  nothing changes. Under R1/R2: the project knowingly runs a trigger whose per-check false-fire rate at D4W1024 is
  above the pre-specified 5% (≈13% observed). False forks dilute Exp2's injected-vs-control contrast with parents
  that had no plasticity loss and move D4W1024's f*_run earlier than D4W1536's for a noise reason, which confounds
  size comparisons of fork timing.
- **Validation required.** Replication fire rate under the unchanged rule; an explicit, dated owner acceptance of the
  FAIL. Under R1/R2, acceptance should also state the expected effect on Exp2 interpretation. The run-level
  false-fork rate cannot be read off independent pairs (checks within a run are correlated), so it stays unknown.
- **Development run.** Not invalidated and not repeated. The pilot (D4W1536, 5/100 passes) runs as planned; its fork,
  if any, remains the positive control's natural trigger.

### 2. Adopt the 95th-percentile threshold
This is the pre-specified fallback, not a post-hoc invention. It is still a change to the trigger, which needs an
owner decision and a dated amendment.
- **Consequences.** Seed-990 t: 1.10e-3 (D4W1024), 1.51e-3 (D4W1536), 7.0e-3 (D2W512). Its in-sample 0/100 is
  *by construction* (t is fitted to those data) and is not evidence. It moves the bar without addressing the
  suspected mechanism (R1/R2): a pair-level variance component the interval ignores. Per-size thresholds make "fires"
  mean different absolute losses at different sizes. A p95 of 100 heavy-tailed values (excess kurtosis 40 at
  D4W1024) is itself poorly determined. Sensitivity is lost by an unknown amount: a real loss must now clear ~1e-3.
  Open scope question: only the failing size, or every size? The rule says "at that size".
- **Validation required, all before adoption.**
  - (a) Out-of-sample fire rate on seed 991 within the existing 5% criterion.
  - (b) Sensitivity: the positive control's natural L_trigger must clear t. The positive control needs a natural trigger,
    whose timing depends on the threshold, so this must be judged on a dev run under the candidate threshold, or
    offline from its stored losses with the fork rerun.
  - (c) A scope decision.
  - Adopting it under R3 would mean changing the trigger after a gate failure that did not reproduce. That is the
    "change the threshold to pass" case and is not supported by the evidence.
- **Development run.** If adopted *before* the pilot: run the pilot under the new value (config change plus amendment).
  If adopted *after* a pilot run at 0.0: the pilot's Exp1 probe data stay valid and its f*_run can be recomputed
  offline, but its fork, and the positive control built on it, **must be repeated**. If t is applied to D4W1536, the
  dev run is directly affected. If it is applied to D4W1024 only, the D4W1536 dev run is unaffected, but D4W1024
  then has no positive-control sensitivity check at its own bar.

### 3. Document the failed gate and defer the threshold decision
- **Consequences.** No methodology change. The D4W1024 FAIL and the replication are recorded. Threshold 0.0 stays in
  force for development, and the D4W1024 grid and Exp2 forks stay blocked until the owner decides. Decisions wait
  for the two missing pieces of evidence (replication reading and positive-control L_trigger) instead of being made
  without either.
- **Validation required.** None now. The later decision needs the replication and the positive control's L_trigger.
- **Development run.** Not invalidated. The D4W1536 pilot can run at 0.0 (its size passed). It would need repeating
  only if option 2 is later adopted *for D4W1536*. Decide that scope before the pilot, since the pilot is meant to
  provide the positive control's natural trigger (`exp1_pilot_readiness.md` checklist).

## How the replication maps onto the options

| Seed-991 outcome | Option 1 (retain) | Option 2 (p95) | Option 3 (defer) |
|---|---|---|---|
| R3 and replication rate ≤ 5% | Best supported; document the original FAIL | Not justified (change after an unreproduced failure) | Acceptable, resolves to 1 |
| R3 but replication rate > 5% | Weak: two calibrations above 5% | Evaluate only on out-of-sample rate plus sensitivity | Preferred until sensitivity is known |
| R1 or R2 | Possible only with explicit acceptance of a known anti-conservative trigger | Candidate if (a)–(c) pass; does not remove the mechanism | Preferred until the positive control's L_trigger exists |
| Replication invalid (`null_followup.py` exit 2) | No new information | No new information | Default |

**Reading for the owner (not a recommendation to change anything):** option 3 is the only one valid under every
outcome now, because neither the replication nor the positive control's L_trigger exists. It also keeps the planned
D4W1536 dev run valid. Whether option 1 or 2 follows depends on those two results as tabulated above. Any change to
the trigger's *interval* itself (for example one that resamples initialisations) would be a new methodology question
outside these three options.
