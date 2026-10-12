# PROPOSED amendment: fresh-critic range validity and identity staging (2026-10-12)

> **STATUS: PROPOSED, NOT APPROVED.** Nothing here is in force. It changes no threshold, probe setting, trigger,
> training configuration or code. If you approve it, append it to the methodology as a new, dated entry *before*
> any confirmatory parent starts, and before any L from a confirmatory run is seen.
> Rationale: [exp12_gate_decision_memo.md](exp12_gate_decision_memo.md).

## A1. Per-cell fresh-critic range validity, from each run's own check 0 (decision D2, option H1)
**Problem:**
- (w) requires P/b ≥ 0.9 "at every critic size" but gives no next step on failure.
- It has been measured in only 2 of the 13 tasks.
- hopper-hop D4W1536 fails at 0.6024.

**Proposed text:**
1. **Measurement.** For each Experiment 1 parent, the range quantity is the check-0 probe already run on the
   untrained critic:
   - IQM over 5 rounds of P, divided by the IQM of b;
   - the configured pool (25,600), with no change to any setting;
   - on HumanoidBench, computed per Q network, and the cell must meet it for both.
2. **Cell rule.** A cell (task × critic size) is *range-valid* if the median over its seeds of the check-0 ratio is
   ≥ 0.9. This is (w)'s threshold, unchanged.
3. **Timing.** Every cell is classified, and the classification recorded, **before** any post-check-0 L of that
   cell is examined. Check 0 precedes all training, so the rule cannot depend on the outcome.
4. **Analysis.** The pre-registered primary Exp1 analysis (final-check L contrast, stratified bootstrap) is
   **unchanged**. A pre-registered sensitivity analysis repeats it with range-invalid strata removed from *both*
   architectures in each scaled-vs-D2W512 contrast. Exp2 reports range-invalid cells' forks separately as a
   sensitivity subset; they stay in the primary analysis [unchanged inclusion rule].
   - *Alternative for the owner:* make the range-valid analysis primary and the full analysis secondary. Choose one
     before launch.
5. **Reporting.** For every cell, report the per-seed check-0 ratios and their loss curves (already saved as
   `check_00.npz` and `probe_curves.npz`).

**Why this is sound:**
- It applies the approved criterion uniformly instead of only where it happened to be checked.
- It uses only pre-treatment data.
- It keeps the registered primary analysis intact.
- It needs no extra GPU time.

**What it does not do:**
- It does not make hopper D4W1536 "pass".
- It does not alter (w).
- It does not identify the direction of any bias in L.

## A2. Identity qualification staged by suite (optional, decision D5)
**Problem:** (b) requires identity "before any Experiment 1 grid run". (p) defines the cases as D4W1024 and D4W1536
in each suite, on the grid's GPU model.

**Proposed text:**
1. D2W512 parents never fork. They may start once the positive control has passed; identity does not apply to them.
2. The forking parents of suite S (D4W1024 and D4W1536) may start once **both** of S's identity cases pass on the
   grid's GPU model.
3. An Exp2 arm of suite S runs only after both of S's cases pass. A failing case blocks only its own suite's forking
   parents and arms.

**Unchanged:** the identity test, its bitwise criterion, the GPU-model requirement and the positive-control
ordering.

## Not proposed
- **No amendment is needed for the D4W1024 null gate.** Both options (adopt the 95th-percentile threshold, or keep
  0) are inside the pre-registered NULL rule.
- **One clarification is needed if the threshold is adopted:** which pairs define the 95th percentile.
  - Recommended: all 200 existing pairs, pooled, using the script's `np.percentile(L, 95)`.
  - Record it in writing before the value is computed.
- **No change** to:
  - the probe budget, pool or optimizer;
  - the trigger's consecutive-check or eligibility rules;
  - the positive-control ordering.
