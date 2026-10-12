# Pilot 22800749: why no natural trigger (read-only analysis plan, 2026-10-12)

**Owner-reported inspector result** (`scripts/exp12_pilot_inspect.py`, commit `0084c20`):
- verdict `NO_NATURAL_TRIGGER`; fired checks `[]`; recomputed f*_run `null`;
- horizon complete; source certificate `COMPATIBLE`.

This environment has no Delta access, so the pilot's per-check numbers are **not** in this document. They come from the command below.

## Command (Delta login node, read-only, lightweight)
**Requirements:** Python 3.10+ and numpy only. The plot is self-contained SVG/HTML, so neither matplotlib nor SciPy is needed. The inputs are the inspector report and the pilot's `probes/` files.

```bash
cd "$FINAL_CHECKOUT/main"          # checkout of this branch's commit
D=/work/hdd/biqc/skaveti1/exp12_decisions
~/exp12_inspect_env/bin/python scripts/exp12_trigger_trajectory.py \
  --report "$D/pilot_inspection.json" --out "$D/pilot_trajectory"
```

**Outputs** (all outside the pilot root; the script refuses paths inside it):
- `trajectory.csv` contains one row per check 0–20:
  - step and budget fraction;
  - IQM(L) and the 95% bootstrap interval with its lower bound;
  - the gap from the lower bound to the firing boundary;
  - whether the check fires under the registered rule;
  - the descriptive near-boundary flag;
  - per-round positivity and spread;
  - fresh and current P with their IQM, b, and fresh/current P/b;
  - fresh and current fit-curve "still improving at budget end";
  - quality flags.
- `summary.json` holds the pattern class, counts, the maximum lower bound and its check, the medians, the trend, missing data and per-check flags.
- `trajectory.html` contains the IQM(L) and interval chart with the dashed boundary (lower bound > 0) and the non-eligible check 20 shaded, a fresh vs current P/b chart, and the full table.

**Firing.** Firing is evaluated only with the registered rule from the run's own config: lower bound > `trigger.null_threshold` (0), two consecutive checks, eligible checks 1–19, IQM over 5 rounds, 10,000-resample percentile bootstrap. The bounds are those the inspector recomputed and checked against the saved records.

**Descriptive definitions.** These were fixed before any pilot number was seen and are not trigger criteria.
- **Near boundary:** the lower bound is ≤ 0 and within 25% of the interval width of 0.
- **Pattern classes**, in precedence order:

| Pattern | Rule (eligible checks) | What it means for "why no trigger" |
|---|---|---|
| `NEGATIVE` | ≥ 75% of IQM(L) < 0 and ≥ 50% of intervals entirely < 0 | The trained critic fits new targets *better* than its fresh reference. The probe finds no plasticity loss in this cell; the absence of a trigger reflects the measurement, not an underpowered test. |
| `POSITIVE_UNDERPOWERED` | ≥ 50% of IQM(L) > 0, none fires, ≥ 1 near-boundary check | There is some loss signal, but the 5-round interval is too wide to clear 0 twice in a row. |
| `NEAR_ZERO_UNCERTAIN` | ≥ 75% of intervals contain 0 | Estimates sit at zero within noise. |
| `UNDETERMINED` | none of the above, or any missing or invalid data | The cause is not established. |

**Quality flags to read alongside the class:**
- the round spread exceeds |IQM|;
- a non-finite round;
- fresh P/b < 0.9 (the range rule (w), descriptive);
- fresh or current fit still improving over the last 10% of the 1,000 steps (the finite-budget effect seen on hopper).

Under `NEGATIVE`, a fresh-fit flag matters: an under-fitted fresh reference pushes L down.

## What the stop-and-consult clause permits
Methodology (e): "D4W1536 … on dog-run …, with a development seed outside 1–5. **If it never triggers, stop and consult. The trigger is never loosened.**" The positive control must come "before the main runs of Experiment 1", and development runs are "excluded from the confirmatory … analyses".

| Action | Status |
|---|---|
| Stop: no positive control, no m, no Exp1 parent (D2W512 under A2 included) | **Required now** |
| Consult the project lead with this trajectory, `summary.json` and the inspection report | **Required; this is the prescribed next step** |
| Read-only analysis of the saved pilot artifacts (this document); report the pilot descriptively as excluded development evidence | **Permitted** |
| Run qualification that does not depend on the positive control (identity cases, the Delta preflight, numeric-bound measurement), so these are ready if the consultation unblocks | **Permitted.** It starts no Exp1 run, but spends GPU time before the outcome is known. |
| Another *natural* development run: same D4W1536, dog-run, settings and trigger, at a different development seed outside 1–5 | **Lead's decision under the consult.** It changes no criterion, but (y) refers to "the same development seed", so the lead should record the reading of (e)/(y). To avoid seed-shopping, it should also fix in advance the number of seeds allowed, and what happens if none triggers. |
| Change the threshold, consecutive count, confidence, rounds or eligibility window | **Prohibited** ("never loosened") |
| Force a trigger or fork for the positive control | **Prohibited** (forced evidence cannot qualify; the code refuses it) |
| Artificial damage to create a degraded critic | **Prohibited** (methodology: "No separate artificial damage procedure") |
| Positive control on another architecture or environment, a longer development budget, a different probe budget or pool, or a positive control taken from confirmatory grid forks | **Amendment required** (changes (e), the probe definition, or the development/confirmatory separation) |
| Start Exp1 before a positive control exists, or drop it | **Amendment required** |

The scientific weight differs by class:
- `NEGATIVE` questions whether the degradation the study assumes occurs in this cell at all. That is a design question for the lead, not a power question. Repeating the run at another seed is less likely to help.
- `POSITIVE_UNDERPOWERED` or `NEAR_ZERO_UNCERTAIN` is where another natural development seed is most informative.

Minor note: a planning entry in `docs/exp12_decisions.md` says "dev seeds 1001+", while the pilot used seed 102. It still satisfies (e), which only requires a seed outside 1–5.
