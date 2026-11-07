# Null-trigger decision memo

Status: owner decision required; threshold remains0 and approved gate remains
at most5 firings among100 valid fresh pairs. No exploratory threshold is adopted.
Authority: `.claude/methodology-exp1-exp2.md` and `exp12_decisions.md`; analysis
framework: `null_followup_procedure.md`, `null_followup_original_results.md` and
`scientific_validation_investigation.md`. Count-level independent reference:
`scripts/exp12_null_decision_reference.py`.

## Established versus reported

The original Block B report and repository investigations report D4W1024
13/100. The owner reports seed991 job22779974 at9d6a82d:100 valid pairs,
6 firings, exploratory p95=.0005196526646614074. There is no Delta access and
the new raw files are not uploaded here. The reported completion, validity,
pairing, source revision and bootstrap recomputation have NOT been independently
verified. Both reported counts FAIL the unchanged empirical≤5/100 gate.

Independent count calculations (exact Clopper–Pearson95% intervals):

| Sample | Rate | Interval | One-sided binomial p under rate=.05 |
|---|---|---|---|
| Original | .13 | [.07107,.21204] | .001464 |
| Seed991 | .06 | [.02233,.12603] | .384001 |

The original sample contradicts a .05 independent-binomial model; the second
alone does not distinguish .05 from modest excess. This inferential statement
does NOT override the prespecified empirical pass/fail rule. The two-sided
Fisher independence illustration gives p=.146385, not a valid paired-design
decision rule. Seed991 was designed to reuse initialization keys on new replay;
its actual pairing must be verified from raw rows. A valid paired comparison
needs discordances, which counts13 and6 alone do not supply. Non-significance
under an independence illustration does not establish equivalence. Do
not pool the two revisions/replay conditions as if scientific exchangeability
were demonstrated. These p-values are descriptive investigation calculations,
not replacement scientific acceptance criteria.

Production forks require TWO consecutive eligible checks among checks1–19.
For independent Bernoulli checks only, probability of any adjacent firing pair
is .04220 at p=.05, .05976 at p=.06 and .24159 at p=.13. This is an exact
recursion illustrating why neither p nor p² alone is the campaign fork rate.
Actual repeated checks reuse the same current/fresh initialization and evolving
replay, creating dependence. Independent fresh-pair counts do not identify this
longitudinal probability. The existing consecutive trigger must not be changed.

A two-sided95% interval with a strict positive lower endpoint has a nominal
one-sided2.5% rate only under a valid zero-effect coverage model. The approved
empirical maximum is5%, not a promise of that nominal rate. No new2.5% gate is
introduced. Distinct fresh initializations can have genuinely different finite
fitting residuals conditional on the pair; the operational fresh-null failure
need not be a type-I error for every conditional zero-effect hypothesis.

## Cause assessment

No implementation root cause is established by the counts. Approved paired
rounds, five-round IQM,10,000 percentile resamples,95% intervals, strict positive
lower bound and FP32 loss definitions remain unchanged. Five empirical rounds
provide limited bootstrap support; skewed fitting losses, initialization-pair
effects and round/replay dependence could yield conditional firing beyond a
nominal confidence interpretation. Independent CPU calculations/tests of IQM,
paired arithmetic and trigger implementation cannot establish CUDA coverage.

The current IQM uses the existing quarter-trimmed mean; with five rounds it
discards one observation from either end and averages the middle three. Its
bootstrap resamples paired losses rather than independently resampling each
critic. This establishes source-level procedure, not empirical coverage. There
is no proposal here to replace it with fractional trimming or unpaired resampling.

Source inspection at9d6a82d and844e3c0 confirms null-row resumption already checks
resolved configuration, code/dirty state, runtime, seed/environment/architecture,
unique pair indices and regenerated bootstrap firings. This reduces a known
stale-artifact risk; it does not prove the reported files passed those checks.
FP32 paired subtraction precedes NumPy float64 statistical aggregation. The
single-Q dog-run null is not explained by the separately corrected HB twin-loss
arithmetic merely because that correction exists. TF32-sensitive fitting/teacher
targets and near-zero lower-bound rounding remain hypotheses needing paired
device evidence; no precision flag is changed.

Original-data investigation reports no overall round-level shift and confounded
initialization/replay effects: all original pairs share one replay. Seed991
changes replay, but count6 alone cannot supply the prespecified matched-pair
Spearman/permutation test or shuffle result. Precision, fitting convergence,
sampling errors and stale files cannot be ruled in or out without raw keys,
rounds, config/source and collection hashes. There is no justification to call
the6-count result proof that the measurement is calibrated.

Smallest missing evidence: original and seed991
`null_pairs_D4W1024.jsonl`, corresponding summary JSON, launch/config/backend
metadata, stdout/stderr and checksums. Each row must contain pair/check/index,
initialization keys, five scores/b/L, validity, bootstrap settings and CI/fired.
Recompute CIs and firings; check distinct keys/ordering, no duplicate or stale
rows, raw pair correspondence across replays, and run the existing prespecified
X and shuffle analyses. Retrieve files from
`/work/hdd/biqc/skaveti1/exp12_null991/main/logs/null_replicate_991_22779974/null_dog_run/`;
no new GPU job is necessary for this analysis. Raw source evidence for numerical
and longitudinal hypotheses may require a separately approved paired-repeat
or consecutive-check experiment; do not infer it from counts.

## Exact owner decision

1. **Retain0 and disposition the failed qualification:** retain the unchanged
   methodology; explicitly decide whether the development pilot may use its
   natural zero-threshold trigger. This does not make the grid gate pass.
2. **Authorize a prespecified independent calibration/validation protocol:**
   choose calibration population/replay/architectures, held-out validation,
   uncertainty and per-check versus consecutive-fork estimand before new results.
   A calibrated effect threshold is a scientific amendment, not an engineering
   fix. Freeze it before held-out validation; never tune on validation firings.
3. **Revise the measurement/statistical procedure:** e.g. rounds, bootstrap or
   hierarchical calibration. These change scientific methodology and require
   approval, independent null qualification and a sensitivity positive control.
4. **Leave the failed gate blocked while acquiring existing raw evidence.** This
   is the current state; it need not block CPU engineering or validation prep.

The exploratory p95 value has no authority. Its use on the same100 pairs is
in-sample threshold selection; even cross-validation is exploratory without a
prespecified design and sufficiently independent data. Any nonzero threshold
can suppress true degradation whose lower CI lies below it. Require the actual
healthy/trigger losses, probe noise and candidate-m positive-control recoveries
to assess sensitivity; none is inferred here. The current0 threshold is the
approved implementation, but its statistical/scientific qualification is still
failed. Neither an agent nor a process exit may dispose that failure.
