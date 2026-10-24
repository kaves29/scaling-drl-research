# Unapplied audit proposals

None of these patches is adopted. Regression tests execute transformed copies
and assert the old failure and candidate behavior. Owner approval is required
before adopting scientific arithmetic changes. No archive backfill is proposed.

| Patch | Bound original | Consequence |
| --- | --- | --- |
| contemporary_score_logging.patch | campaign f597ee9 RunProbes.record_check | Adds current marginal IQM log fields. Does not change score, fit, CI or trigger; historical summaries remain untouched. |
| late_effective_n.patch | Uploaded critic_metric_comparison/analyze.py; full SHA in adjacent excerpt JSON | Adds per-window finite seed counts; retains nominal count and all estimators/CIs. Targets the supplied analysis package, not a repository production file. |
| coupled_seed_grouping.patch | PR11 production f8ad82f pipeline.py | Removes verified coupled seed fields from grouping key, validates coupling, preserves other settings. Changes future aggregation membership/uncertainty. Does not fix native metadata mapping or unknown dirty provenance. |
| extreme_offset_centered_residual.patch | campaign f597ee9 probe.py | Evaluates algebraically centered residuals to avoid materialized-target rounding at extreme offsets. Changes float arithmetic and potentially gradients/measurements; campaign impact unproven, needs owner decision and GPU qualification. |

The source-specific transformations fail if their exact anchors change.
`git apply --check` checks repository-target proposals without applying them;
the analysis-package patch is checked against a temporary copy of the uploaded
file. Never apply these to frozen historical source or overwrite original data.
The numerical candidate is evidence about an edge case, not a stabilizer
recommended to remove an inconvenient result.
