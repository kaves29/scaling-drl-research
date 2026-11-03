# Official Experiment 1 reconciliation — in progress

Owner correction on 3 November 2026 supersedes the earlier pilot-population
assumption: the official population is **11 environments × 10 seeds × 3 critic
sizes = 330 parents**, with ten targets, 2,000 updates and 50 checks. The
four-environment/eight-seed dataset is pilot evidence, not the official population.

The owner clarified: use the legacy thirteen tasks without any HumanoidBench.
That removes h1-reach-v0 and h1-run-v0, leaving eleven, not twelve.
`experiments/exp12/official_exp1.py` freezes those identities and seeds 1–10.
The owner confirmed randomized sine targets, not a sign-function substitution.

A schedule conflict remains: legacy swimmer/hopper budgets are 500,000
raw steps = 250,000 interactions, so 50 checks put check 1 at the 5,000-step
fresh capture. The existing runner rejects that collision. An owner decision
on retaining those budgets versus matching the other tasks' 1,000,000 raw steps
has been requested. Neither budgets nor warmup/schedule were silently changed.
This population is not a qualified/executable launch manifest.

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
  transport. They retain the existing opt-in controls while qualification proceeds;
  default enablement must be source-bound in the final official profile.

Still required before claiming a complete official implementation: resolve the short-task budget/schedule decision, reconcile its versioned source/configuration/certifier/analysis paths,
verify Exp1's existing deferred fork behavior permits proceeding independently of
an unfrozen Experiment 2 decision, and run combined-source validation. Do not
change thresholds, targets, normalization, budgets or teacher definitions merely
because the pilot interpretation is difficult. No Experiment 2/3 proposal is
silently adopted. Do not launch this in-progress candidate.

Independent checks so far: 42 producer/reporting/reference-diagnostic CPU tests passed; four official-profile/population checks passed. These are not full combined certification or GPU qualification. Evidence is outside Git under `/workspace/artifacts/exp1-integration-20261103/`.
