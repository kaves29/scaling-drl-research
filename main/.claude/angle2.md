# Angle 2A

## Per-seed structure

Each scaled critic gets its own separately-initialized, independently-trained
reference agent (R_2x512), trained to its own t*:

D_5x768 → own t* → own R@t*_5x768
D_7x1024 → own t* → own R@t*_7x1024

The two reference agents share no state with each other or with either
scaled critic. This matches research-methodology.md's Angle 2A section
verbatim ("Each scaled critic gets its own independently-trained reference
critic, matched to its own t* - never shared across two scaled
architectures, never averaged across seeds or architectures").

Never average onset times.

CORRECTION (2026-08-28 audit): this file previously specified "ONE shared
R_2x512 trajectory... The reference is shared intentionally," directly
contradicting research-methodology.md and its "Must Never Change Silently"
list. The implementation matched THIS file, not research-methodology.md,
until the 2026-08-28 audit found and fixed it - see
experiments/angle_2_a.py and the End-of-Task Summary for that date.

## Independence

D_5x768 and D_7x1024 are independent.

Their reference agents are independent too - never shared, never the same
trained trajectory snapshotted twice.

Do not share:
- reference agents (across scaled architectures)
- scaled replay buffers
- scaled actors
- scaled critics
- optimizer state
- RNG state

## Probes

10 D-source
10 R-source

20 per matchup.

## MC

15 rollouts/probe.

Force sampled action.

Continue with source agent's actor only.

## Q

Record:
Q_D
Q_R
MC return

Primary diagonal:
E_D = |Q_D - MC_D|
E_R = |Q_R - MC_R|

Off-diagonal Q values are secondary analysis only.

## Onset

Use the exact Angle 1 ledger entry matching:
architecture + seed + environment.

Missing onset = fail/manual review.
Never invent an onset.

## Architecture

No hidden architecture defaults.
CLI/config overrides required.

---

# Angle 2B

## Reuse — no new training

Zero additional training or environment interaction anywhere in 2B.

Reuses, directly from Angle 2A:
- frozen degraded agent (Q_D, π_D) checkpoint @ t*
- frozen reference agent (Q_R, π_R) checkpoint @ t*
- D's own probe-capture buffer
- R's own probe-capture buffer

Never retrain. Never resample from the environment.

## Primary analysis

Actor π_D held fixed. Critic swapped only.

g_{D|D} = ∇_θJ(π_D; Q_D)   ← real signal
g_{D|R} = ∇_θJ(π_D; Q_R)   ← counterfactual

Same actor params, same state batch, same entropy term in both.
Only the critic's Q-term changes.

Probes: exclusively from π_D's own buffer.

Never source probes for this analysis from R's buffer — evaluates π_D on
states it never visited, reintroducing an OOD confound.

## Secondary analysis

Same procedure, roles reversed:

g_{R|R} = ∇_θJ(π_R; Q_R)
g_{R|D} = ∇_θJ(π_R; Q_D)

Probes: exclusively from π_R's own buffer.

Diagnostic robustness check only — tests whether distortion is specific to
π_D or a property of the critic detectable regardless of actor.

Report primary and secondary side by side, explicitly labeled.
Never average into one aggregate statistic.

## Distortion metrics

Computed independently on primary pair and secondary pair:

D_dir  = 1 - cos(g_1, g_2)
D_mag  = log(‖g_1‖ / ‖g_2‖)
D_grad = ‖g_1 - g_2‖

## Null

Must mirror primary analysis's exact structure. Not a raw healthy-vs-healthy
comparison.

Two independently-trained default-architecture agents, A and B (different
seeds; neither critic trained on the other's actor's data).

π_A held fixed. Probes from A's own buffer only.

g_{A|A} = ∇_θJ(π_A; Q_A)   ← real
g_{A|B} = ∇_θJ(π_A; Q_B)   ← foreign healthy critic

Same three metrics, same procedure. Isolates ordinary foreign-critic
unfamiliarity — zero pathology involved on either side.

Multiple independent A/B pairs required (reuse Angle 1 baseline seeds where
possible) — build a distribution, not a point estimate.

Threshold rule: (mean + 2σ), not Angle 1's 95th percentile.

CORRECTION (2026-08-28 audit): this is intentionally a DIFFERENT rule from
Angle 1's onset/propagation threshold. Angle 2B's null has very few points
(≤5, one per baseline seed) — a percentile is unstable/poorly-defined at
that sample size. Angle 1's baseline curves pool many logged timesteps
across all 5 seeds, where a percentile is well-defined. Do not make these
two rules match each other. This divergence has been reviewed and kept
deliberately — see research-methodology.md's Onset Definitions section.

Limitation to state explicitly: assumes foreign-critic-unfamiliarity
magnitude is comparable between a healthy-healthy pairing and the real
degraded-reference pairing. Standard assumption, not verified equivalence.

## Scope boundary

Frozen, single-step measurement only.

Never extend into continued training, forked branches, or multi-step
rollout — this belongs to Angle 3 only, and doing it here reintroduces the
actor/critic co-adaptation confound the frozen design exists to avoid.

Never phrase a 2B result as a downstream performance/outcome claim.
2B shows distortion reaches the gradient. It does not show that distortion
causes worse policy behavior over time.

---

# Angle 2C

## Reuse — no new sampling

Operates on the same g_D/g_R pairs and underlying ∇_aQ values already
computed in 2B.

Never resample (s,a) pairs for this angle.

## Candidate properties

Three, each measured against its own null (same A/B construction as 2B):

Directional corruption:
cos_sim(∇_aQ_D(s,a), ∇_aQ_R(s,a))
Meaningfully below null → degraded critic points actor toward a
substantially different action, not just differently-confident.

Magnitude/bias shift:
‖∇_aQ_D‖ / ‖∇_aQ_R‖
Q_D(s,a) - Q_R(s,a)
Ratio off from 1 + high directional cosine → correct direction,
miscalibrated scale. Consistent with Q-overestimation.

Local instability:
Perturb a by small random offsets δ.
Compute ∇_aQ at each perturbed point, both critics.
Compare variance of perturbed gradients between critics.
Each measured relative to its own null — even a healthy critic isn't
perfectly smooth.

## Co-occurrence

If multiple properties diverge from null simultaneously:
do not assume equal causal contribution.

Reconstruction test:
- build synthetic gradient = one critic's direction + other critic's magnitude
- run through same frozen-actor procedure
- check which reconstruction better reproduces the real observed gradient

Whichever reconstruction wins → that property is doing more causal work.

## Closing the loop

Dominant property's onset timing (when it departs from null across the
degradation window) must be checked against Angle 1's real actor-side
metric onset timing.

Consistency between the two = what makes this a mechanism explanation,
not just a taxonomy.

## Non-result

None of the three separating cleanly from null is a legitimate, reportable
outcome.

Never force a narrative onto data that doesn't support one. Report as:
corruption not well explained by these three interpretable properties.