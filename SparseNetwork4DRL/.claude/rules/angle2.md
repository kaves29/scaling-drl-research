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