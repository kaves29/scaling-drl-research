# Angle 2A

## Per-seed structure

ONE shared R_2x512 trajectory.

D_5x768 → own t*
D_7x1024 → own t*

Reference snapshots:
R@t*_5x768
R@t*_7x1024

Never average onset times.

## Independence

D_5x768 and D_7x1024 are independent.

The reference is shared intentionally.

Do not share:
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