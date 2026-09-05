# Angle 1

## Purpose
Angle 1 measures critic degradation and pathology propagation
under critic architecture scaling.

## Architectures
Baseline: D2W512
Scaled: D5W768, D7W1024

Actor remains fixed at D1W128.

## Seeds
5 seeds per architecture/environment.

## Tracking
critic_degradation=true enables degradation analysis.

pathology_prop=true enables propagation analysis.

## Baseline
Default SimBa D2W512 is the baseline.

Exactly 5 baseline seeds are required.

Baseline/scaled logging grids must align.

Self-baselines are forbidden.

## Analysis
Onset analysis is post-hoc.

Metrics are persisted before onset detection.

CSV onset ledger is canonical.
WandB is secondary.

## Windows
Read current definitions from configs/base_sac.yaml.
Do not resurrect obsolete configuration names.
Do not replace percentage-based definitions with fixed values.