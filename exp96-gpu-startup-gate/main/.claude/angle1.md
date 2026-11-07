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
N (sustain window) and W (propagation/lag window) are calibrated
per-environment from the 5-seed baseline data via ACF/CCF - see
analysis/window_calibration.py and research-methodology.md's Onset
Definitions (2026-09-05 procedure). They are no longer config values at all.

UPDATE (2026-09-05): this note previously said "read current definitions
from configs/base_sac.yaml" and "do not replace percentage-based
definitions with fixed values." Both are now stale and superseded: the
percentage-based sustain_window_fraction/propagation_window_fraction fields
never actually calibrated anything against baseline data (they were fixed
fractions of run length, despite research-methodology.md separately, and
correctly, requiring real calibration) and have been removed from
configs/base_sac.yaml entirely, replaced by the genuine ACF/CCF calibration
above. Do not resurrect the removed sustain_window_fraction /
propagation_window_fraction / sustain_window / propagation_window
config field names - they no longer exist and nothing reads them.