# Fresh-pair null follow-up: results on the original Block B evidence (2026-10-09)

Procedure: [null_followup_procedure.md](null_followup_procedure.md). The analysis code is
`scripts/sci_investigation/null_followup.py`, unchanged from commit `5487f62`, run before any result was seen.
No approved setting, threshold, rule or acceptance criterion was changed. The p95 option is evaluated, not adopted.
No GPU job ran.

## Evidence
- **Archive** `blockB_22706349_null_evidence.tar.gz` (SHA-256 `5819d2be…16fb`): 13 regular files, relative paths only.
  All 13 match the supplied SHA-256 manifest.
- **Provenance:**
  - `commit.txt` = `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`;
  - four NVIDIA A100-SXM4-40GB GPUs;
  - jax 0.4.34 and numpy 1.26.4 (the same numpy as the analysis environment, so exact recomputation is meaningful);
  - the null step passed (exit 0, 6,042 s).
- **One uninterrupted run.** The step log shows three trainer builds, exactly 300 pair rows and 3 summaries, and no
  restart.
- **Analysis outputs** (kept outside git):
  - `null_followup_original.json`, SHA-256 `4c0c1c7d…9ecd`;
  - `null_exploratory_original.json`, SHA-256 `19bd2c0b…adf1`.

## 1. Integrity (pre-specified): all pass
For each of D2W512, D4W1024 and D4W1536:
- 100 rows, pairs 0–99, no duplicates, no missing or extra pairs, no malformed lines;
- arch, env (dog-run), seed (990), `check_index` = pair + 1 and `init_keys` all consistent;
- 0 invalid pairs;
- every stored loss, IQM, interval and `fired` flag recomputed exactly with the production functions;
- every summary field recomputed exactly.

**The reported 13/100 at D4W1024 is supported by valid original data.** Its exact 95% interval is
[7.1%, 21.2%].

## 2. Approved qualification criterion (unchanged)

| Size | Fire rate | Pre-specified NULL rule (≤ 5%, ≥ 100 pairs) |
|---|---|---|
| D2W512 | 3/100 | PASS |
| D4W1024 | 13/100 | **FAIL** |
| D4W1536 | 5/100 | PASS |

## 3. Test W (pre-specified, original data only, α = 0.01): not significant

| Size | Lower-bound fires (p vs no-difference floor 0.048) | Upper-bound fires | **W: permutation p** (between-pair variance of IQM) | ANOVA p (τ/σ_w) | Kruskal–Wallis p | Pairs with all 5 rounds the same sign (6.25 expected; p) |
|---|---|---|---|---|---|---|
| D2W512 | 3 (0.86) | 5 | **0.021** | 0.005 (0.31) | 0.052 | 8 (0.29) |
| D4W1024 | 13 (0.001) | 6 | **0.069** | 0.13 (0.19) | **0.0028** | 12 (0.022) |
| D4W1536 | 5 (0.53) | 7 | **0.153** | 0.15 (0.18) | 0.23 | 11 (0.048) |

**Pre-specified reading, all three sizes:** "no pair-level effect detected within this replay; cross-seed
persistence untested". This reading stands as specified.

**Why W had less power than planned.** W's power was estimated with normal round noise. The real round losses are
far heavier-tailed: excess kurtosis 4.4 (D2W512), 40 (D4W1024) and 63 (D4W1536), with right skew. A few extreme
rounds dominate the variance of IQMs across permutations, so the variance statistic loses power. This is a
limitation of the test as designed, recorded here; it does not change the reading above.

## 4. Post-hoc exploratory checks (not pre-specified; `null_exploratory.py`)
| Size | Observed fires (lower / upper) | Fire count after shuffling the 500 real round losses across pairs (mean; 95th percentile) | **Permutation p, lower fires** | Permutation p, two-sided | Sign-overdispersion p |
|---|---|---|---|---|---|
| D2W512 | 3 / 5 | 5.6; 9 | 0.95 | 0.81 | 0.18 |
| D4W1024 | **13 / 6** | 5.8; 9 | **0.003** | **0.006** | 0.052 |
| D4W1536 | 5 / 7 | 3.9; 7 | 0.37 | 0.29 | 0.15 |

1,000 permutations, seed 20261010, production `bootstrap_interval` and `triggered`.

- **The shuffle keeps the exact empirical distribution** (heavy tails, skew, overall location) and removes only pair
  structure. Under it, D4W1024 averages 5.8 fires, and 13 is reached with p ≈ 0.003. So the excess is not produced by
  the distribution of round losses. It needs rounds that co-vary within a pair, which is a pair-level effect
  conditional on this replay. The Kruskal–Wallis p = 0.0028 and the same-sign excess (12 against 6.25) point the
  same way.
- **Pooled D4W1024 losses show no overall shift:** 52.2% of rounds are positive (sign test p = 0.35), and the median
  is 1.6e-5.
- **The fired pairs** are consistently positive with right-skewed magnitudes (1e-5 to 8e-3), for example pair 81:
  [0.0084, 0.0006, 0.0002, 0.00002, 0.0070].
- **Lower against upper fires** (13 against 6) is consistent with a symmetric effect (two-sided p = 0.17).
- **Controls:** D2W512 and D4W1536 show no fire excess under the same shuffle. Whether D4W1024 really differs from
  them is uncertain: 13 against 5, Fisher p = 0.08; 13 against 3, p = 0.016.
- **Caveat:** these are post-hoc tests chosen after seeing that W was not significant. They are evidence to guide
  the next experiment, not confirmatory findings.

## 5. The pre-specified p95 option (diagnostic only, not adopted)

| Size | t = p95 of pair-level L | t / SD of pair L | In-sample fire rate | Two-fold cross-validated (even→odd, odd→even) |
|---|---|---|---|---|
| D2W512 | 0.00701 | 1.62 | 0/100 | 0/50, 0/50 |
| D4W1024 | 0.001096 | 1.48 | 0/100 | 0/50, 0/50 |
| D4W1536 | 0.00151 | 1.96 | 0/100 | 0/50, 0/50 |

- **It removes null fires in these data,** in-sample and cross-validated alike. Every interval's lower bound sits
  below about 1.5 SD of pair-level L.
- **Out-of-sample** behaviour needs fresh pairs; the seed-991 replication supplies them.
- **Cost in sensitivity cannot be assessed yet.** A real plasticity loss would have to clear a lower bound of about
  1.1e-3 (D4W1024) and 1.5e-3 (D4W1536). Whether that is small or large needs the positive control's L_trigger.

## 6. Can the original data separate a persistent initialisation effect from a replay-dependent one?
**No.** In the original calibration every pair shares one warm-up replay (seed 990), so the initialisation pair and
the replay are completely confounded. Section 4 indicates (post hoc) a pair-level effect conditional on that replay.
Only holding the initialisations fixed while changing the replay can show whether that effect belongs to the
initialisation pairs themselves. That is exactly the seed-991 replication. Because the pre-specified W was not
significant, the replication is also needed to confirm a pair-level effect at all, on new data, with the
pre-specified tests.

## 7. Recommended next experiment (not submitted; needs approval)
**Seed-991 replication, D4W1024, one A100, about 35–40 min.**
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard --archs D4W1024 \
    --null_pairs 100 --seed 991 --out_dir /abs/new/null_replicate_seed991
```
- **Optional size control:** add `D4W1536` to `--archs` (about 1 h more) to test whether the effect is specific to
  D4W1024.

### Readings pre-specified now, before any replication data exists
- **Primary: test X, unchanged.** Spearman ρ of `loss_iqm` across the 100 matched pairs, one-sided permutation
  p < 0.01. It is rank-based on trimmed means, so heavy tails do not affect it.
- **Added before seeing the replication: the fire-count shuffle test of section 4, applied to the replication's own
  rows** (1,000 permutations, seed 20261010, α = 0.01). It tests whether a pair-level effect exists on the new
  replay. W is still reported, with the heavy-tail caveat.
- **Readings** (in addition to `null_followup.py`'s):
  - **Intrinsic initialisation effect:** X significant.
  - **Replay-dependent pair effect:** the shuffle test is significant on the replication but X is not.
  - **No reproducible pair effect:** neither is significant. Then the original 13/100 is a pair-level excess that
    did not reproduce (chance or replay-specific), and the replication's fire rate is reported against the
    unchanged 5% rule.
- **Approved criterion unchanged.** The original D4W1024 FAIL stands whatever the replication shows. Two
  calibrations, and any threshold such as the p95 option, are the lead's decision.

## Reproduction
```bash
cd main
E=/abs/evidence; mkdir -p $E && tar -xzf blockB_22706349_null_evidence.tar.gz -C $E
(cd $E && sha256sum -c /abs/blockB_22706349_null_evidence.sha256)
JAX_PLATFORMS=cpu python -I scripts/sci_investigation/null_followup.py \
    --original $E/logs/blockB_22706349/gpu3/null_dog_run --out /abs/null_followup_original.json       # exit 0, a few minutes
JAX_PLATFORMS=cpu python -I scripts/sci_investigation/null_exploratory.py \
    --original $E/logs/blockB_22706349/gpu3/null_dog_run --permutations 1000 --out /abs/null_exploratory_original.json  # ~14 min
```
