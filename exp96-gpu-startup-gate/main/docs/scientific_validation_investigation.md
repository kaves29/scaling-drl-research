# Scientific qualification blockers: fresh-null, hopper range, positive control, m

Branch `claude/scientific-validation-investigation`, based exactly on `9d6a82d4aef81299901177865e383640439cde2a`
(`codex/exp12-metric-integration`). CPU only; no Delta access, job, push or merge. No configuration, threshold,
statistical rule, optimizer, probe budget or acceptance criterion was changed. Every experiment below calls the
production probe code (`experiments/exp12/probe.py`, `trigger.py`) unchanged on the real seed-990 warm-up replays.

Labels: **CONFIRMED** = reproduced here by running it; **SUPPORTED** = consistent with every measurement here but
not proven for the GPU artefact; **OPEN** = needs data not available here.

## Summary

| Blocker | Finding | Status |
|---|---|---|
| Hopper D4W1536 range, P/b 0.602 < 0.9 | Reproduces in full FP32 on CPU (P/b 0.540 and 0.730 on rounds 0 and 1). Cause: AdamW's first fresh-state steps move every parameter by exactly lr, which for the D4 critics shifts the output by several units (pool MSE 26 to 106 after steps 1–2 against b ≈ 0.5). The fit then sits on a predict-the-mean plateau (loss ≈ b) until about step 630 on hopper (about 350 on dog-run) and the 1,000-step budget ends while the loss is still falling. Not TF32, not a fitter defect. | CONFIRMED (mechanism), CONFIRMED (not GPU-specific) |
| Fresh-pair null, D4W1024 13/100 > 5% | The approved per-check rule fires about 4.8% of the time even with no pair difference at all (exact enumeration). So the 5% gate sits on the rule's own floor and fails by chance in 35% to 46% of 100-pair calibrations per size (73% to 84% for at least one of three sizes). 13/100 is still well above that floor (p ≈ 0.001 to 0.003). The pairing, RNG streams, sign, IQM and bootstrap are correct. Independent, zero-centred round noise cannot reach 13% for any tail shape tested (4.4–5.0%). That leaves a pair-level effect (two particular initialisations that differ in probe-fitting ability on every round, which the rounds-only bootstrap correctly detects) or invalid rows. A six-initialisation CPU measurement at D4W1024 did not detect a pair effect; that test has low power. | Rule floor CONFIRMED; implementation CONFIRMED correct; independent zero-centred rounds EXCLUDED for any tail shape; cause narrowed to a pair-level effect or invalid rows, OPEN |
| Positive control / m | No implementation defect. `select_m` applies the approved arithmetic literally: it returns m (exit 0) when every recovery is negative, and it prefers an overshoot (recovery 2.0) to an exact return to healthy (1.0). Both are gaps in the approved rule, not code errors. | CONFIRMED (behaviour); decision needed |

## 1. Hopper D4W1536 fresh-critic range

### What was run
`scripts/sci_investigation/range_round.py` reproduces one round of `probe_fresh_checks.range_mode`: the trainer's own
seed-990 critic, `probe_rng(990, 0, r)` pool of 25,600, `probe_key(990, 0, r)` teacher and minibatches, AdamW
1e-4 / 1e-2, 1,000 steps, batch 256, target scale 1e5, chunk 2,560. The replay is the real warm-up buffer (identical
for every architecture; checked).

| Case (CPU FP32) | P/b | Final pool MSE | Block B (A100, TF32) |
|---|---|---|---|
| hopper D2W512 r0 | 0.928 | 0.036 | IQM 0.933 |
| hopper D4W1024 r0 | 0.910 | 0.044 | IQM 0.902 |
| hopper D4W1536 r0 | **0.540** | 0.232 | IQM **0.602** (rounds 0.559–0.649) |
| hopper D4W1536 r1 | **0.730** | 0.138 | |
| dog-run D4W1536 r0 | 0.993 | 0.003 | (not measured in Block B; D6W1536 0.993 in Block A) |

The failure reproduces without TF32 or a GPU, so the A100 numerics are not its cause. The per-round spread
(0.54 to 0.73 here, 0.56 to 0.65 on the A100) is large because the outcome depends on when the fit escapes the
plateau described below.

### Mechanism
Mean minibatch loss in 50-step windows (round 0; b ≈ 0.50):

| Steps | 0–50 | 100–150 | 300–350 | 500–550 | 700–750 | 950–1000 |
|---|---|---|---|---|---|---|
| hopper D2W512 | 0.61 | 0.38 | 0.21 | 0.09 | 0.06 | 0.05 |
| hopper D4W1536 | 5.58 | 0.49 | 0.46 | 0.44 | 0.34 | 0.20 |
| dog-run D4W1536 | 10.47 | 0.48 | 0.42 | 0.17 | 0.04 | 0.004 |

First AdamW steps of the approved fit (`first_steps.py`; pool MSE on 2,560 points after each step):

| Case | AdamW step-1 update norm | lr·√N | MSE after steps 1 / 2 / 3 | Output mean shift after step 1 |
|---|---|---|---|---|
| hopper D2W512 | 0.205 | 0.205 | 0.82 / 1.51 / 0.98 | −0.13 |
| hopper D4W1024 | 0.579 | 0.580 | 19.2 / 32.3 / 10.3 | +3.56 |
| hopper D4W1536 | 0.869 | 0.869 | 26.3 / 106.1 / 7.7 | −2.35 |
| dog-run D2W512 | 0.208 | 0.208 | 2.1 / 1.1 / 1.6 | +1.13 |
| dog-run D4W1024 | 0.582 | 0.582 | 21.5 / 13.4 / 13.2 | +4.51 |
| dog-run D4W1536 | 0.871 | 0.872 | 229 / 5.6 / 83.7 | −15.1 |

- **CONFIRMED.** With fresh optimizer state, AdamW's first update is lr·sign(g) for every parameter: the update norm
  equals lr·√N to three digits at every size. The change in the critic's output grows with network size, so the
  D4 critics' predictions jump by several units (mostly a shift of the mean, oscillating in sign over the first
  steps) on both tasks. D2W512 does not blow up.
- **CONFIRMED.** After recovering, D4W1536 sits at loss ≈ b, predicting roughly the mean, for hundreds of steps
  before it starts fitting the high-frequency target. On dog-run it leaves the plateau at about step 350 and
  finishes at MSE 0.003. On hopper it leaves at about step 630 and the budget ends at 0.20, still falling.
- **SUPPORTED (why hopper).** Hopper's probe inputs are 19-dimensional; dog-run's are 261-dimensional. In normalised
  units the median nearest-neighbour distance in the round-0 pool is 1.74 on hopper against 12.2 on dog-run (4,975
  distinct rows each). Memorising effectively random labels (sin(1e5·f) varies over far shorter distances than
  these) on crowded low-dimensional inputs is slower, so the same plateau leaves too little of the budget.
- No fitter defect: the loss trajectories are smooth after the transient. The repository's analytic-fitter reference
  (`plasticity_validation_investigation.md`) and the D2W512/D4W1024 agreement with Block B rule out a generic
  implementation error.

### What this means
- The (w) gate fails because the approved fixed budget cannot absorb D4W1536's early-step instability on hopper's
  inputs. This is a measurement-regime finding about an untrained critic, not evidence of degradation.
- Wider caveat for the lead (all D4 critics, all tasks). For D4 critics the probe score includes how quickly a
  critic recovers from AdamW's first steps. A trained "current" critic and its fresh reference could differ in that
  recovery for reasons other than plasticity, for example weight or feature scale. The probe definition is
  approved; this is a reported limitation, not a change.

## 2. Fresh-pair null (D4W1024 13/100)

### Implementation (CONFIRMED correct)
- `null_mode` draws two independent initialisations per pair (`PRNGKey(10000+pair)` split), shares pool, teacher and
  minibatches per round (`probe_rng`/`probe_key` with check index 1+pair), fits from fresh AdamW state, computes
  L = P(fresh) − P(current) per round in FP32, and applies the production `bootstrap_interval` (10,000 resamples,
  IQM, percentile, lower bound > 0).
- Re-run here with the production code: identical initialisations give L = 0 exactly on every round and never fire;
  swapping the roles negates L bit for bit (`pairing_check.py`).
- The production 10,000-resample interval agrees with the exact 5⁵ enumeration on 400/400 random inputs
  (`stats_null.py`).

### The gate sits on the rule's own floor (CONFIRMED)
- With no pair difference at all (IID symmetric rounds), the approved rule fires on about 4.8% of checks (exact
  enumeration, 11,400 checks; 4.3–5.4% across independent Monte Carlo batches). The nominal one-sided rate is 2.5%.
  The cause is five-round percentile undercoverage: unanimous signs alone fire 1/32 = 3.1%.
- At that rate, a 100-pair calibration exceeds 5/100 with probability ≈ 0.35 to 0.46 for one size (for a no-difference rate of 4.8% to 5.4%), and at
  least one of three sizes does with probability ≈ 0.73 to 0.84. The pre-specified "> 5% at any size" consultation is therefore
  expected to trigger even if the probe were perfect.
- D4W1024's 13/100 is still well above the floor: P(X ≥ 13) ≈ 0.001 to 0.003 at those no-difference rates. Its exact 95% interval is
  [7.1%, 21.2%]. Across sizes (3, 13, 5 of 100) the rates differ (χ² p = 0.014; 13 vs 5, Fisher p = 0.08).

### What can and cannot produce 13/100
- **Independent, zero-centred round noise cannot (CONFIRMED).** If a pair's five round losses are independent and
  centred on zero, the approved rule fires on 4.4–5.0% of checks whatever the tail shape. Tested (`tails_null.py`,
  8,000 checks each): normal 4.8%, Laplace 4.4%, t₃ 4.6%, t₁.₅ 4.7%, Cauchy 4.5%, 13% contamination at 10× 4.8%,
  difference of lognormals 5.0%. The IQM and percentile interval absorb heavy tails.
- **By elimination, 13/100 needs either a pair-level effect or invalid evidence.** A pair-level effect is a component
  shared by a pair's five rounds. Within a null pair the rounds use independent pools, teachers and minibatches; only
  the two initialisations (and the warm-up replay common to every pair) are shared. So either specific
  initialisation pairs differ in probe-fitting ability, conditional on this replay, or the 13/100 rows are stale,
  mixed or invalid. Which one is OPEN; the raw `null_pairs_D4W1024.jsonl` rows or a replicate run decide it.
- **What a pair-level effect would mean.** The rounds-only bootstrap would then correctly detect a real difference
  between two particular critics, and the calibration counts it as a "false" fire. Normal model
  L_r = μ_pair + ε_r with μ_pair ~ N(0, τ²) and ε ~ N(0, σ_w²), under the exact approved rule:

  | τ/σ_w | per-check fire rate | runs with two consecutive fires among checks 1–19 (μ held fixed per run) |
  |---|---|---|
  | 0 | 4.8% | 4.5% |
  | 0.3 | 7.0% | 13.7% |
  | 0.6 | 12.6% | 25.5% |
  | 1.0 | 20.4% | 32.7% |

  13% per check corresponds to τ ≈ 0.6 σ_w: a pair effect about 60% the size of the round noise.
- **Direct CPU measurement did not detect such an effect (low power).** Six null-mode initialisations (pairs 0–2),
  each fitted on the same five rounds of one check (dog-run, production probe, `varcomp.py`, `varcomp_robust.py`):

  | Size | Raw-scale F (init), p | Friedman p | Log-scale p | Rounds > 5× the round median | Directed pair fires (of 30) |
  |---|---|---|---|---|---|
  | D2W512 | 2.7, 0.049 | 0.13 | 0.09 | 1/30 | 3 |
  | D4W1024 | 0.90, 0.50 | 0.28 | 0.38 | 4/30 | 2 |

  - The robust tests find no significant initialisation effect at either size. At D4W1024 the point estimate of τ is
    0. With six initialisations a τ ≈ 0.6 σ_w effect would be detected only about half the time, so this weakly
    argues against a large effect on CPU FP32 but cannot exclude one on the A100.
  - D4W1024's final losses are heavy-tailed: 4 of 30 fits end 10–40× above the bulk (about 2e-4), on different
    initialisations in different rounds, consistent with the plateau-escape variability of section 1. As shown above,
    heavy tails alone do not raise the fire rate.
  - More CPU initialisations would cost about 19 min per D4W1024 initialisation (5 fits) and still say nothing about
    the A100 rows. The existing GPU rows decide this faster.
- **Architecture link (OPEN).** D4 fits depend on when they escape the post-blow-up plateau (section 1), a plausible
  source of pair-specific differences. D4W1536 fired only 5/100 on the A100, so the blow-up alone does not predict
  the rate.

### Is `null_pairs_*.jsonl` needed?
Yes, for three things a CPU run cannot supply:
1. confirming the 13 fires are persistent pair effects (per-pair round signs, within-pair SD, and how many pairs fire
   on the opposite side, `ci_high < 0`);
2. ruling out stale or mixed rows;
3. evaluating the pre-specified alternative threshold (p95 of L = 0.001096), which needs each pair's `ci_low`.

Retrieving it costs no GPU time. Note that a p95 threshold calibrated on those same 100 pairs passes them almost by
construction, so if it is adopted it must be validated on fresh pairs.

### Production relevance
Production compares a trained critic with its own initialisation, so the between-initialisation component of this
calibration does not exist there. Its analogue is any persistent difference in fitting ability that training
creates, which the trigger is designed to detect. The null result therefore says that the per-check test detects
differences as small as initialisation-to-initialisation variation. That is a question of effect-size calibration
for the lead, not a defect.

## 3. Positive control and m (no implementation defect)
`experiments/exp12/m_selection.evaluate` and `scripts/positive_control.py` implement amendments (d) and (q) exactly.
Demonstrated with the production function:

| Input (L per candidate vs L_trigger ≈ 0.8) | Recoveries last / half / all | Result |
|---|---|---|
| every injected critic worse than degraded | −0.125 / −0.25 / −0.375 | chooses `last`, exit 0, Block B report shows "m chosen" |
| overshoot beyond fresh | 1.0 / 2.0 / 1.5 | chooses `half` (2.0 beats exact recovery 1.0) |

- Natural-trigger validation, forced-evidence refusal, finite and shape guards, refusal to overwrite, and stops on
  L_trigger ≤ 0 or noise ≥ 0.10 are implemented and tested (`tests/test_exp12_positive_control.py`, 54 scoped tests
  pass, below).
- Not changed (needs approval): what to do when no candidate recovers (best recovery ≤ 0), and whether "toward the
  healthy reference" means the largest recovery or the one closest to 1. Changing either is a new success criterion.

## 4. Engineering changes
None to production code. No confirmed implementation defect was found in the probe, the null or range modes, the
trigger, the capture script or the positive control. Reproduction scripts were added under
`scripts/sci_investigation/`; they only call production functions. Minor gap in
`scripts/capture_exp12_range_evidence.py`: it does not save the fresh critic's initial parameters, so a CPU and an
A100 fit cannot be compared on bit-identical parameters. The parameters are reproducible from seed 990, but
orthogonal initialisation may round differently across devices. Optional; not changed.

## 5. Test results (CPU, this branch, HEAD 9d6a82d plus this report and scripts)
- `python -m unittest tests.test_exp12_range_capture tests.test_exp12_positive_control tests.test_exp12_probe tests.test_exp12_phase3 tests.test_exp12_reports`
  gives **54 tests, OK** (292 s), with `JAX_PLATFORMS=cpu MUJOCO_GL=disable WANDB_MODE=disabled EXP12_JAX_CACHE_DIR=off`.

## 6. Reproduction (CPU; outputs outside the repository)
```bash
cd main/scripts/sci_investigation
export JAX_PLATFORMS=cpu MUJOCO_GL=disable WANDB_MODE=disabled EXP12_JAX_CACHE_DIR=off
export EXP12_SCI_DATA=/abs/scratch/data EXP12_SCI_OUT=/abs/scratch/out
for s in "2 512" "4 1024" "4 1536"; do set -- $s
  python make_replay.py hopper-hop dmc_medium $EXP12_SCI_DATA/hopper_D$1W$2.npz $1 $2   # ~1 min each
  python make_replay.py dog-run dmc_hard $EXP12_SCI_DATA/dog_D$1W$2.npz $1 $2; done
python range_round.py hopper 4 1536 0 25600     # ~9 min: P/b 0.540
python first_steps.py hopper 4 1536 20          # blow-up table
python varcomp.py dog 4 1024 3 $EXP12_SCI_OUT/varcomp_dog_D4W1024.npz   # ~2 h
python varcomp_analyze.py $EXP12_SCI_OUT/varcomp_dog_D4W1024.npz
python varcomp_robust.py $EXP12_SCI_OUT/varcomp_dog_D2W512.npz $EXP12_SCI_OUT/varcomp_dog_D4W1024.npz
python stats_null.py; python runlevel.py; python tails_null.py; python pairing_check.py
```

## 7. Decisions for the lead
1. **Null gate.** The 5% limit equals the approved rule's own no-difference fire rate, so the gate is uninformative
   at 100 pairs. Options include the pre-specified p95 threshold, validated on fresh pairs, or a different gate.
   Whether a fire caused by initialisation-specific fitting differences counts as false for this study is a scientific
   choice.
2. **Hopper range.** D4W1536 does not reach 0.9 within the approved 1,000 steps, because of an AdamW first-step
   transient. The options are to accept it as a documented limitation (the probe still measures L), or to change the
   budget, optimizer warm-up or learning rate. All of these are methodology changes; none is proposed here.
3. **Positive control.** What a non-positive best recovery means, and whether overshoot counts as better.
4. **Probe validity caveat** (all D4 critics): the probe includes recovery from AdamW's first-step transient.

## 8. Next GPU experiments (not submitted)
1. **No GPU cost (first).** Retrieve `logs/blockB_22706349/gpu3/null_dog_run/null_pairs_{D2W512,D4W1024,D4W1536}.jsonl`
   and the summaries.
2. **Null replicate, one A100.** Needs no code change: null-mode initialisation keys do not depend on `--seed`.
   ```bash
   python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard --archs D4W1024 \
       --null_pairs 100 --seed 991 --out_dir /abs/new/null_replicate_seed991
   ```
   The same 100 initialisation pairs are run on a different warm-up replay and different probe streams. Estimated
   about 35–40 min (Block B: 300 pairs across three sizes in 6,042 s).

   Pre-specified reading, with Block B's seed-990 rows:
   - **Persistent initialisation effect confirmed** if the Spearman correlation of per-pair `loss_iqm` between seeds
     is > 0 with one-sided p < 0.01 (r ≳ 0.23 at n = 100). The 13/100 then reflects real differences between
     critics, and decision 1 applies.
   - **Chance or artefact** if there is no correlation and the seed-991 rate is ≤ 5/100.
   - **Rate confirmed, cause not** if there is no correlation but the seed-991 rate stays high; look next at the
     replay, teacher and numerics.
3. **Hopper capture (optional).** Run `scripts/capture_exp12_range_evidence.py --arch D4W1536` on one A100 (a few
   minutes).
   - **Confirms the CPU mechanism** if the step-1 or step-2 minibatch loss is > 10× the step-0 loss and the loss
     stays ≥ 0.8·b until after step 300 in at least 3 of 5 rounds.
   - **Mechanism not established on the A100** if neither holds; investigate GPU-specific causes.
