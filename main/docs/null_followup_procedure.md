# Fresh-pair null follow-up: retrieval, analysis and pre-specified readings

Companion to [scientific_validation_investigation.md](scientific_validation_investigation.md), section 2. Nothing
here changes an approved setting, threshold or rule, or adopts the p95 option. No Delta job is needed for step 1.

## 1. Where Block B wrote the null evidence (job 22706349, source b4a90cb)
Derived from `scripts/exp12_blockB.sh` and `scripts/sbatch_exp12_blockB.sh` at `b4a90cb`:
- **Checkout:** `/work/hdd/biqc/skaveti1/exp12/main`. The sbatch script `cd`s here, and `OUT=$(pwd)/logs/blockB_${SLURM_JOB_ID}`.
- **Null step:** lane `gpu3`, step `null_dog_run`, which ran
  `probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard --archs D2W512 D4W1024 D4W1536 --null_pairs 100`
  with seed 990 (the script's default; `PROBE_EXTRA` was empty).

Files (paths relative to the checkout):

| File | What it is |
|---|---|
| `logs/blockB_22706349/gpu3/null_dog_run/null_pairs_{D2W512,D4W1024,D4W1536}.jsonl` | one JSON row per pair: arch, env, seed, pair, check_index, init_keys, five-round scores, b and L, loss_iqm, valid, ci_low, ci_high, resamples, confidence, null_threshold, fired |
| `logs/blockB_22706349/gpu3/null_dog_run/null_summary_{D2W512,D4W1024,D4W1536}.json` | the reported rate, p95, IQM and SD |
| `logs/blockB_22706349/gpu3/null_dog_run.log` | the step's stdout and stderr (every row as printed, plus any restart) |
| `logs/blockB_22706349/status/gpu3.tsv` | step result, exit code and wall time |
| `logs/blockB_22706349/{commit.txt,host.tsv,gpus.csv,report.txt}` | source revision, lane layout, GPU model, the reported verdicts |
| `logs/exp12_blockB_22706349.out` | job header: jax version, devices, environment dump |

At `b4a90cb` the row format has no provenance block. Its resume code silently kept the last copy of a duplicated
pair, so the analysis must detect duplicates rather than trust the summary.

## 2. Retrieval (read-only on the evidence; run on a Delta login node)
```bash
cd /work/hdd/biqc/skaveti1/exp12/main
J=22706349; B=logs/blockB_$J; N=$B/gpu3/null_dog_run
ls -la "$N"            # expect exactly 3 null_pairs_*.jsonl and 3 null_summary_*.json
F=""; for f in "$N"/null_pairs_{D2W512,D4W1024,D4W1536}.jsonl "$N"/null_summary_{D2W512,D4W1024,D4W1536}.json \
    "$B/gpu3/null_dog_run.log" "$B/status/gpu3.tsv" "$B/commit.txt" "$B/host.tsv" "$B/gpus.csv" "$B/report.txt" \
    "logs/exp12_blockB_$J.out"; do
  if [ -f "$f" ]; then F="$F $f"; else echo "MISSING $f"; fi; done
sha256sum $F > "$HOME/blockB_${J}_null_evidence.sha256"
tar -czf "$HOME/blockB_${J}_null_evidence.tar.gz" $F
tar -tzvf "$HOME/blockB_${J}_null_evidence.tar.gz"; cat "$HOME/blockB_${J}_null_evidence.sha256"
```
- **Read-only:** it only reads the evidence; its two outputs go to `$HOME`. It does not touch git or any job.
- **What to send me:** the `.tar.gz` (expected well under 1 MB), the `.sha256` text, the `ls -la` listing and any
  `MISSING` lines.

## 3. Analysis (CPU; deterministic)
From `main/`, in the pinned environment (numpy 1.26.x; the versions are recorded in the report):
```bash
mkdir -p /abs/evidence && tar -xzf blockB_22706349_null_evidence.tar.gz -C /abs/evidence
(cd /abs/evidence && sha256sum -c /abs/blockB_22706349_null_evidence.sha256)
JAX_PLATFORMS=cpu python scripts/sci_investigation/null_followup.py \
    --original /abs/evidence/logs/blockB_22706349/gpu3/null_dog_run --out /abs/null_followup_original.json
# after an approved seed-991 replication (section 5):
JAX_PLATFORMS=cpu python scripts/sci_investigation/null_followup.py \
    --original /abs/evidence/logs/blockB_22706349/gpu3/null_dog_run \
    --replication /abs/null_replicate_seed991 --out /abs/null_followup_replication.json
```
- **Exit status:** 0 means every supplied file passed integrity; 2 means at least one integrity error (the statistics
  are still printed, but must not be interpreted).
- **Also check by hand:** `commit.txt` equals `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`, `gpus.csv` shows A100s,
  and the job header shows jax 0.4.34.

### Integrity checks (per architecture and file)
- **Parsing:** every line is a JSON object; blank or malformed lines are errors.
- **Fields:** all required fields are present; each round array holds exactly 5 numbers.
- **Duplicates:** a repeated pair index is an error, and the report says whether the copies are identical.
- **Coverage:** pairs 0–99 present, none missing, none beyond 99.
- **Identity:** `arch` matches the file, `env` = dog-run, `seed` = 990 (991 for the replication),
  `check_index` = pair + 1, and `init_keys` = [10000 + pair, 0, 1].
- **Stored statistics:** `loss_rounds` equals float32(fresh − current) exactly. `valid`, `loss_iqm`, `ci_low`,
  `ci_high` and `fired` are recomputed with the production `iqm`, `bootstrap_interval` and `triggered` (seed, check
  index, 10,000 resamples, 0.95, threshold 0) and must match exactly. Resamples, confidence and threshold must be the
  same in every row.
- **Summary:** `null_summary_<arch>.json` is recomputed from the validated rows (count, rate, p95, IQM, SD).
- **Replication only:** one provenance block per file, from a clean source tree.

### Diagnostics (statistical evidence, α = 0.01 one-sided, pre-specified here)
- **W, within-original pair effect.** Permutation test (20,000 permutations, seed 20261009) of the between-pair
  variance of each pair's five-round IQM. Under no pair effect the 500 round values are exchangeable across pairs.
  Reported alongside, not decisive: lower-side and mirror (`ci_high < 0`) fire counts with a binomial test against
  the no-difference floor 0.048; pairs whose five rounds all share a sign against the 2/32 expected under symmetric
  rounds; the one-way ANOVA F and τ/σ_w estimate; and Kruskal–Wallis.
- **X, cross-seed persistence** (needs the replication). Spearman ρ of `loss_iqm` across the 100 matched pairs,
  with a one-sided permutation p-value. Reported alongside: Pearson r, sign agreement, and a Fisher test of
  re-firing.
- **Power** (simulated with these exact statistics, 100 pairs): the effect size that a 13% rate implies,
  τ/σ_w ≈ 0.6, is detected with probability ≈ 1.00 by both W and X. At τ/σ_w = 0.45 the power is 0.97 (W) and
  0.99 (X); at 0.3 it is 0.46 and 0.73. False-positive rates at τ = 0 were 0.00 and 0.02.

## 4. Pre-specified readings
**A. Evidence integrity (prerequisite).** Any integrity error means the reading is "EVIDENCE INVALID", and no
statistic below is interpreted. If the original file has duplicate or missing pairs, the reported 13/100 itself is
in doubt and the original null step must be re-run unchanged.

**B. Approved qualification criterion (unchanged).** The NULL rule passes when each size's per-check fire rate is
≤ 5% with ≥ 100 pairs.
- The original D4W1024 result is a FAIL, unless integrity shows the stored record differs from its recomputation.
- The follow-up cannot convert that FAIL into a PASS. A seed-991 rate ≤ 5% is additional evidence, not a
  re-qualification: the approved rule has no retest provision, so what to do with two calibrations is the lead's
  decision.
- The approved rule's own no-difference fire rate is about 4.8% (investigation report, section 2). A 5% limit at
  100 pairs is therefore failed by chance 35–46% of the time per size. This is stated for the decision, not applied.

**C. Statistical readings (diagnostic, not qualification).** The script prints exactly one per architecture:

| Reading | Condition | Meaning |
|---|---|---|
| PERSISTENT INITIALISATION-PAIR EFFECT | X: ρ > 0 and p < 0.01 | Specific initialisation pairs differ in probe fitting on a new replay and new probe streams. The 13% then reflects real between-initialisation differences, which the rounds-only bootstrap correctly detects: the per-check test is sensitive at initialisation-to-initialisation scale. This is an effect-size or threshold decision, not a defect. |
| PAIR EFFECT SPECIFIC TO THE ORIGINAL REPLAY/STREAMS | W: p < 0.01, X not | A pair effect conditional on that replay (pair × replay interaction). Same class of consequence, but not intrinsic to the initialisations. |
| NO PAIR EFFECT DETECTED; replication within 5% | neither W nor X; replication rate ≤ 5/100 | The original excess is consistent with chance (p ≈ 0.001 to 0.003, so unusual) or with an artefact the integrity checks did not catch. |
| RATE EXCESS REPRODUCED WITHOUT A PAIR EFFECT | neither; replication rate > 5/100 | Investigate the replay, teacher and numerics next. |
| (original only) pair-level effect within this replay / none detected | W alone | Interim reading before any replication. |

The D2W512 and D4W1536 original files are controls. If W is significant only at D4W1024, the effect is
size-specific; if all three are significant, it is general and the size difference in fire rates is partly chance
(3, 13 and 5 of 100; χ² p = 0.014).

## 5. The pre-specified p95 alternative, evaluated but not adopted
The pre-specified option: if the rate exceeds 5%, the lead may adopt the 95th percentile of pair-level L at that
size as the null threshold (stored for D4W1024: 0.001096). The script reports only:
- **In-sample fire rate** with `ci_low > t`. It is ≤ about 5% almost by construction, because t is calibrated on
  the same 100 pairs, so it is not evidence.
- **Two-fold cross-validated rate** (threshold from even pairs applied to odd, and the reverse).
- **Out-of-sample rate on the seed-991 replication.** This is the informative number.
- **t in units of the SD of pair-level L.**

What the option would cost in sensitivity cannot be judged until the positive control gives L_trigger. It stays
the lead's decision.

## 6. Replication run (needs approval; not submitted)
One A100 of the grid's model, at a clean, reviewed revision:
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard --archs D4W1024 \
    --null_pairs 100 --seed 991 --out_dir /abs/new/null_replicate_seed991
```
- **Same pairs, new replay and streams.** The null initialisation keys (`PRNGKey(10000 + pair)`) do not depend on
  `--seed`. Seed 991 changes only the warm-up replay, the probe pools, teachers and minibatches, and the bootstrap
  streams.
- **Code is comparable.** The single-critic null computation is unchanged between `b4a90cb` and `9d6a82d` (checked:
  only twin-critic, resume-validation and provenance code differ).
- **Estimated cost:** about 35–40 min on one A100 (Block B ran 300 pairs across three sizes in 6,042 s).
- **Package it** with section 2's command, using `N=/abs/new/null_replicate_seed991`.

## 7. Tests of the analysis script
`python -m unittest scripts.sci_investigation.test_null_followup` (from `main/`, CPU, about 30 s) gives 5 tests, OK.
The fixtures use the exact `b4a90cb` row format:
- a clean null;
- a persistent pair effect that reproduces in a replication;
- eight corruptions, each required to produce its own specific error: identical duplicate, different duplicate,
  missing pair, wrong seed, malformed line, tampered `fired`, inconsistent `loss_rounds`, short round array;
- a summary mismatch;
- determinism.

Break-and-restore: five mutations of the script were each caught, and the tests passed again once restored:
duplicate detection off, stored `fired` trusted, permutation p-value inverted, missing-pair message changed,
identity check off.
