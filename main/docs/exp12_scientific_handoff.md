# Exp1/Exp2 scientific handoff (Claude, 2026-10-10)

For the shared integration branch. Every result below names the source commit and the job that produced it.
Nothing here changes an approved setting, threshold, rule or acceptance criterion. Full reports:
[scientific_validation_investigation.md](scientific_validation_investigation.md),
[null_followup_procedure.md](null_followup_procedure.md) and
[null_followup_original_results.md](null_followup_original_results.md).

## Commits

| What | Branch | Commit |
|---|---|---|
| Approved engineering baseline | `codex/exp12-metric-integration` | `9d6a82d4aef81299901177865e383640439cde2a` |
| Block B (the source of every A100 result below), job 22706349 | (submitted from the Delta checkout) | `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309` |
| Scientific investigation: hopper, null, positive control (CPU) | `claude/scientific-validation-investigation` | `f12e00f` |
| Null follow-up tooling (pre-specified analysis, `null_followup.py`) | same | `5487f62` |
| Null results on the original rows (post-hoc `null_exploratory.py`) | same | `8a888ef998dd2055d43faea26b78e8f555d00bc6` |
| Exact-identity audit and job 22765261 profile audit (Codex) | `codex/a100-exact-identity-audit` | `dcb7491b61c0…` |
| Atomic compilation-cache writes (engineering) | `claude/precision-cache-fix` | `d23191befd1d6a3254c61ac1edbad363b9b744e5` |

## 1. Hopper D4W1536 fresh-critic range: FAIL (established)
- **A100 result** (job 22706349 @ `b4a90cb`): configured-pool IQM(P)/IQM(b) = 0.602 against the required ≥ 0.9
  (amendment (w)). D2W512 = 0.933 and D4W1024 = 0.902 pass.
- **Reproduces on CPU in full FP32** (@ `9d6a82d`, scripts @ `f12e00f`): D4W1536 rounds give 0.540 and 0.730. TF32
  and GPU numerics are therefore excluded.
- **Mechanism** (established on CPU):
  - AdamW's first fresh-state step moves every parameter by exactly lr (update norm = lr·√N).
  - The D4 critics' outputs jump: pool MSE 26–229 after steps 1–2, against b ≈ 0.5.
  - The fit then plateaus at loss ≈ b until about step 630 on hopper (about 350 on dog-run, which still reaches
    0.993), so the 1,000-step budget ends mid-descent.
  - Hopper's crowded 19-dimensional probe inputs slow fitting after the plateau (supported, not proven).
- **No implementation defect.** This is a measurement-regime failure of an untrained critic, not degradation.
- **Optional A100 confirmation:** `scripts/capture_exp12_range_evidence.py --arch D4W1536` (a few minutes; pass/fail
  reading in the investigation report, section 8).

## 2. D4W1024 fresh-pair null: FAIL (established); cause narrowed, not resolved
- **Rate:** 13/100 fires (exact 95% CI 7.1%–21.2%) against the pre-specified ≤ 5%. D2W512 3/100 and D4W1536 5/100
  pass. Job 22706349 @ `b4a90cb`.
- **Evidence integrity** (archive SHA-256 `5819d2be…16fb`, analysis @ `5487f62`): all 300 rows valid, no duplicates
  or missing pairs, every stored statistic and summary recomputed exactly, one uninterrupted run.
- **The rule's own floor** (CPU): with no real difference the approved rule fires on about 4.8% of checks for any
  tail shape tested. A 5% gate at 100 pairs therefore fails by chance 35–46% of the time per size. 13/100 is still
  far above that floor (p ≈ 0.001–0.003).
- **Test W** (pre-specified, original data): permutation p = 0.069 at D4W1024, so "no pair-level effect detected"
  at α = 0.01. Round losses are extremely heavy-tailed (excess kurtosis 40), which cost the test power.
- **Post-hoc** (@ `8a888ef`): shuffling the real round losses across pairs gives 5.8 fires on average; the observed
  13 has p = 0.003. Kruskal–Wallis p = 0.003. This points to a pair-level effect conditional on the one replay.
  Controls show no excess (D2W512 p = 0.95, D4W1536 p = 0.37). Whether D4W1024 differs from D4W1536 is uncertain
  (Fisher p = 0.08).
- **95th-percentile option, diagnostic only:** 0/100 in-sample and 0/50 cross-validated at every size. Its
  out-of-sample rate and its cost in sensitivity are unknown. Not adopted.
- **Not separable from the original data:** initialisation and replay are confounded (one replay for every pair).

## 3. Seed-991 replication: status UNVERIFIED
- **Drafted:** the submission commands and the readings fixed in advance are in
  `null_followup_original_results.md`, section 7, and in the agreed instructions: a separate checkout at `9d6a82d`,
  one A100, 1 h.
- **Not verifiable here:** I have no Slurm access. No GitHub branch records a submission or a job ID. I did not
  submit it and cannot confirm whether anyone did.
- **To verify on Delta:**
  ```bash
  sacct -u $USER -S 2026-10-09 --name=exp12_null991 -o JobID,State,Elapsed,Start,End
  ls -d /work/hdd/biqc/skaveti1/exp12_null991/main/logs/null_replicate_991_*
  ```

## 4. Positive control and m: no implementation defect; the success rule is ambiguous
- **Code** (audit @ `9d6a82d`): `m_selection.evaluate` and `scripts/positive_control.py` implement amendments (d)
  and (q) literally. Natural-trigger validation, forced-evidence refusal and the stops on L_trigger ≤ 0 and
  noise ≥ 0.10 are tested.
- **Ambiguity A:** if every injected critic is worse than the degraded one (all recoveries < 0), m is still chosen
  (`last`) with exit 0, and the Block B report shows "m chosen".
- **Ambiguity B:** overshoot ranks above an exact return (recovery 2.0 beats 1.0), although the rule speaks of
  recovery "toward the healthy reference".
- **No qualifying evidence exists.** In Block B the D4W1536 dog-run development run hit its 6 h timeout
  (dev_run TIMEOUT, 21,600 s) without a trigger, so the positive control and the injected arm were skipped. m is not
  frozen.

## 5. Exp1 readiness: NOT READY
Blockers:
1. Null gate FAIL at D4W1024, pending the replication and a decision.
2. Hopper D4W1536 range gate FAIL, pending a decision.
3. No natural positive control and no frozen m, pending an A100 development run within its time budget.
4. GPU test suite at `b4a90cb` failed (`test_check1` plus 11 errors, both modes), with no A100 rerun at `9d6a82d`.
   Four GPU numerical tolerances are still unset.
5. Runtime: Block B's D4W1536 dev run timed out at 6 h. Job 22765261 @ `9d6a82d` (profile mode) projects about
   3.27 h of warm training for a 1M-step D4W1536 dog-run parent. This excludes probes, evaluations, compilation and
   I/O, so it is not a full-run guarantee.
6. Open decisions on entropy-target wording, the single-seed reporting convention and the deployment plan.

## 6. Exp2 fork-identity qualification: NOT ESTABLISHED on an A100
- **Block B:** all seven identity forks that ran (D4W1024 and D4W1536 × dog-run and myo-key-turn × cold and warm
  cache) TIMED OUT; the last cell (D4W1536 myo-key-turn, warm) was skipped for time. HumanoidBench was unavailable.
  No bit-identity verdict exists for any size or suite.
- **At `9d6a82d`:** no identity continuation has run. Codex's audit of job 22765261 (`dcb7491`) projects that the
  D4W1536 `identity_warm` step would not fit its 300 s internal timeout (about 329 s for training, probes and
  evaluations alone).
- **`dcb7491` tightens the comparison to bit-for-bit** (dtype and bytes; NaN not equal to itself), consistent with
  amendment (m). A ±0.0 difference that previously passed would now fail.
- **Required before Exp2 arms:** the identity fork bit-identical for D4W1024 and D4W1536 in each suite (DMC,
  MyoSuite, HumanoidBench twin critics) on the production GPU model, under the cache policy actually used.

## 7. Next actions
**Engineering** (no scientific approval needed):
- Review and merge `claude/precision-cache-fix` into `integration/exp12`. It is a cache race fix that changes no
  compiled results.
- Give the identity-fork qualification a time budget that fits the measured costs, then run it at the integration
  HEAD.
- Rerun the GPU test suite at the integration HEAD.
- Recover any seed-991 job record (section 3).

**Decisions needed from the lead:**
1. **Null gate:** after the replication, decide what to do with the original FAIL, whether to adopt the
   95th-percentile threshold, or another gate.
2. **Hopper D4W1536:** accept it as a documented limitation, or approve a change to the probe budget, optimizer
   warm-up or learning rate (a methodology change).
3. **Positive-control success rule:** Ambiguities A and B.
4. **Development-run budget** for obtaining a natural trigger.
5. Entropy wording, single-seed reporting, the GPU tolerances and the deployment plan.
