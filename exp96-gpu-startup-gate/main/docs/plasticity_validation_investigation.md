# Plasticity validation investigation

Engineering reference: approved corrections at `a3f44aa56455e3aae4b36459a78c97237bad62a8`, integrated with runtime instrumentation on the separate `codex/exp12-integrated-candidate` branch. No calibration criterion, probe budget, optimizer, architecture, precision policy, sampling rule, bootstrap, or trigger rule is changed. This report does not establish a CUDA root cause.

## Evidence and interpretation limits

The authoritative collected Block B report is `/workspace/attachments/45917c23-4022-4d33-b9aa-e34782076ed8/blockB_22706349.txt`, SHA-256 `4f87ba9a4cada6be2eefb7a8b366200e5dce3a57edb167982136409238e258b2`, source `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`. It contains all nine hopper range rows with their five scores, baselines and final MSEs. It contains only architecture-level null summaries, not the 100 individual null-pair rows. Its runtime/provenance is historical, not qualification of the integrated candidate.

Canonical authority is `.claude/methodology-exp1-exp2.md` with chronological amendments, and the recorded approved decisions in `docs/exp12_decisions.md`. In particular: the pre-result NULL rule requires at least 100 fresh pairs per size and lead review if per-check firing exceeds 5%; amendment (w) requires configured-pool `IQM(P)/IQM(b) >= .9`; amendment (l) uses two consecutive production checks; amendment (z) defines the approved literal twin paired-loss order. Historical range and consecutive-check alternatives are superseded.

`scripts/plasticity_validation_cpu.py` independently verifies 68 IQM/10,000-resample CI fixtures, four exact-bootstrap enumeration cases, pairing/eligibility, and all nine embedded range rows. It uses sorting, explicit trimmed averaging, direct linear quantiles and enumeration rather than the production statistical functions for its reference answers. Its simulations are explanatory CPU toys, not CUDA calibration or new acceptance tests.

## Fresh-pair null: 13/100

**Established:** D4W1024 fired 13 of 100 checks; D2W512 fired 3, D4W1536 fired 5. Its across-pair L IQM is `7.418e-6`, SD `.0007396`, and reported p95 `.001096`. These rounded across-pair summaries cannot recover within-pair round variance or CI coverage; their near-zero center does not rule out persistent positive/negative pair effects. D4W1024 fails the approved observed-rate gate. No threshold is automatically adopted. The archived Block B step's shell exit zero does not negate its scientific rule failure.

Source paths `experiments/exp12/probe.py:61`, `:73`, `:98`, `:111`, `:137`, `:146`, `:173`, `experiments/exp12/trigger.py:48`, and `scripts/probe_fresh_checks.py` implement:

- A warm-up-only replay with the critic still untrained; the callback stops before the first SAC update.
- Two independent critic initializations per pair, fixed across that pair's five rounds, each fitted from a fresh optimizer state and its own pre-fit offset.
- One shared replay sample, random teacher/base target and minibatch ordering within each round. Dedicated streams derive from seed/check/round; CPU checks find no collision among the 525 production seed/check/round probe keys.
- FP32 score subtraction within each paired round, then the IQM and percentile bootstrap of the five losses. The strict lower-bound-above-zero rule is retained. Nonfinite losses never fire.

No incorrect pairing, sign, sample replacement rule, fitting-state reuse, bootstrap arithmetic, or scheduling defect is demonstrated in those calculations. Tests supplement source review; they do not certify historical artifacts.

### Ranked explanations

| Explanation | Evidence and what it cannot establish |
|---|---|
| Initialization-dependent fitting differences persistent across rounds, combined with conditional bootstrap inference | Structurally supported: each fresh pair is reused across its five rounds. Distinct fresh initializations need not have the same finite-budget fitting ability. A persistent pair effect makes five round losses dependent marginally. The calibration labels fresh-versus-fresh differences as null fires, whereas a conditional test can correctly detect a nonzero fitting difference between those particular initializations. This estimand distinction is not a demonstrated implementation bug. Actual pair effects require the missing raw rows/curves. |
| Five-round percentile-IQM statistical undercoverage | Established limitation in independent toys. For IID continuous symmetric centered round losses, unanimous signs alone cap two-sided coverage at `1-2/32=.9375`. Exact five-round empirical-bootstrap normal simulations (12,000 checks) give coverage .904083 and positive zero exclusion .048167. The unchanged finite 10,000-resample implementation fires 50/1,000 normal controls. Undercoverage helps explain why nominal 95% two-sided intervals do not imply 2.5% upper-direction firing; it does not alone identify this architecture's 13%. |
| Pool/teacher dependence and architecture-specific finite-fit variability | All pairs use a common warm-up replay; targets, pool rows and minibatches vary by dedicated streams. Replay overlap and heterogeneous fitting can affect both per-check and consecutive-check dependence. The raw rows and actual pools are missing, so a quantitative variance decomposition is unavailable. |
| Numerical effects | The high-frequency target is sensitive to forward rounding. The CPU fixed-parameter experiment below establishes that mechanism locally, not its A100 magnitude or causal contribution to the 13 fires. Shared targets within a paired round are still identical. |
| Stale/reused or inconsistent calibration artifacts | A real resume-integrity gap was reproduced and fixed: stored firing flags were trusted despite contradictory saved losses/CIs. The candidate recomputes stored paired losses, IQM, CI, protocol fields and firing flags before reuse. Existing a3 guards already reject stale source/config/runtime identity and duplicate pair indices. There is **no evidence** that the historical 13/100 was caused by this gap. The original raw rows/source provenance are needed to investigate it. |
| Pure binomial variability | Under an illustrative IID p=.05 model, `Pr(K>=13, n=100)=.00146435`; the exact 95% binomial interval for 13/100 is approximately [.071073, .212041]. Independence and p=.05 are not established for the actual calibration, so these are not new criteria or a causal finding. |

The first two explanations interact. A rounds-only bootstrap quantifies variability conditional on the sampled fresh pair; it cannot automatically incorporate an unmodeled distribution of initialization-dependent mean differences. Replacing independent fresh critics with identical initializations, changing the bootstrap, adding rounds, or calibrating a threshold would change the approved procedure and requires lead approval. The CPU identical-copy pairing control is an implementation check, not such a replacement.

### Per-check versus production triggering

The calibration reports individual checks. Production requires the first pair of contiguous firing checks among checks 1–19; check 20 cannot create an eligible fork. It compares a trained critic to **its own stored initial critic**, unlike the calibration's independently initialized fresh pair. Consequently neither `.13^2` nor the calibration rate is an established production false-fork probability.

For illustration only, an IID recurrence over 19 eligible checks gives probabilities of at least one consecutive pair of .010940, .042199 and .241595 for per-check probabilities .025, .05 and .13 respectively. These are different from the probability of a particular pair firing. Actual longitudinal dependence and initialization linkage prevent transferring those numbers to production.

### Minimum next evidence

First retrieve, without a GPU experiment, `logs/blockB_22706349/gpu3/null_dog_run/null_pairs_D4W1024.jsonl`, its sibling D2W512/D4W1536 files, matching resolved configuration/source/runtime records, and the complete null log. The 100 D4W1024 rows would permit independent CI/firing recomputation, identification of the exact 13 pairs, per-pair means/spread, polarity and validity checks, and detection of duplicate/stale records. They cannot recover fitting curves/targets that were never exported.

If ambiguity remains, repeat **one specified existing firing pair** at its recorded index on the same approved settings/source/device, retaining both critics' five complete loss curves, actual shared inputs/base targets/minibatch keys, initialization fingerprints and replay/normalizer provenance. One repeat distinguishes reproducibility/inconsistent evidence from a stable pair effect; it cannot requalify the 100-pair rate or estimate the full production trigger. A complete unchanged 100-pair rerun is needed for final-source qualification, not an unapproved number of cherry-picked successful pairs. Different initialization controls, longitudinal calibration designs, precision contrasts, or new statistical criteria require approval.

One A100 matching the approved runtime is sufficient for the proposed measurements, subject to actual memory verification. Block B's three-size, 300-pair null took 6,042 seconds in its recorded multi-lane setting; no per-size or per-pair breakdown exists, so a precise D4W1024 time or memory estimate is unavailable. No GPU work was conducted here.

## Hopper fresh range: D4W1536 P/b=.602398

**Established arithmetic:** at pool 25,600, `IQM(P)=.3008451164`, `IQM(b)=.4994128446`, ratio `.6023976348`; final pool MSE IQM `.1999773383`. Each of the five round ratios is below .9 (approximately .559–.649). Independent FP32 subtraction and sorted-trim references reproduce all nine archived rows. D2W512 and D4W1024 configured-pool ratios are .933294 and .901661. The small D4W1536 pools give .986747 (1,600) and .845097 (6,400).

The approved statistic is the **ratio of IQMs**, not the IQM of ratios or `1-IQM(MSE)/IQM(b)`. Trimming is not linear. Alternative arithmetic neither replaces the approved statistic nor explains this large shortfall. `b=Var(base)` uses the actual random sinusoidal target's variation, and the fixed pre-fit offset is translation-invariant in exact arithmetic. Actual offset/target FP32 rounding still needs observation.

This is an **untrained fresh critic's fitting failure under the approved measurement budget**, not evidence that a trained critic lost plasticity. No SAC updates occurred in the range routine. It establishes failure of the measurement qualification gate, not a primary Exp1 degradation or Exp2 intervention conclusion.

### Ranked explanations and source checks

1. **Finite-budget optimization/conditioning interacting with architecture and target complexity** is most directly supported. The large network leaves a substantial true full-pool residual, and the same architecture does much better on a smaller sampled pool. Source `_fit` starts fresh AdamW, makes exactly 1,000 updates with batch256 and evaluates final MSE over the full pool. An independent analytic linear-network reference matches every CPU minibatch loss, the final full-pool loss and the fixed offset; JIT/eager/chunk controls agree on that fixture. This rules out an obvious generic AdamW/update-count/offset/reduction defect there, not a production-size conditioning or representational limitation.
2. **Pool support, weighting and effective fitting effort** are supported mechanisms but confounded in the historical comparison. Sampling is with replacement from 5,000 warm-up rows. Expected distinct original rows are about 1,369, 3,610 and 4,970 for pools 1,600, 6,400 and 25,600; mean draws per pool position over the same fit are 160, 40 and 10. Input coverage, duplicate weighting, teacher evaluations, offsets and minibatch indices all change when pool size changes. Evaluation chunks also change through gcd:320/1280/2560. The smaller-pool result is not an isolated proof that pool size causes the failure, and does not authorize reducing it.
3. **Teacher complexity and numerical sensitivity** are plausible. `sin(1e5*f)` can yield very different targets from small changes in f. In a reproducible CPU D4W32 fixed-parameter synthetic pool, compiled/eager Q differs by at most `2.3841858e-7`, but target values differ by as much as .01497877. One FP32 Q ULP at Q=1 corresponds to roughly .01192 radians before multiplication rounding. This is CPU forward arithmetic, not evidence that CUDA/TF32 is erroneous or a bound on its effects. Within a round both fitted critics receive the same generated base. No precision policy is changed.
4. **Offset/cancellation, replay normalization, optimizer scaling, JAX shape effects or an architecture-specific implementation defect** remain hypotheses. There is no historical loss trajectory, actual target/pool data, offset record or initialization fingerprint sufficient to resolve them. Source uses pre-fit own offsets, frozen normalization, copied functional parameters, population variance and full-pool final loss as specified. The approved fixed learning rate/AdamW decay can behave differently as depth/width grows; changing it would be a scientific intervention, not a repair justified by this failure.

No confirmed range arithmetic or generic fitter implementation bug explains .602. The correct distinction is “does not fit this approved probe sufficiently in this budget” versus “has been scientifically shown to degrade during training.” The available evidence supports only the former.

### Minimum A100 experiment, prepared but unexecuted

`scripts/capture_exp12_range_evidence.py` delegates to the existing hopper range routine at seed990, check0, D4W1536, five rounds, pool25,600, 1,000 steps, batch256, target scale1e5, chunk2,560 and the same optimizer. It does not refit a second time, change any setting, regenerate a target, or consume additional RNG. It records already-returned curves plus the actual observed teacher pools after fitting has finished, and warm-up replay/normalizer/source/runtime/checksum provenance. The observer's CPU tests verify identity/delegation, preserved live tiny-probe arrays, immutable outputs, checksums and refusal of CPU full-setting capture. Mocks do not count as GPU qualification.

On a separately approved A100 allocation, from a fresh detached **integrated candidate** checkout, the command is:

```bash
export JAX_PLATFORMS=cuda,cpu
unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE
export WANDB_MODE=disabled MUJOCO_GL=disable
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
candidate_rev=$(git rev-parse HEAD)
timeout 1800 python -u scripts/capture_exp12_range_evidence.py \
  --arch D4W1536 --expected-commit "$candidate_rev" \
  --out-dir "/absolute/new/evidence/${candidate_rev}_hopper_D4W1536"
```

The absolute output path must be supplied and must not exist. The 1,800-second cap is the existing Block B range-step timeout, not an increase. This is a command for an approved allocation, not authorization to submit one, and is not the frozen five-minute runtime comparison.

Outputs include `range_D4W1536_pool_25600_curves.npz` (5×1,000 minibatch losses, scores, baselines, offsets and final full-pool MSE), `round_0_teacher_pool.npz` through `round_4_teacher_pool.npz` (actual normalized observations/actions, generated base targets and teacher keys), `warmup_replay.npz`, `range_D4W1536.json`, and `capture_metadata.json` with checksums/configuration/runtime/source. Use them to test declining versus plateaued/divergent fits, input/target duplicates, baseline/offset arithmetic, and reproducibility. Plateau or a numerical association is still not by itself a unique causal diagnosis.

The approved .9 criterion applies unchanged **to this size's configured-pool ratio**; all three sizes are still needed for full range qualification. No new curve-slope, convergence, numerical, or runtime acceptance threshold is selected. Block B's full nine-case range step took198 seconds; this is an empirical context, not a guaranteed duration or peak-memory estimate for the new one-size capture. One A100-SXM4-40GB is the proposed matching hardware; actual feasibility/performance remains unverified. More fitting steps, a smaller pool, FP32-only probes, alternate targets/optimizers or changed acceptance criteria require methodological approval before any contrast experiment.

## Confirmed engineering defects versus scientific findings

| Classification | Finding / disposition |
|---|---|
| Confirmed engineering defect, fixed | Check 1 accepted broadcastable malformed panel shapes. Matching nonempty finite FP32 shape/rank guards now reject invalid evidence; the exact-control/identity and64-eps injected calculations remain unchanged. |
| Confirmed evidence-integrity gap, fixed | Null resumption trusted contradictory stored statistics/flags; they are now recomputed using the same serialized FP32 losses and approved CPU bootstrap. Historical attribution remains unverified. |
| Confirmed qualification-input gaps, fixed | Positive control did not independently certify the first natural trigger against saved round arithmetic, schedule, state and horizons. Canonical qualification now validates that evidence before constructing/loading an agent. Test-only/forced evidence still cannot choose m. Existing PC outputs are preserved instead of overwritten. |
| Confirmed report completeness gaps, fixed | Block B could infer required range sizes from present files, and null summaries could pass with fewer than100 pairs. It now applies the full required population/count rule and checks stored flag consistency. Missing/nonzero invalid-pair counts are exposed for review; **no new invalid-pair acceptance criterion** is adopted. |
| Statistical limitation | Five-round percentile-IQM undercoverage and initialization/longitudinal dependence limit transfer from nominal CI coverage to calibration or production false-fork rates. |
| Numerical limitation | High-frequency target amplification is demonstrated on a CPU fixture; CUDA magnitude/impact remains unverified. |
| Genuine measured finding | Archived D4W1024 fresh-null and D4W1536 configured-pool range fail their approved gates. These are measurement-qualification failures; no production scientific conclusions follow from them. |
| Unresolved hypotheses | Stable initialization effects, production-size optimization/conditioning/representation, replay/teacher distribution, precision/chunk effects and stale historical artifacts cannot be separated by the supplied summaries alone. |

## Decisions requiring the lead

1. Decide the disposition of both failed approved measurement gates after reviewing the exact missing raw null evidence and the minimal unchanged range capture. No test-driven change to make either pass is proposed.
2. Approve any discriminating experiment that changes initialization linkage, precision, fitting budget, sampling, target function or statistical procedure. The prepared same-setting retention experiment changes only evidence collection, but still requires separate GPU/job authorization.
3. Clarify how invalid calibration pairs should be treated if any are observed; the implementation retains the approved “invalid checks never fire” behavior and exposes counts without choosing a new acceptable invalid-pair rate.
4. Obtain a genuine natural full-setting positive control, review recovery/noise/sensitivity/reproduction/runtime provenance, and freeze m. Existing approved arithmetic can recommend m even if all recovery values are negative; adding a positive/signed/significance success requirement or a numerical reproduction tolerance would need approval. Neither is silently added.

Until the measurement procedure is qualified, the intended degradation/intervention inference is blocked. These diagnostics do not demonstrate that either scientific hypothesis is false, and CPU passing tests cannot resolve that qualification gap.
