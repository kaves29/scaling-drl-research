# Exp1/Exp2 decision memo: the two failed calibration gates (2026-10-12)

Methodology review only. Base: `integration/exp12` @ `844e3c0`. Evidence read from `codex/exp12-production-readiness`
@ `3d75a05` (`exp12_production_decision_2026_10_11.md`, `exp12_remaining_qualification.md`). That branch is not
modified.

What this review did **not** do:
- change any threshold, method, configuration or inclusion rule;
- submit any job;
- repeat any capture, null run or completed audit.

**Labels:**
- **[M]**: the approved methodology or pre-registered rules;
- **[O]**: an owner decision;
- **[E]**: my recommendation.

The amendment text referred to below is in [exp12_proposed_amendment.md](exp12_proposed_amendment.md). It is
**PROPOSED, not approved**.

## Answers
| Question | Answer |
|---|---|
| What can launch now under the approved protocol? | **No confirmatory Exp1 or Exp2 run.** The positive control must come "before the main runs of Experiment 1" [M]. Identity must pass on CUDA "before any Experiment 1 grid run" [M (b), (p)]. Neither has happened. **Allowed now:** the queued development pilot 22800749, `positive_control.py` on its natural fork, and the six identity qualification cases. All three are protocol-required qualification work, not new diagnostics. |
| What blocks the rest? | Positive control and m; the six identity cases; the grid GPU model; and owner dispositions of the two failed gates. The gates block **only** D4W1024 parents (null) and the 5 hopper-hop D4W1536 parents (range); they do not block the other 125 parents. |
| What needs your approval? | D1 null disposition, D2 hopper disposition, D3 grid GPU model, D4 accepting the pilot-source positive control (the science code is identical; see §3). D5 (optional) per-suite staging of identity. |
| Fastest defensible path | §5. Owner decisions take about an hour and need no GPU. After that, only the protocol-required GPU work remains: the pilot, then a probe-only positive control, plus six identity cases run in parallel. |
| Is any extra GPU diagnostic indispensable? | **No.** The hopper curve capture and a further null calibration would not change either decision (§1.4, §2.4). |

## 1. D4W1024 fresh-null gate (13/100 and 6/100 against ≤ 5/100)
### 1.1 Classification
| Kind | Finding |
|---|---|
| Implementation defect | **None established.** Both JSONL files were re-derived exactly by Codex, including exact enumeration of all 5^5 resamples. The probe and trigger code is unchanged from `b4a90cb` to `3d75a05`. |
| Statistical uncertainty | **Small, and it does not rescue the gate.** Pooled, the two samples give 19/200 = 0.095, with Clopper–Pearson 95% CI [0.058, 0.144]. One-sided binomial p = 0.006 against 0.05. The samples are not detectably different (Fisher p = 0.146; pre-specified X p = 0.023 > 0.01). So the per-check rate exceeds 5% even after allowing for sampling error. |
| Diagnostic limitation | (a) The null is **dog-run only**, but the threshold applies in all 13 tasks. (b) Fresh pairs do not measure two consecutive checks on one evolving critic, so the run-level false-fork rate is **not identified**. Under independence it is 0.060 at p = 0.06, 0.140 at p = 0.095 and 0.242 at p = 0.13 (illustration only). (c) The exploratory shuffle p ≈ 0.003 suggests pair or replay clustering; the cause is unknown. |
| Design decision | The pre-registered NULL rule requires **no amendment**. Its consequence is "the lead decides whether to adopt the 95th-percentile threshold at that size" [M]. Both adopting it and declining it are inside the rule. |

### 1.2 What the gate protects
- **Exp1 primary endpoint:** not protected by this gate. The endpoint is the final-check L contrast, stratified
  bootstrap, 65 runs per architecture [M]. It never uses the threshold.
- **The parent's own trajectory:** not affected. The control continuation is bit-identical to a fork-disabled run
  (tested on CPU; identity on CUDA is pending).
- **What it does protect:** the Exp2 premise that forks occur on *degraded* critics, and the f*_run distribution,
  which is descriptive.
- **Risk of proceeding at threshold 0:**
  - some D4W1024 forks are taken on critics that have not lost plasticity, and forks occur earlier;
  - each fork's injected-vs-control comparison stays a valid paired causal contrast, but the pooled Exp2 effect at
    D4W1024 is diluted toward "injection into a healthy critic";
  - this biases Exp2 toward the null, so it is conservative for the propagation claim.
- **Why the decision must come before D4W1024 parents:** forks are irreversible. Only LATEST and the fork state are
  kept.

### 1.3 Options (at most three)
| # | Option | Rigor | Calendar | Minimum additional evidence | Main risk |
|---|---|---|---|---|---|
| N1 | **Adopt the rule's null-calibrated threshold for D4W1024 only:** `np.percentile(L, 95)` over the pairs' L. Fix the sample *before computing it*; I recommend pooling all 200 pairs, because selecting one sample would be outcome-dependent. D2W512 is unaffected; D4W1536 stays at 0. | Highest. It is the pre-registered remedy and uses the pre-registered estimator. | CPU minutes plus a config change in a reviewed commit | The pooled value, computed on CPU from the two hashed JSONL files (`89846b78…`, `d6dc1269…`). **Missing:** neither value has been reported to this review. | Lower sensitivity at D4W1024: the lower bound must clear the 95th percentile of null point estimates. Sensitivity there is unverified, because the positive control is D4W1536. Possible outcome: few or no D4W1024 forks. That is a valid null result [M], not a validity failure. The threshold is estimated on dog-run and applied to all tasks. |
| N2 | **Decline adoption and keep threshold 0, with a stated limitation.** Pre-register an Exp2 sensitivity subset: D4W1024 forks whose fork-check `ci_low` also exceeds the pooled p95. This can be computed post hoc from the ledger with no reruns. | Medium. It is inside the rule, but runs with a known per-check miss of about 9.5%. | None | None before launch | Diluted Exp2 effect at D4W1024. The sensitivity subset is "forks that also qualified at their own check", not "forks that would have happened under the stricter rule". |
| N3 | **Hold D4W1024 parents** until a third, unchanged, final-source 100-pair null run exists (Codex's conservative route). | Lowest information value | One GPU job plus review; D4W1024 (65 parents, a third of Exp1) waits | One 100-pair GPU run | **It cannot change the decision.** The pooled count is already 19, and pooled ≤ 5% over 300 pairs would need ≤ 15 in total. A third sample read on its own would pass with probability only 0.078 if the rate is 0.095. That pass would be a selection artifact. |

**[E] Recommendation: N1** (pooled, with the sample fixed in writing before computing). Fall back to N2 if you
judge the sensitivity risk worse than the dilution risk. N3 is not recommended.

### 1.4 GPU diagnostics
**None indispensable.** Exactly what is missing for N1:
- the pooled p95 value;
- a written statement of which pairs it uses.

Both are CPU work.

## 2. Hopper-hop D4W1536 range gate (P/b = 0.6024 against ≥ 0.9)
### 2.1 Classification
| Kind | Finding |
|---|---|
| Implementation defect | **None established.** Replay, teacher targets, normalization and arithmetic were independently verified; all 15 round scores and the IQM recompute exactly. |
| Mechanism | Finite-budget underfitting of the fresh D4W1536 critic. AdamW's first fresh-state steps shift D4 outputs, the loss then plateaus until about step 630, and the 1,000-step budget ends while all five curves are still falling. It is pool-dependent: 0.987 at 1,600, 0.845 at 6,400, 0.602 at 25,600 (draws per item 160 → 40 → 10). Hopper D2W512 and D4W1024 pass. |
| Diagnostic limitation | (a) **The range gate was measured in only 2 of 13 tasks:** dog-run (Block A) and hopper-hop (Block B). The other 11 tasks are unmeasured at every size, so "hopper is the only failing cell" is **not known**. (b) A fresh-critic check cannot show the **direction** of any bias in L. It is plausibly toward smaller L (an under-fitted reference), but **not demonstrated**. (c) (w) itself says this check "cannot establish sensitivity"; sensitivity comes from the positive control. |
| Design decision | (w) "gives no next step" [M]. Any disposition is an owner decision. |

### 2.2 What the gate protects
- **Exp1 primary endpoint: yes, partially.** Hopper is 1 of 13 strata in the D4W1536 stratified bootstrap. The
  gate guards the meaning of L there: whether L measures capacity or fitting speed.

Two facts limit the damage:
- the methodology already normalises each run to its own fresh reference "because critics of different sizes fit
  the probe differently from the start" [M];
- 0.60 lies inside the originally pre-registered 10–90% window. The instrument is neither at floor nor at ceiling
  (the original rule is recorded as information only in (w)).

Exp2 is affected too: fork placement in that cell may be later or rarer.

**New, decision-relevant fact (verified in code at `3d75a05`):** every parent already runs the gate's exact
measurement at check 0.
- `run_probes.capture_fresh` calls `run_probe` at `check_index` 0 on the untrained critic, before the first update,
  with the buffer at warm-up length. That is the same call `probe_fresh_checks.py --mode range` makes.
- It writes per-round P, b, the final loss and the **full 1,000-step loss curves** (`check_00.npz`,
  `probe_curves.npz`).

So the configured-pool P/b, and its curves, will exist for **every cell and seed** at no extra GPU cost. It is
measured before any training, so it is independent of the outcome.

### 2.3 Options (at most three)
| # | Option | Rigor | Calendar | Minimum additional evidence | Main risk |
|---|---|---|---|---|---|
| H1 | **Per-cell, pre-treatment validity rule** (proposed amendment A1): run every parent as specified. Before any L is unblinded, classify each cell × size by its parents' check-0 IQM(P)/IQM(b) against the approved 0.9. The registered primary analysis is unchanged. A pre-registered sensitivity analysis drops failing strata. Alternatively, you may make the exclusion primary. | Highest. It applies (w)'s own threshold uniformly to all 39 cells, not only to the 2 tasks someone happened to measure. The criterion is outcome-blind. | None before launch; hopper runs in normal order | None before launch. The data come from the parents' own check 0. | About 5 parent-runs of compute (plus any forks) are spent on a cell that may end up excluded. |
| H2 | **Hold the 5 hopper D4W1536 parents** and schedule them last; decide before the queue reaches them. | Neutral: it only defers the decision. | None, if decided before the queue reaches them; the full Exp1 endpoint needs these 5 runs | None | It leaves the 11 unmeasured tasks unaddressed. |
| H3 | **Exclude hopper D4W1536 now** (design decision). The D4W1536 contrast uses 12 strata (60 runs). | Low–medium. It is fast, but it applies the gate selectively: other cells might fail and would stay in. | None | None | Inconsistent application; a smaller population. |

A probe-budget amendment (more steps or a smaller pool) is **not** offered as an option:
- it re-opens the range and null gates at every size and the positive control;
- it breaks comparability with Blocks A and B;
- it needs substantial GPU work.

That is disproportionate for 5 of 195 parents.

**[E] Recommendation: H1**, with the registered primary analysis unchanged. Use H2 for the 5 hopper parents only if
you are not ready to approve A1 today.

### 2.4 GPU diagnostics
**None indispensable.**
- The proposed same-settings curve capture (`capture_exp12_range_evidence.py`) would re-establish a mechanism that
  is already established (the curves are still decreasing). Under any of H1–H3 it would not change the
  disposition.
- Under H1 the parents themselves record curves for every cell.
- Nothing is missing for H2 or H3.
- For H1, the only missing item is approval of A1's wording.

## 3. Can Exp1 parent training be separated from Exp2 qualification?
**Already separated by design:**

| Item | Needed before Exp1 parents? | Why |
|---|---|---|
| m | **No** | The fork state is m-agnostic; injection happens in the arm job [M (d)]. |
| Check 1 tolerance, source-provenance guard, arm GPU-model check | **No** | They run inside each arm job. |
| Exp2 arm scheduling and census | **No** | Arms are separate jobs, submitted after m is frozen. |

**Not separable without a change:**

| Item | Needed before Exp1 parents? | Why |
|---|---|---|
| Trigger threshold (D4W1024) | **Yes**, before D4W1024 parents | The fork is irreversible; only the fork state is kept. |
| Grid GPU model | **Yes**, before forking parents | Arms and identity must run on the fork's model [M (p)]. |
| Identity (six cases) | **Yes**, before the grid [M (b)] | It also protects Exp1 itself: post-fork Exp1 data come from the *restored* control. |
| Positive control | **Yes**, before the main Exp1 runs [M] | It validates probe sensitivity for the Exp1 endpoint. If it fails, the probe may change and every parent reruns. |

**Possible further separation, not recommended:**
- *Full threshold decoupling:* keep the complete state at every eligible check (1–19) of D4W1024 parents, and take
  forks later under whichever rule is chosen.
  - It needs a production code change, its own qualification, and roughly 19 × ~1.5 GB per parent (about 1.9 TB
    for 65 dog-shaped parents; illustrative).
  - Deciding N1/N2 is faster.
- *Identity staging (D5, optional, amendment A2):*
  - D2W512 never forks, so identity cannot affect its data.
  - Each suite's forking parents could start once that suite's two cases pass. DMC and MyoSuite would then not
    wait for the HumanoidBench environment on Delta.
  - The gain is hours if the six cases run in parallel; more if HumanoidBench lags.
- *Positive-control ordering:* **do not amend.** If the pilot triggers, the positive control is probe-only, a
  matter of hours. If it never triggers, the protocol says stop and consult, and an early grid would be the wrong
  bet.

**Positive-control source (D4):** pilot 22800749 runs `7d0883c`. I verified on CPU with `git diff` that
`experiments/exp12/{probe,trigger,injection,m_selection,run_probes,fork}.py`, `experiments/exp1.py`,
`scripts/positive_control.py` and `configs/base_exp12.yaml` are **byte-identical** across `844e3c0`, `7d0883c` and
`3d75a05`.

From `7d0883c` to `3d75a05`, the only experiment-code change is the two-line source guard in `exp2_arm.py`. The
rest is scheduler and launch tooling.

Consequences:
- A natural pilot fork gives positive-control and m evidence for the same science code. Approve its use once the
  final grid SHA is confirmed to keep these paths identical (one `git diff`, CPU).
- Any **arm** restored from the pilot fork must still run on `7d0883c`, as the new guard enforces.

## 4. Decisions requiring your approval
| ID | Decision | Options (recommended first) | Blocks | GPU needed to decide |
|---|---|---|---|---|
| D1 | D4W1024 null disposition | N1 pooled p95, N2 keep 0 with limitation, N3 hold | 65 D4W1024 parents | No |
| D2 | Hopper D4W1536 range disposition | H1 (approve A1), H2, H3 | 5 hopper D4W1536 parents (H1 also governs analysis of all cells) | No |
| D3 | Grid GPU model | infrastructure choice | all forking parents and identity runs | No |
| D4 | Use the pilot's positive control and m (same science code) | approve once the final SHA keeps those paths identical / require a rerun on the final source | Exp1 GO, then Exp2 | No |
| D5 | (optional) Identity staging per suite (A2) | approve / keep the "all six before any grid run" rule | calendar only | No |

m itself is frozen mechanically by (d) from the positive control's recovery. Your action is to record it [M].

## 5. Fastest defensible path to full launch
1. **Now (no GPU):** decide D1–D4 (and D5).
   - For N1: compute the pooled p95 on CPU and commit the D4W1024 threshold in a reviewed config change.
   - For H1: approve A1 before any parent starts.
2. **GPU, protocol-required, in parallel:**
   - (a) pilot 22800749 → natural trigger → `positive_control.py`, probe-only on the fork → m recorded;
   - (b) six identity cases on the D3 model, ≤ 24 A100-h by Codex's cold cap, in parallel where quota allows.
   - If the pilot never triggers: **stop and consult** [M (e)]. This is the one scientific risk that can stall
     everything, and nothing in this memo works around it.
3. **Exp1 GO** when (a) and (b) pass:
   - all D2W512 and D4W1536 parents (hopper per D2);
   - D4W1024 parents under the D1 rule;
   - the Exp3-candidate parents last, per the earlier audit.
4. **Exp2 GO per fork:** after m is recorded, each arm runs on its fork's GPU model and source.

Engineering items that are **not** scientific prerequisites: G1 shared-mount claim, G2 packing, the W&B archive and
allocation sizing. They gate the scheduler and the allocation, not validity.

## 6. Exactly what is missing
- Pilot 22800749's outcome: queued, unknown.
- The pooled (or per-sample) `would_be_null_threshold_p95_of_L` values.
- The grid GPU model.
- Any identity receipt on the final source.
- Check-0 P/b for the 11 unmeasured tasks (produced by parents under H1).
- The direction of any hopper L bias (not identifiable from fresh-only data).
