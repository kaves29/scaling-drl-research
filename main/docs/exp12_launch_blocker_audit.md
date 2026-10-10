# Exp1/Exp2 launch-blocker audit (2026-10-10)

Scientific audit for the launch deadline. Base: `integration/exp12` @ `844e3c0`; the latest Codex status is read
from `codex/exp12-gate-attempt-accounting` @ `07721bf`. Nothing here changes methodology, thresholds,
hyperparameters or configurations, and no job was submitted. Codex's dog-run D4W1536 development pilot is not
duplicated.

**Labels:** **[M]** = required by the approved methodology (`.claude/methodology-exp1-exp2.md`) or pre-registered
rules (`docs/exp12_decisions.md`); **[E]** = engineering recommendation; **[O]** = owner decision with no
pre-specified answer.

## Verdict
- **Exp1: NO-GO today.** Earliest defensible GO: after (1) the development run's natural trigger, the positive
  control and m frozen; (2) the six mandatory identity forks pass on the grid's GPU model; (3) dispositions are
  recorded for the two failed calibration gates. (3) needs owner decisions, not GPU time.
- **Exp2 injected arms: GO per fork** once m is frozen, identity has passed, and the arm runs on the fork's GPU
  model. Nothing extra is required from Exp1 parents beyond what they already save.

## 1. D4W1024 null-trigger calibration (dog-run)
**What exists:**
- 13/100 at `b4a90cb` (Block B, audited: 300 rows valid).
- 6/100 at `9d6a82d` (seed 991, job 22779974; owner-reported, raw rows **not yet audited**).
- Exact 95% CIs: [0.071, 0.212] and [0.022, 0.126] (Codex memo).
- D2W512 3/100 and D4W1536 5/100 pass the same rule.
- The single-critic probe and trigger code is unchanged from `b4a90cb` to `844e3c0` (`probe.py`'s change only adds
  twin pairing; `trigger.py` is unchanged).

**What it establishes:**
- *The gate:* both samples fail the pre-registered per-check rule ("exceeds 5% at any size") [M]. That rule's only
  consequence is that the lead decides whether to adopt the 95th-percentile threshold at that size [M]. Nothing is
  automatic.
- *Per-check vs fork:* the gate is **per check**; the fork needs **two consecutive** firing checks (l) [M]. Under
  independence (illustration only), any adjacent pair among 19 checks occurs with probability 0.042 at p = 0.05,
  0.060 at p = 0.06, and 0.242 at p = 0.13. Real checks are dependent, so the existing data do **not** identify the
  run-level false-fork rate. The run-level rate is not the gated quantity in any case.
- *Mechanism:* no implementation defect has been found. The pre-specified seed-991 readings (cross-seed X test,
  shuffle test; `scripts/sci_investigation/null_followup.py --replication`) are **unrun**, because the raw rows are
  on Delta.

**Launch impact:**
- *Who is affected:* the threshold affects only fork placement. The Exp1 primary endpoint (final-check L contrast)
  does not use it.
- *Why decide early:* forks happen **inside** the Exp1 parent, and only LATEST plus the fork state are kept. A
  later threshold change for D4W1024 would force parent reruns, so the disposition must precede D4W1024 parents.
- *Unaffected sizes:* D2W512 (never forks) and D4W1536 (gate passed) are not affected.

**Cheapest valid step:** copy the seed-991 `null_pairs_D4W1024.jsonl` and summary, and run `null_followup.py`. It is
CPU-only and takes minutes, needs no GPU and adds no new criterion. Then the owner chooses among the options in
`exp12_null_threshold_decision.md`.

## 2. Hopper D4W1536 range, P/b = 0.602 vs ≥ 0.9
**Definition:** P = b − final pool MSE after the fixed probe budget (5 × 1,000 AdamW steps, pool 25,600); P/b is the
IQM over rounds. Amendment (w) [M] replaced the original 10–90% window with **P/b ≥ 0.9 at every critic size**, and
"gives no next step" on failure (decisions log, Block B rules #5).

**Evidence:**
- Block B A100: D2W512 0.933, D4W1024 0.902, D4W1536 **0.602** (rounds 0.559–0.649).
- Reproduced in CPU FP32: 0.540 and 0.730.
- Mechanism: the first fresh-state AdamW steps shift D4 critics' outputs by several units; the fit plateaus at
  predicting the mean until about step 630 on hopper (about 350 on dog-run). The budget ends while the loss is
  still falling. It is not TF32 and not a computation defect.
- The single-critic path is unchanged to `844e3c0`.

**Interpretation:**
- *What the number means:* the number is correct; it measures what (w) defines. There is no legitimate
  re-reading that makes it pass, and nothing is waived here.
- *Consequence:* in hopper D4W1536 the fresh-critic reference under-fits, so L is biased toward ≤ 0 in that cell
  (fewer or later forks, a smaller Exp1 contrast). Other cells are unaffected by this mechanism. Hopper
  D2W512/D4W1024 pass, and dog-run D4W1536 converges (CPU).

**Launch impact:** an owner disposition is required [O]. The options are those already documented: accept with a
stated limitation; hold or exclude the cell; or a probe-budget amendment, which would re-open the null and range
gates for every size. The **scope** of the hold is also the owner's call. My recommendation [E]: hold only the 5
hopper-hop D4W1536 parents. No GPU run is needed to decide.

## 3. Natural-trigger positive control
- **Requirement [M]:** "run before the main experiment, and before the main runs of Experiment 1". D4W1536 dog-run,
  development seed outside 1–5 (e). If it never triggers, stop and consult; the trigger is never loosened.
- **Measurements [M]:**
  - L_trigger > 0 and probe noise < 0.10 in recovery units (q);
  - recovery for last/half/all, with m = the smallest within 0.10 of the best (d);
  - the one-time shared-offset sign check (c).
- **Smallest sufficient evidence:** Codex's development pilot, **if it triggers naturally**, supplies the
  degraded checkpoint (fork state). `scripts/positive_control.py` then needs only probe evaluations of the degraded
  critic and three injected copies; no further training. Nothing else is required.
- **Collect from the pilot:** the trigger check, the fork state, Check 1 of the in-process control restore at full
  width (it also exercises the parent's fork path), probe overhead and timing, and the shared-offset check.
- **Risk:** the Block B development run reached no trigger before its 6-hour timeout (that was before the
  throughput fixes). With no natural trigger, the methodology says stop and consult. That would block Exp1.

## 4. What must be fixed before the first Exp1 parent (so Exp2 needs no parent reruns)
| Item | Why | Source | Status |
|---|---|---|---|
| Trigger: threshold 0, two consecutive checks, eligible checks 1–19 | the fork happens inside the parent and cannot be redone | [M] (l), config | frozen; D4W1024 waits on §1 |
| Fork plan: horizon 0.25B, total ≤ 1.2B, evaluation every 1% B with 10 episodes, complete-state save at the exact check | defines the Exp2 starting state | [M] (b), F1 | frozen in config |
| **The grid's GPU model** | arms must run on the fork's model (p); identity must run on it | [M] (p) | **owner/infrastructure choice needed** |
| m | **not needed**: the fork state is m-agnostic; injection happens in the arm job | [M] (d) | frozen later by the positive control |
| Parent checkpoint retention | Exp2 needs only the fork state, which is always kept | code | sufficient |

## 5. Identity qualification
- **Mandatory [M]:**
  - (b) "must pass on CUDA for every architecture and suite before any Experiment 1 grid run";
  - (p) D4W1024 and D4W1536 in each suite on the grid's GPU model. That is **6 cases** (2 forking sizes × DMC,
    MyoSuite, HumanoidBench);
  - the arms stay identical over 1,000 further training steps (Identity fork, methodology).
- **Redundant or engineering-only [E]:**
  - running both cold and warm compilation caches is not required by the methodology. Run identity once, in the
    cache mode the grid will actually use; that also avoids the D4W1536 `identity_warm` case currently held by the
    300 s internal cap;
  - the 300 s cap is an engineering budget, not a methodology rule. The owner can authorize an adequate allocation
    instead.
- **Reusable evidence:**
  - none satisfies (b) by itself: Block B's identity forks all timed out;
  - the GPU resume gate PASS at `844e3c0` (job 22783981, owner-reported) and Block B's in-process Check 1
    (bit-exact, tiny, old source) support but do not replace it.
- **Minimum GPU:** 6 identity runs on the grid model, each a short forced-trigger development parent (validation
  only) plus two arms of 1,000 steps. HumanoidBench needs `venv_hb` on Delta.

## 6. Exp3 capture
- Exp3 is **not** part of the approved Exp1/Exp2 methodology.
- If the owner keeps the intended Exp3 sources (scaled parents of dog-run, humanoid-walk and the MyoSuite
  candidates: up to 60 parents), running them unrecorded **irreversibly** loses the U-arm streams and matched
  post-fork states. A later continuation is new evidence, not the historical run.
- Capture code (`claude/exp3-retention-integrity-fixes` @ `459e700`) awaits independent review and GPU gate A.
- **Not a blocker for the other 135 parents.** [O]: Exp3 scope; [E]: schedule those 60 last.

## Decision table
| Category | Item | Exact evidence | Cheapest valid route | Already satisfied? |
|---|---|---|---|---|
| **HARD before Exp1** [M] | Positive control + m | natural trigger (dev seed, D4W1536 dog-run); L_trigger > 0; noise < 0.10; recoveries; m frozen | Codex's pilot + `positive_control.py` (probe-only after the fork) | No |
| **HARD before Exp1** [M] | Identity forks | 6 cases bit-identical over 1,000 steps on the grid GPU model | one cache mode; forced dev forks; adequate allocation | No |
| **HARD before Exp1** [M]+[O] | Null gate failed (D4W1024) | owner disposition per the NULL rule | CPU analysis of seed-991 raw rows, then decide | Evidence yes; decision no |
| **HARD before hopper D4W1536 parents** [M]+[O] | Range gate failed | owner disposition and scope | decision only | Evidence yes; decision no |
| **HARD before forking parents** [M] | Grid GPU model fixed | named model | infrastructure choice | No |
| **HARD before Exp2 arms** [M] | m frozen; fork's GPU model; Check 1 injected ≤ 64 eps (built into the arm) | positive-control result; per-fork device record | already on the Exp1 critical path | No (follows the positive control) |
| **CAN RUN IN PARALLEL with Exp1** | Exp3 capture review + GPU gate A; HumanoidBench environment on Delta; Exp1/Exp2 analysis tooling; seed-991 X/shuffle readings if D4W1024 parents are held | — | — | — |
| **OPTIONAL / DEFER** | cold + warm cache identity; TF32 sensitivity of diagnostics; Exp3 pilot decisions; packing/throughput tuning; dog-run range re-measurement (single-critic path unchanged) | — | — | — |

## Critical path (no new acceptance criteria)
1. **Today, no GPU:**
   - owner decisions: null (after the CPU raw-row readings), hopper scope, the grid's GPU model, Exp3 scope;
   - copy the seed-991 rows and run `null_followup.py`.
2. **In parallel, GPU:**
   - (a) Codex's development pilot, then the positive control, then m;
   - (b) the 6 identity cases on the grid model;
   - (c) if Exp3 is kept, capture gate A.
3. **Exp1 GO:**
   - D2W512, plus D4W1536 for every cell except hopper (subject to its disposition);
   - D4W1024 once the null disposition is recorded;
   - Exp3-candidate scaled parents last, after capture qualifies or the owner releases them.
4. **Exp2 GO per fork:** as forks appear, on the same GPU model.

**Note:** if the development run does not trigger naturally, the methodology requires stopping and consulting before
Exp1. Plan the pilot's allocation so it reaches the eligible checks, rather than relying on a timeout.
