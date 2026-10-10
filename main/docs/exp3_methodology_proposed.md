# Experiment 3: proposed methodology and execution readiness (PROPOSAL, 2026-10-10)

Status: **proposal for owner approval; not methodology.** Kept separate from the approved Exp1/Exp2 protocol
(`main/.claude/methodology-exp1-exp2.md`), which this document does not modify. Branch `claude/exp3-pilot-review`.

**Source labels** (every statement carries one):
- **[A]** approved methodology text;
- **[O]** design intent stated by the owner in task messages (2026-10-10), not yet committed as methodology;
- **[C]** Codex's implementation or documents (`codex/exp3-pilot-implementation` @ `142a7fc`): engineering, not
  authority;
- **[R]** my recommendation or addition;
- **⚑ Dn** an unresolved choice that **requires owner approval**. Nothing marked ⚑ is chosen here.

## 0. What is confirmed
- **[A] (the only approved Exp3 text):** "Actor_grad_cosine is retained as a mechanism diagnostic for Experiment 3
  rather than being treated as one of the primary Experiment 2 actor-side diagnostics" (`methodology-exp1-exp2.md:210`).
- **[O] Pilot 1:** critic-to-actor guidance comparisons on matched states, per-state gradients, causal
  interventions.
- **[O] Pilot 2:** passive learning from ordered U/I experience streams; shared vs independent RNG controls; an
  optional 2×2 design.
- **[O] Pilot 3:** frozen target-policy assignment, matched critic initialization, a fixed target evaluator, and
  isolated target-policy effects.
- **[C] population:** dog-run, humanoid-walk and two MyoSuite tasks, seeds 1–5 (described by Codex as
  "requested"; ⚑ D-POP).
- **Missing entirely:** the Exp3 hypothesis statement, primary outcomes, statistical analysis and success criteria
  (⚑ D-HYP). This document defines *estimands*, meaning what each comparison isolates, as [R]. It does not state
  or alter hypotheses.

## 1. Shared definitions [R, using Exp1/Exp2 terms [A]]
- **Cell:** (task, seed, scaled architecture) whose Exp1 parent produced an actual natural-trigger fork [A]. **F**
  is the complete fork state.
- **U arm:** the Exp1 parent continuing as the Exp2 control after its fork [A]. **I arm:** the Exp2 injected arm
  [A], starting from F plus the Nikishin injection with the frozen m (⚑ D-m, an Exp2 decision) and `seed = cfg.seed`
  [A, code].
- **Matched times:** fork + j·N/20 (j = 1…5, up to `arm_end = fork + 0.25N`). Both arms already save at these
  steps under the existing cadence [A, code]; they become available only through retention (§6).
- **Stream:** the ordered raw arrivals of one arm (transition, the source normalization in force, the update count,
  and the source agent key) from the fork onward [C + R].

## 2. Pilot 1: critic-to-actor guidance
**Intended design [O]:** compare the actor guidance supplied by the untreated and injected critics on matched states
using per-state gradients, and test causal interventions on that guidance.

**Proposed estimand [R]:**
- For a fixed actor A, state s and shared base noise ε (α and normalization fixed), compare g_U(s) and g_I(s). These
  are the critic-dependent parts of the SAC actor gradient at s under critics U and I. Because A, s, ε, α and the
  normalization are identical, any difference is attributable to the critic alone.
- Interventions replace one component of g_U by that of g_I (direction only, magnitude only, or full), with the
  entropy term unchanged. Each is applied as one matched optimizer step from the same actor and Adam state. This
  attributes the update-level effect of the critic difference to its direction and magnitude components.

Matching requirements [O + C, verified on CPU in the review]:
- same actor parameters and Adam state per comparison;
- same states (a common panel);
- same base noise across critics and actors;
- the same α rule and the same normalization rule;
- twin critics aggregated as min(Q1, Q2), as in the SAC actor loss [A, code];
- per-state measurements retained.

Unresolved (⚑):
- **D-P1a** which actors: U, I, and/or the fork actor (Codex measures all three);
- **D-P1b** intervention space: action-space dQ/da substitution vs per-state parameter-space substitution (Codex
  implements both), and the exact definitions of "direction" and "magnitude";
- **D-P1c** panel: source and size; the approved 256-pair fork panel vs new panels;
- **D-P1d** α rule (fork vs per-actor) and normalization rule (common fork vs per-artifact);
- **D-P1e** zero-norm policy (refuse vs keep the baseline);
- **D-P1f** times: the fork only, or the matched post-fork offsets;
- **D-P1g** outcome horizon: one-step descriptive (Codex) vs a longer guided continuation, which would be a new
  design;
- **D-P1h** chunk size, or per-state noise keys (results currently depend on chunk size);
- **D-P1i** storage: full per-state gradients vs summaries;
- **D-PREC** measurement precision (FP32 vs TF32);
- **D-CHK1** fork-time Check 1 on the approved panel against the arm's saved `check1_after.npz` [R].

## 3. Pilot 2: passive learning from ordered streams
**Intended design [O]:** passive learners trained only from ordered U/I experience streams, with shared and
independent RNG controls, and an optional 2×2 design.

**Proposed design [R, consistent with [O]]:**
- Every passive learner starts from F. The injected learner has F's critic replaced by the same injection as the I
  arm (⚑ D-m; the seed convention is enforced against the stream's recorded injection, C1).
- Learners never act. They receive the stream's arrivals in order and perform the source's recorded number of
  ordinary SAC updates per arrival.
- **The 2×2 design (optional, ⚑ D-P2a):** critic ∈ {uninjected, injected} × stream ∈ {U, I}.
  - *Critic contrast:* uninjected vs injected on the **same** stream isolates the effect of the injected critic
    under fixed experience, with exploration feedback removed.
  - *Stream contrast:* U vs I stream with the **same** starting critic isolates the effect of the experience
    distribution.
  - *Interaction:* whether the critic effect depends on the experience distribution.
  - **Stage A** (U stream only) gives only the critic contrast. Codex implements A and the 2×2 as "Stage B" [C].
- **RNG controls [O]:**
  - *Shared* (implemented [C]): both arms of a contrast use identical replay-sampling indices and agent key streams,
    so the comparison is paired. The review confirmed on CPU that, without injection, the arms stay bit-identical
    through every update.
  - *Independent* (**not implemented**): the same contrast with independent sampling and agent keys, plus a
    no-injection pair with independent keys. This gives the noise floor against which a paired critic effect is
    judged. Required if the owner wants effect sizes calibrated against SAC sampling noise (⚑ D-P2b). It is an
    engineering addition, not built here, as instructed.
  - *Source-key replica* (available [R]): with the captured per-arrival source key, a passive uninjected learner on
    the U stream reproduces the active U arm bit for bit (CPU, real Exp1 entry point). This is a positive control
    for passive fidelity (⚑ D-P2c: control only, or the primary key protocol).

Unresolved (⚑):
- **D-P2d** passive horizon (number of arrivals ≤ `arm_end`) and stream length;
- **D-P2e** normalization: the recorded source statistics vs frozen fork statistics;
- **D-P2f** evaluation protocol. [R]: align with Exp2's post-fork evaluation, every 1% of B with 10 episodes
  [A: approved Exp2 config, `fork.eval_every_fraction 0.01` and `eval_episodes 10`, decision F1], using evaluation-only environments and isolated RNG (already implemented [C]);
- **D-P2g** checkpoint cadence;
- **D-P2h** outcomes (returns, Exp2 actor diagnostics, critic plasticity probes?) and their analysis.

## 4. Pilot 3: frozen target-policy assignment
**Intended design [O]:** isolate the effect of the target policy by assigning frozen target policies, with matched
critic initialization and a fixed target evaluator.

**Proposed estimand [R]:**
- Two critic learners start from the **identical** critic and Adam state.
- They are fitted with the existing critic optimizer to Bellman targets computed on identical batches.
- The targets use one fixed evaluator network and a fixed α, with the shared base noise transformed by each frozen
  actor.
- The only difference between the learners is the target policy (U's actor vs I's actor), so the divergence of
  their fits isolates the target-policy pathway.
- No actor, α or evaluator update occurs. Terminal masks, γⁿ and min-of-twins targets follow SAC [A, code; C].

Unresolved (⚑):
- **D-P3a** starting critic (fork, U or I);
- **D-P3b** evaluator (fork, U or I target network), and any alternative evaluator;
- **D-P3c** α source;
- **D-P3d** batch source: fork replay (common data; [R] preferred for isolation) vs each arm's post-fork replay
  (needs replay retention);
- **D-P3e** update budget, batch size and checkpoint cadence;
- **D-P3f** final measurements: guidance on a panel, and which actor (Codex: fork actor only);
- **D-P3g** normalization rule;
- **D-PREC** precision of the final measurements.

## 5. Relation to `actor_grad_cosine` [A + R]
- **What it is [code]:** `actor_grad_cosine` is the mean pairwise cosine between *per-sample* SAC actor-loss
  gradients within one *training batch*. It uses the run's own actor and critic, independent per-sample noise, a
  1e-8 epsilon and cadence `actor_grad_cosine_every = 30`. Exp1/Exp2 already log it as a time series.
- **How Pilot 1 differs:** Pilot 1's cosine compares *two critics' guidance at the same state* (between critics),
  while `actor_grad_cosine` measures *coherence across states* under one critic (within a critic).
- **[R] Bridge:** compute, on Pilot 1's fixed panel, the mean pairwise cosine of each critic's per-state gradients
  (the `actor_grad_cosine` estimator on a common panel). That relates the logged U and I time series to Exp3's
  controlled measurements. Panel vs training batch and per-sample vs batched noise must be stated explicitly.
- **⚑ D-AGC:** whether the approved Exp3 role of `actor_grad_cosine` is (a) analysis of the existing Exp2 U/I time
  series, (b) the panel bridge above, or (c) both. No new capture is needed for (a): Exp2's retained logs contain it.

## 6. Required artifacts, evaluation and outcomes
| Artifact | Source | Pilot 1 | Pilot 2 | Pilot 3 | Today |
|---|---|---|---|---|---|
| F (complete fork state) + `panel.npz` + `check1_*` + `fork.json` | Exp1 parent | ✔ | ✔ | ✔ | written and retained [A, code] |
| I at the fork (post-injection save) + `check1_after.npz` | Exp2 arm | ✔ (fork-time) | — | ✔ (fork-time) | written, then **deleted**; retain |
| U and I agent states at matched offsets (lean) | Exp1 parent / Exp2 arm | ✔ (post-fork) | — | ✔ | written, then **deleted**; retain |
| U stream | Exp1 parent (control) | — | ✔ | — | **never written**; record |
| I stream | Exp2 arm | — | ✔ (2×2) | — | **never written**; record |
| Active outcomes (returns, Exp2 diagnostics, `actor_grad_cosine`) | both | reference | reference | reference | logged [A] |
| Evaluation | Pilot runs | descriptive one-step (⚑ D-P1g) | ⚑ D-P2f | none (critic-only) | — |
| Outcome measurements | Pilot runs | per-state g_U, g_I, norms, cosines, updates, policy changes | passive learner returns and diagnostics, critic/actor divergence | targets, losses, fitted-critic divergence, final guidance | ⚑ D-HYP |

## 7. Execution readiness per pilot
| Pilot | From Exp1 | From Exp2 | Independent runs | Gates |
|---|---|---|---|---|
| 1 at fork | F + panel (✔ already) | I-at-fork save retained + `check1_after` | none | D1 spec, D-P1a–i, D-PREC, D-m, E-tests |
| 1 post-fork | U matched states (retention during the parent run) | I matched states (retention during the arm run) | none, if captured; else separately authorized continuations (equal to history only if GPU training is deterministic) | as above + D-RET |
| 2 Stage A | F; U stream (recorded during the parent run) | — | none, if captured | D-m, D-P2c–h, D-RET, E-tests |
| 2 2×2 | F; U stream | I stream (recorded during the arm run) | none, if captured | + D-P2a, D-P2b (independent-RNG control not yet implemented) |
| 3 | F (batches from fork replay) | I matched/fork-time states | none, if captured | D-P3a–g, D-PREC, D-RET |

Engineering status [C + review]:
- implemented, with CPU parity: pairing, passive delivery, restart, retention, lean snapshots, key capture, C1/C3
  fixes;
- Codex's independent review is pending;
- **GPU unqualified.**

## 8. Can the source population be reduced? (alternatives; none selected)
Key fact [R]:
- recording starts only at a fork, and retention keeps only post-fork saves;
- so capture costs are paid only by parents that actually fork;
- holding or recording a parent that never forks costs nothing but scheduling;
- reductions therefore matter mainly for **pilot compute and storage**, and the realised population is bounded by
  natural forks, whose rate is unknown (the Block B dev run did not trigger).

| Option | Cells (upper bound) | Keeps | Loses / risk |
|---|---|---|---|
| R0 Full (as [C] describes) | 4 tasks × 5 seeds × 2 architectures = 40 | suite and width generality, seed replication | most compute and storage (about 420 GB retention, E) |
| R1 One architecture | 20 | all tasks and seeds; within-width causal contrasts intact | no width comparison of mechanisms; which width is an owner choice |
| R2 Fewer seeds (e.g., 3) | 24 (2 architectures) | tasks, widths | weaker replication; fork attrition may leave < 3 per task |
| R3 Fewer tasks (e.g., 1 DMC + 1 MyoSuite) | 20 | widths, seeds | less suite generality |
| R4 Pilot-tiered: Pilot 1 on all forked cells (cheap measurement); Pilots 2–3 on a subset | as R0 for Pilot 1 | every cell keeps the full guidance measurement; the expensive pilots run only on the subset | the subset must be chosen before seeing results (pre-registered) to avoid selection bias |
| R5 Capture all, run later: record and retain on all candidates; decide pilot execution after forks are known | as realised | no irreversible loss; reduction deferred until fork counts are known | storage for every forked cell (E); a pre-registered rule is still needed |

None of these weakens the *within-cell* causal comparisons. Every pilot contrast is paired inside a cell (same F,
states, noise, data). Reductions reduce generality and replication, not identification.

## 9. Staged A100 validation (minimal GPU time; engineering separated from science)
Measured full-width references:
- **[M]** job 22765261 @ `9d6a82d`, D4W1536 dog-run: 42.44 interaction steps/s (60-step window only), peak 3.5 GiB,
  compilation about 30 s, save 6.1 s / restore 2.5 s (synthetic 475k transitions).
- **[owner-reported, unaudited]** GPU resume gate PASS at `844e3c0` (job 22783981).
- CPU timings are **not** GPU measurements.

| Stage | Type | Needs | Content | Criterion | GPU time |
|---|---|---|---|---|---|
| V0 | engineering | no fork | `exp3_gpu_validation.py`, tiny exactness suites at production precision; repeat determinism | bitwise / `EXACTNESS_PASS` | U (CPU: 541 s per precision, not a GPU figure) |
| V1 | engineering | one real fork (the dev pilot's natural fork, or an owner-authorized forced dev trigger) | recording parity: a short continuation from F, recorded vs unrecorded | final states bitwise equal | about 24 s per 1,000 interaction steps per run (X, from 42.44 steps/s); length ⚑ |
| V2 | engineering | V1's fork | passive exactness (no-injection pair, repeat run, source-key replica vs retained active state) and restart (recorder interrupted between checkpoints; passive and target pilots) | bitwise | ≤ about 2 × V1 per learner pair (X) |
| V3 | engineering | V1's fork | full-width memory and time: one Pilot 1 chunk, a short Pilot 2 window, a few Pilot 3 updates | **report** peak device/host memory, time per unit, bytes per artifact | U |
| S1 | scientific sensitivity | V1 artifacts | every Pilot 1 field and the Pilot 3 targets at FP32 vs TF32 | **report only** → input to D-PREC | U (small: a panel-sized forward/backward) |
| S2 | scientific sensitivity (if D-P2b) | V1 artifacts | independent-RNG noise floor of passive contrasts (needs the not-yet-built control) | **report only** | about passive-window × 2 learners (X) |

Ordering: V0 → V1 → (V2, V3, S1) → S2. A failure at V0, V1 or V2 stops the plan:
- repeat not bitwise → NONDETERMINISTIC, an owner decision;
- repeat bitwise but an exactness check fails → an engineering defect;
- no tolerances or thresholds are chosen.

## 10. Approval checklist, by deadline (prioritized)
**Before any designated Exp1 source parent runs:**
1. **D-POP:** tasks (confirm dog-run and humanoid-walk; choose the 2 MyoSuite tasks), seeds, architecture(s).
   Until then, hold the scaled parents of all 6 candidate tasks (60 parents).
2. **D-CAP:** enable recording and retention on designated parents (an operational amendment: outputs only;
   training unchanged).
3. **D-RET:** retention window and offsets, replay omitted or kept, U stream length, storage allocation.
4. **V0 + V1 passed** (recording parity on the GPU). If V1 fails, capture cannot be used in production, and
   designated parents need either unrecorded runs plus later continuations, or a redesign.
5. Codex's engineering review of the capture code accepted.

**Before Exp2 forks (injected arms) of designated cells:**

6. **D-m** frozen (an Exp2 decision, prerequisite for I).
7. Recording and retention enabled on those arms with the same settings (I stream, I at fork, I matched states).

**Before Pilot 1:**

8. **D1** commit the Exp3 specification, including **D-HYP**.
9. **D-P1a–i**, **D-PREC** (after S1), **D-CHK1**, **D-AGC**.

**Before Pilot 2:**

10. **D-P2a** (2×2 or Stage A), **D-P2b** (independent-RNG control; requires implementation), **D-P2c–h**.
11. V2 passed.

**Before Pilot 3:**

12. **D-P3a–g**, **D-PREC**.

**Population and reduction (any time before pilot execution, ideally pre-registered before forks are known):**

13. Choose among R0–R5 (§8), and a rule for cells without forks.

## 11. Separation from approved Exp1/Exp2
- No Exp1/Exp2 training, trigger, injection, Check, evaluation, statistics or budget rule is changed by anything
  here.
- The only Exp1/Exp2-facing items are:
  - D-CAP and D-RET (additional outputs of designated runs; bit-identical trajectories on CPU, GPU check V1);
  - scheduling holds;
  - D-m, which is already an open Exp2 decision.
- Proposed Exp3 methodology, once approved, should live in its own document (e.g. `main/.claude/methodology-exp3.md`),
  not as amendments to `methodology-exp1-exp2.md`.
