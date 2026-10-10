# Exp3 development pilots: methodology reconciled with the recovered request (PROPOSAL, 2026-10-10)

Supersedes `exp3_methodology_proposed.md` @ `0b70602` and the S1–S9 table in `exp3_owner_decisions.md`. It is a
proposal for the owner and changes no approved Exp1/Exp2 rule.

**Sources:**
- **[REQ]** the recovered original request, `docs/exp3_original_user_request.md` (SHA-256 `3a444073…ab10`). It is
  the authority for Exp3 *design constraints*. It describes **development pilots**, not the final Exp3 experiments,
  and it deliberately leaves numerical settings unspecified ("do not invent panel size, update duration, checkpoint
  cadence or intervention normalization").
- **[A]** approved Exp1/Exp2 methodology and decisions.
- **[C]** current implementation.
- **[R]** my recommendation.
- **⚑** a choice that needs owner approval.

## 1. Discrepancies between my proposal (`0b70602`) and [REQ]
| # | My earlier proposal | [REQ] says | Correction |
|---|---|---|---|
| 1 | **Pilot 1 actors were an open choice** (D-P1a); earlier S3/D4 recommended U and I as primary and F only as a "secondary reference" | "evaluate Q_U and Q_I under: untreated actor π_U, injected actor π_I, original fork-time actor π_F" | **Requirement, not a decision.** All three actors are primary; each gives one within-actor comparison of Q_U vs Q_I. My recommendation is withdrawn. [C] already implements all three. |
| 2 | Pilot 1 timing open: fork only vs post-fork (D-P1f) | "At matched post-fork checkpoints … Include fork-time equivalence assertions" | Post-fork measurement is **required**. The fork appears only as an *equivalence assertion*. Open: **which** post-fork offsets (checkpoint cadence is an unspecified number). |
| 3 | Pilot 1 storage: full per-state gradients **or summaries** (D-P1i) | "Preserve per-state/per-action measurements before aggregate statistics" | Per-state preservation is **required**; "summaries only" is not admissible. Only the storage location and format remain logistics. |
| 4 | Measurement chain not listed as a requirement | Required chain: dQ/da → critic-derived actor parameter gradients → full SAC actor gradients → actual optimizer updates → policy-function changes; direction and magnitude; "optionally local variation where scientifically justified" | The chain, direction and magnitude are **required** ([C] implements them). Local variation is an **optional** ⚑. |
| 5 | Panel choice conflated with the 256-pair Check 1 fork panel (D-P1c) | "One common prespecified state panel, with optional source-specific panels" | The common panel is a **Pilot 1 panel (required structure; size and source ⚑)**, distinct from the Check 1 fork panel used by the fork-time equivalence assertion. |
| 6 | Within-actor α and normalization treated as one open rule (D-P1d) | "match states, reparameterization noise/actions, alpha and actor optimizer state" within each actor | Within-actor matching is **required**. Which α value (fork vs per-actor) and which normalization rule remain ⚑. |
| 7 | Intervention definitions open | The set {direction-only, magnitude-only, full, sanity} is **required**; "intervention normalization" is explicitly unspecified | The set is required; the space (action vs per-state parameter) and normalization (including the zero-norm policy) are ⚑. |
| 8 | **Independent-RNG control labelled [O] as an original Pilot 2 element** (D-P2b) | Pilot 2 requires *matched* "replay sampling keys between passive learners"; no independent-RNG control | **Re-categorized** as a *later optional extension*, not original design. It is not implemented, and it is not required for Stage A or B. |
| 9 | Pilot 3 treated like a full pilot with seven open choices | "OPTIONAL … opt-in diagnostic, not an automatically launched experiment" | Pilot 3 is **optional**. Its choices are needed only if enabled. |
| 10 | Recommended Pilot 3 guidance on the fork actor **and** on U and I | "Optional guidance to **a common actor**" | Withdrawn: guidance targets one common actor (which one ⚑). |
| 11 | Measurement precision framed as a required decision, with FP32 recommended | "Preserve current scientific precision … unless a pilot explicitly requires a documented diagnostic intervention" | The **default is the current precision**: production matmul precision, with the existing FP32 diagnostic policy, which [C] follows. An FP32-measurement rule would be a *documented diagnostic deviation*. Optional ⚑, informed by the TF32 sensitivity measurement. |
| 12 | Missing hypothesis, statistics and success criteria treated as a pilot prerequisite (D-HYP) | Development pilots; no statistics requested | **Not a pilot prerequisite.** Needed only for the final Exp3 experiments. |
| 13 | The `actor_grad_cosine` bridge presented as a Pilot 1 element | Not mentioned in [REQ] | It remains an [R] optional addition. The only link is [A] l.210 (actor_grad_cosine is an Exp3 mechanism diagnostic); the existing Exp2 time series need no capture. |
| 14 | MyoSuite "pair" | "Both existing MyoSuite environments" | The request assumed there are two; the repository registers **four**. A ⚑ choice: select two, or redefine the scope. |
| 15 | Fork-time "I at fork" Check 1 recommendation (D-CHK1) | "Include fork-time equivalence assertions" | Consistent: the assertion is required; using the approved panel and `check1_after.npz` is how to meet it [R]. |

## 2. Reconciled design (requirements [REQ] → implementation status [C])
**Pilot 1, actor guidance across three actors:**
- *Design:* at matched post-fork checkpoints, Q_U vs Q_I under each of π_U, π_I and π_F.
- *Within-actor matching:* states, base noise and actions, α, actor parameters and Adam state.
- *Measurements:* the full chain above, with per-state preservation, on one common prespecified panel plus optional
  source panels.
- *Fork-time:* equivalence assertions.
- *Interventions:* direction-only, magnitude-only, full and sanity, kept separate from ordinary SAC.
- *Outcomes:* an outcome-evaluation interface. Gradient differences are not to be read alone as harm.
- *Status:* implemented [C]. Matching, the chain and interventions were verified on CPU in my earlier review.
  Fork-time and post-fork runs on real-entry retained states are tested.

**Pilot 2, passive learning:**
- *Stage A (required):* P_U←U and P_I←U from the common fork, never adding their own experience.
- *Matched:* initial replay, ordered arrivals, replay-sampling keys between passive learners, update budget and
  timing, initialization, normalization and evaluation protocol.
- Each learner runs ordinary SAC on its own state.
- *Stage B (optional, opt-in):* the 2×2, with active U/U and I/I as references only.
- *Status:* implemented [C]. Isolation is bit-exact on CPU; with captured keys, passive U equals active U on CPU.

**Pilot 3, optional target-policy assignment:**
- *Design:* clone one selected critic and Adam state into two learners. They see identical batches, use one frozen
  evaluator and a fixed protocol-derived α, and differ only in the frozen target actor (π_U vs π_I), with shared
  base noise. Short configurable budget.
- *Measurements:* target differences, critic adaptation, Q and dQ/da changes on fixed panels, and optional guidance
  to a common actor. An alternative-evaluator sensitivity check is included.
- *Status:* implemented [C], including on retained lean real-entry states (Codex test).

**Capture ([REQ]: "opt-in source-stream recording … without altering trajectories"):**
- *Status:* implemented and corrected, with recording parity tested on CPU. GPU parity is outstanding.

## 3. Minimal prioritized decisions requiring owner approval
**Before any designated Exp1 source parent runs:**
1. **Scope:** which MyoSuite tasks (the request's "both" vs the 4 registered tasks) and which scaled source
   architecture(s). Until decided, the preservation envelope is 60 scaled parents.
2. **Capture authorization:** opt-in recording on designated parents, the retention window (`--retain-until-arm-end`
   or a step), replay retained or omitted, and the storage allocation.

**Before designated Exp2 injected arms:**

3. **m** (the existing Exp2 freeze process), and binding passive P_I's injection to it with the Exp2 seed
   convention.

**Before Pilot 1:**

4. Post-fork offsets to measure (cadence).
5. The common panel's size and source, and any source-specific panels.
6. The α value source (fork vs per-actor; matching within each actor is already required) and the normalization
   rule.
7. Intervention space and normalization, including the zero-norm policy.
8. The outcome-evaluation protocol (horizon, episodes).
9. A frozen `chunk_size`, because noise draws depend on it.
10. Optional: whether to include local variation.

**Before Pilot 2:**

11. Stage B on or off.
12. Passive horizon, evaluation cadence and episodes, and checkpoint cadence.
13. Normalization: the recorded source statistics vs frozen fork statistics (both match between learners).

**Only if Pilot 3 is enabled:**

14. Enablement; source critic; evaluator and alternative evaluator; α value; batch source; budget and batch size;
    the common actor for optional guidance.

**Optional deviations or additions (not required by [REQ]):**

15. FP32 measurement precision as a documented diagnostic deviation (decide after the TF32 sensitivity measurement).
16. Independent-RNG noise-floor control for Pilot 2 (needs implementation).
17. Captured-key replica as a passive-fidelity positive control.
18. The `actor_grad_cosine` panel bridge.

Not decisions (requirements under [REQ]): the three Pilot 1 actors; post-fork measurement with fork-time
assertions; per-state preservation; the measurement chain; the intervention set; Stage A; shared replay-sampling
keys; Pilot 3 being optional.
