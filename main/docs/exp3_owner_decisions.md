# Experiment 3: owner decision package (2026-10-10)

Branch `claude/exp3-pilot-review`. Reviewed implementation: Codex `codex/exp3-pilot-implementation` @ `142a7fc`,
plus review fixes on this branch. This package **chooses nothing**. Every row below needs owner approval.
Supporting detail: [review findings](exp3_pilot_review.md), [capture plan](exp3_capture_plan.md).

Labels used for numbers: **M** = measured (source given), **X** = extrapolated from a measurement, **E** = estimate
or analytic bound, **U** = unmeasured.

## 0. The specification these decisions refer to is missing
A search of every branch and of the full Git history found **no original Pilot 1–3 specification text**.
- *Approved methodology on Exp3:* one sentence, `methodology-exp1-exp2.md:210`: "Actor_grad_cosine is retained as a
  mechanism diagnostic for Experiment 3".
- *What that quantity is:* the cosine between per-sample actor gradients within a batch
  (`sac_update.compute_actor_gradient_cosine`). Exp1/Exp2 already log it.
- *A different quantity:* it is **not** the U-vs-I guidance cosine that Codex's Pilot 1 measures.
- *The only other written description:* Codex's `exp3_pilots.md` and `exp3_implementation_plan.md`. They are
  engineering documents, not authority, and this package does not treat them as the specification.

Missing text needed before execution (each item is quoted nowhere approved):

| Missing item | Used by | Currently filled by (non-authoritative) |
|---|---|---|
| Pilot 1 definition: which actors (U, I, fork), which critics, which panel, what "direction", "magnitude", "full", "sanity" mean, intervention space | Pilot 1 | Codex's two interpretations (action-space, per-state parameter-space) |
| Pilot 2 definition: Stage A/B, passive update timing, horizon, normalization, evaluation | Pilot 2 | Codex docs |
| Pilot 3 definition: target-policy assignment, evaluator, alpha source, critic start, budget | Pilot 3 | Codex docs |
| The cell population: dog-run + humanoid-walk + two MyoSuite tasks × seeds 1–5 | all | Codex docs ("requested") |
| How `actor_grad_cosine` (approved hook) enters Exp3, and whether it is the same as Pilot 1's U-vs-I cosine | Pilot 1 | — |
| Outcomes, statistics and success criteria of each pilot | all | none |

**Action:** commit the approved Pilot 1–3 text to `main/.claude/` (the methodology authority). Until then, every
"Changes methodology?" answer below about Exp3 itself is provisional.

## 1. Unresolved scientific choices
| # | Choice | Recommended option | Alternatives | Causal-validity implication |
|---|---|---|---|---|
| D1 (S9) | Exp3 specification | Commit the approved text (§0) | — | Without it, no Exp3 result can be shown to test the approved hypothesis |
| D2 (S6) | Injection m and seed for Pilot 2's passive I | **Owner sets m (Exp2's m is itself not yet frozen).** Recommend tying passive I to the same m and seed convention as the Exp2 I arm (`seed = cfg.seed`). Block Pilot 2 until m is frozen. | A different m, explicitly labelled as a sensitivity analysis | Passive I must start from the critic whose consequences are being decomposed; otherwise passive and active I are different interventions. Stage B is already enforced against the I stream's recorded injection (C1). |
| D3 (S2) | Precision of Exp3 measurements | **Owner decides after the measured TF32-vs-FP32 sensitivity (§6, G5).** Recommendation to consider: measure in full FP32 and keep training updates at production precision. | TF32 throughout; FP32 throughout, including Pilot 3 fits | The instrument's own rounding can change small or near-orthogonal per-state cosines and ratios. CPU cannot measure this. |
| D4 (S3) | Which actor is measured | Spec decides. Recommend U and I own actors as primary, and the fork actor as a common reference (Pilot 1 computes both today; Pilot 3 measures only the fork actor). | Fork actor only; own actors only | Common actor isolates the critic difference; own actors measure the realised guidance |
| D5 (S1) | Passive key stream | Keep independent keys as the primary passive design. Add the source-key replay as a positive control, now possible exactly (captured key per arrival; passive U then equals active U bit for bit, CPU). | Source keys as primary | Isolation of passive U vs I holds either way (bit-exact, CPU). Source keys add a replica check of the active control. |
| D6 (S4) | Pilot 1 outcome horizon | Keep the one-step outcomes descriptive. Pilot 2 covers multi-step consequences. | A k-step guided continuation (new design and budget) | One AdamW step barely changes returns; a longer horizon becomes a new confounded experiment |
| D7 (S5) | Fork-time Check 1 provenance | Use the approved `fork/panel.npz` and the arm's saved `check1_after.npz`, and require a bitwise match | Codex's fresh policy-action panel | Ties every Exp3 "I at fork" to the exact Exp2 injection |
| D8 (S7a) | Noise draw depends on `chunk_size` | Fix `chunk_size` in the protocol | Per-state keys (small engineering change, not made) | A memory knob must not change the measured quantity |
| D9 (S7b) | Pilot 3 batches | Spec decides; recommend fork replay (common data) | Each arm's own post-fork replay (requires replay retention, §4) | Common batches isolate the target-policy effect from the data distribution |
| D10 | Pilot 1 storage | Owner: full per-state parameter gradients (about 8 GB per cell, E) or summaries | — | Summaries lose the ability to recompute other statistics later |
| D11 | U stream length | Owner: stop at `arm_end` (Stage B cannot use later arrivals), or to `control_end` | — | None. The `arm_end` option is not implemented (an engineering option, not a scientific one) |
| D12 (S8) | Pilot 3 log density | Keep Codex's latent-based density. It equals production on unsaturated draws, which a test confirms. | Literal production formula (NaN at saturation) | Avoids dropping saturated states selectively |
| — | Budgets, panel sizes, update counts, evaluation counts, thresholds and success criteria | **Owner** | — | Not proposed here |

## 2. Source-cell selection to approve (dimensions only; no cells chosen)
A source cell is (task, seed, scaled architecture) with an **actual natural-trigger fork** in its Exp1 parent.

| Dimension | Values to approve | Note |
|---|---|---|
| DMC tasks | dog-run, humanoid-walk | Reported by Codex as specified; confirm against your text |
| MyoSuite tasks | **2 of**: myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach | Unselected |
| Seeds | 1–5 (`EXP12_SEEDS`) | — |
| Architecture(s) | D4W1024, D4W1536, or both | Doubles cells and storage if both |
| Missing forks | Cell dropped / replaced / reported as null | Parents that never trigger have no fork; nothing can be substituted without a new rule |
| Exp2 arm | The approved injected arm with frozen m | Depends on D2 |

## 3. Exp1 parents to hold until recording is GPU-qualified
The Exp1 parent of a scaled cell **is** the U arm after its fork. If it runs unrecorded, its U stream and U matched
states are lost (a later U continuation equals it only if GPU training is deterministic, which is still unverified).

| Exp1 parents | Hold? |
|---|---|
| Scaled (D4W1024, D4W1536) × {dog-run, humanoid-walk} × seeds 1–5 | **Hold** (20 parents) |
| Scaled × {4 MyoSuite candidates} × seeds 1–5 | **Hold until the §2 MyoSuite pair is selected**, then hold only the selected pair (40 parents now, 20 after selection) |
| If one architecture is chosen | Halve the above |
| All D2W512 parents (never fork); scaled parents of the other 7 tasks; the seed-102 dev pilot | **Not held**: Exp3 needs nothing from them |

Total held now: **60** of 195 parents; **40** once the MyoSuite pair is chosen; **20** if one architecture is
also chosen. Pre-fork data of every parent is already written and retained. Holding changes only scheduling.

## 4. Minimal checkpoint and stream retention
| Item | Source | When | Content | Status |
|---|---|---|---|---|
| Fork state, `panel.npz`, `check1_pre/control`, `fork.json`, fresh critic | Exp1 parent | at fork | complete | already retained; archive |
| I at the fork | `exp2_arm` first save | at fork, post-injection | agent + normalization + meta (lean) | retain (opt-in) |
| U and I matched states | both, routine saves | fork + j·N/20 up to `arm_end` (j = 1…5) | lean: agent (actor/critic/target/alpha + Adam + counters), `obs_rms`, `meta.pkl` | retain (opt-in); offsets are an owner choice |
| U and I streams | both | fork → `arm_end` (U possibly to `control_end`, D11) | raw transition, source `obs_rms` at add time, update count, source agent key | record (opt-in) |
| Replay in snapshots | — | — | omitted (Pilots 1/3 use fork replay) | needed only if D9 chooses arm replay |

Mechanism: `scripts/exp3_record_source.py` with `--retain-root`, `--retain-until-step` and `--retain-replay`. The
same flags must be passed on every resumed segment, plus `--resume-stream`.

## 5. Storage and runtime budget (N = 500k interaction steps, horizon 125k)
| Quantity | Value | Label |
|---|---|---|
| D4W1536 agent checkpoint | 0.53–0.54 GiB untrained; ≤ about 1.22 GB trained | M (CPU save) / E (analytic) |
| D4W1024 agent checkpoint | 0.24 GiB untrained; ≤ about 0.54 GB trained | M / E |
| Stream bytes per arrival | 613 B (hopper), 2,113 B (myo-reach); about 7.4 KB dog-run, about 2.35 KB humanoid-walk, 1.6–2.1 KB MyoSuite | M / X |
| Per cell, lean, fork + 5 offsets + 2 streams to `arm_end` | D4W1536: about 14–15 GB; D4W1024: about 6.4–8 GB | E |
| 20 cells | about 285 GB (D4W1536), about 135 GB (D4W1024), about 420 GB (both) | E |
| Pilot 1 per-state gradients | about 2.7 MB per state per actor (about 8 GB per 1,000 states × 3 actors) | E |
| Recorder host memory | ≤ one save interval of arrivals: about 185 MB dog-run | E |
| Recording time overhead | no extra env steps, updates, evaluations, saves or RNG draws (bit-identical final state, CPU); per-save hard links plus SHA-256 over about 1.2 GB | M (CPU equality) / U (A100 time) |
| Active training rate, D4W1536 dog-run, A100 | 42.44 interaction steps/s (60-step window only) | M (job 22765261) |
| One arm-length continuation (125k steps) | about 0.8 h training | X |
| Pilot 2 Stage A (2 learners, 2 updates per arrival, no env) | ≤ about 1.6 h for 125k arrivals; Stage B ≤ about 3.3 h | X, upper bound (assumes per-update cost ≈ an active step) |
| Pilot 1, Pilot 3 runtime | — | U |
| Peak GPU memory, one D4W1536 learner | 3.5 GiB | M (job 22765261) |
| Exp3 suite on CPU | 64 tests, 814 s; harness exactness 25 tests, 541 s per precision | M (CPU) |

## 6. GPU validation plan (requires authorization; nothing submitted; no tolerances chosen)
Bitwise is the existing representation-exact criterion; it is not a new threshold. Every non-bitwise quantity is
**reported**, never judged.

| Stage | What | Pass condition | Needs |
|---|---|---|---|
| G0 | `scripts/exp3_gpu_validation.py` on one A100: tiny exactness suites (isolation, passive fidelity, restarts, recording parity, real-entry capture), repeat determinism, oracle and TF32 sensitivity at production width on synthetic states | `EXACTNESS_PASS`; oracle and sensitivity reported | Owner: precisions, sensitivity task/architecture/states, time budget. CPU exactness takes about 9 min per precision, so 300 s is unlikely to suffice (U) |
| G1 | **Recording parity at production width:** from one actual fork, run a short continuation recorded and unrecorded | final states bitwise equal | A real fork, from the dev pilot or an owner-authorized forced dev trigger; length set by the owner |
| G2 | **Passive exactness:** passive pair without injection stays bitwise identical; a repeat run is bitwise; captured-key passive U equals the active retained snapshot | bitwise | same fork |
| G3 | **Restart correctness:** interrupt a recorded continuation between checkpoints and resume; interrupt and resume passive and target pilots | stream and states equal the uninterrupted run | same |
| G4 | **Full-width memory and time:** one Pilot 1 chunk, Pilot 2 Stage A for a short window, a few Pilot 3 updates | report peak device/host memory, time per arrival/update, bytes per checkpoint and arrival | owner sizes |
| G5 | **FP32 vs TF32 on real artifacts:** every Pilot 1 measurement field and the Pilot 3 targets at both precisions | report differences and cosine sign flips → input to D3 | — |

Interpretation:
- **Repeat not bitwise:** NONDETERMINISTIC. That is an owner decision, not a defect.
- **Repeat bitwise but an exactness check fails:** a defect for engineering.
- **Missing outputs or a timeout:** INCOMPLETE.

## 7. Changes to approved Exp1/Exp2 methodology implied by these recommendations
- **Training, triggers, injection, Check rules, statistics and budgets:** **none.** No recommendation alters them.
- **Outputs of designated runs:** retention and recording add files to designated Exp1 parents and Exp2 arms. They
  do not change their trajectories (bit-identical on CPU; GPU unverified, G1). Recommend recording this as a dated
  operational amendment ("designated runs retain post-fork routine saves and record arrivals"), approved by the
  owner, not as a methodology change.
- **Scheduling:** holding the parents in §3 changes order only. The 195-parent population is unchanged.
- **If G1 fails** (recording changes trajectories on GPU): capture cannot be used in production runs without
  changing Exp1/Exp2. The alternative is separately authorized continuations, with the caveat on GPU determinism.
- **Dependencies on open Exp1/Exp2 decisions** (not changes): frozen m (D2), and natural-trigger forks that depend
  on the unresolved fresh-null and positive-control gates.
