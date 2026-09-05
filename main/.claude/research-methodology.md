EchoCritic Research Methodology

Scientific Objective

This is a systematic empirical and causal study, not a method paper. The primary contribution is establishing the mechanism by which critic-side optimization pathology, induced by scaling, propagates into actor optimization under SimBa-based SAC — and whether this propagation is detectable through actor-side functional metrics even when actor-side structural metrics (SRank, dormant ratio) appear unremarkable. A corrective method is a secondary, contingent objective: the design of any correction depends on which distortion property is identified as dominant, and cannot be finalized in advance. Rather than relying on theoretical analysis or observational correlation alone, the study uses controlled architectural comparisons, frozen-batch counterfactual intervention, and (where applicable) short-horizon forked training to establish that critic pathology causes actor degradation, not merely co-occurs with it.

Locked Experimental Variables

Actor architecture: fixed at depth=1, width=128 across every experiment, every angle.

Critic architectures: baseline depth=2/width=512 (cites SimBa's own default configuration); scaled depth=5/width=768 and depth=7/width=1024 (not 1028 — this typo has recurred and must not be reintroduced).

UTD (updates-per-interaction-step): fixed at 5 across all experiments and all architectures. This is a stated limitation, not a second independent variable — findings reflect architecture scaling at UTD=5, not architecture scaling in isolation.

Seeds: 5 for Angle 1; 5 for Angle 2A/2B; 10 for Angle 2C (power there comes from within-run (s,a) sampling density, not seed replication); 5 for Angle 3.

Environments: Angle 1 and Angle 2 share an identical core set of 10 environments (6 DMC + 4 MyoSuite) — Angle 2 must never be run on an environment Angle 1 has not already established degradation in. Angle 3 = same 10 core + 4 explicitly-labeled held-out generalization environments (2 DMC + 2 MyoSuite), never merged into the core set.

Step counts (UTD=5-adjusted): Hard-tier environments (Dog-run, Dog-trot, Humanoid-run) = 2.5M steps; Easy/Medium-tier environments (Cheetah-run, Quadruped-run, Manipulator-bring_ball) = 1.25M steps. Same total step count required across all architectures within a given environment for a fair comparison.

Confidence intervals: 95% as default throughout. Any comparison run at n=3 or lower must be explicitly caveated as underpowered rather than presented with the same implied confidence as 5-seed results.

Angle 1

Systematic evaluation of how scaling critic architecture (holding actor fixed) causes the development of optimization pathologies in the critic that propagate into the actor network and degrade the policy. Establishes, per seed per architecture, the interaction step at which the critic is classified as "degraded" (see Onset Definitions), and characterizes the relationship between functional and structural metrics in terms of when each identifies emerging degradation relative to the other. This angle is the foundation every downstream angle depends on — Angle 2's reference-critic timing and Angle 3's degraded/reference arms all inherit their t* values from here.

Angle 2A

Purpose: determine whether the degraded scaled critic exhibits greater policy-conditioned functional error than a matched healthy reference critic, at the onset of pathology propagation.

Reference critic construction: baseline architecture (depth=2/width=512), trained online normally (its own actor, its own replay buffer, no buffer-sharing, no actor-transplanting) for the exact number of timesteps it took the matched scaled critic to reach its own individually-logged t*. Each scaled critic gets its own independently-trained reference critic, matched to its own t* — never shared across two scaled architectures, never averaged across seeds or architectures.

Evaluation set: 20 state-action pairs per matchup — 10 sampled from the degraded agent's replay buffer, 10 from the reference agent's replay buffer, each tagged with its source.

Ground truth: for each (s,a) pair, reset to s, force action a, then continue rollout using only the actor belonging to that pair's source agent — 15 rollouts, averaged, to obtain Q̂_MC^{π_D}(s,a) (for D-buffer pairs) or Q̂_MC^{π_R}(s,a) (for R-buffer pairs).

Error computation — diagonal only

E_D = |Q_D(s,a) − Q̂_MC^{π_D}(s,a)|

on D-buffer probes only.

E_R = |Q_R(s,a) − Q̂_MC^{π_R}(s,a)|

on R-buffer probes only.

Both critics' raw Q-estimates may be logged on every probe for secondary/off-diagonal analysis, but only the matched (diagonal) pairing counts as an "error" measurement. Q_R evaluated against Q̂_MC^{π_D}, or vice versa, is not a valid accuracy measurement — no common ground-truth Q-function exists across two different policies, since Q^π is defined relative to a specific policy. The primary claim is: does the degraded agent's critic exhibit greater policy-conditioned return-estimation error than the reference agent's critic, each evaluated against the return function induced by the policy it actually serves — not "is critic D objectively less accurate than critic R as approximations of one shared true Q-function" (that quantity is not identifiable from these agents).

Angle 2B

Purpose: determine whether the functional error established in Angle 2A actually reaches the actor's optimization signal — not just whether the critic's beliefs are wrong, but whether that wrongness distorts what the actor is trained on, and whether that distortion exceeds normal healthy-critic variability. This is necessary because a critic can be biased in value while producing an unaffected local gradient (a uniform offset does not change slope); 2A cannot show this on its own. Angle 2B reuses Angle 2A's existing infrastructure (frozen degraded agent (Q_D, π_D) and frozen reference agent (Q_R, π_R) checkpoints, matched to t*) — no additional training or environment interaction occurs anywhere in Angle 2B.

Primary analysis: a frozen, single-step counterfactual. Hold the degraded agent's actor π_D completely fixed. Using identical state-action inputs, entropy coefficient, and sampled evaluation conditions, compute two counterfactual actor gradients:

g_{D|D} = ∇_θJ(π_D; Q_D)   (the real signal, using the critic actually present in the degraded agent)

g_{D|R} = ∇_θJ(π_D; Q_R)   (the counterfactual signal, same actor, critic swapped to the reference)

Only the critic changes between these two; π_D's parameters, the state batch, and the entropy term are held identical in both.

Distortion metrics — compute all three, on the primary analysis pair:

D_dir = 1 − cos(g_{D|D}, g_{D|R})   (direction)

D_mag = log(‖g_{D|D}‖ / ‖g_{D|R}‖)   (magnitude)

D_grad = ‖g_{D|D} − g_{D|R}‖   (raw gradient displacement)

Secondary robustness analysis: repeat the identical counterfactual procedure using the frozen reference actor π_R instead — g_{R|R} = ∇_θJ(π_R; Q_R), g_{R|D} = ∇_θJ(π_R; Q_D) — and compute the same three distortion metrics on this pair. This is a diagnostic robustness check only, never averaged with the primary analysis: its purpose is to determine whether the observed distortion is specific to the actor that emerged from the degraded training condition, or is a property of the critic detectable regardless of which actor probes it. Report both results side by side, explicitly labeled primary vs. secondary, never combined into a single aggregate statistic.

Null distribution: the null must mirror the primary analysis's structure exactly, not just involve "a healthy critic." Select two independently-trained default-architecture SimBa agents, A and B (different seeds; neither critic has trained on the other's actor's data — essential, not optional). Using π_A held fixed, compute g_{A|A} = ∇_θJ(π_A; Q_A) (real signal, A's own critic) and g_{A|B} = ∇_θJ(π_A; Q_B) (foreign healthy critic swapped in), then compute D_dir, D_mag, and D_grad on this pair exactly as in the primary analysis — never a raw comparison between two healthy critics' outputs. This isolates the distortion caused by ordinary "foreign critic" unfamiliarity alone, with no pathology involved on either side. Repeat across multiple independent healthy A/B pairs (reuse baseline critics from Angle 1 seeds where possible rather than retraining) to build a real distribution, not a single point estimate. Compare the degraded-critic distortion (g_{D|D} vs g_{D|R}) against this null distribution using the (mean + 2σ) rule (implemented in experiments/angle_2b/statistics.py) — do not invent a new statistical threshold for this step. This is now a DIFFERENT rule from Angle 1's onset/propagation threshold (a 95th percentile — see the 2026-08-28 correction in Onset Definitions above), not "the same criterion established elsewhere in this study" as this section previously implied: (mean + 2σ) remains the appropriate choice here specifically because Angle 2B's null distribution has very few points (at most one A/B pair per baseline seed, ≤5 total), where an empirical percentile is poorly defined/unstable, unlike Angle 1's baseline curves which pool many logged timesteps across all 5 seeds. Do not silently make these two rules consistent with each other by changing either implementation — each is the right choice for its own sample size, and this divergence has been deliberately reviewed and kept as of the 2026-08-28 audit. State explicitly, as a limitation, that this comparison assumes the magnitude of ordinary foreign-critic unfamiliarity is comparable between a healthy-healthy pairing and a degraded-reference pairing — a standard assumption underlying null-baseline comparisons throughout this methodology, not a fully verified equivalence.

Scope boundary — do not overclaim: 2B establishes that critic inaccuracy reaches and measurably distorts the actor's training signal at a single frozen point, relative to the healthy-critic null. It does not establish that this distortion produces worse downstream policy behavior over continued training — a single-step distortion could in principle wash out under momentum-based optimizers or self-correct over many updates. That causal claim belongs to Angle 3 only. Angle 2B must never be extended into a continued-training or multi-step design — doing so collapses it into Angle 3's experiment and reintroduces confounds (actor/critic co-adaptation over time) that the frozen design exists specifically to avoid.

Angle 2C

Purpose: given that Angle 2B establishes the actor's gradient is distorted, determine what property of the distortion is responsible — direction, magnitude, or local landscape stability. Operates on the same g_D/g_R pairs (and the underlying ∇_aQ values) already computed in 2B — no new state-action sampling required.

Three candidate properties

Directional corruption: cos(∇_aQ_D(s,a), ∇_aQ_R(s,a)). A value meaningfully below the null baseline indicates the degraded critic points the actor toward a genuinely different action, not merely a differently-confident version of the same direction.

Magnitude/bias shift: ‖∇_aQ_D‖ / ‖∇_aQ_R‖ (slope steepness ratio) and the raw offset Q_D(s,a) − Q_R(s,a) (level offset, ties directly to Q-overestimation tracking). A magnitude ratio departing from 1 paired with high directional cosine similarity indicates a correctly-directed but miscalibrated signal — consistent with overestimation bias.

Local instability: perturb a by small random offsets δ, compute ∇_aQ at each perturbed point for both critics, compare the variance of these nearby gradients between critics (each measured relative to its own null, since even a healthy critic is not perfectly smooth). Elevated variance in the degraded critic indicates a locally jagged landscape near the actor's actual operating point, producing inconsistent updates for near-identical inputs.

Multiple properties diverging simultaneously: do not assume co-occurrence implies equal causal contribution. Use a reconstruction test — construct synthetic gradients combining one critic's direction with the other's magnitude (and vice versa), run each synthetic version through the same frozen-actor procedure, and check which reconstruction better reproduces the real agent's actual observed gradient. Whichever reconstruction is closer indicates which property is doing more of the causal work.

Closing-the-loop criterion: whichever property is identified as dominant, its onset timing (when it departs from null across the degradation window) must be checked against Angle 1's real actor-side metric onset timing. Consistency between the two is what turns 2C from "here is a taxonomy of possible distortions" into "here is the specific distortion that explains what Angle 1 already measured."

None of the three cleanly separating from null is a legitimate, reportable outcome — do not force a narrative onto data that doesn't support one. Report it honestly as a finding that the corruption is not well explained by these three interpretable properties, rather than picking the closest-looking candidate.

Angle 3

Purpose: test whether the mechanism identified in Angle 2 has real consequences for training outcomes over time — the only angle in this study where training is continued rather than frozen at a single step.

Three-arm design, per seed, per scaled architecture

Real/degraded: normal continued training, mechanism fully active, critic degrades naturally as characterized in Angle 1.

Synthetically-corrected: identical setup, but the specific corrupted property identified as dominant in Angle 2C is intercepted and corrected at every actor update (e.g., substituting the reference critic's direction while retaining the real critic's magnitude, or vice versa, depending on which property 2C identified). This is an explicitly artificial, undeployable intervention used only to test the causal chain — it is not a proposed real-world fix, and must not be described as one.

Reference-ceiling: actor trained against the baseline-architecture reference critic's own trajectory (matched steps, not an early checkpoint of the scaled critic — this avoids reintroducing the staleness confound solved in Angle 2A's reference-critic design). Serves as an upper bound: how much of the gap between arms 1 and 3 does arm 2's correction actually recover.

Non-Q-pathway specificity control: to rule out "any sufficiently large perturbation would do this, regardless of source" as an alternative explanation, construct a fourth condition where Gaussian noise — magnitude-matched per-checkpoint to the real Angle 2B divergence — is injected into the combined gradient (after the Q-computation and entropy term have already been added together), never into the Q-term itself. Injecting noise into the Q-term directly would conflate this control with the real condition and defeat its purpose as a specificity check.

Environments: core 10 (shared with Angle 1/2) for the full mechanistic trace, plus 4 explicitly-labeled held-out environments testing generalization of the consequence only (not the full mechanistic trace, which was never established there).

Horizon: bounded, not run to the full original training length — long enough to observe onset-timing and severity differences between arms, short enough to avoid distribution-mismatch artifacts from sustained synthetic gradient splicing in arm 2.

Scope boundary: Angle 3 is the only place in this study where "the correction improves performance" is a claim that can legitimately be made. Nothing in Angle 2 (A, B, or C) supports that claim — conflating a 2B/2C finding with a performance claim is a category error the write-up must avoid.

Null Baseline

Constructed from two independently-trained, matched-timestep baseline-architecture critics (no scaled critic involved) — reusable, at no extra compute cost, from two of the 5 seeds already run in Angle 1. Run through the identical Angle 2A procedure to establish the natural degree of critic-to-critic divergence expected from ordinary training/seed variance alone, absent any pathology. Every reported divergence (E_D vs. E_R, and downstream in 2B/2C) must be interpreted relative to this null, not as a raw, unscaled number — a gap that falls within the null's natural spread is not evidence of degradation.

Onset Definitions

Critic degradation onset

The first interaction step at which td_error_var exceeds the 95th percentile of the baseline-architecture critic's own td_error_var distribution at the matched step, sustained for ≥ N consecutive logging intervals (to filter transient spikes from genuine, sustained shifts).

Pathology propagation

The first interaction step at which an actor-side functional metric (not structural — this is the core dissociation claim) exceeds the 95th-percentile baseline of a default-architecture actor's own distribution at the matched step, within a bounded lag window W of the interaction step at which the scaled critic was classified as degraded.

CORRECTION (2026-08-28 audit): this section previously specified "(mean + 2σ)" as the threshold rule. The implementation (analysis/baseline_calibration.py, analysis/onset_detection.py) has always computed an empirical 95th percentile of the baseline distribution instead (`aligned.quantile(percentile / 100.0, axis=1)`, configs/base_sac.yaml's `onset_detection.baseline_percentile: 95`) — there is no mean/std computation anywhere in analysis/*.py. A percentile threshold makes no distributional-normality assumption, unlike (mean + 2σ), and is not numerically equivalent to it on skewed data. The 2026-08-28 audit found this discrepancy; per explicit instruction, the implementation is being kept and this document corrected to match, rather than rewriting the implementation, satisfying the "must never change silently... without recalibrating and stating so explicitly" requirement below.

N and W are not assumed values — both are calibrated against the 5-seed default-architecture (baseline) data before being finalized, and the calibration procedure itself is reported in methods.

Accepted Limitations

Width/depth conflation

The scaling grid varies critic width and depth jointly, not independently. Prior work is cited as evidence that depth is the dominant driver of degradation, but this study's own results cannot attribute observed effects to either axis in isolation — that disentanglement is left to future work.

UTD fixed, not isolated

Findings reflect architecture scaling specifically at UTD=5; UTD is a known independent driver of similar pathologies in the literature and was deliberately not varied alongside architecture to avoid conflating two causal levers.

Angle 2A's estimand is policy-conditioned, not architecture-independent

Because Q_D and Q_R are each evaluated only against their own policy's ground truth, a result favoring the reference critic reflects "less error relative to the policy it actually serves," not proof of one critic being objectively more accurate in some policy-independent sense — that quantity is not identifiable from this design.

Untested architectural scale

Depth=5/768 and depth=7/1024 exceed the largest configuration validated in the original SimBa scaling ablation (depth=4/1024) — this study operates outside any previously published stability curve for this architecture. Mitigated by confirming clean, non-divergent training at baseline and at each scaled configuration before attributing observed instability to the mechanism under study rather than unvalidated scale.

MyoSuite state-reset fidelity

MyoSuite's musculoskeletal dynamics include muscle activation/tendon state beyond standard joint kinematics; incomplete state capture during reset could silently corrupt any rollout-based ground-truth procedure. Mitigated by a pre-registered determinism smoke test (reset, roll forward twice, confirm identical trajectories) on each MyoSuite task before it is used in any rollout-dependent analysis.

Held-out environment set is small

4 of 20 total environments in Angle 3 — sufficient to indicate a generalization trend, not to establish one conclusively.

Things Claude Must Never Change Silently

Averaging, sharing, or reusing a reference critic or its t* across different scaled architectures or across seeds. Every scaled critic gets its own independently-timed, independently-trained reference critic.

Comparing a critic's Q-estimate against Monte Carlo ground truth generated by a different agent's policy (any cross-policy / off-diagonal comparison being reported or treated as an "error" metric).

Changing UTD, seed counts, environment lists, or step counts without explicitly flagging the change and why — these are locked, not defaults.

Reintroducing width=1028 (typo) in place of width=1024.

Redefining or re-thresholding the onset/propagation criteria (the 95th-percentile rule — see the 2026-08-28 correction in Onset Definitions above — N, W) without recalibrating against the 5-seed baseline data and stating so explicitly.

Conflating the roles of Angle 2A, 2B, and 2C, or letting Angle 2B/2C's frozen, single-step analysis silently turn into a continued-training experiment (that boundary belongs to Angle 3 only).

Treating Angle 2A's diagonal-only result as a claim about architecture-independent critic accuracy rather than the narrower, correctly-scoped policy-conditioned claim it actually supports.

Injecting Angle 3's non-Q-pathway control noise directly into the Q-term rather than into the combined (post-entropy-addition) gradient — this collapses the control into the real condition and defeats its purpose.

Using an early checkpoint of the scaled critic as Angle 3's "reference-ceiling" arm instead of an independently-trained, matched-steps baseline-architecture critic — this reintroduces the staleness confound Angle 2A's reference-critic design already solved.

Describing Angle 3's synthetically-corrected arm as a proposed real-world fix or deployable method — it is an artificial, undeployable intervention used only to test the causal chain.

Claiming a performance/outcome improvement anywhere in Angle 2 (A, B, or C) — that claim is only supportable within Angle 3's continued-training design.