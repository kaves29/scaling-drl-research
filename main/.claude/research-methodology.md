EchoCritic Research Methodology

Scientific Objective

This is a systematic empirical and causal study, not a method paper. The primary contribution is establishing the mechanism by which critic-side optimization pathology, induced by scaling, propagates into actor optimization under SimBa-based SAC — and whether this propagation is detectable through actor-side functional metrics even when actor-side structural metrics (SRank, dormant ratio) appear unremarkable. A corrective method is a secondary, contingent objective: the design of any correction depends on which distortion property is identified as dominant, and cannot be finalized in advance. Rather than relying on theoretical analysis or observational correlation alone, the study uses controlled architectural comparisons, frozen-batch counterfactual intervention, and (where applicable) short-horizon forked training to establish that critic pathology causes actor degradation, not merely co-occurs with it.

Locked Experimental Variables

Actor architecture: fixed at depth=1, width=128 across every experiment, every angle.

Critic architectures: baseline depth=2/width=512 (cites SimBa's own default configuration); scaled depth=5/width=768 and depth=7/width=1024 (not 1028 — this typo has recurred and must not be reintroduced).

UTD (updates-per-interaction-step): fixed at 5 across all experiments and all architectures. This is a stated limitation, not a second independent variable — findings reflect architecture scaling at UTD=5, not architecture scaling in isolation.

Seeds: 5 for Angle 1; 5 for Angle 2A/2B; 10 for Angle 2C (power there comes from within-run (s,a) sampling density, not seed replication); 5 for Angle 3.

Environments: Angle 1 and Angle 2 share an identical core set of 10 environments (6 DMC + 4 MyoSuite) — Angle 2 must never be run on an environment Angle 1 has not already established degradation in. Angle 3 = same 10 core + 4 explicitly-labeled held-out generalization environments (2 DMC + 2 MyoSuite), never merged into the core set.

CONFIRMED (2026-09-05 audit, DMC held-out gap closed 2026-09-05) — full environment/tier/step-count table, canonicalized in code as DMC_MED/DMC_HARD/DMC_HELDOUT2 (scale_rl/envs/dmc.py) and MYOSUITE_CORE4/MYOSUITE_HELDOUT2 (scale_rl/envs/myosuite.py):

DMC (8 total): 6 core, shared Angle 1/2 — Dog-run (Hard, 2,500,000), Dog-trot (Hard, 2,500,000), Humanoid-run (Hard, 2,500,000), Cheetah-run (Medium, 1,250,000), Quadruped-run (Medium, 1,250,000), Manipulator-bring_ball (Medium, 1,250,000); held-out-2 for Angle 3 only — Hopper-hop (Medium, 1,250,000), Fish-swim (Medium, 1,250,000).

MyoSuite (6 total): core-4 for Angle 1/2 — MyoElbowPose1D6MRandom (Medium, 1,250,000), MyoHandReachFixed (Hard, 2,500,000), MyoHandKeyTurnFixed (Hard, 2,500,000), MyoLegWalk (Hard, 2,500,000); held-out-2 for Angle 3 only — MyoHandPenTwirlFixed (Hard, 2,500,000), MyoHandBaodingBallsP1 (Hard, 2,500,000; myosuite's own registry has no "BaodingBallsP1"-style id — the only matching registered env is under its "Challenge" naming, myoChallengeBaodingP1-v1, not the -v0 convention every other entry here uses — confirmed via gym registry inspection, see scale_rl/envs/myosuite.py).

Total: 10 core (6 DMC + 4 MyoSuite, Angle 1/2) + 4 held-out (2 DMC + 2 MyoSuite, Angle 3 only) = 14, matching the original spec in the paragraph above and in the Angle 3 section's own Environments line further down exactly. The previously-flagged conflict (DMC held-out pair missing from this table) is resolved as of 2026-09-05: Hopper-hop and Fish-swim, both confirmed real dm_control tasks (see tests/test_dmc_heldout2.py). Enforced in code identically to the MyoSuite split: scale_rl/envs/dmc.py's validate_dmc_not_heldout() and scale_rl/envs/myosuite.py's validate_myosuite_core4(), called from experiments/angle_1.py's run() and experiments/angle_2a/agent_runner.py's check_single_env_type() (Angle 2A is the only other angle that ever instantiates a live environment — 2B/2C never do; check_single_env_type gained MyoSuite support 2026-09-07, having originally only accepted env_type='dmc' — see Accepted Limitations' "MyoSuite state-reset fidelity" entry below).

Step counts (UTD=5-adjusted): Hard-tier environments (Dog-run, Dog-trot, Humanoid-run) = 2.5M steps; Easy/Medium-tier environments (Cheetah-run, Quadruped-run, Manipulator-bring_ball) = 1.25M steps. Same total step count required across all architectures within a given environment for a fair comparison.

Confidence intervals: 95% as default throughout. Any comparison run at n=3 or lower must be explicitly caveated as underpowered rather than presented with the same implied confidence as 5-seed results.

Angle 1

Systematic evaluation of how scaling critic architecture (holding actor fixed) causes the development of optimization pathologies in the critic that propagate into the actor network and degrade the policy. Establishes, per seed per architecture, the interaction step at which the critic is classified as "degraded" (see Onset Definitions), and characterizes the relationship between functional and structural metrics in terms of when each identifies emerging degradation relative to the other. This angle is the foundation every downstream angle depends on — Angle 2's reference-critic timing and Angle 3's degraded/reference arms all inherit their t* values from here.

Angle 2A

Purpose: determine whether the degraded scaled critic exhibits greater policy-conditioned functional error than a matched healthy reference critic, at the onset of pathology propagation.

Reference critic construction: baseline architecture (depth=2/width=512), trained online normally (its own actor, its own replay buffer, no buffer-sharing, no actor-transplanting) for the exact number of timesteps it took the matched scaled critic to reach its own individually-logged t*. Each scaled critic gets its own independently-trained reference critic, matched to its own t* — never shared across two scaled architectures, never averaged across seeds or architectures.

Evaluation set: 20 state-action pairs per matchup — 10 sampled from the degraded agent's replay buffer, 10 from the reference agent's replay buffer, each tagged with its source.

Ground truth: for each (s,a) pair, reset to s, force action a, then continue rollout using only the *stochastic* actor belonging to that pair's source agent — R rollouts (calibrated per environment, see below; historically a fixed, unvalidated R=15), averaged, to obtain Q̂_MC^{π_D}(s,a) (for D-buffer pairs) or Q̂_MC^{π_R}(s,a) (for R-buffer pairs). Report the standard error of this mean (σ_rollout_sample/√R) alongside every such point estimate in all persisted results, regardless of what R turns out to be for that environment — visible uncertainty is valuable even where R happens to already be close to the old fixed value.

CORRECTION (2026-09-06): rollout continuation previously used the actor's deterministic mean action (temperature=0), not genuine sampling from π. Combined with dm_control's otherwise-deterministic dynamics, this made repeated rollouts from the identical (s,a) pair bit-identical — confirmed directly (σ=0.0 exactly across 10 repeated rollouts) — i.e. no actual Monte Carlo estimate was being computed at all, since Q̂_MC^{π_D}(s,a) denotes the expected return under the actual stochastic trained policy π_D, not its greedy/mean action. Fixed in experiments/angle_2a/probes.py's run_rollouts_for_probe (temperature=1, i.e. genuine policy sampling) — this affects every MC rollout in Angle 2A (main comparison, prereq check, and R calibration below), all of which route through this one function.

Monte Carlo rollout-count (R) calibration (added 2026-09-06)

R is calibrated per environment rather than fixed, replacing the previous unvalidated R=15: the smallest integer R such that the standard error of the MC estimate, σ_rollout/√R, stays at or below half (the safety margin) of this environment's existing Angle 2A null distribution's 95th-percentile threshold — reusing the null-baseline machinery already built for the healthy-vs-healthy comparison rather than inventing a new significance criterion. Applies uniformly to every MC rollout in this environment's Angle 2A analysis: the main diagonal comparison and the pre/post construct-validity check above both use it, never a separately-configured value.

No such "null distribution + 95th-percentile threshold" artifact existed anywhere in Angle 2A before this (only raw per-probe diagonal errors from optional null-baseline matchups, never aggregated into a distribution or a percentile computed from them), and reusing the main comparison's own null-baseline data directly would be circular in any case — R calibration must finalize before the main comparison runs, but that comparison's null-baseline matchups are what would produce that data. Resolved (project decision, 2026-09-06) with a dedicated, one-time-per-environment healthy-vs-healthy reference pair (both sides the default/reference architecture, trained to the same burn-in-fraction cutoff used elsewhere — see experiments/angle_2a/r_calibration.py), decoupled from any specific main-comparison or prereq-check seed. Cached under results/angle_2a_r_calibration/{environment}/ so it runs once per environment, not once per invocation. This dedicated pair's own diagonal errors (which only exist to define the threshold, not to be reported as an MC estimate anyone relies on) are computed using the historical fixed rollout count — calibrating R against the very quantity used to define R's own calibration target would be circular.

σ_rollout measurement: for 5 representative (s,a) pairs sampled from the dedicated calibration pair's own buffer, run 50 repeated rollouts from each exact pair using the same actor throughout, and take the standard deviation WITHIN that one pair's own 50 returns. This is computed separately per pair and never pooled across pairs before combining — pooling raw returns across different (s,a) pairs first would conflate ordinary between-state value differences (some states simply having higher or lower value than others) with genuine rollout-to-rollout noise, artificially inflating the variance estimate and the resulting R requirement; this distinction is enforced in the implementation (sigma_rollout_from_per_pair_returns computes each pair's std independently before combining), not left as a risk, and tested directly with a synthetic case designed to fail if pooling were used instead. The 5 per-pair standard deviations are combined via max, not average — conservative, since a calibration meant to protect every probe's precision should be sized for the noisiest representative state actually observed, not a typical one.

Hard cap and underpowered flag: the calibrated R is capped at 35 regardless of what the margin above would otherwise require. If the uncalibrated ("ideal") R would exceed 35, the cap is used instead and every persisted result for that environment is explicitly flagged as underpowered relative to the target margin, rather than silently spending unbounded compute or silently loosening the margin to make a larger R look acceptable. Project decision, confirmed 2026-09-06.

Per-environment only, like every other calibrated parameter in this study (baseline thresholds, N, W) — never pooled or averaged across environments.

Error computation — diagonal only

E_D = |Q_D(s,a) − Q̂_MC^{π_D}(s,a)|

on D-buffer probes only.

E_R = |Q_R(s,a) − Q̂_MC^{π_R}(s,a)|

on R-buffer probes only.

Both critics' raw Q-estimates may be logged on every probe for secondary/off-diagonal analysis, but only the matched (diagonal) pairing counts as an "error" measurement. Q_R evaluated against Q̂_MC^{π_D}, or vice versa, is not a valid accuracy measurement — no common ground-truth Q-function exists across two different policies, since Q^π is defined relative to a specific policy. The primary claim is: does the degraded agent's critic exhibit greater policy-conditioned return-estimation error than the reference agent's critic, each evaluated against the return function induced by the policy it actually serves — not "is critic D objectively less accurate than critic R as approximations of one shared true Q-function" (that quantity is not identifiable from these agents).

Construct-validity prerequisite check (added 2026-09-06)

Before the main D-vs-R comparison runs, a within-subject pre/post check verifies that the td_error_var-based degradation threshold used to define t* actually corresponds to an independently-measured decline in functional accuracy, using a different method (MC-validated diagonal error, the same E_D already defined above) than the one that defined t* (td_error_var exceedance). This is a check on the onset-detection methodology itself, not a restatement of the main comparison.

Seed independence: run on 3 seeds dedicated to this check alone, for each scaled architecture (scaled_a and scaled_b independently — each architecture's own onset-threshold construct validity is checked against its own dynamics, not extrapolated from the other). These seeds are never reused in, drawn from, or otherwise connected to the main 5-seed comparison, and are chosen and evaluated before any main-analysis result exists. This separation exists specifically to prevent any appearance of cherry-picking seeds that happen to look favorable after seeing the main comparison's results — the construct-validity claim would be worthless if the seeds validating it could have been selected with knowledge of what they'd show.

Procedure, per (scaled architecture, dedicated seed): train one fresh D agent. Save a real checkpoint ("pre") at the burn-in cutoff already established for ACF/CCF window calibration — configs/base_angle2a.yaml's angle_2_a.prereq_check.burn_in_fraction, intended to match configs/base_sac.yaml's onset_detection.acf_ccf_burn_in_fraction (analysis/window_calibration.py's BURN_IN_FRACTION) but not automatically synced with it — i.e. the end of the first 25% of this agent's own planned training length. Continue training, unperturbed, to this seed's own t* (looked up from the onset ledger exactly as the main comparison does), and save a second checkpoint ("post"). At each checkpoint independently, sample 10 fresh probes from D's own buffer as it exists at that checkpoint — pre and post each get their own separately-sampled probes, never the same probes reused across the two — and compute E_D the same diagonal-only, source-actor-only way E_D is defined above, with this environment's calibrated R rollouts per probe (see Monte Carlo rollout-count (R) calibration below), reported with its standard error like every other MC-based estimate here.

"Unperturbed" is a real implementation guarantee, not just an intention: evaluating E_D at the pre checkpoint requires running MC rollouts on the same live agent/environment that training is about to resume on, which would otherwise leave the environment's physics state and the agent's JAX RNG stream wherever the last rollout happened to end, not where training actually left them. experiments/angle_2a/prereq_check.py explicitly saves and restores both (reusing the pre checkpoint itself to restore agent state, and env_state.py's capture/restore for physics state) before letting training resume, so training after the pre checkpoint proceeds bit-identically to how it would have without this check ever running — verified directly (not just reasoned about) in tests/test_angle2a_prereq_check.py by comparing a checkpointed run against an uninterrupted control run of the same seed.

Reporting is descriptive, not an automated gate: E_D(pre), E_D(post), their difference, and whether the direction is consistent with genuine decline (E_D(post) > E_D(pre)) are reported per seed per architecture, never averaged across the 3 seeds given the small sample size. There is no statistical pass/fail threshold and no code path that blocks or alters the main comparison based on this check's outcome — the main comparison always proceeds regardless of what this check finds, and results are surfaced (printed, and persisted to results/angle_2a_prereq/{environment}/seed{seed}/{architecture_label}/) for human judgment rather than an algorithmic decision, since n=3 does not support a principled statistical threshold either way.

Resumability (added 2026-09-07): run_matchup accepts checkpoint_dir/checkpoint_interval, both required together for anything to actually be checkpointed. Training resumability (D and R, each independently, to onset_step) and MC-validation resumability (per-rollout, not per-probe or whole-phase) are both implemented — see experiments/angle_2a/agent_runner.py's train_agent_to_step/run_training_loop and experiments/angle_2a/probes.py's run_monte_carlo_rollouts. A crash/kill resumes from exactly where it left off on the next identical invocation. See "Pending / Open Items" below for what this feature has not yet been validated against.

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

Baseline distributions are per-environment, never pooled across environments (LOCKED, clarified 2026-09-05): td_error_var and the actor-side functional metric are not on a comparable scale across different environments — task difficulty, reward structure, and dimensionality differ. Pooling across environments would let a naturally noisier environment inflate the shared threshold, making it too lenient for quieter environments and risking missed degradation there. Each environment's threshold must be built using only the 5 baseline-architecture seeds run on that same environment — never seeds or data from any other environment — for both critic degradation onset and pathology propagation. This applies identically to both metrics above; it is not a separate rule for one and not the other.

CORRECTION (2026-08-28 audit): this section previously specified "(mean + 2σ)" as the threshold rule. The implementation (analysis/baseline_calibration.py, analysis/onset_detection.py) has always computed an empirical 95th percentile of the baseline distribution instead (`aligned.quantile(percentile / 100.0, axis=1)`, configs/base_sac.yaml's `onset_detection.baseline_percentile: 95`) — there is no mean/std computation anywhere in analysis/*.py. A percentile threshold makes no distributional-normality assumption, unlike (mean + 2σ), and is not numerically equivalent to it on skewed data. The 2026-08-28 audit found this discrepancy; per explicit instruction, the implementation is being kept and this document corrected to match, rather than rewriting the implementation, satisfying the "must never change silently... without recalibrating and stating so explicitly" requirement below.

CONFIRMATION (2026-09-05 audit): the per-environment scoping above was audited and found to already be correctly implemented, with three independent guarantees, none of which required a code change: (1) experiments/angle_1.py's call site builds `baseline_identities` with `environment=cfg.env_name` fixed across all 5 seeds - the baseline is always constructed for the same environment as the run being analyzed; (2) analysis/baseline_calibration.py's `calibrate_baseline()` has an explicit runtime guard (`environments = {ident.environment for ident in baseline_identities}; if len(environments) != 1: raise ValueError(...)`) that hard-fails rather than silently pooling if this were ever violated; (3) the on-disk baseline cache is keyed by `{architecture}/{environment}/...` (`_cache_path`), so even a caching bug could not cross-contaminate between environments. Verified with a new test (tests/test_baseline_calibration.py::test_two_environments_get_distinct_not_pooled_thresholds) constructing two synthetic environments with deliberately different variance scales (values ~1-5 vs ~1000-5000) and confirming each environment's threshold matches its own 5 seeds exactly (4.8 and 4800 respectively) rather than a pooled value (~4550, which is what the low-variance environment's threshold would have wrongly become under pooling) - see that file for the exact assertions.

N and W are not assumed values — both are calibrated against the 5-seed default-architecture (baseline) data before being finalized, and the calibration procedure itself is reported in methods. Genuine data-derived calibration replaced an earlier hardcoded-fraction-of-run-length placeholder as of the 2026-09-05 audit (see CORRECTION note below); the procedure is documented in full here.

N (sustain window) — Autocorrelation Function (ACF) method. N answers: how long can an ordinary noise fluctuation in td_error_var plausibly persist, before a sustained shift becomes distinguishable from noise? For each baseline-architecture seed, for each environment, independently: (1) take that seed's td_error_var series over only the latter 75% of the aligned trajectory, discarding the first 25% as an early, non-stationary settling period — this 75%/25% split is an explicit, practical, conservative convention adopted given project time constraints, not a statistically derived cutoff, and is documented as such rather than presented with false precision; (2) compute the ACF of that trimmed series; (3) find the lag at which the ACF first decays below 1/e (≈0.368) of its value at lag 0 — that seed's decorrelation-time estimate, in units of consecutive recorded (logging-interval) points; (4) average this decorrelation-time estimate across all 5 baseline seeds for that environment — this average is N for that environment, rounded to the nearest whole recorded point. If a seed's ACF never decays below 1/e within the available post-burn-in segment (a genuinely long-memory signal, or ordinary residual non-stationarity given the burn-in cutoff is itself only a practical convention — not assumed to indicate rare pathology), that seed's estimate is capped at the full segment length, a warning is logged naming the seed, and calibration proceeds using the capped value in the average — chosen over hard-blocking on manual review specifically so the pipeline does not stall repeatedly if this triggers more often than a rare edge case.

W (propagation/lag window) — Cross-Correlation Function (CCF) method, not ACF. W answers a different question from N: not "how long does one signal's own noise persist," but "what is the natural timing relationship between the critic-side signal and the actor-side signal, even under healthy conditions?" The 1/e decay rule does not apply here. For each baseline-architecture seed, for each environment, independently, over the same latter-75%-of-trajectory segment as N: (1) compute the cross-correlation between td_error_var and the actor-side functional metric (actor_grad_cosine) across a symmetric range of candidate lags, searched from −20% to +20% of the (post-burn-in) segment length; (2) find the lag maximizing the cross-correlation magnitude for that seed. The search is symmetric (both negative and positive lags) as an implementation-correctness diagnostic only, not a test of critic-vs-actor causal direction: critic-leads-actor holds by construction, since the actor's gradient at a given update step is computed from the critic's Q-output at that same step. If a seed's peak lands at a negative lag, that seed is excluded entirely from the average (not folded in via absolute value or otherwise) and flagged for manual review, since a negative-lag peak far more likely indicates a logging/indexing misalignment between the two signals than a genuine causal reversal. Only non-negative per-seed peak lags count toward the average; (3) average the valid (non-negative-lag) per-seed natural-lag estimates across up to 5 baseline seeds for that environment — this average, converted from recorded-points units to raw interaction steps (multiplied by logging_per_interaction_step, to match the existing raw-step contract for the propagation-window bound check), is W for that environment. If all 5 seeds for an environment have a negative-lag peak, W cannot be calibrated for that environment at all — this is treated as a strong signal of a systemic logging/indexing misalignment, not an isolated per-seed anomaly, and fails loudly requiring manual review rather than being silently skipped.

Both N and W are calibrated strictly per-environment and independently per environment — never pooled or shared across environments, consistent with the per-environment (never-pooled) baseline distribution rule already established above for the onset/propagation percentile threshold itself. See analysis/window_calibration.py for the implementation and tests/test_window_calibration.py for verification.

Accepted Limitations

Width/depth conflation

The scaling grid varies critic width and depth jointly, not independently. Prior work is cited as evidence that depth is the dominant driver of degradation, but this study's own results cannot attribute observed effects to either axis in isolation — that disentanglement is left to future work.

UTD fixed, not isolated

Findings reflect architecture scaling specifically at UTD=5; UTD is a known independent driver of similar pathologies in the literature and was deliberately not varied alongside architecture to avoid conflating two causal levers.

Angle 2A's estimand is policy-conditioned, not architecture-independent

Because Q_D and Q_R are each evaluated only against their own policy's ground truth, a result favoring the reference critic reflects "less error relative to the policy it actually serves," not proof of one critic being objectively more accurate in some policy-independent sense — that quantity is not identifiable from this design.

Untested architectural scale

Depth=5/768 and depth=7/1024 exceed the largest configuration validated in the original SimBa scaling ablation (depth=4/1024) — this study operates outside any previously published stability curve for this architecture. Mitigated by confirming clean, non-divergent training at baseline and at each scaled configuration before attributing observed instability to the mechanism under study rather than unvalidated scale.

Clipped Double-Q (CDQ) critic is non-functional and unused (CONFIRMED, 2026-09-05 audit)

SACClippedDoubleCritic (scale_rl/agents/sac/sac_network.py) fails to even initialize under this project's installed jax==0.4.34/flax==0.8.4: its nn.vmap(..., in_axes=None) rejects the observations/actions kwargs with "Expected None, got (Array...)" — confirmed by reproducing it in total isolation from any experiment code (see tests/test_angle_2b_smoke.py's make_agent_cfg comment for the exact failure and where it was first isolated). critic_use_cdq is wired to ${env.episodic} in configs/agent/sac_simba.yaml, and every current environment config (configs/env/dmc_hard.yaml, dmc_medium.yaml, myosuite_hard.yaml, myosuite_medium.yaml, "mujoco .yaml") sets episodic: false — so critic_use_cdq resolves to false everywhere in this study, and CDQ is never actually exercised. Left as-is deliberately: not planned to be used for this study, and fixing the underlying nn.vmap incompatibility is out of scope.

MyoSuite state-reset fidelity (RESOLVED for core-4, 2026-09-07 - see Angle 2A's "Environment support" note below)

Originally flagged as a concern about musculoskeletal dynamics (muscle activation/tendon state) exceeding standard joint kinematics. Investigation (2026-09-07) found the real mechanism was different and more specific: (1) MyoSuite's own set_env_state() silently advances physics by one step instead of restoring it (bypassed); (2) MyoSuite's Robot interface steps physics on its own mj_data object, distinct from the env's own mj_data (a passive mirror) - restoring the wrong one silently no-ops; (3) some MyoSuite tasks (myoLegWalk-v0 confirmed) maintain plain-Python step counters outside any mj_data field, used in observations. All three are handled in experiments/angle_2a/env_state.py, verified bit-exact via a standing, permanent determinism smoke test (tests/test_angle2a_env_state_determinism_smoke.py) covering all 14 environments in this study (not a one-time manual check). 13 of 14 pass exactly; myo-baoding-p1 (Angle-3-only held-out, never used by Angle 1/2) has its own unfixed task-specific counter (self.counter) causing a small residual - not a blocker for Angle 1/2/2A, would need fixing before Angle 3 could rely on exact-state rollouts for it, if that's ever needed.

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

Reusing an Angle 2A construct-validity prerequisite check seed in the main 5-seed comparison (or vice versa), or turning that check's descriptive per-seed report into an automated pass/fail gate without explicitly flagging the change — both were deliberate project decisions (2026-09-06), not omissions.

Reverting Angle 2A's MC rollout continuation to the deterministic/mean action (temperature=0), or computing sigma_rollout by pooling raw returns across representative (s,a) pairs before taking a standard deviation — both were confirmed, deliberate corrections (2026-09-06); reintroducing either would silently make R calibration meaningless (zero measured variance, or an inflated between-state-conflated one) without any error to signal it.

Injecting Angle 3's non-Q-pathway control noise directly into the Q-term rather than into the combined (post-entropy-addition) gradient — this collapses the control into the real condition and defeats its purpose.

Using an early checkpoint of the scaled critic as Angle 3's "reference-ceiling" arm instead of an independently-trained, matched-steps baseline-architecture critic — this reintroduces the staleness confound Angle 2A's reference-critic design already solved.

Describing Angle 3's synthetically-corrected arm as a proposed real-world fix or deployable method — it is an artificial, undeployable intervention used only to test the causal chain.

Pending / Open Items

Angle 2A resumability has no end-to-end determinism/correctness test against a real Angle 2A run (raised 2026-09-07, explicitly deferred by project decision — not resolved, not forgotten). Unit-level coverage (tests/test_angle2a_resumability.py) confirms the mechanics in isolation: a resumed run reaches the correct final step, already-captured probe data survives a simulated interruption byte-for-byte, an already-fully-done MC probe is not recomputed, and a partially-done MC probe resumes from exactly the interrupted rollout. What is explicitly NOT tested: bit-exact reproducibility of a resumed run versus an uninterrupted control (seed_global_rng_for_agent is called unconditionally on resume, same as Angle 1's own precedent, so post-resume RNG continuity is not bit-exact — an accepted, pre-existing-pattern limitation, not new), and behavior under a real, full-length Angle 2A matchup rather than the short synthetic runs used in unit tests. Raise this again the next time Angle 2A's status is discussed — the user will provide a decision once real Angle 2A runs exist to evaluate against.

Claiming a performance/outcome improvement anywhere in Angle 2 (A, B, or C) — that claim is only supportable within Angle 3's continued-training design.