# EchoCritic investigation — available-evidence report

Date: 24 October 2026. **Scope incomplete: seven-run pilot analyzed; complete multi-seed campaign inaccessible.**

## 1. Executive summary

The available evidence establishes architecture-dependent fitting transients within this pilot and an environment-dependent relationship between probe endpoints and policy returns. It does not establish seed robustness, causality, or a novel scientific contribution. The full investigation cannot be completed without the campaign export or a working Forge credential. A stored W&B credential was tried against the explicitly supplied Forge API and returned HTTP 401 invalid api key. No secret is included in outputs.

## 2. Inventory and provenance

The immutable input is `/Users/shouryakaveti/VS_Projects/echo_pilot_local_partial.zip`. Its MANIFEST explicitly identifies it as a partial local pilot archive, not a complete W&B export. All listed payload hashes were verified. The seven unique pilot conditions all use seed 1: dog-run has three architectures, myo-key-turn has three, humanoid-run has only D4W1024. There is no myo-pen-twirl. No pilot condition is counted as an Exp96 replicate. The reported 103 W&B records, 95 finished and eight crashed are owner-supplied counts, not independently verified. Crash causes, restarts, duplicates, canonical seeds and campaign completion remain unknown.

The recorded pilot source revision is f8619dd5f494399d9febc79fc9092010234c714d. Probe, scheduler, SAC updates and base configuration were recovered from GitHub at that exact revision and saved in source/. This source is evidence for the pilot only; campaign implementation identity is unverified. Run inventory retains each recorded commit and status. Recorded complete status is not proof of campaign membership.

## 3. Experimental methodology

The pilot has 500,000 interaction steps, corresponding to the recorded 1,000,000 raw environment-step budget. Evaluation records occur at 0 and every 50,000 interactions (11 observations). The base source specifies actor D1W128, two updates per interaction, five evaluation episodes, 20 paired probe checks, five repetitions, pool 25,600, batch 256 and 2,000 fitting updates. These are source defaults, not a substitute for missing resolved per-run configs. Exact effective optimizer overrides, parameter counts and dimensions require resolved configurations or model-state evidence. Do not import later campaign settings retrospectively.

## 4. Metric definitions and validation

The recovered probe fits a critic copy to a + sin(100000 * teacher(s,a)), with a fixed offset equal to that fitted critic's initial pool-mean prediction. Each round has its own pool, teacher and minibatch stream; current and fresh share those within the round but use their own offsets. Fresh parameters are captured before SAC updates and reused. The optimizer is fresh-state AdamW configured from critic settings; its state continues through all 2,000 fit steps. The first 1,000 and subsequent 1,000 minibatches use separate deterministic streams. Targets are not variance-normalized.

The fitting curve stores pre-update MSE on changing minibatches. Its last point differs from the post-fit full-pool MSE. P = Var(base targets) minus final pool MSE; this is a variance-referenced difference, not a ratio. L = P(fresh) minus P(current), calculated in stored FP32 before summarization. Positive L indicates worse current adaptation. IQM uses scipy trim_mean with 25% removed per tail; with five rounds it averages the middle three. Differences of separate IQMs need not equal the IQM of paired differences. Score identities and final IQMs are independently recomputed in derived/. Probe-repeat intervals are conditional on one trained critic and are not between-seed intervals.

SAC critic loss is squared Bellman residual on training minibatches, distinct from probe MSE. Targets include reward, discounted target-network Q and entropy adjustment, masked by termination. Actor loss averages alpha times log probability minus Q. Q means are function outputs; gradient/parameter norms describe parameter-space quantities. Recorded metric presence alone does not validate every actor diagnostic or evaluation implementation; these definitions remain partly unaudited.

## 5–7. Learning performance, scaling and environment interactions

| run_key | environment | architecture | seed | eval_count | final_return | final_window_mean | peak_return | peak_step | peak_to_final | late_to_peak_ratio | observed_auc | observed_end | initial_fresh | final_deficit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp1_D2W512_dog-run_seed1 | dog-run | D2W512 | 1 | 11 | 612.6442756870747 | 589.6746143004286 | 612.6442756870747 | 500000.0 | 0.0 | 0.9625073434973568 | 170176765.16212803 | 500000.0 | 0.4929951628049214 | 0.2411078860362371 |
| exp1_D2W512_myo-key-turn_seed1 | myo-key-turn | D2W512 | 1 | 11 | 1038.8698043921506 | 1036.8854650989242 | 1041.5745813562792 | 400000.0 | 2.7047769641285413 | 0.9954980504120512 | 450897471.76242125 | 500000.0 | 0.4973500470320384 | 0.4884779552618662 |
| exp1_D4W1024_dog-run_seed1 | dog-run | D4W1024 | 1 | 11 | 446.43675992565727 | 428.8429339828098 | 446.43675992565727 | 500000.0 | 0.0 | 0.9605905527453042 | 136347179.07280263 | 500000.0 | 0.4934489826361338 | -0.0135018030802408 |
| exp1_D4W1024_humanoid-run_seed1 | humanoid-run | D4W1024 | 1 | 11 | 190.20415942994913 | 185.5228692721018 | 190.20415942994913 | 500000.0 | 0.0 | 0.9753880768334542 | 61400442.730222106 | 500000.0 | 0.4952596525351206 | -0.0547762016455332 |
| exp1_D4W1024_myo-key-turn_seed1 | myo-key-turn | D4W1024 | 1 | 11 | 1007.4818735200508 | 1024.9429133704225 | 1042.403953220794 | 450000.0 | 34.9220797007431 | 0.9832492578367332 | 451817945.9866239 | 500000.0 | 0.4901328285535176 | 0.4324833452701568 |
| exp1_D4W1536_dog-run_seed1 | dog-run | D4W1536 | 1 | 11 | 468.9899953391548 | 443.8432723058691 | 468.9899953391548 | 500000.0 | 0.0 | 0.9463811098675984 | 132737240.85709731 | 500000.0 | 0.495572030544281 | -0.0219487448533376 |
| exp1_D4W1536_myo-key-turn_seed1 | myo-key-turn | D4W1536 | 1 | 11 | 544.3941380754541 | 913.2877325564226 | 1285.4705159646237 | 400000.0 | 741.0763778891696 | 0.7104696072092224 | 527549433.5621976 | 500000.0 | 0.4949985841910044 | 0.5984749794006348 |

All summaries describe individual trajectories. Final window is the mean of observed evaluations at >=90% of each trajectory's observed endpoint (two points here). AUC is trapezoidal integration of observed evaluations, explicitly a numerical integral across sparse observations, with units return × interaction steps. It does not reconstruct unobserved behavior. Peak is susceptible to evaluation noise. Across-seed SD, confidence intervals and population IQM are not estimable with one seed per cell; reporting degenerate bootstrap intervals would mislead.

Dog-run's small critic ends at 612.64 versus 446.44 and 468.99 for the scaled critics. Medium versus large is nonmonotonic in final return. Myo-key-turn's large critic has higher peak/AUC but worse final return, so an endpoint-only ranking misses its earlier advantage. Humanoid architecture comparisons are unavailable.

## 8. Fresh fitting dynamics

| run_key | arm | median_amplification | max_amplification | median_final_pool | median_peak_index |
| --- | --- | --- | --- | --- | --- |
| exp1_D2W512_dog-run_seed1 | current | 1.0 | 1.2141062 | 0.0509774 | 0.0 |
| exp1_D2W512_dog-run_seed1 | fresh | 1.7819318 | 4.730617 | 0.010193745 | 2.0 |
| exp1_D2W512_myo-key-turn_seed1 | current | 1.1430905 | 1.8214769 | 0.57585335 | 2.0 |
| exp1_D2W512_myo-key-turn_seed1 | fresh | 1.3658154 | 5.8888955 | 0.063421726 | 2.0 |
| exp1_D4W1024_dog-run_seed1 | current | 1.0 | 1.3517617 | 0.005087622 | 0.0 |
| exp1_D4W1024_dog-run_seed1 | fresh | 31.526655 | 69.00778 | 0.019305358 | 2.0 |
| exp1_D4W1024_humanoid-run_seed1 | current | 1.0096564 | 1.3200654 | 0.0070477063 | 1.0 |
| exp1_D4W1024_humanoid-run_seed1 | fresh | 18.756504 | 46.98257 | 0.049720917 | 2.0 |
| exp1_D4W1024_myo-key-turn_seed1 | current | 1.1279801 | 2.256575 | 0.529822 | 1.0 |
| exp1_D4W1024_myo-key-turn_seed1 | fresh | 29.353558 | 76.13102 | 0.08497616 | 1.0 |
| exp1_D4W1536_dog-run_seed1 | current | 1.5230125 | 4.7651525 | 0.005126712 | 10.5 |
| exp1_D4W1536_dog-run_seed1 | fresh | 101.86913 | 322.01605 | 0.029729333 | 2.0 |
| exp1_D4W1536_myo-key-turn_seed1 | current | 1.0589744 | 1.9386538 | 0.6790842 | 1.0 |
| exp1_D4W1536_myo-key-turn_seed1 | fresh | 88.40151 | 308.67865 | 0.1428349 | 2.0 |

Medians pool checkpoints and repetitions descriptively, not inferentially. Full per-repetition peak, amplification, peak index, first recovery below initial loss, integrated excess and final pool residual are in fit_dynamics.csv. Recovery is the first crossing and does not guarantee persistent recovery. Integrated excess is a discrete sum over update-index losses, not a held-out generalization measure. Figure fitting curves preserve individual repetition traces without smoothing. This demonstrates a transient in observed minibatch errors, not by itself numerical divergence or permanent plasticity loss.

## 9–10. Adaptation and output scale

| run_key | recomputed | recorded | absolute_error | fresh_initial | fresh_final |
| --- | --- | --- | --- | --- | --- |
| exp1_D2W512_dog-run_seed1 | 0.24110788106918335 | 0.2411078860362371 | 4.967053740534411e-09 | 0.49299517273902893 | 0.48801830410957336 |
| exp1_D2W512_myo-key-turn_seed1 | 0.4884779453277588 | 0.4884779552618662 | 9.93410742555767e-09 | 0.4973500669002533 | 0.4281679093837738 |
| exp1_D4W1024_dog-run_seed1 | -0.013501803390681744 | -0.0135018030802408 | 3.10440943784851e-10 | 0.49344897270202637 | 0.4787208139896393 |
| exp1_D4W1024_humanoid-run_seed1 | -0.05477620288729668 | -0.0547762016455332 | 1.241763476766966e-09 | 0.4952596426010132 | 0.44198885560035706 |
| exp1_D4W1024_myo-key-turn_seed1 | 0.43248334527015686 | 0.4324833452701568 | 5.551115123125783e-17 | 0.49013280868530273 | 0.40291714668273926 |
| exp1_D4W1536_dog-run_seed1 | -0.021948745474219322 | -0.0219487448533376 | 6.208817210362483e-10 | 0.4955720007419586 | 0.4701153039932251 |
| exp1_D4W1536_myo-key-turn_seed1 | 0.5984749794006348 | 0.5984749794006348 | 0.0 | 0.494998574256897 | 0.3676798641681671 |

The scaled dog-run critics end with negative deficits, while D2W512 has a positive deficit of 0.2411 despite better policy return. Myo-key-turn deficits are positive for all three sizes and largest for D4W1536 (0.5985), but medium is lower than small. Neither deficit-size nor deficit-return relationships are globally monotonic. Fresh-reference scores can change because the replay pool, teacher and minibatches change across checks even when fresh parameters are fixed. This protocol does not isolate replay-distribution effects from target-task effects.

A large mean Q alone does not algebraically imply large initial probe MSE: the per-critic mean offset removes the pool mean prediction. Initial error depends on centered prediction variation and covariance with the sine target. Output dispersion and conditioning, rather than mean alone, must be evaluated to establish an output-scale mechanism. Stored mean Q cannot identify that decomposition.

## 11–13. Timing, propagation and collapse

Evaluations are 50,000 interactions apart and cannot locate a precise collapse onset. The D4W1536 myo-key-turn peak at 400,000 interactions is 1285.47; final is 544.39, a drop of 741.08 (57.65%). Its final-window mean remains 913.29, showing sensitivity to endpoint choice. At most two subsequent evaluations exist: long-term persistence and recovery are unresolved. The other two myo-key-turn runs decline by 2.70 and 34.92 from their own peaks. This makes the large-critic case distinctive within the pilot, not representative across seeds.

Lagged prediction with seven heterogeneous single-seed runs would be unstable and cannot validate generalization. Repeated checkpoint correlations would confound training stage and environment. No validated critic-to-actor predictor or causal propagation claim is available. Figures expose return, TD loss, Q and actor traces for inspection; they do not establish event ordering precisely enough for causality.

## 14–17. Counterexamples, mechanisms, robustness and strongest finding

Architecture-dependent optimizer sensitivity explains fresh spikes and their size dependence, but minibatch changes and unknown conditioning remain alternatives. Function-space overshoot is consistent with large transient MSE but requires fixed-pool intermediate predictions to separate it from changing-minibatch variability. Persistent representation loss cannot explain why fresh critics spike, and scaled dog-run deficits contradict universal trained-critic deterioration. Mean-Q mismatch is weakened by the fitted critic's own offset. A 2,000-step budget limitation remains plausible for residual errors; full-pool residuals at multiple budgets would distinguish it. Numerical instability is unproven without nonfinite outputs or numerical traces. Actor exploitation and representation deterioration remain hypotheses requiring interventions.

No cross-seed statistical claim, equivalence claim, crash-instability claim or novelty claim is warranted. No confirmatory hypothesis test is performed, so multiple-testing corrections are inapplicable to these descriptive pilot outputs. Selection of this pilot and exploratory endpoints limits external validity. The strongest defensible current statement is: this pilot contains architecture-dependent fresh fitting transients, environment-dependent trained-versus-fresh endpoints, and a policy/probe dissociation. Campaign replication is essential before publication.

## 18–19. Follow-up and research direction

First recover the complete campaign records, unsampled histories, resolved configs, files, logs and source commits. This CPU-only evidence acquisition has greater immediate information value than a new training launch. Then use training seed as the unit, pair architectures only when seed coupling is justified, compare within environments, and bootstrap complete seed trajectories. Keep missing runs explicit; analyze common observed horizons and complete-horizon subsets separately. Predeclare estimands and sensitivity thresholds before confirmatory inference.

Do not change research direction based on these seven runs. The owner retains that decision. The separate discovery ledger and roadmap specify conditional tests. No GPU jobs, new training, frozen-source modifications or engineering corrections were performed.
