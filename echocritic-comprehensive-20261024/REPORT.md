# EchoCritic: complete-record multi-seed scientific investigation

**24 October 2026. CPU analysis of all 103 W&B records.** This report supersedes the seven-run report's scientific coverage, while preserving that preliminary report unchanged. It is an exploratory observational investigation, not a preregistered confirmatory analysis and not authorization to change the methodology.

## 1. Executive summary

The strongest result is a dissociation between critic adaptation diagnostics and policy utility. Critic scaling reliably changes the probe, but does not produce one universal direction of policy improvement, instability or plasticity loss. At the common 70% training horizon, larger critics have lower mean trained-versus-fresh deficits than the small critic in every environment. Yet dog-run favors the small critic in policy return, humanoid-run shows uncertain performance differences, myo-key-turn has similar mean policy utility with intermittent drops, and myo-pen-twirl combines highly variable learning success with some of the largest deficits in its better-performing conditions.

Three findings limit a simple critic-to-actor pathology story. First, fresh-reference scores deteriorate in all 96 conditions by 70%, despite no SAC training of those fresh parameters. Second, large humanoid critics acquire a strong apparent adaptation advantage mainly because the fresh reference gets worse, rather than because the trained critic's score improves. Third, held-out future-return prediction benefits primarily from Q information in the Myo tasks; a plasticity-deficit feature alone has weak and uncertain incremental value. These patterns are consistent with evolving task difficulty, optimization and value-scale interactions. They do not identify a unique causal mechanism.

The available evidence cannot establish multi-seed replication of the pilot's early fitting-loss amplification. Full fitting curves are absent from the 103 W&B run file inventories. One separately archived original campaign checkpoint is available and numerically bound to its W&B scores; it demonstrates a large trained/fresh fitting discrepancy for one small critic, not an architecture-by-seed replication. This limitation is explicit rather than being filled with score-based guesses.

All eight crashes involve large critics, seeds 7–8, on one host. Last histories cluster within 351.68 seconds of wall-clock time, despite different start times and training fractions. A shared external interruption is plausible; numerical failure is not established. The eight missing terminal values are never set to zero. At the terminal horizon large-critic n=6; common-70% analyses retain n=8 for every cell.

## 2. Exact dataset inventory and provenance

The read-only API export contains 103 unique run IDs, 95 finished and eight crashed. It consists of seven source-f861 pilot records and 96 source-f597 campaign records. The campaign exactly fills four environments × three architectures × seeds 1–8, with one run ID per condition. Seven display names occur across the two cohorts; they are not used as identifiers. No duplicate campaign condition was found. The record census, complete configurations, unmodified histories, file lists and runtime metadata are preserved under raw_api/. A SHA256 manifest covers 697 files. Every exported history count matches the API's reported historyLineCount. No checkpoint row or probe round is counted as a trained seed.

Campaign source: `f597ee9898e4d401912092c336c3e5330cf2fda1`. Pilot source: `f8619dd5f494399d9febc79fc9092010234c714d`. Original probe, SAC, trainer and logging implementations were retrieved directly from those Git objects; 52 initial source snapshots and their checksums are retained. W&B runtime Git labels and launch arguments agree with the campaign source. These labels do not establish a full clean-worktree certificate for every run; runtime metadata does not supply all archived tree hashes. Configured optimizer settings are verified for all 103 records, while exact restored optimizer states are not in the export.

A crucial overlap: each of the seven pilot conditions has a campaign seed-1 counterpart with **7,989 exactly matching observed training/evaluation values**. All 55,923 compared values match exactly. Runtime metadata describes separate source-bound launches, but this is the same observed seeded training trajectory and cannot independently corroborate the pilot. Pilot records are excluded from all campaign inference; exclusion of campaign seed 1 is a further robustness analysis. This does not establish that checkpoints were reused, nor does it establish independence merely from different run IDs.

| pilot | campaign | environment | architecture | shared_values | exact_matches |
| --- | --- | --- | --- | --- | --- |
| ecls9oz9 | 3hfsq13w | dog-run | D4W1536 | 7989 | 7989 |
| eoja0jna | 7egofbq8 | dog-run | D2W512 | 7989 | 7989 |
| es456sr1 | onrxf1xo | myo-key-turn | D4W1536 | 7989 | 7989 |
| hy3xeazk | lcmc7x3b | myo-key-turn | D2W512 | 7989 | 7989 |
| iwapnp9m | 5z1wpev1 | dog-run | D4W1024 | 7989 | 7989 |
| xcyqf9a1 | jbcstd01 | humanoid-run | D4W1024 | 7989 | 7989 |
| xthjkzd5 | 86itjkh3 | myo-key-turn | D4W1024 | 7989 | 7989 |


The eight crashed records and their wall-clock evidence are:

| id | env | seed | last_timestamp | last_iso | runtime | start | host |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 46znu9zr | humanoid-run | 7 | 1.7927e+09 | 2026-10-22T19:33:06.004885+00:00 | 60957 | 2026-10-22T02:37:08.596106Z | gpuc05.delta.ncsa.illinois.edu |
| 6apfwv2e | myo-pen-twirl | 7 | 1.7927e+09 | 2026-10-22T19:29:11.650210+00:00 | 60904 | 2026-10-22T02:37:09.203579Z | gpuc05.delta.ncsa.illinois.edu |
| 6dwfq82o | humanoid-run | 8 | 1.7927e+09 | 2026-10-22T19:31:13.130834+00:00 | 53633 | 2026-10-22T04:39:50.566903Z | gpuc05.delta.ncsa.illinois.edu |
| 7blwjacl | myo-pen-twirl | 8 | 1.7927e+09 | 2026-10-22T19:32:31.450840+00:00 | 53559 | 2026-10-22T04:39:51.674805Z | gpuc05.delta.ncsa.illinois.edu |
| 7yxtlkdl | dog-run | 7 | 1.7927e+09 | 2026-10-22T19:27:48.793407+00:00 | 60817 | 2026-10-22T02:37:09.203468Z | gpuc05.delta.ncsa.illinois.edu |
| fxb32tqc | myo-key-turn | 8 | 1.7927e+09 | 2026-10-22T19:30:51.031036+00:00 | 53621 | 2026-10-22T04:39:51.674706Z | gpuc05.delta.ncsa.illinois.edu |
| imgo2az5 | dog-run | 8 | 1.7927e+09 | 2026-10-22T19:28:52.446238+00:00 | 53503 | 2026-10-22T04:39:51.856085Z | gpuc05.delta.ncsa.illinois.edu |
| xdlcj5d9 | myo-key-turn | 7 | 1.7927e+09 | 2026-10-22T19:33:40.472926+00:00 | 60991 | 2026-10-22T02:37:09.300394Z | gpuc05.delta.ncsa.illinois.edu |


All lack uploaded output.log files. Their file inventories contain runtime metadata and requirements plus an artifact marker, not an exception trace. They stop at 78.4–92.0% of raw budget. The shared host and synchronized endings favor investigating scheduler/node/allocation evidence first. Because that evidence is absent, the definitive cause remains unknown. A read-only Delta attempt was made to recover additional original files; existing SSH authentication was unavailable. No job was launched and no credentials or historical artifacts were modified.

## 3. Experimental methodology reconstructed from configurations and source

All campaign tasks have 1,000,000 raw environment steps, one training environment, action repeat 2, and therefore 500,000 interactions. Critic architectures are D2W512, D4W1024 and D4W1536; actor architecture is fixed D1W128. UTD is two updates per sample-ready interaction, replay capacity one million, warmup/buffer minimum 5,000 interactions, batch 256. Number of actual optimizer updates depends on sample readiness and the observed endpoint; the logged W&B step is not an optimizer counter. Gamma is .99 for dog/humanoid and .95 for Myo; tau is .005. Configured actor/critic AdamW learning rates are 1e-4, weight decay .01; temperature learning rate is 1e-4 with zero decay and initial .01. All use the single critic (`critic_use_cdq=false`), so q1_mean and q2_mean are duplicate reporting views, not independent critics. Observation normalization is enabled. Mixed precision is false; the source and runtime precision metadata, rather than a generic claim of GPU determinism, define arithmetic provenance.

Parameter counts in the W&B configurations are rounded strings, retained in run_inventory.csv. They distinguish actor and critic capacity, but they are not exact integer leaf counts. Original parameter trees and complete environment dimension specifications are missing for most runs, so no fabricated exact parameter counts are reported. Environment identifiers, full env settings and installed package versions are retained in per-run metadata/configuration/requirements; software-version consistency is not equivalent to numerical equivalence across GPU executables.

Campaign probes use 50 checks at 2% budget intervals and ten paired rounds per check; check 0 is fresh-only at 5,000 interactions. Each fit uses 2,000 continuous optimizer updates, batch 256, a 25,600-sample replay pool, sine frequency 100,000 and pool evaluation chunks of 2,560. Pilot probes use five rounds and 20 checks, making direct endpoint pooling scientifically invalid. Each fresh-reference parameter set is captured before SAC updates and reused; it is not a newly resampled fresh initialization at every checkpoint. Teacher, replay pool and minibatch streams change across rounds/checks.

Evaluation is every 50,000 interactions, giving 11 observations including initialization for complete runs. **Intermediate evaluations use five episodes, but the terminal evaluation uses 50 episodes**, as explicitly implemented by the source trainer. Consequently transient evaluation dips and terminal means have different measurement noise. Episode-level returns are not uploaded, so episode uncertainty cannot be reconstructed. Terminal-success attribution has a known historical implementation defect corrected later in PR #10; this report does not treat archived success fields as corrected outcomes. Episode returns themselves are accumulated independently of that success-attribution logic.

## 4. Metric definitions and numerical validation

Training TD loss is minibatch squared Bellman residual. The single-critic target is reward + gamma^n (1-terminated) × [target Q(next state, sampled next action) − alpha log pi(next action)]. Time-limit truncation is not treated as terminal in that expression. Target-network updates use the original SAC implementation. Actor loss averages alpha log pi − Q. Thus actor loss is intrinsically value-scale dependent and cannot be interpreted as a dimensionless pathology score. Mean Q is a function-space output; parameter norms and gradients are parameter-space quantities. Q mean alone does not measure value error or overestimation. TD-error variance is logged, but Q variance/extrema, held-out Bellman target statistics and calibrated rollout-based Q errors are not generally available. Actor gradients, entropy, temperature, KL and churn are available; full critic-gradient distributions are not.

For a probe, `g(s,a) = offset + sin(100000 × teacher(s,a))`. The offset is the fitted critic's own initial mean prediction on the fixed pool, held constant during fitting. Fresh and trained fits share pool, teacher/base targets and minibatch index sequence, but their offsets can differ; the literal target arrays are therefore not generally identical. Each fit receives a fresh AdamW optimizer state using the critic's configured optimizer. Parameters and AdamW state continue through all 2,000 updates. The first 1,000 draws are preserved and the subsequent stream is separately generated. Logged fitting trajectories, where available, contain pre-update loss on changing minibatches. Final fitting error is evaluated after optimization on the full pool. These are different measurements.

Baseline b is variance of sine targets; targets are not normalized to unit variance. The score is P=b−final pool MSE, a difference in squared-target units. The paired deficit is L=P_fresh−P_current; because b is shared within a pair, it corresponds to current minus fresh final pool error, subject to source FP32 subtraction. Larger positive values mean a trained disadvantage; negative values mean a trained advantage on this conditional fitting task. A low or negative L does not by itself imply good absolute trainability, policy utility or retained representation quality.

The ten-round campaign uses a fractional empirical IQM: sorted positions 2 and 7 each receive weight .1, positions 3–6 weight .2. This is different from simply dropping two observations at each end. Five-round pilot IQM averages the middle three. Using the exact campaign rule after FP32 paired subtraction reproduces **all 4,738 nonzero-check logged deficits exactly**. W&B logs fresh IQM only at check 0; later fresh IQMs in this analysis are explicitly reconstructed from the ten raw scores. Using the stale summary fresh-IQM field as a late observation would be wrong. All 103 history counts and step integrity checks pass. Independently reconstructed final returns agree within 2.28e-13. The separately archived original probe matches its ten fresh/current W&B round scores to machine precision and satisfies P=b−MSE exactly in stored FP32. VALIDATION.json records all 12 checks.

## 5. Multi-seed learning performance

Terminal means, descriptive 95% percentile seed-bootstrap intervals, medians, IQMs and between-seed sample SD are shown below. n is independent trained trajectories per cell; the large-critic terminal columns use only seeds 1–6. The pilot is never added as a ninth seed.

| environment | architecture | n | mean | ci_low | ci_high | median | iqm | sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dog-run | D2W512 | 8 | 551.46 | 513.63 | 590.42 | 533.6 | 543.55 | 59.045 |
| dog-run | D4W1024 | 8 | 477.1 | 445.71 | 504.27 | 488.44 | 481.93 | 44.925 |
| dog-run | D4W1536 | 6 | 470.88 | 439.1 | 499.9 | 480.2 | 477.9 | 42.018 |
| humanoid-run | D2W512 | 8 | 174.47 | 160.25 | 191.71 | 167.72 | 169.44 | 24.583 |
| humanoid-run | D4W1024 | 8 | 179.92 | 168.78 | 193.58 | 172.46 | 174.92 | 19.596 |
| humanoid-run | D4W1536 | 6 | 172.1 | 161.88 | 182.39 | 171.41 | 171.94 | 13.855 |
| myo-key-turn | D2W512 | 8 | 1123.9 | 1048.9 | 1199.9 | 1119.3 | 1121.5 | 117.13 |
| myo-key-turn | D4W1024 | 8 | 1147.8 | 1070 | 1227 | 1150.1 | 1144.8 | 121.7 |
| myo-key-turn | D4W1536 | 6 | 1049.6 | 826.43 | 1219 | 1091.4 | 1116 | 274.54 |
| myo-pen-twirl | D2W512 | 8 | 1571 | 838.52 | 2130.3 | 2047.1 | 2000.8 | 970.17 |
| myo-pen-twirl | D4W1024 | 8 | 750.79 | 190.88 | 1392.4 | 189.87 | 433.6 | 945.3 |
| myo-pen-twirl | D4W1536 | 6 | 742.21 | 219.98 | 1340.8 | 601.22 | 649.91 | 808.67 |


Dog-run's small critic ends around 551.46, versus 477.10 for medium and 470.88 for observed large completers. The paired medium-minus-small difference is −74.36 reward units [−126.83, −19.10], with the medium higher in only 1/8 seed pairs. At 70%, all eight seeds contribute: medium-minus-small is −71.49 [−107.57, −29.44]. The large critic's mean observed terminal performance is also lower, but common-horizon paired uncertainty and missing terminal runs limit a universal ranking of the two scaled architectures.

Humanoid means are 174.47, 179.92 and 172.10. Paired differences remain uncertain; medium-minus-small terminal difference is 5.45 [−15.98, 23.01]. This is not evidence of equivalence. Scaling shifts adaptation measurements much more strongly than these policy endpoints.

Myo-key-turn means are 1123.89, 1147.82 and 1049.61 among completers. Its large-critic interval is wide because of intermittent drops, including the pilot-overlapping seed-1 terminal decline. Medium-minus-small difference is 23.92 [−64.30, 124.60]. Mean return is not monotonically improved or degraded by size. Myo-pen-twirl is much more heterogeneous: means 1571.02, 750.79 and 742.21, with broad intervals and mixtures of successful and near-failed learning. Small-minus-medium differences are clearest for earlier integrated performance; medium-minus-small AUC through 70% is −679.79 [−1095.78, −270.63] normalized-budget × return units. The final paired difference is less precise [−1580.56, 16.85].

## 6. Architecture scaling: speed, peak, variability and sensitivity

AUC is trapezoidal integration over observed evaluations. It does not fabricate unobserved evaluations; all intervals between sparse evaluation points remain unobserved. The primary common window is [0,.7], before every crashed trajectory stops. Final-window return averages the observed .9 and 1.0 evaluations for complete runs. All seed summaries—including means/medians over observed evaluations, temporal variability, peaks and AUC—are in seed_endpoints.csv; observed temporal variability is not a new independent-run sample size.

| environment | architecture | metric | n | mean | ci_low | ci_high | sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dog-run | D2W512 | peak_return | 8 | 553.93 | 515.48 | 591.73 | 58.021 |
| dog-run | D2W512 | auc70 | 8 | 189.05 | 171.72 | 208.18 | 28.327 |
| dog-run | D4W1024 | peak_return | 8 | 486.07 | 467.39 | 505.55 | 30.104 |
| dog-run | D4W1024 | auc70 | 8 | 174.49 | 166.47 | 181.48 | 11.79 |
| dog-run | D4W1536 | peak_return | 8 | 462.94 | 435.62 | 489.12 | 42.281 |
| dog-run | D4W1536 | auc70 | 8 | 158.57 | 143.66 | 174.88 | 24.465 |
| humanoid-run | D2W512 | peak_return | 8 | 174.47 | 160.55 | 191.72 | 24.583 |
| humanoid-run | D2W512 | auc70 | 8 | 49.463 | 41.747 | 56.982 | 11.557 |
| humanoid-run | D4W1024 | peak_return | 8 | 179.92 | 169.03 | 194.22 | 19.596 |
| humanoid-run | D4W1024 | auc70 | 8 | 52.962 | 42.422 | 63.122 | 15.865 |
| humanoid-run | D4W1536 | peak_return | 8 | 174.97 | 166.35 | 183.56 | 13.609 |
| humanoid-run | D4W1536 | auc70 | 8 | 49.946 | 44.053 | 55.223 | 8.8003 |
| myo-key-turn | D2W512 | peak_return | 8 | 1173.6 | 1100.2 | 1242.8 | 110.2 |
| myo-key-turn | D2W512 | auc70 | 8 | 619.28 | 573.41 | 668.25 | 72.768 |
| myo-key-turn | D4W1024 | peak_return | 8 | 1155.1 | 1083.4 | 1227.9 | 112.81 |
| myo-key-turn | D4W1024 | auc70 | 8 | 602.21 | 536.6 | 676.7 | 107.88 |
| myo-key-turn | D4W1536 | peak_return | 8 | 1204.2 | 1124 | 1281.9 | 124.4 |
| myo-key-turn | D4W1536 | auc70 | 8 | 622.45 | 569.63 | 667.97 | 76.228 |
| myo-pen-twirl | D2W512 | peak_return | 8 | 1582.5 | 838.15 | 2135.2 | 967.31 |
| myo-pen-twirl | D2W512 | auc70 | 8 | 824.95 | 399.08 | 1192.9 | 607.35 |
| myo-pen-twirl | D4W1024 | peak_return | 8 | 758.37 | 191.45 | 1407.5 | 957.62 |
| myo-pen-twirl | D4W1024 | auc70 | 8 | 145.16 | 15.99 | 398.29 | 356.69 |
| myo-pen-twirl | D4W1536 | peak_return | 8 | 1228.2 | 563.09 | 1873.3 | 1030.7 |
| myo-pen-twirl | D4W1536 | auc70 | 8 | 543.98 | 147.22 | 939.91 | 589.1 |


The direction of medium-versus-large effects is often nonmonotonic. Medium dog-run has higher common-horizon AUC than large, while final differences are uncertain. Pen-twirl medium has especially poor early AUC even relative to large; its final outcomes reflect delayed learning in some seeds rather than one simple divergence phenotype. Large key-turn can peak strongly and still suffer late or intermediate drawdowns. Peak comparisons have different observed horizons for crashed runs and are selection-biased maxima of noisy evaluations; the complete-case/common-horizon contrasts carry more inferential weight.

Leave-one-seed sensitivity is reported for each main contrast. The mean 70% deficit reduction of scaled versus small critics survives removing any one seed in every environment. Excluding pilot-overlapping seed 1 also preserves these directions. For pen-twirl performance, individual seeds substantially change magnitudes, so wide uncertainty and bimodality matter more than a single group mean. No environment-normalized reward aggregate is used to hide incompatible reward scales.

## 7. Environment × architecture interactions

The data support strong interaction in the *meaning of probe scores*: the scaled DMC critics generally have negative deficits, while all 48 Myo runs have positive deficits at their final observed probe. This is not merely reward rescaling, because the signs and architecture rankings differ within the same environment. Nonetheless reward/Q scales are much larger in Myo. Near 60–70% training, the median of run-level reward/Q/TD summaries is:

| environment | architecture | reward | q | td | tdvar |
| --- | --- | --- | --- | --- | --- |
| dog-run | D2W512 | 0.46027 | 82.733 | 0.033773 | 0.033338 |
| dog-run | D4W1024 | 0.44354 | 82.747 | 0.062629 | 0.061775 |
| dog-run | D4W1536 | 0.39527 | 80.606 | 0.057249 | 0.056433 |
| humanoid-run | D2W512 | 0.12064 | 26.53 | 0.0094208 | 0.0092877 |
| humanoid-run | D4W1024 | 0.14278 | 27.892 | 0.0087109 | 0.0085673 |
| humanoid-run | D4W1536 | 0.13031 | 27.75 | 0.014148 | 0.013927 |
| myo-key-turn | D2W512 | 19.529 | 532.56 | 2.0909 | 2.0751 |
| myo-key-turn | D4W1024 | 18.638 | 487 | 2.8062 | 2.787 |
| myo-key-turn | D4W1536 | 19.319 | 543.96 | 3.2218 | 3.1911 |
| myo-pen-twirl | D2W512 | 60.477 | 2097.6 | 899.1 | 894.2 |
| myo-pen-twirl | D4W1024 | 1.0064 | 26.401 | 0.10596 | 0.10511 |
| myo-pen-twirl | D4W1536 | 24.8 | 974.18 | 1230.9 | 1222 |


A particularly revealing contrast is pen-twirl: the medium critic's median replay reward is about 1.01 and Q about 26.40, while the small critic's are about 60.48 and 2097.60. Its much smaller TD loss and deficit partly coexist with failure to reach the rewarding state distribution. Thus low fitting/TD errors can reflect an easier, low-reward visitation distribution rather than a better critic. This is a measured association, not proof that replay distribution alone causes the difference. Identifying the mechanism requires state/replay/target cross-controls.

Task complexity, exact state/action dimensions across all environments, replay-state distribution distances, intermediate target variances and loss curvature are not established by scalar histories. They are missing explanatory measurements, not license to invent an environment-complexity explanation.

## 8. Fresh-critic optimization dynamics

The W&B project supplies endpoint scores but not the full fitting histories needed to test architecture-dependent transient replication across all seeds. Therefore the pilot's approximately 31–102× median scaled dog-run amplification is **not promoted to a campaign conclusion**. It also cannot be rejected from endpoint scores: a recovered early spike and a smooth fit can share a final score. No new fitting was performed to fill this gap.

One original campaign probe is available from the previously checksummed Delta evidence: D2W512 pen-twirl seed 5, final check, ten rounds. Its numerical scores exactly match the campaign W&B record. Median fresh initial minibatch MSE is .87561, median fresh peak/initial ratio 2.3443 and median final full-pool MSE .14227. The trained copy has median initial minibatch MSE 164001.53 and median final pool MSE 14.8014; median pool-offset is 1998.706. This is one small architecture and one training seed. The enormous trained error reduces by orders of magnitude during fitting, yet leaves a large residual relative to fresh. It demonstrates finite-budget fit difficulty and a trained/fresh contrast for that task; it does not isolate capacity, conditioning, representation damage or architecture scaling.

Full per-round initial/peak/recovery/excess/final values and the unsmoothed figure are retained. The changing-minibatch traces do not provide intermediate fixed-pool losses. A transient peak can result from actual function overshoot, minibatch sampling or both. No nonfinite fitting behavior is present in this archived probe. The failed A100 segmented/continuous diagnostic is a separate historical engineering limitation; this report uses original arrays, not a refit from that failed runner.

## 9. Trained-versus-fresh adaptation dynamics

The shared 70% checkpoint avoids missing terminal values. Mean deficits and seed-bootstrap intervals are:

| environment | architecture | n | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| dog-run | D2W512 | 8 | 0.1206 | 0.094125 | 0.148 |
| dog-run | D4W1024 | 8 | -0.012484 | -0.016242 | -0.0089458 |
| dog-run | D4W1536 | 8 | -0.029829 | -0.032711 | -0.02691 |
| humanoid-run | D2W512 | 8 | -0.022373 | -0.026592 | -0.017971 |
| humanoid-run | D4W1024 | 8 | -0.051968 | -0.060683 | -0.042653 |
| humanoid-run | D4W1536 | 8 | -0.15121 | -0.16162 | -0.14138 |
| myo-key-turn | D2W512 | 8 | 0.59231 | 0.53927 | 0.64381 |
| myo-key-turn | D4W1024 | 8 | 0.45892 | 0.44513 | 0.47427 |
| myo-key-turn | D4W1536 | 8 | 0.46428 | 0.41366 | 0.51503 |
| myo-pen-twirl | D2W512 | 8 | 7.434 | 4.2158 | 10.304 |
| myo-pen-twirl | D4W1024 | 8 | 0.69795 | 0.29716 | 1.4529 |
| myo-pen-twirl | D4W1536 | 8 | 2.8463 | 0.81461 | 4.9046 |


Scaling reduces mean deficits relative to the small critic in **all four environments**, including both Myo tasks. This directly contradicts a universal prediction that larger critics lose more measured adaptation. At 70%, small-minus-medium gaps are .13308 in dog, .02960 in humanoid, .13339 in key-turn and 6.73602 in pen-twirl. Leave-one-seed-out directions persist in every case. However, these are conditional probe endpoints; they do not establish that the larger critic has greater useful plasticity for the actor.

At the final observed check, dog small has positive deficits in 8/8 runs, dog medium in 3/8 (small magnitudes), and dog large in 0/8. All humanoid cells have negative final deficits. All Myo cells have positive final deficits. Source-style two-consecutive-positive-CI onset can appear early in humanoid small before reversing, demonstrating that crossing a detector threshold need not be permanent pathology. Key-turn usually crosses by check 2 (4% budget); pen-twirl typically by checks 3–4. Such early, persistent positivity alone cannot time later policy collapses occurring across much of training.

The reference decomposition uses arithmetic means of paired round scores, *not a difference of IQMs*, to preserve an exact additive identity. From check 1 to check 35:

| environment | architecture | fresh_change | current_change | mean_deficit_change |
| --- | --- | --- | --- | --- |
| dog-run | D2W512 | -0.0057443 | -0.12592 | 0.12017 |
| dog-run | D4W1024 | -0.010523 | -0.0023605 | -0.0081625 |
| dog-run | D4W1536 | -0.025535 | -0.00030285 | -0.025232 |
| humanoid-run | D2W512 | -0.030981 | -0.0093573 | -0.021623 |
| humanoid-run | D4W1024 | -0.053712 | -0.0028863 | -0.050826 |
| humanoid-run | D4W1536 | -0.14726 | -0.0035044 | -0.14375 |
| myo-key-turn | D2W512 | -0.053704 | -0.2414 | 0.1877 |
| myo-key-turn | D4W1024 | -0.060474 | -0.14324 | 0.082767 |
| myo-key-turn | D4W1536 | -0.12603 | -0.19514 | 0.069107 |
| myo-pen-twirl | D2W512 | -0.081507 | -8.5323 | 8.4508 |
| myo-pen-twirl | D4W1024 | -0.15292 | -0.85646 | 0.70353 |
| myo-pen-twirl | D4W1536 | -0.2132 | -3.0251 | 2.8119 |


For large humanoid, fresh mean score changes by −.14726, current by only −.00350; consequently the mean paired deficit changes by −.14375. Most of this apparent trained advantage is reference deterioration. Dog small shows the opposite balance: current deteriorates by −.12592 while fresh changes only −.00574, producing a growing trained disadvantage. Pen small shows a much larger trained change (−8.53232) than fresh change (−.08151), so changing reference difficulty cannot explain away its large deficit. These examples separate two distinct phenomena hidden by a single deficit trajectory.

## 10. Fresh-reference drift, Q scale and fitting error

All 96 runs have negative initial-to-70% fresh-score change. The mean decline grows with size within every environment:

| environment | architecture | n | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| dog-run | D2W512 | 8 | -0.0062944 | -0.007642 | -0.0049084 |
| dog-run | D4W1024 | 8 | -0.014691 | -0.016899 | -0.012689 |
| dog-run | D4W1536 | 8 | -0.032513 | -0.035924 | -0.028682 |
| humanoid-run | D2W512 | 8 | -0.03217 | -0.034547 | -0.029773 |
| humanoid-run | D4W1024 | 8 | -0.056003 | -0.063444 | -0.047768 |
| humanoid-run | D4W1536 | 8 | -0.15345 | -0.16314 | -0.1446 |
| myo-key-turn | D2W512 | 8 | -0.061934 | -0.066752 | -0.057601 |
| myo-key-turn | D4W1024 | 8 | -0.070216 | -0.076188 | -0.063536 |
| myo-key-turn | D4W1536 | 8 | -0.14315 | -0.15113 | -0.13476 |
| myo-pen-twirl | D2W512 | 8 | -0.089573 | -0.1076 | -0.074419 |
| myo-pen-twirl | D4W1024 | 8 | -0.15701 | -0.20158 | -0.1137 |
| myo-pen-twirl | D4W1536 | 8 | -0.24645 | -0.31054 | -0.18583 |


The fresh critic's parameters have no SAC training exposure, so its deterioration cannot be attributed directly to its own training-induced representation loss. The probe's replay input distribution, observation normalization, teacher and target realization change across checks; their interactions with fixed initialization and optimizer can change task difficulty. The data demonstrate nonstationarity of this reference measurement; they do not uniquely attribute it to replay distribution. A fixed pool/teacher/normalization cross-control is needed.

Within-run Q associations remain after removing linear and quadratic training-time trends. The following confidence intervals cluster the three architecture trajectories by their shared seed label (eight clusters), rather than treating thousands of checks as independent:

| environment | target | n_runs | n_seed_groups | mean_within_run_time_adjusted_r | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| dog-run | deficit | 24 | 8 | -0.26441 | -0.33911 | -0.18116 |
| dog-run | fresh | 24 | 8 | -0.35603 | -0.41149 | -0.30148 |
| humanoid-run | deficit | 24 | 8 | -0.18518 | -0.30356 | -0.085135 |
| humanoid-run | fresh | 24 | 8 | 0.32082 | 0.21057 | 0.43647 |
| myo-key-turn | deficit | 24 | 8 | 0.62936 | 0.54902 | 0.7145 |
| myo-key-turn | fresh | 24 | 8 | -0.63533 | -0.6587 | -0.61058 |
| myo-pen-twirl | deficit | 24 | 8 | 0.62003 | 0.55093 | 0.68403 |
| myo-pen-twirl | fresh | 24 | 8 | -0.62342 | -0.714 | -0.52391 |


Positive Q-deficit associations in Myo contrast with negative associations in DMC. A pooled cross-environment correlation would therefore conflate both reward scales and opposite relationships. Removing simple time trends helps but does not remove evolving policy/replay confounding. The fixed offset cancels the fitted critic's mean prediction when forming initial residuals. Therefore a statement that large *mean* Q automatically produces large initial probe MSE is algebraically incorrect. Centered output variation and covariance with the sine target matter initially. Over optimization, weight decay can break translation invariance and optimizer conditioning can still depend on scale; neither is measured by mean Q alone. Exact attribution needs predictions, target arrays and optimizer state.

Across the full campaign, initial and individual final pool MSE are not logged, so a general initial→final error regression cannot be reconstructed. L identifies relative final fitting error; it does not identify the two absolute errors without b. The one archived probe is a valid counterexample to any claim that offset centering alone makes trained fitting easy, but is not a multi-seed scale mechanism test.

## 11. Temporal pathology onset and definition sensitivity

Adaptation onset follows the source-style rule of two consecutive positive lower confidence bounds; tables retain all crossings and later signs instead of treating the first crossing as irreversible. Critic-loss/Q/KL event sensitivity uses a trailing five-observation median, ratios 2/3/5, and one versus three consecutive observations, starting after 20% budget. These operational events are exploratory change indicators, not statistical change-point certificates. Ratios are undefined when the prior median is nonpositive and are not rescued with an arbitrary epsilon. Results are in critic_event_sensitivity.csv.

Policy events use an observed return decline of at least 20/30/50% from a preceding positive running peak, after 20% training and after reaching at least half the run's own eventual observed peak. This explicitly post hoc gate removes low-initialization denominator artifacts but also uses future information, making it unsuitable as a deployable detector. The event counts are 20/17/12, respectively. Two consecutive qualifying evaluations occur in only 3/2/0 cases at those thresholds. Sparse five-episode evaluations mean a one-point drop can be evaluation variability or a transient policy change. A final-point event cannot demonstrate persistence.

Probe checks are every 2% budget while evaluations are every 10%; onset ordering is interval-censored at this cadence. A probe change that appears first may reflect observation frequency rather than causal direction. Table cases align preceding critic observations only through the evaluation before a drop; they do not silently use post-event metrics as predictors.

## 12. Critic-to-actor propagation and predictive signatures

An exploratory, fixed-penalty ridge model predicts the *next evaluation's return change*. Baseline covariates are current return, previous return change, training fraction and architecture indicators. Added critic covariates are deficit, log(1+absolute Q mean), and log(1+TD loss). Leave-one-seed-number-out validation holds out all three architectures with that seed within an environment. Training-fold normalization avoids test leakage. The model is deliberately small, with no hyperparameter search; its results remain post hoc and do not constitute an independently confirmed predictor of collapse.

With all eight seeds, adding combined critic covariates reduces MSE by about .6% in dog, worsens it about 4% in humanoid, and reduces it about 5.3% in key-turn and 24.5% in pen-twirl. The more stringent exclusion of pilot-overlapping seed 1 gives:

| environment | model | n_seeds | relative_mse_reduction | gain_ci_low | gain_ci_high |
| --- | --- | --- | --- | --- | --- |
| dog-run | baseline | 7 | 0 | 0 | 0 |
| dog-run | deficit_only | 7 | -0.0014135 | -6.1944 | 1.3083 |
| dog-run | q_only | 7 | 0.012447 | -24.847 | 45.176 |
| dog-run | td_only | 7 | -0.011711 | -41.217 | 1.3218 |
| dog-run | combined | 7 | 0.0059939 | -28.425 | 38.146 |
| humanoid-run | baseline | 7 | 0 | 0 | 0 |
| humanoid-run | deficit_only | 7 | -0.014014 | -8.5239 | -1.1555 |
| humanoid-run | q_only | 7 | -0.006318 | -7.5887 | 0.8837 |
| humanoid-run | td_only | 7 | -0.042108 | -33.96 | -0.24414 |
| humanoid-run | combined | 7 | -0.060599 | -43.881 | -2.6464 |
| myo-key-turn | baseline | 7 | 0 | 0 | 0 |
| myo-key-turn | deficit_only | 7 | 0.041262 | -744.03 | 3978.1 |
| myo-key-turn | q_only | 7 | 0.058323 | 1632.1 | 3979 |
| myo-key-turn | td_only | 7 | 0.013783 | -323.98 | 1437 |
| myo-key-turn | combined | 7 | 0.052864 | 1126.3 | 3290.5 |
| myo-pen-twirl | baseline | 7 | 0 | 0 | 0 |
| myo-pen-twirl | deficit_only | 7 | 0.0032465 | -5703 | 6157.6 |
| myo-pen-twirl | q_only | 7 | 0.2361 | 5989.9 | 76630 |
| myo-pen-twirl | td_only | 7 | 0.17176 | -9249.2 | 68159 |
| myo-pen-twirl | combined | 7 | 0.23295 | 4703.3 | 77581 |


Q alone beats the combination slightly in both Myo tasks after seed-1 exclusion: about 5.83% versus 5.29% reduction in key-turn and 23.61% versus 23.30% in pen-twirl. Deficit-only reductions are about 4.13% (uncertain seed-bootstrap gain) and .32% (uncertain), respectively. Humanoid added diagnostics worsen prediction; dog improvements are tiny and uncertain. Thus the data support environment-dependent predictive information in Q histories, not a reliable incremental propagation signal specific to adaptation deficit. Future-return-change prediction includes both learning and degradation and must not be relabeled a validated collapse classifier. Even held-out prediction does not identify causality.

## 13. Performance-collapse case studies: complete qualifying catalog

Every 30% operational event is listed below; no dramatic example is substituted for the denominator of 96 runs. Seventeen runs qualify, including one low-peak pen small run whose 25.46-unit absolute decline is not comparable in practical magnitude to the >1000-unit collapses. The catalog includes architecture, seed, timing, persistence and recovery. Full preceding Q, TD loss, entropy, temperature, KL and actor gradients are in collapse_case_catalog.csv and the 96-page seed atlas.

| id | environment | architecture | seed | state | onset_fraction | peak_return | max_absolute_decline | final_drawdown | persistent_two_evals | preceding_deficit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14fmtkz6 | myo-pen-twirl | D4W1536 | 5 | finished | 0.3 | 1965.5 | 1193.1 | 0.071345 | False | 3.9051 |
| 1ygj8vna | myo-pen-twirl | D2W512 | 2 | finished | 0.4 | 2018 | 792.21 | 0 | False | 17.465 |
| 2vybacn1 | myo-key-turn | D4W1536 | 4 | finished | 0.4 | 1352.6 | 982.57 | 0.14855 | False | 0.46207 |
| 5bp87gdz | myo-key-turn | D4W1024 | 8 | finished | 0.8 | 1236.2 | 659.62 | 0 | False | 0.47093 |
| 7blwjacl | myo-pen-twirl | D4W1536 | 8 | crashed | 0.8 | 1943.3 | 1947.5 | 1.0022 | False | 5.4305 |
| 9j0g5bee | myo-key-turn | D4W1536 | 5 | finished | 0.5 | 1044.5 | 361.9 | 0.012764 | False | 0.37304 |
| bftpzg4k | myo-key-turn | D4W1024 | 6 | finished | 0.8 | 1294.2 | 1353.8 | 0 | True | 0.49913 |
| dce5c6uv | myo-key-turn | D2W512 | 4 | finished | 0.9 | 1292.1 | 447.27 | 0 | False | 0.67153 |
| deaf606s | myo-key-turn | D4W1536 | 2 | finished | 0.9 | 1289.4 | 1392.3 | 0 | False | 0.44875 |
| f2itkrl6 | myo-pen-twirl | D2W512 | 4 | finished | 0.7 | 2076.2 | 1841.4 | 0 | False | 12.428 |
| fxb32tqc | myo-key-turn | D4W1536 | 8 | crashed | 0.2 | 1153.9 | 663.53 | 0 | False | 0.46036 |
| hnth93hd | myo-key-turn | D2W512 | 5 | finished | 0.8 | 1244.9 | 378.26 | 0.029277 | False | 0.65924 |
| kqkedgtk | myo-key-turn | D4W1536 | 6 | finished | 0.6 | 1029.7 | 308.56 | 0.024168 | False | 0.4007 |
| nzs2f1az | myo-pen-twirl | D2W512 | 3 | finished | 1 | 31.417 | 25.462 | 0.81047 | False | 0.38901 |
| onrxf1xo | myo-key-turn | D4W1536 | 1 | finished | 1 | 1285.5 | 741.08 | 0.5765 | False | 0.51651 |
| qlxfxfy9 | myo-key-turn | D2W512 | 3 | finished | 0.6 | 1293.9 | 991.83 | 0.23791 | False | 0.6498 |
| stv2sg3f | myo-pen-twirl | D4W1536 | 2 | finished | 0.8 | 2078 | 1209.1 | 0.43621 | True | 5.4025 |


The most severe sustained observed pen case is D4W1536 seed 2: peak 2078.02, maximum decline 1209.13, two-evaluation persistence beginning at 80%, and final return 1171.57 (43.62% below peak). Its preceding deficit is 5.4025, Q about 2085.84 and TD loss about 2973.28. Seed 8 large pen drops from an observed peak 1943.26 to −4.25 by its last evaluation, but then crashes before completion; persistence and terminal recovery are unobserved. Its preceding deficit is 5.4305, Q about 1921.41 and TD loss 3841.61. Their similarity is suggestive but two cases do not identify a mechanism.

The large key-turn seed-1 terminal decline is the exact pilot-overlapping trajectory: peak 1285.47 at 80% budget, final 544.39, a 741.08 or 57.65% decline. It is a real observed terminal decline, but not representative of all large seeds. Other large key seeds suffer substantial intermediate drawdowns and then recover. Medium key-turn seed 6 has two qualifying evaluations near 80–90%, then finishes at its own peak. Small key-turn seeds also drop and recover. These counterexamples argue against assigning all return instability to the largest architecture or treating an operational drop as irreversible collapse.

A more useful scientific distinction is failure to learn versus loss after learning. Several medium pen runs have small deficits/TD errors because they never attain large rewards; their lack of large drawdowns is not evidence of superior stability at comparable policy skill. Conversely, successful small pen runs can show very large adaptation deficits while maintaining high final return. Comparing only collapsing runs would miss both forms of selection bias.

## 14. Contradictions and comparison with the preliminary seven-run report

* Survives beyond seed 1: dog small commonly outperforms scaled critics despite worse probe deficits; all seed-1 exclusions preserve mean adaptation rankings. Fresh-reference deterioration is much broader than the pilot could show—96/96 conditions at 70%—but its mechanism remains unidentified.
* Fails as a general interpretation: larger trained critics do not show monotonically larger adaptation deficits. At 70%, scaled mean deficits are lower than small in every task. The pilot's large-key largest-deficit ordering is not the campaign ordering; small key has the highest mean deficit.
* Survives only as a single trajectory: the late large-key seed-1 decline recurs exactly because its observed training path matches the pilot. Other independent seeds show drops and recoveries, not a uniform terminal decline.
* Remains untested: fresh architecture-dependent fitting spikes across seeds. Endpoints are insufficient and no full campaign curve collection is available through W&B. The original pilot fitting summaries cannot substitute for absent independent trajectories.
* Revised mechanism: high mean Q does not automatically imply initial probe error because each fitted critic uses its own pool-mean offset. Scale may affect variation, conditioning and weight-decay dynamics; absolute initial-error causation remains unproven.
* New counterexample: large humanoid has a strongly negative deficit despite essentially stable current fitting score, because fresh-reference score deteriorates. An apparent benefit of training can be induced by reference difficulty.
* New evidence against simple propagation: same-stage key-turn deficit and return correlations are positive within all architectures (Spearman .905 small, .714 medium, .833 large; n=8 each). These unadjusted exploratory associations do not prove beneficial pathology, but contradict interpreting a high deficit as a universal indicator of poor actor performance.

## 15. Ranked competing mechanistic explanations

1. **Changing diagnostic task/reference difficulty — strongest direct measurement support.** Explains fresh decline in all 96 runs and most large-humanoid negative-deficit drift. Fails to explain the large trained-specific pen-small deterioration, where trained score drops by >8 while fresh changes only ~.08. Teacher, replay and normalization contributions remain confounded. Prediction: holding all task components fixed should remove the fresh longitudinal trend. Cheapest decisive next step is archival task recovery and cross-evaluation, without parent training.
2. **Value/reward/visitation-scale coupling — moderate correlational support.** Explains large Myo Q and deficits, opposite within-task scale relationships across domains, and low-deficit pen policies that fail to discover reward. Fails as a mean-offset-only explanation. Prediction: compare centered initial prediction variance, fixed-pool residuals and targets under shared replay/normalization; these should account for variation better than mean Q alone. Existing campaign scalars cannot identify this.
3. **Architecture-dependent optimization/finite fitting budget — plausible, partly supported.** One archived small pen critic fits from ~164k initial minibatch error down to ~14.8 final pool error while fresh reaches ~.142. This makes optimization progress and residual distinct. The pilot's spikes support the hypothesis only preliminarily. Prediction: reducing diagnostic learning rate should shrink fixed-pool overshoot, whereas increasing continuous fitting horizon should reduce a budget-limited residual. No approved optimizer or budget is changed in this report.
4. **Training-induced representation/conditioning loss — unresolved.** Consistent with persistent positive Myo deficits, especially pen small, but not uniquely identified by them. Fresh deterioration, own-offset targets and optimizer sensitivity are confounders. Prediction: after controlling initial function scale/task and optimizer, trained copies retain slower held-out adaptation across multiple task families; representation interventions improve both that phenotype and controlled downstream utility.
5. **Actor exploitation of critic errors — plausible but unmeasured.** Q histories predict future return changes in Myo, but Q magnitude is not Q error. No rollout-calibrated bias, action-gradient error or randomized intervention is available. Prediction: calibrated value/action-gradient error rises before degradation and a critic-only intervention improves matched actor outcomes.
6. **Shared infrastructure termination — plausible for crashed-record status, independent of scientific instability.** Common host and near-synchronous history endings explain missing terminal data better than assuming eight independent numerical blowups. No scheduler/node logs establish the cause. Prediction: scheduler accounting or system logs identify one common stopping event. Read-only log retrieval is the cheapest decisive check.
7. **Evaluation variability and selection — supported as a limitation, not a complete explanation.** Intermediate evaluations use five episodes; many large drawdowns recover. Final 50-episode means and sustained cases prevent dismissing every decline as noise. Prediction: retained episode distributions distinguish high return variance from persistent policy changes without training new parents.

No mechanism is assigned a causal confidence level. Numerical instability, overestimation, irreversible plasticity loss and critic-to-actor propagation remain hypotheses where their defining measurements/interventions are absent.

## 16. Statistical robustness and limitations

Independent seeds are the primary units. Probe rounds only define a conditional within-critic diagnostic; their source bootstrap intervals are not multi-seed intervals. Checkpoints are repeated observations. Main architecture contrasts are within environment and use paired seed labels to preserve design coupling where possible. With only eight seeds, intervals and exact permutation tests are coarse and sensitive to outliers; the source trajectories, matched denominators and leave-one-out results are supplied.

The report uses 10,000 deterministic percentile resamples. These intervals are descriptive and pointwise; learning-curve bands are not simultaneous confidence regions. IQM across training seeds uses the standard finite-sample trimmed-mean estimator; this is explicitly distinct from the source's special ten-round fractional probe IQM. All broad paired exploratory comparisons also receive a single Holm adjustment; none survives that deliberately broad family at .05. Unadjusted interval exclusions are therefore not described as confirmatory discoveries. Neither a nonsignificant contrast nor a narrow observed range proves equivalence.

Missingness is architecture/seed-specific. The shared 70% horizon retains all eight seed pairs; terminal large results retain six. No zero substitution, interpolation, silent smoothing or fabricated post-crash trajectories are used. Final peak/AUC observed-horizon summaries are labeled as such. The broad history export is complete relative to W&B, not complete relative to all original disk artifacts: full fitting curves, individual evaluation episodes, most replay snapshots, optimizer states and clean-tree certificates are missing.

Time detrending and seed-held-out validation reduce two major correlation pitfalls but cannot eliminate policy/replay feedback, reward-scale effects, architecture interactions or post hoc model choice. Cluster uncertainty uses eight seed groups; these are few clusters. Apparent prediction gains require prospective or independently held-out confirmation. The report does not attribute a causal effect to event ordering or to a diagnostic score's association with reward.

## 17. Strongest scientifically defensible findings

**Descriptive claim, high confidence for this dataset:** Under the frozen single-critic SAC settings, critic capacity produces environment-dependent adaptation-score behavior that is not a monotonic proxy for actor performance. Fresh-reference drift and trained fitting deterioration contribute differently to the same deficit statistic, and separating them changes its interpretation. This is replicated across the campaign seed grid, with missing endpoints explicitly retained as limitations.

**Correlational claim, moderate confidence:** Within-run Q information has environment-dependent associations with adaptation and improves next-return prediction in the Myo tasks under simple seed-held-out exploratory models. The adaptation deficit adds little consistent information beyond Q in those models. This supports studying scale/task/optimization confounding before claiming propagation of plasticity loss.

**Unestablished:** a unique cause of fresh fitting spikes; a general seed-replicated architecture-transient law; overestimation; irreversible representation loss; an intervention that changes critic plasticity and then changes actor learning. Publication-quality causality requires controlling those alternatives. Research direction remains the principal researcher's decision; these findings motivate a probe-validity and policy-utility question without authorizing a project pivot.

## 18. Prioritized low-compute research roadmap

The separate ROADMAP.md gives exact interventions, controls, seeds, outputs, conditional compute assumptions and falsification criteria. Highest priority is retrieving existing probe arrays, task manifests and scheduler/episode logs. Second is a fixed-task reference decomposition using archived parents rather than new parent training. Third is an optimizer/horizon mechanism control on a small set of original saved tasks. Only after a reproducible critic phenotype is isolated should a matched actor-continuation experiment be considered. No experiment in this roadmap was launched.

## 19. Implications, literature and final synthesis

The literature already studies curvature-related plasticity loss and multiple interacting mechanisms; observing a poor adaptation score alone is not a novel causal contribution. The relevant primary context is [Lyle et al. 2023](https://arxiv.org/abs/2303.01486) and [Lyle et al. 2024](https://arxiv.org/abs/2402.18762). The latter explicitly supports considering multiple mechanisms rather than presuming one scalar score diagnoses one cause. Few-seed uncertainty, robust aggregation and interval reporting follow the motivation of [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264). These sources contextualize this investigation; they do not validate EchoCritic's mechanism or establish novelty. No broad literature claim is made beyond this limited primary-source comparison.

Direct answers to the original fifteen synthesis questions:

1. **Five most important patterns:** probe-policy dissociation; fresh-reference deterioration in 96/96 conditions; environment-specific Q/deficit relationships; recoverable rather than universally persistent policy drops; and structured missing terminal data plus exact pilot/campaign trajectory overlap.
2. **Robust across seeds:** fresh decline at 70%, scaled-versus-small mean-deficit reductions that survive leave-one-out/seed-1 exclusion, and positive Myo final-observed deficits. Policy rankings are less universal and less precise.
3. **Environment/architecture dependence:** dog small policy advantage; weak humanoid policy contrasts despite strong probe shifts; key-turn intermittent drops across architectures; pen-twirl heterogeneous success and strong value-scale coupling.
4. **Most surprising contradictions:** the worst trained-fit deficit can coexist with better policy utility; a negative trained/fresh deficit can be driven by a worsening fresh reference; low TD/probe error can accompany failure to learn rewarding behavior.
5. **Fresh fitting spikes:** not identified causally or replicated campaign-wide from available arrays. Optimizer overshoot, changing minibatches and conditioning remain distinct possibilities.
6. **Large trained deficits:** excess final pool error after a finite fit budget is established; its allocation to initial output variation, conditioning, representation change and task difficulty is unresolved.
7. **What the probe measures:** conditional finite-budget adaptation under a particular task, initialization and optimizer. It combines useful learning ability with optimization/task sensitivity; its scalar deficit does not uniquely measure representational plasticity.
8. **Propagation:** no causal evidence. Q-related predictive information in Myo survives seed exclusion, while deficit-specific added prediction is weak. This warrants a targeted intervention, not a propagation conclusion.
9. **Severe collapses:** several distinct patterns exist, including large pen decline, key-turn recoveries and a pilot-overlapping late endpoint drop. Sparse evaluation and missing episode/critic-error evidence prevent one explanation.
10. **Strongest alternatives:** changing reference/task difficulty, scale/visitation coupling and optimization/budget limits, followed by unresolved representation and actor-exploitation mechanisms.
11. **Preliminary conclusions that survive:** dog probe-policy dissociation, changing fresh difficulty, and the specific large-key decline as an observed trajectory.
12. **Misleading/unsupported preliminary interpretations:** monotonic size→deficit loss, mean-Q→initial-error necessity, assuming fresh spikes imply plasticity loss, and treating the campaign seed-1 match as independent replication.
13. **Strongest current claim:** capacity-dependent adaptation diagnostics and policy utility are dissociated under this protocol; decomposing fresh and trained behavior is necessary to interpret that diagnostic.
14. **Essential before publication:** original per-step/fixed-pool curves across seeds, resolved full provenance/parameter counts, episode-level outcomes, missing-run accounting, controlled task/optimizer contrasts, and an intervention for causal claims.
15. **Next investigation:** recover archived curves and run fixed-task reference/scale/optimizer controls on existing checkpoints. Preserve the researcher's authority over any subsequent direction or GPU allocation.

## Artifact index

Raw API export: raw_api/ and raw_api_manifest.json. Original archived probe: raw_archived/ with provenance JSON. Original source snapshots: source/. Machine-readable results: derived/*.csv and histories.pkl. Independent checks: VALIDATION.json. Figures: figures/ (including the complete 96-page seed atlas). Discovery ledger: DISCOVERY_LEDGER.md and discovery_ledger.json. Follow-up design: ROADMAP.md. Reproduction commands and dependency versions: README.md and requirements-analysis.txt. No actual experiment data are stored in Git, and no original raw input is edited.
