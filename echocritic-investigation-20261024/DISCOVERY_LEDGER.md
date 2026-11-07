# Ranked exploratory discoveries

All findings are pilot-only, one training seed per condition; no novelty assessment against literature has been completed.

| Rank | Finding and evidence | Support/counterexamples | Mechanisms and confidence | Best test |
|---|---|---|---|---|
| 1 | Fresh median amplification in dog-run: 1.78× small, 31.53× medium, 101.87× large | Three architectures, seed 1; current critics have much smaller median amplification | Strong descriptive evidence for transients; optimizer sensitivity and changing minibatches unresolved | Intermediate fixed-pool MSE and predictions with identical saved task |
| 2 | Probe-policy dissociation: dog small final return 612.64 and deficit +0.241; scaled returns 446.44/468.99 with deficits −0.0135/−0.0219 | Three seed-1 conditions; myo deficits differ | Strong within-pilot contradiction of a universal deficit-to-return mapping; no causal claim | Replicate within environment across campaign seeds |
| 3 | Large myo peak 1285.47, final 544.39; decline 741.08 | One run; smaller myo declines 2.70/34.92 | Strong observed decline, weak persistence/generalization evidence | More evaluation episodes and independent seeds; compare same-stage controls |
| 4 | Deficit-size association depends on environment | Dog scaled negative; myo all positive, medium lower than small | Descriptive interaction, confounded by seed and environment | Paired within-environment contrasts with seed uncertainty |
| 5 | Mean-Q alone cannot explain initial probe error | Source gives each fitted critic its own mean offset | Strong algebraic constraint; empirical centered variance missing | Recover full-pool predictions and target covariance |
