# Trained probe versus initial fresh reference — 24 October 2026

## What was computed

The 96 campaign runs already contain probes every10,000 training interactions. The fresh-only initial check occurs at5,000 interactions, immediately before the first SAC updates. All fits use2,000 adaptation updates. No new probe optimization or GPU work was run. All4,738 archived nonzero checks are included, preserving each environment/architecture/training-seed identity. The eight crashed trajectories remain truncated; large-critic late group n declines from8to6.

For each run/check we reproduce the source fractional IQM over ten rounds for the current/trained score and fresh score. Define the requested anchored contrast as:

`initial-reference deficit(t) = IQM(P_fresh at check0) − IQM(P_trained at t)`.

Positive means the trained critic's score on its current probe is worse than its fresh initialization scored on the initial probe. Zero means matching scores, and negative means a higher trained score. This is a newly labeled descriptive contrast; it does not replace or redefine the original detector. Scores use P=target variance−final pool MSE and may be negative.

We subtract separately calculated IQMs across different-time tasks. We do not falsely pair check0 teacher roundr with checkt teacher roundr. This differs from the original source deficit, which is IQM of paired score differences within the same checkpoint. Both are retained in per_seed_check_comparison.csv. The decomposition `anchored contrast = [IQM(fresh_t)−IQM(trained_t)] + [IQM(fresh_0)−IQM(fresh_t)]` holds numerically within2.2e-15; replacing the first bracket with the original paired-difference IQM is not an exact identity.

## Results and implications

At350,000 interactions all12 cells have eight training seeds. Large humanoid's trained mean is0.4880 versus initial fresh0.4905 and contemporary fresh0.3370. Its anchored deficit is0.00246 with a pointwise95% seed-bootstrap interval[−0.00189,0.00704]. Thus the large apparent trained advantage against contemporary fresh mostly disappears against the initial anchor. Medium humanoid's anchored deficit is0.00364[0.00012,0.00759]; small is0.00965[0.00505,0.01449]. The chart also shows early humanoid deterioration followed by recovery toward the initial fresh score.

Dog medium/large are near their initial fresh score at350k: anchored deficits0.00211 and0.00247. Small dog has a substantially larger gap0.12752[0.10132,0.15480]. Thus switching reference removes most apparent scaled-critic advantage over fresh without producing a universal large-critic deterioration pattern. Small dog remains more impaired on the probe while often learning a better actor.

Myo effects remain substantial under the initial reference. Key-turn anchored deficits are0.65411 small,0.52931 medium,0.60814 large. Pen-twirl deficits are7.52590,0.85502,3.08622. Baseline drift therefore does not explain all trained-versus-initial differences. In particular, the large trained pen-small deficit persists under this reference choice.

## Does this establish that the tasks are getting harder?

It establishes that the changing checkpoint probe pipeline yields worse scores for the same stored fresh-critic initialization. All96 conditions deteriorate by70%. It does not isolate an intrinsic change in task difficulty or confirm a software defect. The fresh reference was captured once and reset to those same parameters for each fit; it is not a newly sampled initialization at every checkpoint. Replay samples, observation-normalization state, teacher parameters and minibatch draws change across checks. Fresh behavior can respond to any combination of them, and architecture/optimizer interactions can amplify the response.

The anchored comparison still evaluates the trained critic on a *different task/input distribution* from the initial fresh task. It therefore does not remove changing task difficulty. It makes the reference drift visible and tests how the descriptive ranking depends on the reference. To isolate task difficulty, cross-evaluate the same fixed fresh initialization on saved early versus late tasks with optimizer/minibatch/horizon held fixed. To isolate trained-critic change, evaluate early and late trained parameters on one identical saved task. A crossed learner-state×task-time design distinguishes those effects; it remains a proposed experiment, not an executed or silently approved modification.

## Charts and raw numbers

- anchored_deficit_by_architecture.png/pdf:requested initial-anchor comparison, all architectures in each environment, pointwise95% seed-bootstrap bands. Symmetric-log y-axis displays near-zero and very large gaps without clipping negatives.
- trained_fresh_initial_scores.png/pdf:12 panels with trained score, contemporary fresh score and initial fresh reference. At late checks the initial-reference group mean is matched to the same available seeds as the trained observations; it can shift when runs become unavailable.
- all_96_seed_comparisons.pdf:individual seed panels,12pages, eight seeds per architecture/environment. No across-seed averaging hides outliers.
- selected_checkpoints.csv:10k/250k/350k/500k summaries and intervals.
- per_seed_check_comparison.csv:all4,738 run/check observations and all contrast definitions.
- architecture_trajectories.csv:all12architecture/environment trajectories, counts and confidence intervals.
- analyze.py and VALIDATION.json:reproduction and numerical checks.

## Reading guide to the complete investigation

1. ../REPORT.md:the complete technical analysis. Read sections1(executive summary),14(contradictions and pilot comparison),17(strongest claims) and19(fifteen direct synthesis answers) first. Sections8–12give fitting/probe/scale/temporal details.
2. ../DISCOVERY_LEDGER.md:ranked findings with numerical evidence, counterexamples, confidence and follow-up tests; quickest findings overview.
3. This supplement and its charts:the new initial-reference comparison requested after the main report.
4. ../ROADMAP.md:proposed low-compute studies, not findings or launched experiments.
5. ../README.md:artifact map, methods contracts and reproduction commands; not the main scientific narrative.

This supplement is saved separately so the original report and its original estimands remain intact. It supplies the requested sensitivity analysis, rather than retroactively changing the protocol.
