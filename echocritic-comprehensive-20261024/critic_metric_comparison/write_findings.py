from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;S=pd.read_csv(R/'per_seed_summaries.csv');G=pd.read_csv(R/'architecture_summaries.csv');C=pd.read_csv(R/'within_run_associations.csv')
def table(d):
 def f(v):return f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
rows=[]
for (env,arch),d in S.groupby(['environment','architecture']):
 rows.append({'environment':env,'architecture':arch,'n':len(d),'dormancy_tau02_percent':d['train/critic_DR0.2_common70'].mean(),'initial_rank_count':d['train/critic_srank_initial'].mean(),'rank_count_60_70':d['train/critic_srank_common70'].mean(),'gradient_10_20':d['train/critic_gnorm_early'].mean(),'gradient_60_70':d['train/critic_gnorm_common70'].mean(),'initial_reference_probe_gap70':d.anchored_gap70.mean(),'rank_down_seeds':int((d['train/critic_srank_from_initial']<0).sum()),'gradient_up_seeds':int((d['train/critic_gnorm_change']>0).sum())})
T=pd.DataFrame(rows);T.to_csv(R/'comparison_at_common_horizon.csv',index=False)
text='''# Critic degradation proxies versus 2,000-update adaptation probes

24 October 2026. Additive analysis of all96 canonical campaign runs; pilot records excluded. No fits, training, GPU jobs, scientific-setting changes or historical corrections were performed.

## Main conclusion

**The logged metrics do not show universal chronic dormancy plus collapsing gradients in DMC while the probe falsely remains unchanged.** DMC generally shows an early dormancy/gradient/rank transient followed by recovery in several metrics. Large humanoid retains a lower feature-rank statistic than its initial value and than the smaller architectures, but its gradient norm recovers and its dormancy declines from the early peak. In dog-run, the large critic retains a high rank count and has low later dormancy. This mixed pattern is insufficient to establish an implementation defect in the probe.

Myo shows a clearer persistent feature-rank decrease, especially at larger sizes. Its trained probe score also remains below the initial fresh score. Nevertheless the size rankings disagree: pen-small has the largest adaptation deficit but less rank decline than pen-medium/large. Thus the metrics capture overlapping but distinct aspects of learned functions and representations. None of them, alone or collectively without controls, automatically identifies irreversible loss of useful learning ability.

## What was measured and how

The independent logged metrics are: critic dormant ratios atτ=.1and.2; feature rank statistic; training critic gradient norm; final encoder feature norm; parameter norm; TD loss; TD-error variance; and mean Q. All nine are analyzed. `critic_pnorm` is another global parameter-L2 view emitted in update windows; `critic_wnorm` is the structural-snapshot view, so these are not independent evidence for two distinct pathologies.

Source was extracted from archived commit f597ee9898e4d401912092c336c3e5330cf2fda1, not inferred from metric names. Seven relevant source files and checksums are retained under source/ and source_manifest.json. The main call path is trainer logging→agent.get_metrics→get_critic_with_metrics for structural snapshots; update_critic→Trainer.apply_gradient for the gradient norm. Snapshot inputs are a replay minibatch of256observations/actions, with agent preprocessing. Inputs/replay distribution change over training; these are not fixed-reference representation measurements.

- **Dormancy:** for each captured non-predictor Dense output, calculate mean absolute output per unit over examples; divide by its layer's mean unit score plus1e-9; count normalized values≤τ. Concatenate unit masks across captured Dense layers for the network percentage. This is neuron-count weighted, not an equal mean of layer percentages. The units are percentage points:1means1%, not100%. Both thresholds are reported. A scalar ratio does not identify whether the *same* units remain dormant; chronic individual-neuron identity cannot be established from these logs.
- **Measurement location matters:** `capture_intermediates=True` captures Dense-module outputs. In residual blocks the expanded Dense output is captured before the following functional ReLU. Dense outputs after the residual branch's second affine map and the input projection are also included; predictor output is excluded. The metric is therefore not literally a count of dead post-ReLU neurons. No claim is made here about whether the paper's released implementation uses the same hooks; that requires a separate source comparison.
- **The logged `srank` is not conventional stable rank.** It returns the smallest number of singular values whose cumulative *sum* reaches99%of the total singular-value sum. It neither calculates Frobenius-norm-squared/spectral-norm-squared nor a99%-squared-energy rank. It uses final encoder features, reshaped to examples×width. With batch256and all widths≥512its maximum is256, so dividing it by hidden width would create a misleading size effect. This report retains its original values and labels it “99%-singular-value count.”
- **Gradient norm:** full critic training-loss gradient L2 before AdamW update; all campaign critics are dense. This is not optimizer-update displacement, a layerwise gradient profile or a probe-fitting gradient. Training values are aggregated over the logging window. A large raw norm does not establish healthy plasticity, and a decrease can reflect lower TD error or changing reward scale.
- **Feature norm:** mean L2 of final encoder outputs over replay examples. Initialization values22.6274/32/39.1918equal√512/√1024/√1536, consistent with the final normalization. Width and learned normalization scale affect it; a decline is not a calibrated plasticity-loss scale.
- **Weight norm:** global parameter L2 across stored network parameters. Raw norms depend on architecture. Overview charts show ratio to that run's initial norm as an explicitly derived display, while raw values are preserved. Norm growth is observed, but its causal or pathological significance is not established by growth alone.
- **Probe comparison:** the added plot uses initial fresh score minus current trained score, separately reconstructing source fractional IQMs over ten rounds. Positive values indicate worse score relative to the initial fresh reference. It compares different checkpoint tasks; it is not an isolated representation-loss measurement. The original paired contemporary deficit is retained in aligned tables.

## Comparison without crashed-run endpoint bias

Every cell has eight training seeds through70%training. Structural/gradient values below are per-seed medians in the60–70%window, then means across seeds. Early gradients use10–20%; initial rank is the first available pre-SAC structural snapshot, at2,000interactions. Probe gaps are at350,000interactions exactly. These distinct time summaries are labeled rather than silently presented as simultaneous raw observations.

'''+table(T)+'''

### DMC: a transient followed by recovery, with a persistent rank exception

**Dog-run:** at60–70%, dormant ratioτ=.2is .0022%small, .0817%medium and .6786%large. Both thresholds show generally low later dormancy. All eight seeds of every architecture have lessτ=.2dormancy in this window than at10–20%. The gradient norm increases from the early window in8/8small,8/8medium and7/8large runs. Large rank count is237.81versus initial234.25, not a group-level rank collapse. Medium has some rank reduction; small retains high rank but acquires the largest anchored probe deficit. This contradicts a simple claim that worse DMC probe behavior should follow monotonically larger dormancy or smaller gradient norms.

**Humanoid-run:** large rank count falls from234.25to201.75, a13.9%group-level reduction; all8large seeds have a lower value than initially. Medium is219.94versus233.13, and small230.69versus229.50. This is a genuine architecture-dependent representation-statistic change worth investigating. However, all8large seeds have higher gradient norms in60–70%than10–20%, and all8have lowerτ=.2dormancy. Mean gradient norm grows .1573→.5331; dormancy falls5.0686%→1.5941%. The plots show an early rank trough followed by partial recovery, alongside the probe's early deterioration and recovery toward baseline. This is not a clean “all independent proxies degrade while the probe remains flat” counterexample.

Within-run rank/probe relationships in humanoid are also informative: mean Spearman correlations between rank and the anchored deficit are −.901small,−.894medium,−.605large. After separately removing linear/quadratic time trends they are approximately−.847,−.792,−.633. These are averages of per-run repeated-observation correlations, not thousands of independent observations or confirmatory p-values. Higher rank generally coincides with less probe impairment in this task. That temporal agreement weighs against assuming the probe is simply disconnected or frozen, while leaving implementation audit questions open.

### Myo: stronger persistent rank reduction, without a uniform gradient-collapse signature

**Key-turn:** rank declines in all24runs, with group initial→60–70%values229.25→200.75small,232.38→172.88medium and233.25→167.00large. The large critic also develops more later dormancy atτ=.2than small. Its gradient falls from63.62to56.12between the selected early/late windows;7/8large seeds have lower gradient norm. This is more consistent with deterioration proxies than the DMC result, but the gradient remains nonzero and the probe-size ordering still differs: small has the largest initial-reference gap .6541, medium .5293, large .6081. Neither rank nor dormancy uniquely predicts the adaptation endpoint across architectures.

**Pen-twirl:** all24runs lose rank. Large falls230.25→94.69(about58.9%); medium229.25→109.63(52.2%); small226.25→190.31(15.9%). Yet small has the worst anchored probe gap7.5259, compared with .8550medium and3.0862large. Gradients in successful high-value runs can grow enormously rather than collapse: group means of per-seed window medians are about22,031small,4,502medium and25,846large. Broad bands and individual-seed panels reveal mixtures of low-reward runs and high-gradient/high-value runs. It would be misleading to interpret these raw mean gradients as uniformly healthy or uniformly damaged plasticity.

These Myo observations support investigating representation/scale/optimization changes, but do not independently prove that those changes cause the probe residual or actor degradation. Rank can decrease when visitation becomes more concentrated, and global gradients scale with targets/rewards. Shared trends and task changes remain confounders.

## What does this imply about the suspected implementation problem?

A source audit remains warranted. However, the current evidence does not meet the proposed premise that chronic dormancy and *all* other proxies demonstrate plasticity loss while the large DMC probe shows none. Dog large shows limited supporting evidence for that premise. Humanoid large shows persistent relative rank reduction but recovery in dormancy and gradients, with a probe trajectory that partly follows rank recovery. Myo has substantial rank changes and poor probe scores, but conflicting architecture rankings.

Metric disagreement can identify a measurement or implementation question without identifying its answer. In particular:

1. Verify that dormant-neuron hooks match the intended activation locations before comparing percentages with another implementation.
2. Keep the unusual `srank` definition explicit. A misleading metric name is an interpretation hazard, not proof its numeric calculation is wrong.
3. Inspect full adaptation learning curves/fixed-pool intermediate errors: a2,000-update endpoint can recover even if early adaptation slows.
4. Use fixed-replay/fixed-task cross-evaluations to distinguish changes in input distribution from changes in critic parameters.
5. Compare layerwise and update-space gradients if a gradient-collapse mechanism is the target; global training-loss L2 is an incomplete proxy.

No production code, hooks, scientific thresholds, hyperparameters or original measurements were changed here. The next audit should test these concrete paths rather than select whichever proxy agrees with the expected result.

## Figures, data and reproduction

- **DMC_overview.png/pdf:**dog/humanoid, dormancyτ=.2, rank, training gradients, anchored probe gap and relative parameter norms in aligned panels.
- **Myo_overview.png/pdf:**the same for key-turn/pen-twirl.
- **critic_DR0p1/critic_DR0p2/critic_srank/critic_gnorm/critic_fnorm/critic_wnorm/critic_loss/td_error_var/q1_mean.pdf:**separate readable four-environment charts, with faint individual seeds and group mean/pointwise95%seed-bootstrap bands. PNG copies included.
- **all_96_seed_metric_comparisons.pdf:**24pages covering all96training seeds, with dormancy, rank, gradients and probe gap side by side. No smoothing, interpolation or filled post-crash values.
- **per_seed_summaries.csv:**all per-run baseline/window/change/ratio statistics. Source metric histories remain unchanged in the parent raw export.
- **architecture_summaries.csv:**10,000-resample deterministic seed-bootstrap intervals for each metric/window/change. Ratios are undefined when the early value is zero; no epsilon is inserted.
- **aligned_probe_metrics.csv:**4,738probe checks aligned by their explicit interaction counter to an observation at or before the check, at most2,000interactions old.
- **within_run_associations.csv:**raw and time-residual Spearman associations, explicitly repeated-measurement exploratory descriptors. Constant metrics have undefined correlations rather than fabricated zero values.
- **metric_coverage.csv, VALIDATION.json, source_manifest.json:**coverage, invariants and original source identities. Structural metrics have23,704observations each; training gradients23,512. There are no duplicate probe rows; both dormancy thresholds are ordered correctly and allrankvalues lie in1–256.

Reproduce from the parent package's pinned analysis environment:

`python critic_metric_comparison/analyze.py`

`python critic_metric_comparison/write_findings.py`

Analysis seed20261026; NumPy1.26.4,pandas2.2.3,SciPy1.14.1,Matplotlib3.9.0. Seed, checkpoint and environment populations are retained separately. All late curves explicitly show the large-critic n decrease from8to6. Full-horizon group composition can change after crashes; common70%window comparisons use all8seeds.
'''
(R/'FINDINGS.md').write_text(text);print('FINDINGS_WORDS',len(text.split()))
