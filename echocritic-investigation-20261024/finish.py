from analyze import *
# Validate the endpoint using stored FP32 scores, preserving subtraction before IQM.
rows=[]
for n in z.namelist():
 if not n.endswith('/run.csv'):continue
 base=n.rsplit('/',1)[0]; r=pd.read_csv(z.open(n)).iloc[0]; a=np.load(io.BytesIO(z.read(base+'/probe_curves.npz')))
 delta=a['check_20/fresh_score']-a['check_20/current_score']; estimate=trim_mean(delta,.25)
 rows.append(dict(run_key=r.run_key,recomputed=estimate,recorded=r.final_loss_iqm,absolute_error=abs(estimate-r.final_loss_iqm),fresh_initial=trim_mean(a['check_00/fresh_score'],.25),fresh_final=trim_mean(a['check_20/fresh_score'],.25)))
pd.DataFrame(rows).to_csv(OUT/'endpoint_validation.csv',index=False)
s=pd.read_csv(OUT/'performance.csv'); f=pd.read_csv(OUT/'fit_summary.csv'); v=pd.DataFrame(rows)
def table(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
report='''# EchoCritic investigation — available-evidence report

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

'''+table(s)+'''

All summaries describe individual trajectories. Final window is the mean of observed evaluations at >=90% of each trajectory's observed endpoint (two points here). AUC is trapezoidal integration of observed evaluations, explicitly a numerical integral across sparse observations, with units return × interaction steps. It does not reconstruct unobserved behavior. Peak is susceptible to evaluation noise. Across-seed SD, confidence intervals and population IQM are not estimable with one seed per cell; reporting degenerate bootstrap intervals would mislead.

Dog-run's small critic ends at 612.64 versus 446.44 and 468.99 for the scaled critics. Medium versus large is nonmonotonic in final return. Myo-key-turn's large critic has higher peak/AUC but worse final return, so an endpoint-only ranking misses its earlier advantage. Humanoid architecture comparisons are unavailable.

## 8. Fresh fitting dynamics

'''+table(f)+'''

Medians pool checkpoints and repetitions descriptively, not inferentially. Full per-repetition peak, amplification, peak index, first recovery below initial loss, integrated excess and final pool residual are in fit_dynamics.csv. Recovery is the first crossing and does not guarantee persistent recovery. Integrated excess is a discrete sum over update-index losses, not a held-out generalization measure. Figure fitting curves preserve individual repetition traces without smoothing. This demonstrates a transient in observed minibatch errors, not by itself numerical divergence or permanent plasticity loss.

## 9–10. Adaptation and output scale

'''+table(v)+'''

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
'''
(ROOT/'REPORT.md').write_text(report)
(ROOT/'README.md').write_text('Run analyze.py then finish.py using Python with numpy, pandas, scipy, matplotlib. Inputs are read directly from the unchanged pilot ZIP. Source snapshots are pinned to the pilot SHA. All outputs are pilot-only. Full campaign access remains required.\n')
(ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={__import__(p).__version__}' for p in ['numpy','pandas','scipy','matplotlib'])+'\n')
print('Report saved')
