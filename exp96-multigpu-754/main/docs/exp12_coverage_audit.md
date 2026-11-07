# Experiment 1/2 coverage audit

Audit date: 2026-10-06. Baseline inspected: `2f67915cf9243d75f56299645ccd3961e9c0c44e`,
branch `claude/eloquent-fermat-inxqlt`. This is an engineering audit, not approval to launch
the grid. The source of truth is [the methodology](../.claude/methodology-exp1-exp2.md).
Later amendments supersede earlier requirements; historical records are retained.

**Most important unresolved gate: NEEDS CUDA — exact fork/identity validation on the
final tree, with the prescribed architectures, suites, device model and precision.**
CPU equality on tiny networks does not establish A100 equality. The documented Block B
job `22706349` used `b4a90cb`, before twin support. Its scheduler status and original
outputs have not been inspected here. No CUDA execution occurred during this audit.

## Deviations and missing coverage first

| Requirement / issue | Implementation evidence | Test evidence / missing protection | Status / consequence |
|---|---|---|---|
| Exp 1 primary comparison requires 65 runs per architecture | `analysis/exp1_analysis.py:score_matrix` pivots observed seeds/environments, rejecting NaNs inside that pivot; it does not assert the prescribed seed/environment sets | `Exp1AnalysisTest.test_missing_run_is_refused` removes a cell, not a whole seed/environment. Entire absent dimensions are untested | **DEVIATES**: an incomplete grid can produce a smaller primary estimate. Keep the prescribed grid; missing-data handling needs the lead's decision before repair |
| Twin per-network probe observations must be recorded, amendment (z) | `run_probes.per_network_fields` retains them in local `probes/probe_checks.csv` and checkpoint records; `ledger.write_run` reindexes to `CHECK_COLUMNS`, dropping them from canonical Exp 1 `checks.csv` | Twin end-to-end checks inspect local records; no test verifies canonical per-Q CSV retention | **DEVIATES** in canonical reporting, not complete data loss. Raw NPZ and local records survive; no silent schema repair |
| Probe learning curves must support descriptive inspection for every architecture/suite | `twin.combine` creates aggregate score/final-loss/b, but no aggregate `losses`; NPZ contains `current_q1_losses`, etc. `exp1_analysis.plot_learning_curves` requires `current_losses` / `fresh_losses` | Pipeline tests do not request `curves_for` / `--curves` on twin outputs | **NOT IMPLEMENTED** for HB plotting: that optional path raises a missing-key error; individual Q curves remain saved |
| Exp 2 completeness must mean the intended evaluation grid is present | `exp2_analysis.load` checks only the number of unique evaluation indices in each arm | No adversarial test substitutes an incorrect index, removes episodes within an evaluation, or misaligns arm grids | **DEVIATES** in validation: 26 distinct wrong indices can satisfy completeness; no observed production corruption is asserted |
| Original-to-restored fork panel must match exactly, (m) | Baseline wrote `check1_pre.npz`, but neither control nor arm consumed it; arm recomputed `pre` after restore | Baseline error-injection test perturbed only the second arm evaluation, so it could not catch a common restore discrepancy | **DEVIATES at baseline; repaired with approval**. The original/control comparison now runs with zero tolerance before `FORK_READY`, on ready-fork parent resume and before arm creation. Two regressions cover common restore drift and older ready forks; final validation below records the outcome |
| Final twin panel numerical test must pass without weakening it | `TwinCheck1Test.test_panel_values` compares compiled panel values/gradients against an eager reference | A restored break check failed during this audit; independent rerun results are recorded below | **DEVIATES in validation evidence** until resolved; a red reference comparison does not by itself establish a wrong SAC estimator |
| HB validation job must expose every failed step | `scripts/sbatch_exp12_hb.sh:step` records exit codes but continues; final pipeline through `tee` can return success despite earlier failures | No shell test asserts nonzero aggregate exit for a failed HB step | **NOT IMPLEMENTED** aggregate failure gate. Inspect `status.tsv`, every identity JSON and test skips; job completion is not a pass |
| Diagnostic CUDA tolerances must be measured and approved | Four `GPU_TOLERANCES` entries in `tests/exp12_helpers.py` are `None` | `check_gpu_tolerance` records measurements and skips; twin isolation uses the same update tolerance keys | **NEEDS CUDA**, then lead decision. A skip is not an accepted numerical bound |
| Grid-wide one-GPU-model restriction, (u) | Fork-specific model equality is enforced; manifests group arm jobs by model | No global validator prohibits Exp 1 cells being distributed across models | **NOT IMPLEMENTED** as a programmatic grid gate; operational requirement remains binding |

No implementation changes to these reporting/statistical issues are implied by this document.
The approved fork-boundary repair is separate from the audit commit.

## Research flags requiring lead decisions

**RESEARCH FLAG — incomplete Exp 1 population**

- What I found: primary analysis can accept an entirely absent seed or environment.
- Why it may matter: the resulting estimate concerns fewer than the prescribed 65
  runs per architecture and can be mistaken for the confirmatory comparison.
- Evidence: `exp1_analysis.score_matrix` checks missing cells only within observed
  dimensions; `test_missing_run_is_refused` does not remove a whole dimension.
- Severity: **high; potentially invalidating if reported as the prescribed grid**.
- Possible options: refuse anything except the prescribed population; or specify a
  different missing-data/population analysis with explicit methodological approval.
- What decision I need from you: authorize the enforcement rule and its adversarial
  tests before changing analysis. No such repair was made during this audit.

**RESEARCH FLAG — incomplete Exp 2 evaluations**

- What I found: completeness counts distinct evaluation indices without validating
  the exact grid, arm alignment or every evaluation's episode count.
- Why it may matter: malformed/partial evaluations could enter the primary paired
  return graphs and change which observations define the comparison.
- Evidence: `exp2_analysis.load` uses `eval_index.nunique()==n_evals`; pairing later
  groups by evaluation and steps since fork. No adversarial completeness test exists.
- Severity: **high if malformed outputs occur**; this audit does not assert they did.
- Possible options: enforce the declared aligned grid and ten episodes; or define a
  separate explicit partial-evaluation inclusion policy.
- What decision I need from you: choose the completeness/inclusion rule before repair.

**RESEARCH FLAG — twin numerical reference remains red**

- What I found: the twin Check 1 test fails its exact Q reference; separately
  evaluating its later gradient assertion also fails the existing bound.
- Why it may matter: a failed numerical oracle prevents a clean validation claim,
  even when original/control/identity comparisons pass on tiny CPU runs.
- Evidence: independent-process repeats and compiled/eager comparisons in the
  validation record below; the min-gradient mutation does not restore to green.
- Severity: **high validation blocker**; no wrong SAC estimator is established by
  these rounding differences alone.
- Possible options: retain the red gate while investigating independent/CUDA evidence;
  or separately review a justified numerical reference criterion. Exact fork/identity
  requirements must remain intact under either option.
- What decision I need from you: approve any reference-test criterion change only
  after its independent correctness argument. No bound was changed here.

**RESEARCH FLAG — incomplete twin reporting**

- What I found: canonical Exp 1 checks drop per-Q observations, and optional twin
  learning-curve plotting expects absent aggregate loss-array keys.
- Why it may matter: a publication report based only on canonical outputs can conceal
  individual-Q behavior or fail when requesting descriptive HB curves.
- Evidence: `ledger.CHECK_COLUMNS`, `twin.combine` and
  `exp1_analysis.plot_learning_curves`; local records and per-Q NPZ curves survive.
- Severity: **medium**; raw information is retained, so this is recoverable.
- Possible options: retain existing per-Q fields/plot those curves; or explicitly
  define a new aggregate curve if that is desired scientifically.
- What decision I need from you: select the reporting representation before repair.

**RESEARCH FLAG — CUDA identity and diagnostics bounds unresolved**

- What I found: no final-tree CUDA evidence was obtained here, and four diagnostic
  tolerance entries are None, causing measurement followed by skip on GPU.
- Why it may matter: CPU equality cannot validate A100 TF32/autotuning, full-size twin
  injection, HB simulator integration or a final confirmatory grid.
- Evidence: CUDA gate matrix below, CPU-only execution metadata, historical Block B
  commit predating twin support, and `tests/exp12_helpers.py:GPU_TOLERANCES`.
- Severity: **high prerequisite**, potentially invalidating a claimed CUDA validation.
- Possible options: obtain final-tree CUDA outputs for the mapped gates and approve
  measured bounds; or defer confirmatory execution. Existing jobs are left untouched.
- What decision I need from you: review actual measurements/bounds and explicitly
  authorize any Slurm submission. No launch or tolerance selection occurred here.

## Evidence, scope and status meanings

`CPU VERIFIED` means an identified passing CPU test or a directly reproduced CPU/source
check, within its stated scope. It does not mean production-size, full-budget or CUDA
verification. `GPU VERIFIED` requires an actual CUDA result for the relevant tree and
boundary; no final-tree item receives that label here. `NEEDS CUDA` denotes an unresolved
device-dependent gate. `DEVIATES` and `NOT IMPLEMENTED` identify implementation or
coverage gaps, including verification gaps explicitly identified as such.

Tests use real Hydra configuration and tiny networks/replay/probes, with WandB mocked.
Most integration tests reduce the production 5 × 1,000 probe budget and 25,600 pool.
These tests establish mechanics, not sensitivity under the full experimental workload.
`tests/exp12_subprocess_runner.py`, foundations subprocess helpers and preflight test
helpers explicitly select CPU; enclosing them in an outer GPU suite does not make those
subprocesses CUDA tests. Missing HB also omits some loop subcases without a separate skip.

Module shorthand used in the traceability table:

| Shorthand | File / responsibility |
|---|---|
| CONFIG | `configs/base_exp12.yaml`, `configs/agent/sac_simba.yaml`, env config groups |
| ENTRY | `experiments/exp1.py`, `experiments/exp2_arm.py` |
| TRAIN | `experiments/exp12/trainer.py:Exp12Trainer` |
| ENV | `experiments/exp12/envs.py` |
| SAC | shared `scale_rl/agents/sac/{sac_agent,sac_network,sac_update}.py` |
| PROBE | `experiments/exp12/{probe,run_probes,twin}.py` |
| TRIGGER | `experiments/exp12/trigger.py` |
| FORK | `experiments/exp12/{fork,state}.py` |
| INJECT | `experiments/exp12/injection.py` |
| PC | `scripts/positive_control.py`, `experiments/exp12/m_selection.py` |
| LEDGER | `experiments/exp12/{ledger,exp2_ledger}.py` |
| A1 / A2 | `analysis/exp1_analysis.py` / `analysis/exp2_analysis.py` |
| DIAG | `experiments/exp12/diagnostics.py`, SAC actor update and window buffers |
| GRID | `generate_manifest.py`, `scripts/claim_launcher.py` |
| PREC | `experiments/exp12/precision.py` |

Test identifiers below omit the `tests.` prefix. Exp 1/2 class names locate tests in
the `test_exp12_*.py` modules; explicitly named older SAC/cadence tests live in the older suite; the complete inventory appears later in this audit.

## Actor diagnostics: exact implementation

Let `t` index optimizer updates, `s` a normalized replay observation, `a_t` the
reparameterized tanh-Gaussian action, `B=256` the production update batch, and `W` the
updates actually collected within a logging window. Logs close every 2,000 interaction
steps; UTD is 2 after warm-up. Partial windows survive checkpoints. Window statistics
are over observed updates, not over a fixed fabricated count of 4,000.

| Diagnostic / CSV column | Implemented equation, samples, cadence and units | Single / twin behavior | CPU, CUDA, twin tests | A2 use |
|---|---|---|---|---|
| Policy churn: `train/policy_churn`; window alias `train/churn` | Per update `mean_s norm_2(tanh(mu_t(s))-tanh(mu_(t-1)(s)))`. Fixed first normalized training batch (`B=256`), held for the run, not resampled. Mean over updates per window. Action-coordinate L2 units | Same actor equation; twin affects actor gradients through min Q. Diagnostic forward passes use local `highest` | Existing Angle 1 churn/parity tests; diagnostics precision IR test; training on/off equality. CUDA follow-up required. Twin isolation compares resulting actor states, not an independent churn formula | Plots `train/churn`; shared axis currently uses KL rather than this alias |
| KL churn: `train/policy_kl` | `mean_s sum_j [log(sigma_old/sigma_new)+(sigma_new^2+(mu_new-mu_old)^2)/(2 sigma_old^2)-1/2]`, i.e. `KL(pi_new, pi_old)`. 256 replay states sampled with replacement once per window from `(seed,KLRF,window)`, normalized once and held fixed. Per-update KL, then window mean; nats per state | Same closed-form diagonal-Gaussian KL; tanh bijection preserves KL. Both policy forward passes `highest`; training retains TF32 | `KnownAnswerTest.test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old`, near-deterministic report-only test, `ReferenceBatchTest`, `DiagnosticsPrecisionTest`, on/off tests. CUDA deviations/tolerances pending. Twin: isolation test only, no independent twin KL formula test | Plotted post-fork and on shared time axis |
| Gradient norm: `train/actor_gnorm` | `norm_2(grad_theta mean_s [alpha log pi(a_t given s)-Q(s,a_t)])`, before AdamW. One stochastic actor action per replay state, `B=256`; every update, window mean. Loss per parameter-coordinate units | Q single versus min of twin. Includes all actor parameter gradients, not optimizer update norm | `Trainer.apply_gradient` plus actor/update reference/parity tests. Twin actor-loss/min tests and diagnostics isolation; no dedicated analytic gnorm oracle. CUDA update/training isolation pending | Plotted post-fork/shared axis |
| Gradient norm SD: `train/actor_gnorm_std` | `sqrt(mean_(t in W) (g_t-mean_W g)^2)`, NumPy population SD (`ddof=0`) of the norms above. Once per window, host floats after existing metric flushes. Same units as gnorm | Same collector/state for both; depends on respective Q estimator through gradients | `KnownAnswerTest.test_gnorm_std_is_the_population_sd_of_the_window`; reference/checkpoint/on-off mechanics. No twin-specific SD oracle or completed CUDA measurement | Plotted post-fork/shared axis |
| Mean absolute action: `train/actor_action` | `mean_(s,j) abs(a_t(s)_j)`, sampled actor-update actions (`256 × action_dim`), not deterministic environment actions. Every update/window mean; dimensionless normalized action units | Same equation and sample source; twin changes learned actor | Actor-update tests and saturation known answer use the actual sampled actions; no separate analytic action-mean CUDA/twin test | Plotted post-fork/shared axis |
| Saturation: `train/actor_saturation` | `mean_(s,j) 1[abs(a_t(s)_j) > 0.99]`, strict inequality on the same sampled components. Every update/window mean; fraction in `[0,1]`, not percent | Same equation for both; enabled by optional diagnostics kwargs | `KnownAnswerTest.test_saturation_is_the_fraction_of_sampled_components_beyond_the_threshold`, logged-window tests. Twin isolation covers optional path, not separate saturation oracle. CUDA pending | Plotted post-fork/shared axis |
| Gradient cosine: `train/actor_grad_cosine` | Per-state gradients of the stochastic actor objective, independent per-state keys split from that update's actor key. Mean off-diagonal cosine: `(sum(C)-trace(C))/(B(B-1))`, `C_ij=(g_i dot g_j)/(norm_2(g_i)*norm_2(g_j)+1e-8)`. Every 30 optimizer updates; mean of computed values per window; missing scan positions are NaN and filtered. Dimensionless | Single Q versus min of twin, now correctly propagated through the cosine path. It is not a primary Exp 2 endpoint | Older `ActorGradCosineCadenceTest`; twin `test_actor_grad_cosine_uses_min_of_the_two_networks`, scan cadence test; break mutations for missing CDQ argument/Q1-only. CUDA references still needed | Exported in metric CSVs; not in A2 `DIAGNOSTICS` or `SHARED_AXIS`, so not plotted |
| Entropy: `train/entropy` | Per update `-mean_s log pi(a_t given s)` on sampled tanh-Gaussian actions; window average via agent buffer/logger. Monte Carlo differential-entropy estimate, nats; may be negative | Same definition; alpha update uses the estimator; target is `-action_dim/2` in SAC sign convention | Older SAC/parity/scan tests, twin actor/update tests. No dedicated numerical entropy oracle on CUDA/twin | Exported, unused in final A2 plots |
| Actor-loss variance: `train/actor_loss_var` | `var_(t in W)(ell_t)`, population variance (`jnp.var`) of per-update batch-mean actor losses; not variance across individual states or gradient coordinates. Once per window; squared actor-loss units | Objective uses single Q/min twin respectively; buffer mechanism same | Older agent window/scan/parity tests; no dedicated twin variance oracle or CUDA gate | Exported, unused in final A2 plots |

`train/policy_churn` and `train/churn` represent the same underlying update statistic
through two logging routes; they are not independent measurements. Structural actor
features/dormancy/rank are legacy supplementary metrics, not an actor plasticity probe.

**Actor plasticity probe:** the methodology does not require one, and none exists.
The critic probe has a scalar Q regression target. A policy probe would require the
lead to define a state distribution, a target action/distribution/objective, a distance
or loss (including stochastic variance and tanh treatment), a fitting budget and a
fresh-policy reference. Its construct would differ from critic target-fitting
plasticity. No such target or probe has been invented in this task.

## CSV, NPZ and metadata field inventory

Schema groups below enumerate every emitted field or explicitly defined expansion.
`r=0..4`, `q=1,2`; dynamic metric presence depends on whether learning/evaluation
occurred in that window. Pandas unions rows, so missing measurements become blank/NaN.
No legacy fields have been deleted.

| Artifact / fields | Producer | Consumer / final-analysis use |
|---|---|---|
| Run-local `logs/<run_name>.csv`: `env_step`, step-zero `interaction_step`, and every metric below | TRAIN `start/train/write_logs` | Ledgers use in-memory `metrics_rows`; local CSV is archival/review, not read by A1/A2 directly. `interaction_step` is sparse locally; Exp 1 ledger derives it for all rows |
| Run-local `logs/*_eval_episodes.csv`: `interaction_step`, `env_step`, `episode`, `return`, `length`, `success` | TRAIN `_evaluate` | Archival raw nominal evaluations. A2 primary uses separate post-fork evaluations; no normalization applied |
| Exp 1 `run.csv`: `run_key`, `experiment`, `run_role`, `architecture`, `environment`, `seed`, `budget_env_steps`, `num_interaction_steps`, `num_checks`, `status`, `initial_fresh_score_iqm`, `final_loss_iqm`, `f_star_check`, `f_star_interaction_step`, `f_star_fraction`, `fork_interaction_step`, `code_commit` | LEDGER `write_run/RUN_COLUMNS` | A1 role/completion filters, endpoint, f-star tables and plot identity; A2 joins role/architecture/environment/seed/fork identity. Initial score/budget/commit retained for audit rather than primary estimator |
| Exp 1 `checks.csv`: identity fields `run_key`, `experiment`, `run_role`, `architecture`, `environment`, `seed`; `check_index`, `interaction_step`, `budget_fraction`; `score_current_r{r}`, `score_fresh_r{r}`, `loss_r{r}`; `score_current_iqm`, `score_fresh_iqm`, `loss_iqm`, `ci_low`, `ci_high`, `triggered`, `valid` | LEDGER `CHECK_COLUMNS` | A1 trajectories/every-seed exports and A2 shared axis use `loss_iqm`; round scores/intervals/trigger validity are preserved audit evidence. A1 endpoint comes from run.csv. Check zero legitimately lacks current/loss/CI fields |
| Local `probes/probe_checks.csv`: check fields above excluding ledger identity/budget fraction; optional `forced`; twin `score_current_q{q}_r{r}`, `score_fresh_q{q}_r{r}`, `loss_q{q}_r{r}`, `loss_q{q}_iqm` | PROBE `capture_fresh/record_check/per_network_fields` | PC reads local records; tests and manual review. Forced hook is dev-only. Canonical Exp 1 ledger drops `forced` and per-Q fields; A1 does not read these directly |
| Exp 1 `metrics.csv`: local training metrics plus `run_key`, derived `interaction_step`, `budget_fraction` | LEDGER `write_run` | A2 `load_metrics/shared_time_axis`; only shared-axis diagnostics and `eval/avg_return` are plotted |
| Exp 2 arm `checks.csv`: local probe fields, `arm`, `steps_since_fork`; starts at fork check and extends same scheduled grid | LEDGER `write_arm` | A2 exports all checks, plots aggregate `loss_iqm`; per-Q values retained but not plotted |
| Exp 2 arm `metrics.csv`: all metric fields plus `arm`, `steps_since_fork`; rows from fork onward | LEDGER `write_arm` | A2 attaches run identity, exports all, plots six selected diagnostics. Control may have rows beyond the fixed evaluation horizon; the primary return comparison remains bounded by evaluation grid |
| Exp 2 arm `eval_episodes.csv`: `arm`, `eval_index`, `steps_since_fork`, `interaction_step`, `episode`, `return`, `length`, `success` | FORK `post_fork_eval` | A2 primary averages episode returns within each arm/evaluation then subtracts; success/length are preserved and unused by the primary return plots |
| Probe NPZ per network: `losses`, `final_loss`, `offset`, `b`, `score`, prefixed by `fresh/current` or `fresh_q{q}/current_q{q}`; combined twin `fresh/current` only `score`, `final_loss`, `b` | PROBE `_probe`, `twin.combine` | LEDGER copies to `probe_curves.npz` with `check_XX/` prefix; single-Q learning-curve plot uses `*_losses`. Twin plotting gap above. Check 2 curves use `injected/control/fresh` and their Q prefixes in separate `check2_curves.npz` |
| `fork/fork.json`: `fork_step`, `fork_check_index`, `num_interaction_steps`, `horizon_steps`, `arm_end_step`, `control_end_step`, `eval_every_steps`, `eval_episodes`, `run_key`, `run_dir`, `fresh_critic_dir`, `device` (`platform`, `device_kind`, `device_count`) | FORK `fork_plan/write_fork` | ENTRY restore/endpoints, device guard, manifest grouping. Exp 2 ledger fork.json contains the logical plan without path/device informational fields |
| `fork/panel.npz`: `buffer_index`, `observation_raw`, `observation`, `action`; panel result files `check1_pre.npz`, `check1_control.npz`, arm `check1_after.npz`: `q`, `dq_da` | FORK panel/save functions | Check 1, identity comparison, audit. Twin q flattened to 512 values; dq_da is gradient of min on 256 actions |
| Check 1 JSON: `tolerance_eps`, `after_is_injected`, `matmul_precision`, `pairs`, `max_eps_units`, `pass`; pair keys `pre_vs_after`, `pre_vs_control`, `after_vs_control`; each `max_abs_dq`, `max_abs_d_dq_da`, `tolerance_eps`, `dq_eps_units`, `d_dq_da_eps_units`, `pass` | FORK `check1`; ENTRY writes `check1_injected/identity.json`; approved boundary guard writes `check1_control.json` | A2 reads injected pass/max only; pair details/identity/control are audit/validation outputs. Device and configured precision live in fork/run metadata rather than duplicated in each Check 1 JSON |
| Check 2 JSON: `check_index`; `score_{injected,control,fresh}_rounds`, `score_{injected,control,fresh}_iqm`; `loss_injected_rounds`, `loss_control_rounds`; `paired_difference_rounds`, `paired_difference_iqm`, `paired_difference_ci_low`, `paired_difference_ci_high`, `confidence`, `bootstrap_resamples`, `pass`; twin `score_{injected,control,fresh}_q{q}_rounds` | `exp2_arm.check2` | A2 check table/secondary pass filter uses aggregate difference, CI, pass. Per-Q scores and full round distributions retained for audit; not separate endpoints |

Auxiliary validation outputs below are consumed by calibration/validation reports,
not by the confirmatory A1/A2 estimators. Their numerical runtime is recorded separately
from scientific measurements; no projected runtime is a measured full-run duration.

| Artifact / complete field expansion | Producer | Consumer / use |
|---|---|---|
| `positive_control.json`: `created_at`, `run_dir`, `run_identity`, `healthy_reference`, `noise_statistic`, `noise_threshold`, `similar_within`, `allow_any_setting`, `status`, `stop`, `chosen_m`; once forked: `fork`, `trigger_check` (`check_index`, `interaction_step`, `loss_rounds_recorded`, `loss_iqm_recorded`, `forced`), `probe` per critic (`score_rounds`, `score_iqm`, `b_rounds`, `offset_rounds`), `loss_rounds`, `loss_iqm`, `trigger_reproduction_max_abs_diff`, `shared_offset_sensitivity` per mode (`loss_rounds`, `loss_iqm`, `offset_rounds`, `sign_agrees`), `selection` (`l_trigger`, `l_injected`, `noise_sd`, `noise_series`, `chosen_m`, `stop`, conditional `recovery`, `noise`). Curves NPZ prefixes each critic onto the existing probe fields | PC and `m_selection.evaluate` | Calibration selection/noise/stop audit; frozen m passed explicitly to manifests. No automatic confirmatory retuning |
| Identity comparison JSON: `run_dir`, `arm_dir`, `pass`, then either `error` or `differences`, `control_snapshot`, `identity_snapshot`, `interaction_step`, `identity_interaction_step`, `fork_step` | `compare_identity_fork.compare/main` | CUDA gate/report and identity tests; no A1/A2 observation |
| Numerics JSONL: `name`; KL records `diagnostic`, `closed_form`, `rel_diff`, `sigma_min`, `sigma_max`; isolation records `deviation`, `tolerance`, `updates` or `interaction_steps`, with optional `twin`, `differing_components` | Test `record_numerics/check_gpu_tolerance` | CUDA measurements and bound review. Missing tolerance causes skip, not PASS |
| Range JSON rows: `arch`, `env`, `seed`, `pool_size`, `is_configured_pool`, `score_rounds`, `b_rounds`, `final_loss_rounds`, `score_iqm`, `b_iqm`, `score_over_b`, `score_std`, `score_range`, legacy `within_10_90_pct_of_b` | `probe_fresh_checks.range_mode` | `exp12_reports.range_criterion` uses configured-pool P/b>=.9; old 10–90% boolean retained as historical information |
| Null pairs JSONL: `arch`, `env`, `seed`, `pair`, `check_index`, `init_keys`, `score_fresh_rounds`, `score_current_rounds`, `b_rounds`, `loss_rounds`, `loss_iqm`, `valid`, `ci_low`, `ci_high`, `resamples`, `confidence`, `null_threshold`, `fired`; summary: `arch`, `env`, `null_pairs`, `per_check_fire_rate`, `fire_rate_exceeds_5pct`, `would_be_null_threshold_p95_of_L`, `iqm_of_L`, `std_of_L`, `note` | `probe_fresh_checks.null_mode` | Null/calibration report only. Suggested percentile threshold is not adopted automatically |
| Profile JSON: `arch`, `env`, `device`, `train_it_per_s_probes_off`, `probe_check_s`, `probe_check_s_all`, `num_interaction_steps`, `projected_run_train_h`, `projected_run_probe_h`, `probe_overhead_pct_of_wallclock`, `peak_device_bytes`, `recommended_jobs_per_gpu_upper_bound`; optional `train_it_per_s_diagnostics_off`, `diagnostics_overhead_pct`, `fork_buffer_transitions`, `fork_save_s`, `fork_restore_s`, `fork_state_bytes`, `post_fork_eval_s`, `post_fork_evals_per_arm`, `post_fork_eval_episodes`, `post_fork_eval_total_h_per_arm`, `post_fork_arm_train_h`, `post_fork_eval_overhead_pct`, `angle_1_wall_s`, `exp1_probes_off_wall_s`, `ratio_exp1_over_angle1` | `profile_exp12.profile_arch` and optional cost helpers | Packing/runtime decision evidence; projected hours and job-count bound are derived estimates, not launch authorization |

All training metric columns:

| Fields (each name is an emitted column) | Producer | A1/A2 consumer / role |
|---|---|---|
| `train/actor_loss`, `train/actor_pnorm`, `train/actor_gnorm`, `train/entropy`, `train/actor_action`, `train/actor_saturation` | SAC `update_actor`, shared Trainer gradient norm | A2 plots gnorm/action/saturation; entropy/loss/pnorm archived legacy/supplementary |
| `train/policy_churn`, `train/policy_kl` | SAC `_sac_update` | A2 plots KL; churn update column archived (window alias plotted) |
| `train/churn`, `train/actor_loss_var` | SAC window flush/get_metrics | A2 plots churn only; variance archived legacy |
| `train/actor_gnorm_std` | DIAG `window_metrics` | A2 post-fork/shared-axis plot |
| `train/actor_grad_cosine` | SAC scan/cosine update path | Retained mechanism diagnostic, not plotted in A2 |
| `train/critic_loss`, `train/td_error_var`, `train/q1_mean`, `train/q2_mean`, `train/rew_mean`, `train/critic_pnorm`, `train/critic_gnorm` | SAC `update_critic` | Exported supplementary/legacy; A1 primary uses probe loss, not TD loss. Single critic logs q2_mean as alias of q1_mean |
| `train/td_error_q1_var`, `train/td_error_q2_var` (twin only) | SAC `update_critic` | Per-Q audit; A2 exports, does not plot |
| `train/temperature`, `train/temperature_loss`, `train/temperature_gnorm` | SAC `update_temperature` | Archived optimization controls; unused final A1/A2 estimator |
| `train/actor_DR0.1`, `train/actor_DR0.2`, `train/actor_fnorm`, `train/actor_wnorm`, `train/actor_srank` | SAC `get_actor_with_metrics` | Archived legacy structural metrics; unused final A1/A2 plots |
| `train/critic_DR0.1`, `train/critic_DR0.2`, `train/critic_fnorm`, `train/critic_srank` (single); `train/critic_wnorm` (both) | SAC `get_critic_with_metrics` | Archived legacy; injected critic structural metrics are explicitly NaN because its module intermediates differ |
| `train/critic_q1_DR0.1`, `train/critic_q1_DR0.2`, `train/critic_q2_DR0.1`, `train/critic_q2_DR0.2`, `train/critic_q1_fnorm`, `train/critic_q2_fnorm`, `train/critic_q1_srank`, `train/critic_q2_srank` (twin) | SAC `get_critic_with_metrics` | Per-Q legacy structural outputs; injected twin structural metrics explicitly NaN, not zero; unused final A1/A2 |
| `eval/avg_return`, `eval/avg_length`, `eval/avg_success` | TRAIN `evaluate_episodes/_evaluate` | A2 shared-axis nominal raw return; length/success archived. Primary A2 uses per-episode post-fork rows |

Metric dictionaries do not define a single immutable CSV schema; the enumerated union
is emitted by the current Exp 1/2 paths. No generic logger/WandB timing or media fields
are fabricated as scientific CSV fields here.

Run metadata (`run_metadata.json`) is produced by ENTRY and `utils/run_metadata.py`:
resolved config, identity/role, code version, simulator versions, critic count, hardware
and numerical runtime information. PREC records JAX/JAXlib, backend/platform version,
GPU model, CUDA plugin/PJRT versions, matmul precision and cache directory. It is used
for resume/parent configuration guards and audit, not as observations in the estimator.
The metadata schema is `schema_version`, `identity`, `created_at`, `code`, `settings`,
`protocol`, `protocol_hash`, `resolved_config`, `config_hash`, `launches`. `identity`
contains `experiment`, `architecture`, `environment`, `seed`, `run_key`, `run_role`;
`code` contains `commit`, `dirty`. `settings` contains `architecture`, `env_id`,
`simulator_versions`, `discount`, `seed`, `num_env_steps`, `num_interaction_steps`,
`updates_per_interaction_step`, `action_repeat`, `gamma`,
`logging_per_interaction_step`, `actor_grad_cosine_every`,
`evaluation_per_interaction_step`, `num_eval_episodes`. `protocol` contains the
resolved agent/env/buffer subtrees and `utils/run_metadata.py:PROTOCOL_KEYS`, plus
`env_id`, `simulator_versions`; configuration subtrees are data, not a second fixed
metric schema. Launch entries contain `started_at`, `resumed`, `device`, `runtime`,
`commit`, `dirty`, `hostname`, `simulator_versions`, and `critic_count=2` only for
twins (single-Q has no explicit count field). Runtime fields are `jax`, `jaxlib`,
`platform`, `device_kind`, `backend_platform_version`, `jax_cuda12_plugin`,
`jax_cuda12_pjrt`, `matmul_precision`, `compilation_cache_dir`.

Checkpoint `meta.pkl` contains `interaction_step`, `update_step`, `update_counter`,
`numpy_rng_state`, `python_rng_state`, `train_env`, `eval_env`, `observations`,
`timestep`, `meters`, `media`, `metrics_rows`, `eval_rows`, `wandb_run_id`,
`extra_state`, `agent_window_buffers`, `actor_diagnostics`. Diagnostics state has
`window`, `kl_ref`, `gnorm`; window buffers are `actor_loss_buffer`,
`actor_entropy_buffer`, `churn_buffer`. Extra state holds `probe_records`, `f_star`,
and when applicable `fork`, `arm`, `post_fork_evals`, `injection` (`m`, `seed`).
Agent Orbax state has `rng`, `actor`, `critic`, `target_critic`, `temperature`,
`churn_ref_batch`; normalization pickle has `mean`, `var`, `count`. Replay and
environment restoration fields are in `state.save_buffer` / `envs.env_restore_state`,
including the vector action-space RNG and pre-reset simulator RNG/action history.
These are persistence/control data, not additional analysis observations.

## Twin coverage and limits

Tested paths: independent two-network initialization and parameter axis; actor min;
shared target min and summed critic loss; per-Q TD variance and their mean; min-based
actor-gradient cosine and scan cadence; Polyak over both targets; per-Q structural
metrics; twin checkpoint round-trip; probe expansion slices and mean combination;
independent injected heads in both Qs and both targets; frozen heads; trunk/new-head
optimizer rules; both-Q Check 1 and min action gradients; twin fork metadata;
identity snapshot; post-fork local per-Q records; aggregate/per-Q Check 2; injected
arm continuation; probes-on/off and diagnostics-on/off state equality.

Meaningful missing or incomplete paths:

- Real HB pipeline, Reach goal/RNG isolation, environment restore and full-size
  twin identity on CUDA; two explicit HB skips plus loop omissions locally.
- Twin Check 1 compiled/eager reference comparison is red in this CPU environment.
  Q fails before the gradient assertion; a separate evaluation also fails the existing
  gradient criterion. The detailed validation record distinguishes these from the
  passing compiled original/control/identity comparisons.
- Canonical per-Q Exp 1 ledger, twin learning-curve rendering and complete-grid
  rejection tests.
- Twin kill/resume matrix, including an interrupted twin injected-arm save.
  Existing kill matrix uses one Q; tiny twin checkpoint/identity tests do not replace it.
- Full-size twin probe sensitivity/null and twin positive-control transfer.
  `probe_fresh_checks.range_mode/null_mode` call single-network `run_probe` directly,
  rather than `run_probe_networks`; twin expansion in those auxiliary checks is
  **NOT IMPLEMENTED** and untested. A future twin null needs explicit adaptation. The
  prescribed m is calibrated on single-Q dog-run and frozen globally; this audit does
  not retune it or claim that calibration establishes HB rescue.
- Independent twin known-answer diagnostics for churn, KL, SD, action mean,
  saturation, entropy and loss variance. Existing twin isolation tests are narrower.
- Structural metrics for injected heads are not computed; they remain NaN by design.
- Full-budget multi-process CUDA grid execution, restart, cache and packed-node
  behavior on the final tree.

## Remaining CUDA gates and acceptance evidence

Every row below is **NEEDS CUDA** on the final tree. Skips, missing output, timeout,
CPU-only subprocess results and successful sbatch completion do not establish PASS.
The documented Block A measurements are historical results, not final-tree coverage.

| Gap | Execution mapping | PASS evidence | FAILURE / unresolved evidence |
|---|---|---|---|
| Original/control panel and identity fidelity | Block B lane 2 `identity_{cold,warm}_D4W{1024,1536}_{suite}`; HB six `identity_*` steps. Final tree must include approved boundary guard | Guard passes; comparison JSON `pass=true`, no state/panel differences after 1,000 interaction steps, same device model; inspect both cold and warm cases where specified | Any difference, failed guard, wrong model, missing case or CPU backend. HB currently does not explicitly split cold/warm cases; do not silently change the identity measurement |
| HB environment integration and Reach randomness | HB `tests`: HB pipeline plus `HumanoidBenchReachEvalSeedingTest`; B HB step if environment supplied | Actual GPU backend recorded; tests run without HB skips, goals vary with global seed, eval restores training RNG and yields paired seeded evaluations | Skip, missing HB, renderer/init exception, changed RNG or mismatched paired evaluation |
| Twin SAC/probe/injection/Check 1 numerical paths | HB `tests.test_exp12_twin_critic`; B final-tree suites only in a future rerun | All tests pass in appropriate precision mode; both-Q values, min gradients, independent heads, optimizer/target invariants verified | Any failed assertion/error; identity pass cannot excuse incorrect estimator or head logic |
| Diagnostic update isolation | B lane 2 `tests_default`, `tests_deterministic`, A1 follow-up; HB twin isolation test | Measurements emitted, lead-approved `diagnostics_update/{tf32,highest_deterministic}` tolerances populated and deviations within bound | Four tolerance keys currently None: measurement followed by skip remains unresolved; violation fails |
| Diagnostic training isolation | B lane 2 same suites/follow-up | Approved `diagnostics_training/*` bounds and training/reference differences within them | Missing/None tolerance, skip, divergence above bound |
| Diagnostic KL/churn precision | B A1 follow-up and diagnostic tests; twin in HB tests | Compiled IR preserves local highest passes and TF32 training; known-answer well-conditioned KL passes configured criterion; near-deterministic case reported separately | Wrong compiled precision or known-answer mismatch. Do not turn near-deterministic report-only data into a new gate |
| Probe sensitivity, m selection and noise | B lane 0 `dev_run`, `positive_control`; lane 1 injected watcher | Natural eligible degradation; full-settings report for each m, positive L_trigger, recovery/noise rules satisfied, selected m frozen before grid | No trigger, L_trigger<=0, noise>=0.10 (exit 3), failed Check 1; rescue failure must be reported, not forced |
| Fresh range | B lane 2 `range_hopper`, historical A2 | Configured pool 25,600, P/b>=0.9 for each current architecture; final-loss spread reported | Any missing size or ratio below 0.9. Historical D6W1536 is not D4W1536 validation |
| Fresh-pair null | B lane 3 `null_dog_run`; future twin null after adapting the auxiliary wrapper | All 100-pair records per size retained; rate/noise summary inspected under pre-specified decision rules | Missing cases or excessive null firing requiring lead consultation; no threshold retuning by agent |
| Preflight persistence | B lane 0 `preflight_D*`; future HB/production CUDA process checks | Actual CUDA process restores complete state; scaled cells include passing Check 1/injected arm | CPU-forced unittest wrapper alone is insufficient; missing/failing direct CUDA preflight |
| Performance, memory and multi-job packing | B lane 2 packing/A1, lane 3 suite speed; HB `speed_h1-run-v0` | Finite measured rates/probe timings/peak memory for all current sizes, no OOM; overhead decision rules reviewed | OOM/timeout/missing size; estimates are not measurements. Overhead above accepted rule requires lead review, never reduced probes |
| Full final suite and break checks | Future final-tree CUDA validation plus CPU subprocess results separately labeled | All runnable tests and mutations pass/restored; no hidden tolerance or HB skips | Present red twin panel restoration, any new failure, CPU-only result mislabelled as CUDA |

Block B's submitted checkout must remain untouched. The HB script's four-hour request
and documented runtime estimates have not been validated by this session.

## Persistent compilation cache investigation (JAX 0.4.34)

The official `jax-v0.4.34` versions of `compiler.py`, `compilation_cache.py`,
`lru_cache.py` and `config.py` were downloaded and compared byte-for-byte with the
installed files: all four matched. Inspection therefore concerns the pinned implementation, not current
JAX behavior. Sources below use the immutable tag.

The warning originates in `compiler._cache_read`: `get_executable_and_time` raises,
the default `jax_raise_persistent_cache_errors=false` catches it, emits a warning and
returns `(None,None)`. `compile_or_get_cached` then follows its miss branch and
compiles; `_compile_and_write_cache` returns that newly compiled executable. If the
raise flag is true, the exception instead propagates. Corrupted compressed bytes are
not used as a partially decoded executable. This establishes fallback behavior, not
the bit-identity of independently autotuned CUDA compilations.
[Pinned compiler source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/compiler.py).

`compilation_cache.get_executable_and_time` reads bytes, decompresses with zlib (or
zstandard if installed), extracts compile time/serialized executable and deserializes.
The truncated-stream message is raised before deserialization. The compression type
participates in the cache key. A failed read is not an invalidation/repair operation.
[Pinned compilation-cache source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/compilation_cache.py).

`LRUCache.put` checks existence and then calls `cache_path.write_bytes(val)` directly
on the final file: **no temporary-file/atomic-rename publication**. At the default
unlimited `max_size=-1`, get/put acquire no lock. Another reader can encounter a
partially written file; competing writers can both pass the existence check. A
killed writer/storage failure is another plausible cause. This is an inference from
the code, not proof of the cause on Delta. Existing entries are not replaced, so a
corrupt entry can survive later recompilations. `get` always writes an atime sidecar,
even with eviction disabled, so a genuinely read-only warmed directory does not
provide a working read-only cache through this implementation.
[Pinned LRU source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/lru_cache.py).

The tagged documentation recommends shared storage for **one coordinated distributed
JAX computation**, where rank zero writes. Block B launches independent programs;
each is its own process zero, so that restriction does not serialize their writes.
Shared-cache documentation is not evidence of atomic independent-writer publication.
The public finite-size setting enables `filelock` around get/put/eviction: it provides
mutual exclusion, not atomic crash-safe replacement, and depends on filesystem locking.
There is no supported atomic-write switch or public custom-cache setter in this
version. A private monkeypatch or a JAX upgrade is outside a semantics-preserving
infrastructure fix for this task.
[Tagged documentation](https://github.com/jax-ml/jax/blob/jax-v0.4.34/docs/persistent_compilation_cache.md),
[pinned configuration source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/config.py).

CPU reproduction: a fresh process compiled `sin(x @ x.T)` and its input preparation
with caching enabled; five bytes were removed from each of four cache entries. A
second process emitted the exact zlib `Error -5` warnings, recompiled, and produced
both array-equal and byte-equal output to the saved first result. This is measured
CPU fallback evidence only. It does not reconstruct Block B's race or prove CUDA
identity. Reproducer and stdout were saved under `/tmp/exp12-cache-repro*`.

| Option | Benefit | Limitation / judgment |
|---|---|---|
| Per-lane writable cache | Sequential tasks can reuse compiled entries | Safe from other lanes only if the lane never overlaps writers; a timed-out/killed writer can leave a bad entry for the next task |
| Per-process fresh writable cache | Eliminates concurrent writers and reuse of an aborted process's partial files | Recompilation and duplicate disk use; simplest isolation. Use a unique directory for every restart, not PID alone across nodes |
| Node-local cache | Avoids shared parallel-filesystem cache traffic | Shared local writers still race. Combine with process ownership; allocation-local cache does not survive rescheduling |
| Shared read-only warmed cache | Conceptually permits common immutable binaries | Stock 0.4.34 still writes atime on reads. Instead publish an immutable warmed template after its producer exits, then copy into private writable caches |
| Persistent cache disabled | Removes persistent entry races/corruption entirely | Every process recompiles; does not make CUDA compilation deterministic. Existing `EXP12_JAX_CACHE_DIR=off` should be used in a clean process without an inherited JAX cache setting |
| Finite-size shared cache plus filelock | Public setting serializes cooperative read/write calls | Lock contention, extra dependency, filesystem lock semantics, eviction and interrupted-write leftovers; not atomic publication |
| Atomic custom publication | Could prevent partial-file visibility with same-filesystem rename | No supported public implementation hook in 0.4.34; private patch/upgrade requires separate review |

**Recommendation, not implemented:** for normal HB validation processes, unique private
writable caches under allocation-local storage (`SLURM_TMPDIR`, if provided; otherwise
a verified node-local temporary path), or disable persistent caching for short isolated
checks. Preserve explicit cold/warm identity cases and their intended sharing exactly;
do not let unrelated validation processes write those directories. For the large grid,
use private writable process caches, optionally initialized from a validated immutable
warmed template copied only after its sole producer finishes. Record cache provenance
and measure compile/copy cost before packing decisions. This avoids changing pinned
JAX or sharing active writers. No cache execution code or submitted Block B files were
changed in this audit. CUDA identity validation is still required before rollout.

## Verified HumanoidBench parameter limitation

The lead approved reporting the two tasks separately after the source-based dimensions
differed. H1 has `dof=26`; Run uses `2*dof-1=51` observations, Reach adds six goal
coordinates for 57. Both pinned MuJoCo models were loaded without a renderer and
reported `nq=26`, `nv=25`, `nu=19`. These dimension checks use HumanoidBench commit
`cb1189039151c8aadaaa987b442da54383c87fab`, not an installed/running HB environment:
[pinned robot source](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/robots.py),
[pinned Run source](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/envs/basic_locomotion_envs.py),
[pinned Reach source](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/envs/reach.py).

| Task | Actor D1W128 | Critic | Parameters per Q | Twin total | Per-Q / actor | Twin / actor |
|---|---|---|---|---|---|---|
| h1-run-v0 | 143,782 | D2W512 | 4,239,361 | 8,478,722 | 29.48 | 58.97 |
| h1-run-v0 | 143,782 | D4W1024 | 33,658,881 | 67,317,762 | 234.10 | 468.19 |
| h1-run-v0 | 143,782 | D4W1536 | 75,654,145 | 151,308,290 | 526.17 | 1,052.35 |
| h1-reach-v0 | 144,550 | D2W512 | 4,242,433 | 8,484,866 | 29.35 | 58.70 |
| h1-reach-v0 | 144,550 | D4W1024 | 33,665,025 | 67,330,050 | 232.90 | 465.79 |
| h1-reach-v0 | 144,550 | D4W1536 | 75,663,361 | 151,326,722 | 523.44 | 1,046.88 |

Counts sum actual Flax initialization parameter shapes traced with `jax.eval_shape`
using the repository's `SACActor`, `SACCritic` and `SACClippedDoubleCritic`, residual
blocks and float32. The twin count was traced independently and checked against twice
the single-Q count. This avoids materializing full orthogonal initializations on the
local CPU; it is structural evidence, not a full-size learning or CUDA test. Reproduce
from `main/` with the pinned environment:

```python
import math
import jax
import jax.numpy as jnp
from scale_rl.agents.sac.sac_network import SACActor, SACCritic, SACClippedDoubleCritic

key = jax.random.PRNGKey(0)
def count(module, *args):
    params = jax.eval_shape(module.init, key, *args)["params"]
    return sum(math.prod(x.shape) for x in jax.tree.leaves(params))

for obs_dim in (51, 57):
    obs = jax.ShapeDtypeStruct((1, obs_dim), jnp.float32)
    act = jax.ShapeDtypeStruct((1, 19), jnp.float32)
    actor = count(SACActor("residual", 1, 128, 19, jnp.float32), obs)
    for depth, width in ((2, 512), (4, 1024), (4, 1536)):
        per_q = count(SACCritic("residual", depth, width, jnp.float32), obs, act)
        twin = count(SACClippedDoubleCritic("residual", depth, width, jnp.float32), obs, act)
        assert twin == 2 * per_q
        print(obs_dim, depth, width, actor, per_q, twin, per_q / actor, twin / actor)
```

These are online networks before injection; targets and optimizer state are excluded.
Only the approved limitation was added to amendment (z). The ratios quantify imbalance
and do not establish a causal explanation for pathology.

## Validation record

Executed code revision: `7d1bda6d6aeeea22fef1692be46bf56409ade38e` (after the
approved guard, fixture isolation and ratio amendment). The audit-only commit follows
these runs and changes no executable/test/config file. Baseline was
`2f67915cf9243d75f56299645ccd3961e9c0c44e`, matching the cached remote-tracking branch;
no fetch, push, Slurm submission or Delta checkout modification occurred. The branch
was 29 commits ahead of `main` at baseline; merge base `6ffccba`. The inherited
shared SAC changes were exercised by the complete older suite.

All measured validation here uses CPU, Python 3.12.13, JAX/JAXlib 0.4.34, Flax 0.8.4,
Optax 0.2.3, Orbax 0.5.3, NumPy 1.26.4, pandas 2.1.4, MuJoCo 3.6.0, dm_control
1.0.38 and MyoSuite 2.12.2. HB was not installed in the test environment. The local
pinned `rliable==1.2.0` dependency was missing; it and supporting packages were
installed into `/tmp/exp12-validation-deps` with `--no-deps`, without changing the
repository or `.venv`. Supporting versions: arch 7.2.0, seaborn 0.13.2, statsmodels
0.15.0, patsy 1.0.3, formulaic 1.2.2, interface_meta 2.0.1, narwhals 2.26.0.

| Run | Code revision | Executed | Pass | Fail | Error | Skip | Seconds |
|---|---|---|---|---|---|---|---|
| Exp 1/2 baseline | `2f67915` | 147 | 143 | 1 | 1 | 2 | 610.345 |
| Exp 1/2 final | `7d1bda6` | 149 | 146 | 1 | 0 | 2 | 565.515 |
| Older/non-Exp-1/2 | `2f67915; shared SAC unchanged afterward` | 258 | 257 | 1 | 0 | 0 | 318.279 |

The final inventory has 149 tests: the baseline 147 plus two approved guard regressions.
The two explicit skips are `HumanoidBenchReachEvalSeedingTest` and
`PipelinePerSuiteTest.test_humanoid_bench`; missing-HB loop omissions are additional
unexecuted subcases. There were no expected-failure decorations added. The 258 older
tests include dynamically generated cases; a static count of function definitions
would undercount the suite. The targeted Hydra-order pair passed 2/2; the targeted
single/twin fork classes passed 19/19.

Mutation evidence (a mutant must fail and its restoration must pass):

| Run | Original checks | Added checks | OK | PROBLEM | Mutated pass | Restored fail |
|---|---|---|---|---|---|---|
| Baseline | 71 | 0 | 70 | 1 | 0 | 1 |
| Final | 71 | 1 | 71 | 1 | 0 | 1 |

The final 72 mutants fail as 68 assertion failures and four runtime errors. Of those,
67 assertion mutations and all four error mutations restore successfully. The original
twin min-gradient mutation restores to the already-red `TwinCheck1Test.test_panel_values`,
so that row remains PROBLEM. The four error-detected mutations concern the old twin
vmap initialization, unset TD-error variable in the twin branch, omitted CDQ cosine argument,
and absent twin metadata count. A runtime error detects breakage but does not establish
the same semantic specificity as a numerical assertion. The added original/control
comparison mutation fails by assertion and restores to PASS.

Failure evidence and repeatability:

- **Twin panel reference, deterministic in this environment.** Baseline and two fresh
  named-test processes produce the same Q mismatch: 67/512 values, maximum absolute
  difference `7.1525574e-07`, maximum elementwise relative difference `0.00017119`.
  The final suite and restored mutation retain the failure. The Q assertion runs
  before the gradient assertion. A separate read-only investigation evaluated that
  existing gradient criterion (`rtol=1e-5`, `atol=1e-7`) and also failed it: 3/768
  elements outside tolerance, max absolute `9.536743e-07`, max relative
  `5.5496555e-05`; 601 elements differ at all. Independently compiled Q and gradient
  references match the panel exactly on this CPU. This is consistent with differing
  compiled/eager floating-point evaluation, not proof of CUDA correctness or grounds
  to relax a gate. No assertion/tolerance was changed; any reference-test revision
  requires the lead's decision and an independent correctness argument.
- **Older MyoSuite determinism, reproducible.**
  `TestMyosuiteDeterminism.test_myo_baoding_p1` fails at step zero: 12/86 observation
  elements differ, max absolute `0.0029`, max relative `1.0158`. The complete older
  run and an independent fresh named-test rerun fail identically. This is in an older
  task outside the four selected Exp 1/2 MyoSuite tasks. It remains red; this session
  does not establish its historical origin or change simulator semantics to hide it.
- **Baseline Hydra error, repaired as fixture isolation.**
  `Angle1ParityTest.test_same_params_optimizer_state_rng_and_metrics` errored because
  earlier configuration composition initialized global Hydra. It passed alone.
  Clearing Hydra at that fixture's start preserves every existing scientific assertion;
  the ordered two-test check passes, and final full-suite status is in the inventory
  below. No training/SAC implementation change was made for this error.

Warnings observed across completed validation, grouped without suppressing them:

- Orbax `SaveArgs.aggregate` deprecation, and inferred sharding because RestoreArgs
  lacks sharding information. These were frequent checkpoint warnings.
- Matplotlib/pyparsing `oneOf`, `parseString`, `resetCache`, `enablePackrat`
  deprecations, and one baseline font-cache initialization notice.
- Gymnasium Box bounds cast from float64 to float32; the imported older Gym package
  also prints its unmaintained/NumPy compatibility banner.
- rliable/arch bootstrap `random_state` deprecation. The test-only dependency versions
  are recorded above; the bootstrap implementation was not changed.
- ResourceWarning for an unclosed temporary checkpoint `meta.pkl` reader in the
  Exp 1/2 tests, with the accompanying tracemalloc hint.
- Older-suite analysis warnings from its synthetic/incomplete fixtures: missing run
  metadata (settings not machine-verified), absent persisted `td_error_variance`
  baselines/onset analysis failure, stale baseline cache recalibration, and negative
  CCF-lag seeds excluded from window calibration. These belong to older analyses;
  they are not passed off as warnings from confirmatory Exp 1/2 result data.

Initial attempts without `MUJOCO_GL=disable` aborted inside GLFW initialization before
any tests ran. The older runner initially launched from stdin and could not be
re-imported by multiprocessing spawn; that invocation was stopped and replaced with
a real guarded Python file. These environmental/runner failures are not included as
executed tests in the completed counts above. CPU was selected explicitly; the local
Metal backend cannot substitute for CUDA validation.

Commands used from `main/` (stdout/stderr captured together):

```bash
export JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off
export PYTHONPATH=/tmp/exp12-validation-deps
export MPLCONFIGDIR=/tmp/exp12-mpl XDG_CACHE_HOME=/tmp/exp12-xdg
../.venv/bin/python -u -X faulthandler -m unittest discover -s tests -p 'test_exp12_*.py' -v
../.venv/bin/python -u -X faulthandler tests/exp12_break_checks.py
../.venv/bin/python -u -X faulthandler /tmp/exp12-run-older-tests.py
```

The older runner adds `Path.cwd()` to `sys.path`, and under an
`if __name__ == '__main__':` guard builds a unittest suite with
`TestLoader.discover('tests', pattern=p.name)` for each sorted `test_*.py` file excluding
`test_exp12_*`, then returns nonzero unless the result succeeds. The equivalent
fresh-process named-test repeat commands use `-m unittest` with the exact failing
identifiers listed above. Logs and reproducers under `/tmp` are local audit evidence,
not committed research outputs. Completed raw logs are fingerprinted here:

| Local log | SHA-256 |
|---|---|
| `/tmp/exp12-tests.log` | `80806e6e1b29c4c9f558de574a6037931e5d984b28c30dcad9c8a7acfd81a104` |
| `/tmp/exp12-tests-final.log` | `b3fd3d752d87e5beaad398da32bb5d48a307fb7bc6f75dfe6f23cadfe3bbf279` |
| `/tmp/exp12-older-tests-final.log` | `ab572f11d8f7567891017613156b8ebdc888fb4174ac43bbc1da64139a743aec` |
| `/tmp/exp12-break-checks.log` | `9642456ca8cea9e0b37622aef5dd665a03499f3a1492e79db285e9fa1d431d90` |
| `/tmp/exp12-break-checks-final.log` | `17d3a6ed305112d364b5fb909ebb9fb61f80c281ee78d6a91f13b90a719b4981` |

## Decisions still reserved for the project lead

The audit does not repair the reporting/statistical deviations. Before a confirmatory
grid or numerical-gate revision, the lead must decide:

| Issue | Available direction / consequence |
|---|---|
| Missing whole Exp 1 seeds/environments | Enforce the prescribed 5×13 population and refuse incomplete estimates (recommended); any alternative population/missing-data rule would change analysis and require explicit specification |
| Exp 2 completeness | Validate the prescribed aligned evaluation indices and episode counts (recommended); accepting partial evaluations would need a new declared inclusion rule |
| Twin canonical records and curves | Preserve the existing per-Q data in the canonical ledger and plot the already-saved per-Q curves; defining an aggregate learning curve would require an explicit rule |
| Red compiled/eager reference | Keep the current gate red pending independent correctness/CUDA evidence; any proposed bounded numerical reference must be reviewed separately while exact original/control/identity boundaries stay intact |
| CUDA diagnostics bounds | Inspect actual final-tree measurements, then approve bounds using the documented rule; None/skip cannot authorize the grid |
| Cache rollout and launch plumbing | Use isolated private writable caches for ordinary processes while preserving identity cases; verify CUDA identity and the job status outputs before launch. No job was submitted or running checkout touched |

## Requirement-by-requirement traceability

The table below references source line numbers preserved through the ratio limitation addition;
amendment labels and quoted opening phrases remain the stable identifiers. Original
superseded bullets are mapped to their current replacement, not silently reinterpreted.
Each original bullet/list item and each amendment/sub-bullet has its own source row.

<!-- TRACEABILITY -->
| Source item | Requirement (opening phrase; full source remains authoritative) | Implementation | Tests / evidence | Status | Evidence / qualification |
|---|---|---|---|---|---|
| M L29 | 5 seeds throughout the study | GRID.EXP12_SEEDS | Exp12GridTest.test_the_195_run_grid | **DEVIATES** | Seeds 1..5 generated. Analysis does not independently enforce all five; see primary completeness deviation. |
| M L100 | Pool the final-checkpoint comparison across environments using rliable's stratified bootstrap (65 runs per architecture) → this produces the primary architecture-level estimate and confidence interval for sc… | A1.score_matrix/interval_estimate | Exp1AnalysisTest.test_missing_run_is_refused (cell gaps only) | **DEVIATES** | rliable stratified independent architecture resampling, 50k. Prescribed 5×13 shape not enforced; missing whole dimensions can pass. |
| M L102 | Every seed in every environment is shown descriptively, and the individual-run $f^*_{run}$ values are retained in the structured ledger | LEDGER.write_run; A1.plot_every_seed/f_star_table | LedgerTest; Exp1AnalysisTest | **DEVIATES** | Every observed seed plotted and f-star retained. Canonical per-Q fields lost for twin; local records survive. |
| M L188 | Compute Q-values and $\nabla_aQ$ on the fixed panel for (1) the critic just before injection, (2) the injected critic immediately after injection, and (3) the control critic immediately after the fork → all … | FORK.check1/panel_q_and_grad; ENTRY guard | ForkUnitTest Check1 tests; ForkEndToEndTest.test_check1; test_control_stops_before_ready_if_restore_changes_panel; test_arm_rechecks_original_panel_for_existing_forks; TwinCheck1Test | **DEVIATES** | Highest precision; control/identity exact, injection64 eps separate Q/gradient scales. Approved original/control zero-tolerance guard added, with common-restore-error regression; existing arm comparisons retained. Twin eager reference remains red. |
| M L524 | Probe: each network is probed against its own fresh copy, on the shared | PROBE.twin.expand/combine/per_network_fields; LEDGER | TwinProbeTest; TwinForkEndToEndTest.test_probe_records_are_the_mean_of_both_networks | **DEVIATES** | Shared pool/base/minibatches; each Q its own fresh and own offset; trigger meanL. Canonical Exp1 per-Q CSV retention gap. |
| M L95 | Use the probe learning curves descriptively to determine whether the plasticity loss manifests as the slower/shallow new-target learning described by Lyle et al. 2023 → do not introduce a new slope formula | RunProbes._probe; A1.plot_learning_curves | Exp1AnalysisTest.test_full_analysis_writes_outputs (single only) | **NOT IMPLEMENTED** | Curves retained; no slope metric. Twin plotting keys unsupported, see deviation. |
| M L419 | (u) Limitation: the whole grid, including both arms of every fork, must run | Methodology operational rule; FORK.check_same_device (per-fork only) | No global-model enforcement test | **NOT IMPLEMENTED** | One GPU model wholegrid remains an operational precondition; programmatic global validation absent. |
| M L80 | h1-reach - hard | Amendment (a); ENV.HUMANOID_BENCH_TASKS | PipelinePerSuiteTest.test_humanoid_bench (skipped locally) | **NEEDS CUDA** | Current IDs h1-reach-v0 and h1-run-v0. Real HB stepping/restore on final CUDA tree unresolved. |
| M L110 | Twenty scheduled checks create repeated opportunities for an operational trigger to occur → the first detected crossing should therefore be interpreted as an intervention point rather than an exact estimate … | TRIGGER; scripts/probe_fresh_checks.py null mode | TriggerTest.test_null_false_trigger_rate_is_measured | **NEEDS CUDA** | Repeated-look operational trigger recorded; actual full-settings CUDA fresh-pair null still pending. |
| M L140 | h1-reach - hard | Amendment (a); ENV.HUMANOID_BENCH_TASKS | PipelinePerSuiteTest.test_humanoid_bench (skipped locally) | **NEEDS CUDA** | Current IDs h1-reach-v0 and h1-run-v0. Real HB stepping/restore on final CUDA tree unresolved. |
| M L148 | Obtain a naturally degraded scaled critic checkpoint from a preliminary SAC/SimBa development run using the Experiment 1 degradation criterion and probe it → the score should show clear plasticity loss | PC.run; ENTRY dev run | PositiveControlEndToEndTest; SharedOffsetTest | **NEEDS CUDA** | Naturally trained trigger checkpoint, no artificial damage. Tiny forced-trigger mechanics do not establish actual degradation/rescue; Block B dev/PC needed. |
| M L149 | Apply plasticity injection to the naturally degraded critic and probe it again → the score should recover toward the fresh critic's level | PC.run; ENTRY dev run | PositiveControlEndToEndTest; SharedOffsetTest | **NEEDS CUDA** | Naturally trained trigger checkpoint, no artificial damage. Tiny forced-trigger mechanics do not establish actual degradation/rescue; Block B dev/PC needed. |
| M L151 | Use one scaled critic architecture and one environment for the positive-control calibration | CONFIG.positive_control; PC.run checks | PositiveControlEndToEndTest.test_refuses_wrong_setting | **NEEDS CUDA** | D4W1536 dog-run dev seed outside1..5, 1M raw budget. Real calibration outcome pending. |
| M L152 | This validates the plasticity-injection intervention in SAC critics, SimBa, and continuous control; Lyle et al. 2023 establishes the plasticity probe in their value-learning settings, not ours | PC.run; ENTRY dev run | PositiveControlEndToEndTest; SharedOffsetTest | **NEEDS CUDA** | Naturally trained trigger checkpoint, no artificial damage. Tiny forced-trigger mechanics do not establish actual degradation/rescue; Block B dev/PC needed. |
| M L153 | Step 4 is also the first test of the injection | PC.run; ENTRY dev run | PositiveControlEndToEndTest; SharedOffsetTest | **NEEDS CUDA** | Naturally trained trigger checkpoint, no artificial damage. Tiny forced-trigger mechanics do not establish actual degradation/rescue; Block B dev/PC needed. |
| M L154 | No separate artificial damage procedure is used; the degraded critic comes from the same SAC/SimBa setting being studied | PC.run; ENTRY dev run | PositiveControlEndToEndTest; SharedOffsetTest | **NEEDS CUDA** | Naturally trained trigger checkpoint, no artificial damage. Tiny forced-trigger mechanics do not establish actual degradation/rescue; Block B dev/PC needed. |
| M L202 | Fork without injecting either branch → both must remain numerically identical within an implementation-appropriate tolerance over 1000 further training steps | scripts/compare_identity_fork.py; FORK snapshots | ForkEndToEndTest.test_identity_fork_is_bit_identical; IdentityValidationTest; TwinForkEndToEndTest | **NEEDS CUDA** | Tiny CPU equality passes where indicated; prescribed1000-step final full-size CUDA gate remains pending. |
| M L203 | This tests fork fidelity (RNG, replay buffer, optimizer state, environment state/reproducibility, and relevant training state) separately from injection | scripts/compare_identity_fork.py; FORK snapshots | ForkEndToEndTest.test_identity_fork_is_bit_identical; IdentityValidationTest; TwinForkEndToEndTest | **NEEDS CUDA** | Tiny CPU equality passes where indicated; prescribed1000-step final full-size CUDA gate remains pending. |
| M L231 | (a) HumanoidBench tasks. HumanoidBench uses h1-reach-v0 and h1-run-v0, the | ENV.HUMANOID_BENCH_TASKS; GRID | PipelinePerSuiteTest.test_humanoid_bench (local skip) | **NEEDS CUDA** | No-hands h1-reach/run; historical blocked text outdated; final CUDA integration not established. |
| M L236 | (b) Fork design. At the trigger the complete training state is saved. The | FORK.write_fork; ENTRY.enter_control; TRAIN.restore | ForkEndToEndTest; KillMatrixTest; IdentityValidationTest | **NEEDS CUDA** | Parent restores actual saved state, injected job separate; full-size per-suite CUDA identity is a launch gate. |
| M L269 | (e) Positive control. D4W1536 (was D6W1536; amendment (y)) on dog-run (DMC hard, 1M environment steps), with | CONFIG.positive_control; PC.run | PositiveControlEndToEndTest.test_never_triggered_run_stops; test_refuses_wrong_setting | **NEEDS CUDA** | D4W1536 dog-run, external dev seed; no natural trigger -> stop, never loosen trigger. Real outcome pending. |
| M L318 | HumanoidBench needs an EGL offscreen context on GPU nodes | ENV._import_humanoid_bench; HB sbatch exports | HB integration/Reach tests (skipped locally) | **NEEDS CUDA** | GPU needs egl + matching PYOPENGL_PLATFORM. Historical CPU verification is not repeated here; all final CUDA work remains pending. |
| M L320 | The HumanoidBench integration is verified on CPU only; it is untested on GPU. | ENV._import_humanoid_bench; HB sbatch exports | HB integration/Reach tests (skipped locally) | **NEEDS CUDA** | GPU needs egl + matching PYOPENGL_PLATFORM. Historical CPU verification is not repeated here; all final CUDA work remains pending. |
| M L330 | (m) Check 1 tolerance. The control arm, and the identity arm used for | FORK.check1; highest panel; ENTRY gate | ForkUnitTest; ForkEndToEndTest; new boundary regressions | **NEEDS CUDA** | Control/identity exact; injection<=64eps*maxabs per observable; no TF32 rejection after(v). Original-panel baseline gap repaired only with approval; CUDA still needed. |
| M L362 | (p) Device model. Both arms of a fork run on the GPU model that produced the | FORK.check_same_device; GRID arm grouping | ForkUnitTest.test_both_arms_must_run_on_the_fork_device_model; Exp12GridTest | **NEEDS CUDA** | Guard platform/device_kind on control resume and arm; six final architecture×suite identity cells need CUDA. |
| M L424 | (v) Matmul precision (replaces the earlier FP32-everywhere version of (v)). | PREC.set_matmul_precision; all Exp12 GPU entry points | DiagnosticsPrecisionTest compiled-IR test; source inspection | **NEEDS CUDA** | GPU tensorfloat32; CPU no-op; compute casts float32. Runtime CUDA kernels not tested here. |
| M L425 | Every float32 matrix multiplication in Experiment 1 and 2 jobs uses TF32 | PREC.set_matmul_precision; all Exp12 GPU entry points | DiagnosticsPrecisionTest compiled-IR test; source inspection | **NEEDS CUDA** | GPU tensorfloat32; CPU no-op; compute casts float32. Runtime CUDA kernels not tested here. |
| M L428 | It is set explicitly at the start of every GPU entry point: | PREC.set_matmul_precision; all Exp12 GPU entry points | DiagnosticsPrecisionTest compiled-IR test; source inspection | **NEEDS CUDA** | GPU tensorfloat32; CPU no-op; compute casts float32. Runtime CUDA kernels not tested here. |
| M L458 | New criterion: at the configured pool, P/b ≥ 0.9 at every critic size. | exp12_reports.range_criterion; probe_fresh_checks.py | RangeCriterionTest.test_a_size_below_0_9_or_missing_fails | **NEEDS CUDA** | New P/b>=.9 every current size at configured pool; missing size fails. Actual D4W1536 fullsettings CUDA range still pending. |
| M L462 | What this check can and cannot show: it cannot establish sensitivity. | PC; scripts/probe_fresh_checks.py null | PositiveControlEndToEndTest mechanics; BlockB pending | **NEEDS CUDA** | Fresh fit establishes solvability, not sensitivity; natural degradation/rescue and fresh-pair noise measured separately. |
| M L532 | Check 1: Q of both networks, and dQ/da of min(Q1, Q2), with the | FORK.panel_q_and_grad; exp2_arm.check2 | TwinCheck1Test (red); TwinForkEndToEndTest.test_check1_both_networks; test_check2_uses_the_mean_and_records_both_networks | **NEEDS CUDA** | Both Q outputs512 values; gradient of min; Check2 meanP difference plus both perQ scores. CUDA pending. |
| M L535 | The fork state, restore, run metadata (critic count 2) and the identity | TRAIN.save/restore; metadata launches[].critic_count; identity snapshots | TwinCriticPathTest.test_checkpoint_round_trip_keeps_both_networks; TwinForkEndToEndTest | **NEEDS CUDA** | Tiny twin CPU checkpoint/identity covered; HB/full-size CUDA identity missing; twin killmatrix missing. |
| M L12 | MUST USE PERCENTILE BOOT STRAPS → rliable | A1.interval_estimate; TRIGGER.bootstrap_interval; A2.bootstrap_band | Exp1AnalysisTest; TriggerTest.test_statistic_is_iqm_not_mean; BandTest | **CPU VERIFIED** | A1 uses rliable 50,000 stratified percentile resamples. Trigger and A2 use approved explicit NumPy percentile resampling (10,000), not rliable wrappers. |
| M L13 | Use interquartile-mean if you have distributions with low amounts of samples → reduces sensitivity to outliers and computes a robust mean-like aggregate | PROBE.iqm; A1 metrics.aggregate_iqm; A2.STATISTICS | ProbeArithmeticTest.test_iqm_matches_rliable_definition; BandTest | **CPU VERIFIED** | 25% trim. Five rounds retain middle three values; operational small-n limitations remain. |
| M L14 | Using the hyper parameters from other papers in field is justified because we are not aiming to compare different architectures, but rather understand. Furthermore, the hyper parameters were tested on a larg… | CONFIG; A2.paired_returns | Exp12GridTest; SyntheticResultsTest | **CPU VERIFIED** | Design rationale, not an executable claim of generalizability or causal sufficiency. The paired intervention is implemented; no additional cross-algorithm comparison invented. |
| M L15 | Include baselines to give the reader an understanding of your results → healthy/default architecture in Experiment 1 and paired no-injection control in Experiment 2 | PROBE fresh reference; A1.primary_endpoint; A2.paired_returns | ProbePairingTest; Exp1AnalysisTest; SyntheticResultsTest | **CPU VERIFIED** | Same-run fresh baseline, architecture baseline and paired control serve different purposes. |
| M L16 | Development/calibration runs used to validate the injection and select the injection head are kept separate from the confirmatory main runs | ENTRY.run_identity; LEDGER.load; A2.load; PC.run | LedgerTest.test_round_trip_and_dev_excluded_by_default; Exp1AnalysisTest.test_dev_runs_do_not_change_the_confirmatory_result; SyntheticResultsTest | **CPU VERIFIED** | run_role=dev excluded by default. Frozen m supplied to manifest generation; no retuning in arm. |
| M L17 | “Another common approach is to isolate the precise modification made in the proposed | CONFIG; A2.paired_returns | Exp12GridTest; SyntheticResultsTest | **CPU VERIFIED** | Design rationale, not an executable claim of generalizability or causal sufficiency. The paired intervention is implemented; no additional cross-algorithm comparison invented. |
| M L31 | Where actor architecture is held constant at depth=1 and width=128 | CONFIG actor_num_blocks/actor_hidden_dim; SACActor | Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | D1W128 production config; tests often override smaller actor. |
| M L33 | depth=2 & width=512 → 4.2 million parameters | GRID.EXP12_ARCHS; CONFIG; SACEncoder | Exp12GridTest.test_the_195_run_grid; parameter shape-count reproduction | **CPU VERIFIED** | D2W512 / D4W1024 / D4W1536 per Q. Actual count depends on observation/action dimensions; y supersedes D6. |
| M L34 | depth=4 & width=1024 → 33.6 million parameters | GRID.EXP12_ARCHS; CONFIG; SACEncoder | Exp12GridTest.test_the_195_run_grid; parameter shape-count reproduction | **CPU VERIFIED** | D2W512 / D4W1024 / D4W1536 per Q. Actual count depends on observation/action dimensions; y supersedes D6. |
| M L35 | depth=4 & width=1536 → 76 million parameters [changed from depth=6 & width=1536 | GRID.EXP12_ARCHS; CONFIG; SACEncoder | Exp12GridTest.test_the_195_run_grid; parameter shape-count reproduction | **CPU VERIFIED** | D2W512 / D4W1024 / D4W1536 per Q. Actual count depends on observation/action dimensions; y supersedes D6. |
| M L40 | Holding constant across experiments at 2 to isolate the effects of scaling → default used by the research papers | CONFIG.updates_per_interaction_step; TRAIN.train | Angle1ParityTest; Exp12GridTest | **CPU VERIFIED** | UTD 2 after replay warm-up, fused scan. Fractional counter is persisted even though production UTD is integral. |
| M L41 | Could possibly change in ablations but that is later | CONFIG (no ablation selected) | Exp12GridTest | **CPU VERIFIED** | Future ablation is not a current implementation requirement; no UTD change made. |
| M L45 | Learning rate: $0.0001$ | configs/agent/sac_simba.yaml; SAC._init_sac_networks | Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | Actor/critic/temp learning rates 1e-4; no separate schedule introduced. |
| M L46 | Batch Size: 256 | CONFIG buffer.sample_batch_size; PROBE.probe_config | Exp12GridTest; ProbePairingTest | **CPU VERIFIED** | Training and probe batch size 256; tiny fixtures reduce these explicitly. |
| M L47 | Discount Factor: Heuristic → TD-MPC2 heuristic (Hansen et al. 2023), as used in SimBa | CONFIG.eff_episode_len/gamma; ENV.create_envs | DiscountAndHorizonTest.test_gamma_per_suite | **CPU VERIFIED** | TD-MPC2 heuristic: .95 MyoSuite, .99 DMC/HB; action repeat included in effective horizon. |
| M L48 | Target Soft Update: 0.005 | SAC.update_target_network | InjectionInvariantsTest.test_target_injection_and_polyak; TwinCriticPathTest.test_polyak_update_covers_both_target_networks | **CPU VERIFIED** | tau .005; applies elementwise to full single/twin/injected target tree. |
| M L49 | Action Repeat: 2 | CONFIG env.action_repeat; ENV.RepeatAction | Exp12GridTest; DiscountAndHorizonTest | **CPU VERIFIED** | Two raw physics steps per interaction; early termination/truncation can shorten a repeat. |
| M L50 | Entropy: abs(A)/2 | SACAgent temp_target_entropy_coef; update_temperature | Exp12GridTest; twin actor/update tests | **CPU VERIFIED** | Target differential entropy is -action_dim/2; absolute action dimensionality in source is a dimensionality convention, not average sampled action. |
| M L51 | Temperature Learning Rate: 1e-4 | configs/agent/sac_simba.yaml; SAC._init_sac_networks | Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | Actor/critic/temp learning rates 1e-4; no separate schedule introduced. |
| M L52 | Weight Decay: 1e-2 | SAC._init_sac_networks; INJECT.injected_optimizer | InjectionInvariantsTest.test_optimizer_state_rules; TwinInjectionTest.test_training_keeps_frozen_heads_and_optimizer_rules | **CPU VERIFIED** | AdamW lr 1e-4 / weight decay .01 actor and critic; temperature weight decay 0 as inherited SimBa config. |
| M L53 | Optimizer: AdamW | SAC._init_sac_networks; INJECT.injected_optimizer | InjectionInvariantsTest.test_optimizer_state_rules; TwinInjectionTest.test_training_keeps_frozen_heads_and_optimizer_rules | **CPU VERIFIED** | AdamW lr 1e-4 / weight decay .01 actor and critic; temperature weight decay 0 as inherited SimBa config. |
| M L62 | dog-run - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L63 | dog-trot - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L64 | humanoid-run - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L65 | humanoid-walk - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L66 | humanoid-stand - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L67 | swimmer-swimmer15 - medium | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L68 | hopper-hop - medium | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; EnvRestoreAcrossProcessesTest | **CPU VERIFIED** | Each named DMC task generated; older DMC environment tests and tiny hopper pipeline. Exact-restore loops cover suite task list, not full-budget learning. |
| M L72 | myoHandKeyTurnFixed-v0 - hard | GRID.EXP12_ENVS; ENV.make_myosuite_env; MYOSUITE_TASKS_DICT | Exp12GridTest; MyoSuiteSeedingTest; DiscountAndHorizonTest | **CPU VERIFIED** | Aliases map to registered Fixed/Random task IDs. Seed passed at construction. Registered TimeLimit retained. |
| M L73 | myoHandPenTwirlFixed-v0 - hard | GRID.EXP12_ENVS; ENV.make_myosuite_env; MYOSUITE_TASKS_DICT | Exp12GridTest; MyoSuiteSeedingTest; DiscountAndHorizonTest | **CPU VERIFIED** | Aliases map to registered Fixed/Random task IDs. Seed passed at construction. Registered TimeLimit retained. |
| M L74 | myoHandPoseRandom-v0- close to hard | GRID.EXP12_ENVS; ENV.make_myosuite_env; MYOSUITE_TASKS_DICT | Exp12GridTest; MyoSuiteSeedingTest; DiscountAndHorizonTest | **CPU VERIFIED** | Aliases map to registered Fixed/Random task IDs. Seed passed at construction. Registered TimeLimit retained. |
| M L75 | myoHandReachFixed-v0 - medium | GRID.EXP12_ENVS; ENV.make_myosuite_env; MYOSUITE_TASKS_DICT | Exp12GridTest; MyoSuiteSeedingTest; DiscountAndHorizonTest | **CPU VERIFIED** | Aliases map to registered Fixed/Random task IDs. Seed passed at construction. Registered TimeLimit retained. |
| M L79 | h1-shelf-place - hard | Amendment (a); GRID.EXP12_ENVS | Exp12GridTest | **CPU VERIFIED** | Superseded: shelf-place removed, h1-run-v0 selected. Historical source list is not active design. |
| M L82 | Environment Steps: 500k for medium DMC and 2 million for Humanoid Bench, everything else is 1 million | GRID.EXP12_BUDGETS; CONFIG num_interaction_steps | Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | 500k raw DMC medium; 1M DMC hard/Myo; 2M HB. N=raw/(1 env × repeat 2). |
| M L85 | Train one of each of the 3 unique critics within 3 separate agents | GRID.add_exp12_grid; ENTRY.run | Exp12GridTest.test_manifests_never_share_a_checkpoint_dir | **CPU VERIFIED** | Three separate agents/output identities; no critic sharing across architecture cells. |
| M L86 | Before critic training, after the initial replay buffer has been populated, run the plasticity probe on the fresh, untrained critic to establish the run's initial plasticity score. Store the fresh critic par… | TRAIN.before_first_update; RunProbes.capture_fresh | Exp1LedgerEndToEndTest; SimbaRandomWarmupTest | **CPU VERIFIED** | At min replay 5000, before critic update; fresh params saved once and reloaded on resume. |
| M L87 | Periodically check for plasticity loss for all of them as introduced by Lyle et al. 2023. For each period check, we will perform 5 different rounds of evaluation with 1000 step-probe each. Periodically means… | PROBE.check_steps/run_probe/_fit; RunProbes.maybe_check | ProbePairingTest; Exp1LedgerEndToEndTest | **CPU VERIFIED** | 20 even kN/20 checks; 5 paired rounds ×1000 updates, 25,600 pool, production batch256/chunk2560. Fixtures shrink cost. |
| M L88 | The probe is performed on a copy of the critic so that the probe does not modify the training critic | PROBE._fit functional params; dedicated streams | ProbeDoesNotChangeTrainingTest; TwinProbeIsolationTest | **CPU VERIFIED** | No training optimizer/RNG/replay mutation; probes-on/off full state equality on tiny CPU runs. |
| M L89 | For each period check, use the same probe inputs and targets for the current critic and the stored fresh critic parameters so the comparison measures parameter plasticity change rather than changes in the pr… | PROBE.sample_pool/probe_round; twin.expand | ProbePairingTest.test_identical_inputs_targets_and_minibatches; TwinProbeTest | **CPU VERIFIED** | Shared pool/random function/minibatch order. Targets differ by prescribed critic-specific constant offset (c). |
| M L90 | Each probe round evaluates the critic's ability to fit a new target, following the procedure introduced by Lyle et al. 2023, adapted to the SAC critic setting | PROBE._base_targets/_fit | ProbeArithmeticTest; ProbePairingTest | **CPU VERIFIED** | Fresh random same-architecture scalar function; sin(1e5*f), critic own offset, fresh probe optimizer state. |
| M L91 | Take the IQM across the probes of each period check to get one plasticity score per agent for that period | PROBE.iqm/summarize | ProbeArithmeticTest.test_iqm_matches_rliable_definition | **CPU VERIFIED** | One IQM score per check; raw five observations retained. |
| M L92 | For each run, compute the loss of plasticity relative to the same run's fresh, untrained parameters using the plasticity-loss formulation from Lyle et al. 2023 → handles that every network can lose some plas… | PROBE.summarize; twin.combine | ProbeArithmeticTest.test_sign_convention_plasticity_loss_positive_when_current_is_worse; TwinProbeTest.test_combined_loss_is_the_mean_of_the_two_networks | **CPU VERIFIED** | P=b-final pool MSE; L=P_fresh-P_current. Twin L is mean of paired Q losses. |
| M L93 | Use percentile bootstrap confidence intervals to determine whether the observed plasticity loss is statistically distinguishable from zero at the 95% level → this is used as an operational fork trigger, not … | TRIGGER.bootstrap_interval/triggered | TriggerTest.test_known_loss_is_detected; test_edge_cases; test_statistic_is_iqm_not_mean | **CPU VERIFIED** | 10k paired-round IQM bootstrap, 95% percentile CI, strictly lower>0; not population inference. |
| M L94 | The bootstrap is performed on the paired probe-level plasticity-loss observations for that check | TRIGGER.bootstrap_interval/triggered | TriggerTest.test_known_loss_is_detected; test_edge_cases; test_statistic_is_iqm_not_mean | **CPU VERIFIED** | 10k paired-round IQM bootstrap, 95% percentile CI, strictly lower>0; not population inference. |
| M L96 | The first check where an individual run's measured plasticity loss is statistically distinguishable from zero is called $f^*_{run}$ and is used as the operational fork point in Experiment 2 | TRIGGER.f_star | TriggerTest.test_consecutive_checks_rule; test_f_star_first_eligible_check_only | **CPU VERIFIED** | Superseded literal first-crossing by (l): completion of first pair of consecutive positive checks, k=2..19. |
| M L97 | $f^*_{run}$ is an operational eligibility trigger and is separate from the architecture-level statistical comparison used to determine whether scaling produces excess plasticity loss | TRIGGER.f_star versus A1.primary_endpoint | TriggerTest; Exp1AnalysisTest | **CPU VERIFIED** | Operational per-run fork trigger is separate from architecture-level endpoint. Null runs remain Exp 1 observations. |
| M L98 | For the main Experiment 1 scientific comparison, compare the seed-level plasticity-loss trajectories between each scaled architecture and the default architecture | A1.trajectories/plot_trajectories/plot_every_seed | Exp1AnalysisTest.test_full_analysis_writes_outputs | **CPU VERIFIED** | Descriptive time axis is fraction of original budget; final endpoint predefined separately. |
| M L99 | Use the final scheduled checkpoint as the primary pre-specified endpoint for the overall scaling-related plasticity-loss comparison → difference = scaled plasticity loss − default plasticity loss | LEDGER.write_run final check; A1.primary_endpoint | Exp1AnalysisTest.test_known_effect_is_recovered_and_null_effect_is_not | **CPU VERIFIED** | Final scheduled check20; IQM(scaled)-IQM(default). Continued control training does not move endpoint. |
| M L101 | The full plasticity-loss trajectories are also shown descriptively with the time axis in fractions of the training budget | A1.trajectories/plot_trajectories/plot_every_seed | Exp1AnalysisTest.test_full_analysis_writes_outputs | **CPU VERIFIED** | Descriptive time axis is fraction of original budget; final endpoint predefined separately. |
| M L103 | If a run has not reached the operational plasticity-loss trigger by 95% of the nominal training budget, $f^*_{run}$ is null and the run is reported as a null result for Experiment 2 | TRIGGER.last_eligible_check/f_star; ENTRY | TriggerTest.test_f_star_first_eligible_check_only; ForkUnitTest.test_fork_plan | **CPU VERIFIED** | No eligible trigger through 95% -> null, no shortened arm; not an error or discarded Exp 1 run. |
| M L104 | The architecture-level result and the individual-run trigger are kept separate → a scaled architecture can show greater average plasticity loss than D2/W512 even when some individual runs never trigger | TRIGGER.f_star versus A1.primary_endpoint | TriggerTest; Exp1AnalysisTest | **CPU VERIFIED** | Operational per-run fork trigger is separate from architecture-level endpoint. Null runs remain Exp 1 observations. |
| M L108 | Every network is expected to lose some plasticity over training (Lyle et al. 2023) → the fresh same-architecture reference measures the change within each run, while the separate scaled-vs-default comparison… | PROBE fresh/current pairing; A1.primary_endpoint | ProbePairingTest; Exp1AnalysisTest | **CPU VERIFIED** | Documented construct limitation. Tests verify arithmetic/baselines, not that every network must empirically lose plasticity. |
| M L109 | The individual-run trigger is an operational rule rather than a population-level statistical test → it is used only to determine when a run is eligible for the Experiment 2 intervention | TRIGGER.f_star versus A1.primary_endpoint | TriggerTest; Exp1AnalysisTest | **CPU VERIFIED** | Operational per-run fork trigger is separate from architecture-level endpoint. Null runs remain Exp 1 observations. |
| M L111 | With few probe rounds and 5 seeds, bootstrap intervals can be wide or unstable → all seed-level results and probe distributions are retained and shown descriptively | LEDGER raw rounds; A1 every-seed plots; methodology limitations | LedgerTest; Exp1AnalysisTest | **CPU VERIFIED** | Five-seed/round uncertainty explicitly retained. Bootstrap coverage is not guaranteed by a passing synthetic test; twin canonical field caveat above. |
| M L112 | Individual runs can reach the operational trigger at different points in training → Experiment 2 therefore aligns the post-fork analysis by steps since fork rather than absolute training step | A2.paired_returns; FORK.post_fork_eval | SyntheticResultsTest.test_paired_difference_is_injected_minus_control_mean_return | **CPU VERIFIED** | Align by steps since each run fork, not absolute training step. |
| M L113 | Some runs may never reach the trigger within 95% of the nominal budget → this is a valid null result and means there is no eligible fork for that run | TRIGGER.last_eligible_check/f_star; ENTRY | TriggerTest.test_f_star_first_eligible_check_only; ForkUnitTest.test_fork_plan | **CPU VERIFIED** | No eligible trigger through 95% -> null, no shortened arm; not an error or discarded Exp 1 run. |
| M L114 | 5 seeds matches SimBa's ablations, but SimBa used 10 seeds for its main SAC comparisons → confidence intervals will be wide | LEDGER raw rounds; A1 every-seed plots; methodology limitations | LedgerTest; Exp1AnalysisTest | **CPU VERIFIED** | Five-seed/round uncertainty explicitly retained. Bootstrap coverage is not guaranteed by a passing synthetic test; twin canonical field caveat above. |
| M L122 | dog-run - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L123 | dog-trot - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L124 | humanoid-run - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L125 | humanoid-walk - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L126 | humanoid-stand - hard | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L127 | swimmer-swimmer15 - medium | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L128 | hopper-hop - medium | GRID.EXP12_ENVS; ENV.make_dmc_env | Exp12GridTest.test_the_195_run_grid; PipelinePerSuiteTest.test_dmc | **CPU VERIFIED** | Exp 2 inherits the same DMC tasks/budgets; only scaled eligible runs fork. |
| M L132 | myoHandKeyTurnFixed-v0 - hard | GRID.EXP12_ENVS; ENV.make_myosuite_env | Exp12GridTest; PipelinePerSuiteTest.test_myosuite | **CPU VERIFIED** | Exp 2 restores identical MyoSuite task configuration; registered-horizon caveats apply. |
| M L133 | myoHandPenTwirlFixed-v0 - hard | GRID.EXP12_ENVS; ENV.make_myosuite_env | Exp12GridTest; PipelinePerSuiteTest.test_myosuite | **CPU VERIFIED** | Exp 2 restores identical MyoSuite task configuration; registered-horizon caveats apply. |
| M L134 | myoHandPoseRandom-v0- close to hard | GRID.EXP12_ENVS; ENV.make_myosuite_env | Exp12GridTest; PipelinePerSuiteTest.test_myosuite | **CPU VERIFIED** | Exp 2 restores identical MyoSuite task configuration; registered-horizon caveats apply. |
| M L135 | myoHandReachFixed-v0 - medium | GRID.EXP12_ENVS; ENV.make_myosuite_env | Exp12GridTest; PipelinePerSuiteTest.test_myosuite | **CPU VERIFIED** | Exp 2 restores identical MyoSuite task configuration; registered-horizon caveats apply. |
| M L139 | h1-shelf-place - hard | Amendment (a); GRID.EXP12_ENVS | Exp12GridTest | **CPU VERIFIED** | Superseded: shelf-place removed, h1-run-v0 selected. Historical source list is not active design. |
| M L142 | Environment Steps: 500k for medium DMC and 2 million for Humanoid Bench, everything else is 1 million | GRID.EXP12_BUDGETS; CONFIG num_interaction_steps | Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | 500k raw DMC medium; 1M DMC hard/Myo; 2M HB. N=raw/(1 env × repeat 2). |
| M L146 | Probe a fresh critic | PC.run; amendment (q) | PositiveControlEndToEndTest.test_full_report | **CPU VERIFIED** | Steps1/2 both refer to check0 fresh critic under (q), replacing normally-trained earlier healthy reference. |
| M L147 | Probe a normally trained critic | PC.run; amendment (q) | PositiveControlEndToEndTest.test_full_report | **CPU VERIFIED** | Steps1/2 both refer to check0 fresh critic under (q), replacing normally-trained earlier healthy reference. |
| M L155 | The preliminary development run and all results used to validate the injection or select $m$ are excluded from the confirmatory main Experiment 1 and Experiment 2 analyses | ENTRY.run_identity; LEDGER.load; A2.load; PC.run | LedgerTest.test_round_trip_and_dev_excluded_by_default; Exp1AnalysisTest.test_dev_runs_do_not_change_the_confirmatory_result; SyntheticResultsTest | **CPU VERIFIED** | run_role=dev excluded by default. Frozen m supplied to manifest generation; no retuning in arm. |
| M L159 | Head = last m residual blocks + post-layer-norm + output layer | INJECT.head_blocks/split_params | InjectionInvariantsTest.test_head_boundary_per_label | **CPU VERIFIED** | Last m residual blocks + post-LN + output, not input projection; trunk path remains trainable. |
| M L160 | Since SimBa has no dedicated encoder, this head boundary is defined explicitly for the injection | INJECT.head_blocks/split_params | InjectionInvariantsTest.test_head_boundary_per_label | **CPU VERIFIED** | Last m residual blocks + post-LN + output, not input projection; trunk path remains trainable. |
| M L161 | Test m = last block only, half of the total residual blocks, and all residual blocks | INJECT.M_LABELS; PC.run | MSelectionArithmeticTest; PositiveControlEndToEndTest | **CPU VERIFIED** | last/half/all candidates evaluated on copies; depth4 =>1/2/4 blocks; all means all residual blocks, not input projection. |
| M L162 | For each m, apply the plasticity injection to the naturally degraded critic and run the plasticity probe | INJECT.M_LABELS; PC.run | MSelectionArithmeticTest; PositiveControlEndToEndTest | **CPU VERIFIED** | last/half/all candidates evaluated on copies; depth4 =>1/2/4 blocks; all means all residual blocks, not input projection. |
| M L163 | Select m based on the observed recovery toward the healthy reference, with the effect size/recovery reported for every candidate → if multiple candidates produce similar recovery, keep the smallest m | m_selection.select_m/evaluate | MSelectionArithmeticTest.test_select_smallest_m_within_tolerance; test_evaluate_chooses_m | **CPU VERIFIED** | Smallest within .10 best recovery, healthy L=0 per(q). Stop noise>=.10; candidate effects all reported. |
| M L164 | Record which m was chosen and the evidence (probe scores and effect sizes for each candidate) in a results file so the decision can be audited later | PC.run positive_control.json/positive_control_curves.npz | PositiveControlEndToEndTest.test_full_report | **CPU VERIFIED** | Candidate scores/recovery/noise/selection retained for audit, not just selected label. |
| M L165 | Once selected, m is frozen and is not retuned per architecture, environment, or main-run seed | GRID --injection-m; arm config guard | Exp12GridTest.test_arm_jobs_only_for_completed_forks_grouped_by_device; ForkEndToEndTest.test_arm_refuses_bad_inputs | **CPU VERIFIED** | m must be supplied/frozen globally; code does not pick a new per-run m. Operational freeze before confirmatory runs remains lead responsibility. |
| M L169 | Record whether critic pathology precedes actor-side degradation as a descriptive result → measured visually and not deep empirically | A2.shared_time_axis | SyntheticResultsTest.test_tables_and_files; ForkEndToEndTest.test_exp2_analysis_runs_on_the_real_outputs | **CPU VERIFIED** | Visual timing only, marked f-star; no automatic lag/onset/causal proof statistic introduced. |
| M L170 | Create a scaled critic architecture agent | ENTRY.run_identity; CONFIG.fork.architectures | Exp12GridTest; ForkEndToEndTest | **CPU VERIFIED** | Only D4W1024/D4W1536 eligible for production fork; D2W512 default remains comparator. |
| M L171 | When an individual run reaches $f^*_{run}$, fork the complete training state into two identical agents. The fork is performed once per run | FORK.write_fork; TRAIN.save/restore; ENTRY.enter_control | ResumeExactnessTest; KillMatrixTest; ForkEndToEndTest | **CPU VERIFIED** | Exact scheduled step; one immutable complete-state fork; parent rebuilt/restored as control. Approved original/control comparison enforced before FORK_READY and on ready-fork/arm entry; two regressions and a new mutation check. |
| M L172 | If an individual run has not reached the operational plasticity-loss trigger by 95% of the nominal training budget, no fork is performed → reported as a normal null result | TRIGGER.last_eligible_check/f_star; ENTRY | TriggerTest.test_f_star_first_eligible_check_only; ForkUnitTest.test_fork_plan | **CPU VERIFIED** | No eligible trigger through 95% -> null, no shortened arm; not an error or discarded Exp 1 run. |
| M L173 | After the fork, continue both agents for a fixed post-fork horizon of 25% of the nominal training budget for that environment, measured from the fork point → ensures every run has the same amount of time to … | FORK.fork_plan/eval_due; ENTRY arm end | ForkUnitTest.test_fork_plan; ForkEndToEndTest.test_arm_records | **CPU VERIFIED** | Both evaluated for .25N from fork, max1.2N, control trains to max(N,fork+.25N). |
| M L174 | The maximum total training budget is 120% of the nominal training budget → a fork must therefore occur by 95% to receive the full 25% post-fork horizon | FORK.fork_plan/eval_due; ENTRY arm end | ForkUnitTest.test_fork_plan; ForkEndToEndTest.test_arm_records | **CPU VERIFIED** | Both evaluated for .25N from fork, max1.2N, control trains to max(N,fork+.25N). |
| M L175 | To isolate whether the critic pathology is propagating to the actor, perform plasticity injection (Nikishin et al. 2023) to the critic of one of the agents and not the other | TRAIN.inject; ENTRY.exp2_arm | ForkEndToEndTest.test_injected_arm_frozen_head_unchanged_and_new_head_trained; TwinForkEndToEndTest | **CPU VERIFIED** | Injected branch only; control same restored pre-intervention state. Downstream empirical effect is not forced. |
| M L176 | The two branches begin from the same complete training state at the fork → model parameters, optimizer state, replay buffer, RNG state, reproducible environment state, and relevant training counters are dupl… | FORK.write_fork; TRAIN.save/restore; ENTRY.enter_control | ResumeExactnessTest; KillMatrixTest; ForkEndToEndTest | **CPU VERIFIED** | Exact scheduled step; one immutable complete-state fork; parent rebuilt/restored as control. Approved original/control comparison enforced before FORK_READY and on ready-fork/arm entry; two regressions and a new mutation check. |
| M L177 | Plasticity injection: freeze the critic's head, and add two newly initialized copies of it, one trainable and one frozen. The head's output = frozen old head + trainable new head − frozen copy of the new hea… | INJECT.InjectedSACCritic/inject | InjectionInvariantsTest.test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance; test_parameter_counts; test_gradients_reach_earlier_blocks_through_the_frozen_head | **CPU VERIFIED** | Q=old+(new-copy); new/copy identical; frozen parameter gradients only; trunk input gradient flows; trainable count same, total +2heads. |
| M L178 | SimBa has no dedicated encoder, so the head is defined by choice → head = [last m residual blocks + post-layer-norm + output layer] | INJECT.head_blocks/split_params | InjectionInvariantsTest.test_head_boundary_per_label | **CPU VERIFIED** | Last m residual blocks + post-LN + output, not input projection; trunk path remains trainable. |
| M L179 | The target critic is also injected using the same construction, and the normal SAC Polyak target update continues from the post-injection state | INJECT.inject/inject_twin; SAC.update_target_network | InjectionInvariantsTest.test_target_injection_and_polyak; TwinInjectionTest | **CPU VERIFIED** | Lagged target old/trunk preserved; new/copy matched online; entire target tree Polyak updated afterward. |
| M L180 | New trainable injection parameters continue with newly created optimizer state while existing trainable parameters keep their existing AdamW state; frozen parameters are excluded from the optimizer and do no… | INJECT.injected_optimizer/_carry_trunk_state | InjectionInvariantsTest.test_optimizer_state_rules; TwinInjectionTest.test_training_keeps_frozen_heads_and_optimizer_rules | **CPU VERIFIED** | Trunk Adam moments/count retained; new head own fresh count/moments; frozen old/copy zero updates/no decay. |
| M L181 | Inject the single Q critic used by this SAC implementation [DMC and MyoSuite; on HumanoidBench both | SACClippedDoubleCritic; CONFIG env.episodic; INJECT.inject_twin | TwinCriticDefectTest; TwinInjectionTest | **CPU VERIFIED** | Current amendment(z): one Q DMC/Myo; two Q HB, both targets injected. |
| M L183 | Use a fixed panel of 256 state-action pairs sampled once from the replay buffer at the fork and fixed thereafter for the injection correctness checks | FORK.sample_panel/write_fork | ForkEndToEndTest.test_fork_plan_and_files | **CPU VERIFIED** | 256 replay (state,action) pairs with replacement from dedicated stream; normalized once and saved, fixed thereafter. |
| M L184 | All injected/control forks remain in the primary analysis regardless of whether the immediate plasticity-rescue check succeeds | A2.load/paired_returns/run_analysis | SyntheticResultsTest.test_secondary_is_the_check2_success_subset | **CPU VERIFIED** | All complete forks in primary regardless Check2; success-only secondary explicitly labeled. Completeness validation weakness separately listed. |
| M L192 | Run the plasticity probe on a copy of each critic → the injected critic should show improved plasticity relative to the control | exp2_arm.check2; PROBE.run_probe_networks | ForkEndToEndTest.test_check2_written_and_paired_with_the_exp1_check; TwinForkEndToEndTest.test_check2_uses_the_mean_and_records_both_networks | **CPU VERIFIED** | Independent copies, same fork-check pool/random targets/minibatches; aggregate P_inj-P_ctrl and per-Q scores. |
| M L193 | This is a validation of the intended plasticity-rescue effect, not something to force through implementation changes | exp2_arm.check2; TRAIN; A2 both-arm plots | SyntheticResultsTest; ForkEndToEndTest | **CPU VERIFIED** | No rescue/divergence/redegradation is forced. Empirical outcome interpreted by lead; tests establish machinery only. |
| M L194 | Whether Check 2 succeeds or fails does not determine inclusion in the primary analysis → a success-only subset may be reported separately as a secondary analysis | A2.load/paired_returns/run_analysis | SyntheticResultsTest.test_secondary_is_the_check2_success_subset | **CPU VERIFIED** | All complete forks in primary regardless Check2; success-only secondary explicitly labeled. Completeness validation weakness separately listed. |
| M L198 | Q-values, actor updates, and returns are expected to diverge → this is the result being measured, not something to force | exp2_arm.check2; TRAIN; A2 both-arm plots | SyntheticResultsTest; ForkEndToEndTest | **CPU VERIFIED** | No rescue/divergence/redegradation is forced. Empirical outcome interpreted by lead; tests establish machinery only. |
| M L204 | We will sustain the plasticity loss checks within both critics after the fork → verifies whether the injection changes the critic's plasticity trajectory and checks that the critic itself is not silently dri… | RunProbes.extend_to; both ENTRY after_step hooks | ForkEndToEndTest.test_arm_records; TwinForkEndToEndTest | **CPU VERIFIED** | Same N/20 grid continues into post-fork region; fixed fresh reference retained; no probe retuning. |
| M L205 | Observe the performance of the two agents thereafter, and measure for the development of actor-side degradation through observing overall agent performance and recording the following actor-side pathology/op… | DIAG + SAC updates; TRAIN metrics; A2 plots | KnownAnswerTest; ReferenceBatchTest; DiagnosticsPrecisionTest; TrainingRunTest | **CPU VERIFIED** | Exact equations/state sets/cadences and limits in actor table. CPU tests are tiny; CUDA tolerances unresolved. |
| M L206 | Policy churn → measures rapid changes in actor behavior and policy updates (Schaul et al. 2022; Tang & Berseth 2024) | DIAG + SAC updates; TRAIN metrics; A2 plots | KnownAnswerTest; ReferenceBatchTest; DiagnosticsPrecisionTest; TrainingRunTest | **CPU VERIFIED** | Exact equations/state sets/cadences and limits in actor table. CPU tests are tiny; CUDA tolerances unresolved. |
| M L207 | Actor gradient norm → measures changes in the magnitude and stability of the actor optimization signal (Bjorck et al. 2021) | DIAG + SAC updates; TRAIN metrics; A2 plots | KnownAnswerTest; ReferenceBatchTest; DiagnosticsPrecisionTest; TrainingRunTest | **CPU VERIFIED** | Exact equations/state sets/cadences and limits in actor table. CPU tests are tiny; CUDA tolerances unresolved. |
| M L208 | Action saturation / average absolute action → measures whether the actor's outputs become abnormally concentrated near the action bounds (Bjorck et al. 2021) | DIAG + SAC updates; TRAIN metrics; A2 plots | KnownAnswerTest; ReferenceBatchTest; DiagnosticsPrecisionTest; TrainingRunTest | **CPU VERIFIED** | Exact equations/state sets/cadences and limits in actor table. CPU tests are tiny; CUDA tolerances unresolved. |
| M L209 | The primary actor outcome is the return/performance trajectory; the actor-side pathology/optimization diagnostics are supporting diagnostics rather than separate primary endpoints | A2.paired_returns/paired_bands; methodology limitations | SyntheticResultsTest; BandTest | **CPU VERIFIED** | Raw-return paired intervention is primary; diagnostics supporting, not independent proof. Exploration confounding remains a documented limitation, not fixed by instrumentation. |
| M L210 | Actor_grad_cosine is retained as a mechanism diagnostic for Experiment 3 rather than being treated as one of the primary Experiment 2 actor-side diagnostics | SAC.compute_actor_gradient_cosine; A2.DIAGNOSTICS | ActorGradCosineCadenceTest; TwinCriticDefectTest.test_actor_grad_cosine_uses_min_of_the_two_networks | **CPU VERIFIED** | Retained CSV; not an Exp2 primary endpoint and not plotted by A2. |
| M L214 | Plasticity, exploration, and performance interact in RL (Nikishin et al. 2023) → a change in the injected agent's reward after the fork could partly come through changes in the actor/environment interaction,… | A2.paired_returns/paired_bands; methodology limitations | SyntheticResultsTest; BandTest | **CPU VERIFIED** | Raw-return paired intervention is primary; diagnostics supporting, not independent proof. Exploration confounding remains a documented limitation, not fixed by instrumentation. |
| M L215 | The injection intervention tests whether rescuing critic plasticity after degradation changes the downstream actor trajectory, but it does not necessarily remove the original reason the scaled critic became … | exp2_arm.check2; TRAIN; A2 both-arm plots | SyntheticResultsTest; ForkEndToEndTest | **CPU VERIFIED** | No rescue/divergence/redegradation is forced. Empirical outcome interpreted by lead; tests establish machinery only. |
| M L216 | With 5 seeds, a visible gap in the injected vs. control trajectories can still be affected by seed variation → plot percentile bootstrap confidence bands over paired seed-level differences | A2.bootstrap_band/paired_bands | BandTest; SyntheticResultsTest.test_band_per_environment_over_complete_confirmatory_forks | **CPU VERIFIED** | IQM over seed-level paired returns; percentile bands10k; same resampled seed indices across time. |
| M L217 | Individual runs fork at different absolute training steps → all post-fork comparisons are therefore aligned by steps since fork, with the same 25% post-fork horizon | A2.paired_returns; FORK.post_fork_eval | SyntheticResultsTest.test_paired_difference_is_injected_minus_control_mean_return | **CPU VERIFIED** | Align by steps since each run fork, not absolute training step. |
| M L218 | If degradation occurs after 95% of the nominal training budget, the run is reported as a null result rather than receiving a shorter post-fork window | TRIGGER.last_eligible_check/f_star; ENTRY | TriggerTest.test_f_star_first_eligible_check_only; ForkUnitTest.test_fork_plan | **CPU VERIFIED** | No eligible trigger through 95% -> null, no shortened arm; not an error or discarded Exp 1 run. |
| M L219 | The 25% post-fork horizon may not capture effects that emerge only after a longer period of continued training → conclusions are limited to the fixed post-fork observation window | FORK.fork_plan; A2 evaluation grid | ForkUnitTest.test_fork_plan | **CPU VERIFIED** | Only .25N post-fork horizon; no claim about later emergence; no extension selected. |
| M L220 | The actor-side pathology/optimization diagnostics are supporting measurements and are not individually treated as proof of propagation; the causal conclusion comes from the paired intervention and the downst… | A2.paired_returns/paired_bands; methodology limitations | SyntheticResultsTest; BandTest | **CPU VERIFIED** | Raw-return paired intervention is primary; diagnostics supporting, not independent proof. Exploration confounding remains a documented limitation, not fixed by instrumentation. |
| M L221 | Policy churn and action saturation require definitions appropriate to continuous stochastic SAC rather than being copied directly from discrete-action or different actor-critic settings; the implementation s… | DIAG + SAC updates; TRAIN metrics; A2 plots | KnownAnswerTest; ReferenceBatchTest; DiagnosticsPrecisionTest; TrainingRunTest | **CPU VERIFIED** | Exact equations/state sets/cadences and limits in actor table. CPU tests are tiny; CUDA tolerances unresolved. |
| M L222 | Development/calibration runs are excluded from the confirmatory analysis → the selected injection configuration is frozen before the main Experiment 2 runs | ENTRY.run_identity; LEDGER.load; A2.load; PC.run | LedgerTest.test_round_trip_and_dev_excluded_by_default; Exp1AnalysisTest.test_dev_runs_do_not_change_the_confirmatory_result; SyntheticResultsTest | **CPU VERIFIED** | run_role=dev excluded by default. Frozen m supplied to manifest generation; no retuning in arm. |
| M L246 | (c) Probe target offset. In each probe round both the current critic and the | PROBE._base_targets/_fit/summarize; PC shared-offset mode | ProbePairingTest; SharedOffsetTest | **CPU VERIFIED** | Own mean offset; P=b-MSE; L=P_fresh-P_current; identical base sine targets, constant-shift limitation; 5×1000 smaller than cited10×2000; dev-only shared-offset sign report. |
| M L253 | (i) the current and fresh critics' targets differ by a constant shift; | PROBE._base_targets/_fit/summarize; PC shared-offset mode | ProbePairingTest; SharedOffsetTest | **CPU VERIFIED** | Own mean offset; P=b-MSE; L=P_fresh-P_current; identical base sine targets, constant-shift limitation; 5×1000 smaller than cited10×2000; dev-only shared-offset sign report. |
| M L254 | (ii) the probe budget (5 rounds x 1000 steps) is smaller than Lyle et al.'s | PROBE._base_targets/_fit/summarize; PC shared-offset mode | ProbePairingTest; SharedOffsetTest | **CPU VERIFIED** | Own mean offset; P=b-MSE; L=P_fresh-P_current; identical base sine targets, constant-shift limitation; 5×1000 smaller than cited10×2000; dev-only shared-offset sign report. |
| M L260 | (d) m-selection rule, fixed before any result is seen. | m_selection.select_m/evaluate; amendment(q) | MSelectionArithmeticTest | **CPU VERIFIED** | Smallest recovery within .10 of best. Healthy denominator is L_trigger under(q); noise>=.10 stops, not just >.10. |
| M L273 | (f) [SUPERSEDED by (q), 2026-10-04] Healthy reference. The "normally trained | Amendment(q); PC.run | MSelectionArithmeticTest.test_recovery_toward_the_fresh_critic | **CPU VERIFIED** | Superseded: earlier normally-trained healthy reference is not used; check0 fresh has L=0. |
| M L277 | (g) No normalization in Experiment 2. No returns are normalized anywhere, and no | A2.identity/paired_returns/run_analysis; FORK.post_fork_eval | SyntheticResultsTest.test_normalize_hook_is_applied_per_environment; test_paired_difference_is_injected_minus_control_mean_return | **CPU VERIFIED** | Graphs and raw paired returns; identity normalize hook default; raw success/return episodes retained; no scalar performance summary. |
| M L282 | raw per-episode returns (and per-episode success where the environment | A2.identity/paired_returns/run_analysis; FORK.post_fork_eval | SyntheticResultsTest.test_normalize_hook_is_applied_per_environment; test_paired_difference_is_injected_minus_control_mean_return | **CPU VERIFIED** | Graphs and raw paired returns; identity normalize hook default; raw success/return episodes retained; no scalar performance summary. |
| M L284 | analysis scripts take a normalize(env, values) function that defaults to | A2.identity/paired_returns/run_analysis; FORK.post_fork_eval | SyntheticResultsTest.test_normalize_hook_is_applied_per_environment; test_paired_difference_is_injected_minus_control_mean_return | **CPU VERIFIED** | Graphs and raw paired returns; identity normalize hook default; raw success/return episodes retained; no scalar performance summary. |
| M L286 | one metadata file per suite records benchmark-defined constants as data only. | configs/suite_metadata/{dmc,myosuite,humanoid_bench}.yaml | Source inspection; normalization-hook test | **CPU VERIFIED** | Benchmark constants stored as data only; not consumed to normalize returns in A2. |
| M L290 | (h) Initial random actions. As in SimBa's released code, Experiments 1 and 2 | TRAIN.train random action branch | SimbaRandomWarmupTest.test_random_until_min_length_then_policy | **CPU VERIFIED** | Uniform actions through replay warmup5000; obs RMS still sees observations; fresh probe before first update. Sample-action key consumption remains compatible with existing path. |
| M L296 | (i) MyoSuite horizon and discount. Matching SimBa exactly, every MyoSuite task | CONFIG env/myosuite_simba.yaml; ENV.TimeLimit | DiscountAndHorizonTest.test_gamma_per_suite; test_keyturn_is_truncated_at_100_raw_steps | **CPU VERIFIED** | Myo outer100 raw =>gamma.95; DMC/HB1000=>.99; KeyTurn original200 truncated100. |
| M L302 | (j) HumanoidBench single Q. SimBa's released code and paper use clipped | Amendment(z); SACClippedDoubleCritic | TwinCriticDefectTest; Exp12GridTest | **CPU VERIFIED** | Superseded single-Q HB limitation; HB now twin, DMC/Myo single. |
| M L312 | (k) HumanoidBench limitations (h1-reach-v0, h1-run-v0): | ENV._import_humanoid_bench; requirements pins | Source/version inspection; HB skips documented | **CPU VERIFIED** | HB physics pinned MuJoCo3.6.0 rather than3.1.6; not bit-comparable to published SimBa HB numbers. |
| M L313 | Physics runs on mujoco 3.6.0, the repo's pinned version shared with DMC and | ENV._import_humanoid_bench; requirements pins | Source/version inspection; HB skips documented | **CPU VERIFIED** | HB physics pinned MuJoCo3.6.0 rather than3.1.6; not bit-comparable to published SimBa HB numbers. |
| M L316 | SimBa's released code lists the with-hands `h1hand-*` tasks, while its | GRID IDs; pinned HB robots/tasks/XML | Parameter shape/count reproduction | **CPU VERIFIED** | No-hands H1 nu19 measured from XML; source-table discrepancy is a documented limitation, not task substitution. |
| M L324 | (l) Trigger: two consecutive firing checks. A check fires when the lower bound | TRIGGER.f_star; CONFIG.trigger | TriggerTest.test_consecutive_checks_rule; test_shipped_config_is_the_current_rule | **CPU VERIFIED** | Two completed consecutive positive checks, earliest2/latest19; gaps break sequence; no first-crossing fallback. |
| M L342 | (n) Check 2 is reported as the paired difference P(injected) - P(control) on | exp2_arm.check2 | ForkEndToEndTest; TwinForkEndToEndTest | **CPU VERIFIED** | Paired P_inj-P_control IQM CI lower>0; report-only, five-round bootstrap coverage caveat retained. |
| M L352 | (o) Positive control. [SUPERSEDED by (q), 2026-10-04] | Amendment(q); m_selection.evaluate | MSelectionArithmeticTest | **CPU VERIFIED** | Superseded healthy pooling/noise denominator rule. Current fresh L=0, only real-settings degraded+3injected series enter pooled noise. |
| M L353 | Healthy reference: the IQM of the per-round L pooled over all checks | Amendment(q); m_selection.evaluate | MSelectionArithmeticTest | **CPU VERIFIED** | Superseded healthy pooling/noise denominator rule. Current fresh L=0, only real-settings degraded+3injected series enter pooled noise. |
| M L356 | Probe noise: the pooled SD of the per-round L over every probe evaluation | Amendment(q); m_selection.evaluate | MSelectionArithmeticTest | **CPU VERIFIED** | Superseded healthy pooling/noise denominator rule. Current fresh L=0, only real-settings degraded+3injected series enter pooled noise. |
| M L359 | Stop and consult (exit 3) if L_trigger - L_healthy is non-positive, if it | Amendment(q); m_selection.evaluate | MSelectionArithmeticTest | **CPU VERIFIED** | Superseded healthy pooling/noise denominator rule. Current fresh L=0, only real-settings degraded+3injected series enter pooled noise. |
| M L369 | (q) Positive control: healthy reference, noise and stop rule (replaces (f), (o) | PC.run; m_selection.recovery/pooled_sd/evaluate | MSelectionArithmeticTest noise/boundary tests; PositiveControlEndToEndTest | **CPU VERIFIED** | Lhealthy=0; recovery=(Ltrigger-Lafter)/Ltrigger; sample pooled within-series SD over degraded+3 candidates /Ltrigger; Ltrigger<=0 or noise>=.10 exit3; shared-offset repeats excluded. |
| M L382 | (r) Experiment 2 bands. The per-environment bands of the paired return | A2.STATISTICS/bootstrap_band | BandTest.test_iqm_is_the_default_and_unknown_statistics_are_refused; test_resampled_seed_sets_are_shared_across_points | **CPU VERIFIED** | Default IQM; all seed lines visible; --statistic mean exists as an explicit override, not the confirmatory default. |
| M L386 | (s) Time limits and truncation, matching SimBa's released code | ENV._wrap; TRAIN final_observation; SAC.update_critic | DiscountAndHorizonTest truncation/PenTwirl tests | **CPU VERIFIED** | Registered inner horizons retained; outer Myo100 before repeat; only terminated masks bootstrap; final terminal/truncated observation stored, not reset observation. |
| M L388 | (a) MyoSuite builds the registered task, keeping its own registered time | ENV._wrap; TRAIN final_observation; SAC.update_critic | DiscountAndHorizonTest truncation/PenTwirl tests | **CPU VERIFIED** | Registered inner horizons retained; outer Myo100 before repeat; only terminated masks bootstrap; final terminal/truncated observation stored, not reset observation. |
| M L399 | (b) Time-limit truncation is bootstrapped and termination is not. The | ENV._wrap; TRAIN final_observation; SAC.update_critic | DiscountAndHorizonTest truncation/PenTwirl tests | **CPU VERIFIED** | Registered inner horizons retained; outer Myo100 before repeat; only terminated masks bootstrap; final terminal/truncated observation stored, not reset observation. |
| M L412 | (t) Checkpoints. One routine save per probe check (every N/20 interaction | state.new_state_dir/commit_state_dir; GRID checkpoint interval | KillMatrixTest; Exp12GridTest.test_every_job_composes_with_the_methodology_budget | **CPU VERIFIED** | Unique save directory; atomic LATEST publication then old routine deletion; latest + fork + fresh retained. Save crash keeps previous pointer. |
| M L431 | No other mixed precision is used (no bf16 or fp16). Everything that is not | CONFIG mixed_precision:false; SAC float32 | Config composition + parameter shape tracing | **CPU VERIFIED** | No fp16/bf16, x64 default disabled; source dtype guarantee does not require replay normalization host statistics to use float32. |
| M L433 | The one exception is Check 1 (and the A0 error measurement), which | FORK.panel_q_and_grad; A0 matmul_precision_report | ForkUnitTest; DiagnosticsPrecisionTest; TwinCheck1Test (red eager-reference comparison) | **CPU VERIFIED** | Local highest for Check1/A0, plus later(x) additional diagnostic exceptions; no alteration of64eps/exact boundary. |
| M L438 | The earlier rule that Check 1 stops an arm when TF32 is detected is | ENTRY.record_metadata; PREC.runtime_info | Exp12GridTest; TwinForkEndToEndTest metadata test | **CPU VERIFIED** | No stop merely for configured TF32; device/runtime/precision recorded. Check1 JSON itself records highest; runtime details in metadata. |
| M L444 | (w) Fresh-critic range check (A2 and B2): acceptance rule replaced. | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L445 | Original rule (pre-specified 2026-10-04, kept here for the record): at | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L448 | Original result (Block A, job 22667743, A100-SXM4-40GB, dog-run): FAILED | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L451 | Why it is replaced: | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L452 | Lyle et al. (2023) chose the probe budget so that networks from random | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L455 | The rule and its fallback ladder (smaller pool, then more steps) assumed | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L457 | The replacement was decided without any degradation data. | Methodology(w); exp12_reports.range_criterion | RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule | **CPU VERIFIED** | Historical rule/result/rationale retained. Test uses recorded BlockA numbers; no fresh CUDA reproduction claimed. |
| M L461 | The probe settings are unchanged (pool, steps, rounds, offsets). | CONFIG.probe; PROBE | ProbePairingTest; source config inspection | **CPU VERIFIED** | No pool/round/update/offset retuning when acceptance criterion changed. |
| M L466 | The spread of the per-round final loss is reported per size and pool. L = | scripts/probe_fresh_checks.py range JSON | RangeCriterionTest; source inspection | **CPU VERIFIED** | Per-round final-loss spread per pool/size emitted; constant offsets leave target variance unchanged mathematically; finite-precision measurements remain logged. |
| M L472 | (x) Precision of the actor diagnostics. In every Experiment 1 and 2 job, the | SAC._sac_update diagnostics_precision; PREC | DiagnosticsPrecisionTest; KnownAnswerTest; TrainingRunTest; Angle1ParityTest | **CPU VERIFIED** | Only diagnostic actor forward passes highest; training/action/probe/eval TF32 on GPU; Angle1 kwargs absent preserves previous precision. Near-zero update resolution motivates exception; no new threshold selected. |
| M L477 | Everything else keeps amendment (v)'s TF32: training updates, action | SAC._sac_update diagnostics_precision; PREC | DiagnosticsPrecisionTest; KnownAnswerTest; TrainingRunTest; Angle1ParityTest | **CPU VERIFIED** | Only diagnostic actor forward passes highest; training/action/probe/eval TF32 on GPU; Angle1 kwargs absent preserves previous precision. Near-zero update resolution motivates exception; no new threshold selected. |
| M L479 | Training is unchanged. The Angle 1 path, which has no diagnostics, is | SAC._sac_update diagnostics_precision; PREC | DiagnosticsPrecisionTest; KnownAnswerTest; TrainingRunTest; Angle1ParityTest | **CPU VERIFIED** | Only diagnostic actor forward passes highest; training/action/probe/eval TF32 on GPU; Angle1 kwargs absent preserves previous precision. Near-zero update resolution motivates exception; no new threshold selected. |
| M L481 | Reason: a per-update parameter change is often below TF32's resolution, so | SAC._sac_update diagnostics_precision; PREC | DiagnosticsPrecisionTest; KnownAnswerTest; TrainingRunTest; Angle1ParityTest | **CPU VERIFIED** | Only diagnostic actor forward passes highest; training/action/probe/eval TF32 on GPU; Angle1 kwargs absent preserves previous precision. Near-zero update resolution motivates exception; no new threshold selected. |
| M L486 | Checked by `DiagnosticsPrecisionTest`: under the TF32 run setting the | SAC._sac_update diagnostics_precision; PREC | DiagnosticsPrecisionTest; KnownAnswerTest; TrainingRunTest; Angle1ParityTest | **CPU VERIFIED** | Only diagnostic actor forward passes highest; training/action/probe/eval TF32 on GPU; Angle1 kwargs absent preserves previous precision. Near-zero update resolution motivates exception; no new threshold selected. |
| M L493 | (y) Critic grid. The three critic sizes are D2W512, D4W1024 and D4W1536 | GRID.EXP12_ARCHS; CONFIG; INJECT.head_blocks; PC | Exp12GridTest; InjectionInvariantsTest; actual Flax parameter shape counts | **CPU VERIFIED** | D4W1536 replaces D6 in active settings; old measured records retained. Depth4 m1/2/4. Parameter ratios depend on task dimensions; no architecture change in audit. |
| M L500 | Reason: the largest critic was too large relative to the actor, and the | GRID.EXP12_ARCHS; CONFIG; INJECT.head_blocks; PC | Exp12GridTest; InjectionInvariantsTest; actual Flax parameter shape counts | **CPU VERIFIED** | D4W1536 replaces D6 in active settings; old measured records retained. Depth4 m1/2/4. Parameter ratios depend on task dimensions; no architecture change in audit. |
| M L502 | Parameters per Q network (dog-run, 223 observation and 38 action | GRID.EXP12_ARCHS; CONFIG; INJECT.head_blocks; PC | Exp12GridTest; InjectionInvariantsTest; actual Flax parameter shape counts | **CPU VERIFIED** | D4W1536 replaces D6 in active settings; old measured records retained. Depth4 m1/2/4. Parameter ratios depend on task dimensions; no architecture change in audit. |
| M L506 | m candidates for depth 4: last = 1 block, half = 2 blocks, all = 4 blocks | GRID.EXP12_ARCHS; CONFIG; INJECT.head_blocks; PC | Exp12GridTest; InjectionInvariantsTest; actual Flax parameter shape counts | **CPU VERIFIED** | D4W1536 replaces D6 in active settings; old measured records retained. Depth4 m1/2/4. Parameter ratios depend on task dimensions; no architecture change in audit. |
| M L508 | Earlier records that name D6W1536 (Block A's measurements, the original | GRID.EXP12_ARCHS; CONFIG; INJECT.head_blocks; PC | Exp12GridTest; InjectionInvariantsTest; actual Flax parameter shape counts | **CPU VERIFIED** | D4W1536 replaces D6 in active settings; old measured records retained. Depth4 m1/2/4. Parameter ratios depend on task dimensions; no architecture change in audit. |
| M L515 | (z) Critic per suite, following SimBa (replaces (j)). DMC and MyoSuite keep the | CONFIG env.episodic -> agent.critic_use_cdq; SAC | TwinCriticDefectTest; TwinCriticPathTest actor/target tests | **CPU VERIFIED** | HB two independent networks; per-Q sizes; actor min; shared target minimum; critic loss sums two MSEs. |
| M L520 | Critic sizes (D2W512, D4W1024, D4W1536) are per Q network; a twin critic is | CONFIG env.episodic -> agent.critic_use_cdq; SAC | TwinCriticDefectTest; TwinCriticPathTest actor/target tests | **CPU VERIFIED** | HB two independent networks; per-Q sizes; actor min; shared target minimum; critic loss sums two MSEs. |
| M L528 | Injection: the same head (the last m residual blocks, the post-LayerNorm | INJECT.inject_twin/InjectedClippedDoubleCritic | TwinInjectionTest three tests; TwinForkEndToEndTest.test_injected_arm_both_networks | **CPU VERIFIED** | Independent splitkeys, same m in both online/target heads; optimizer tree stacked elementwise. |
| M L537 | Logged twin metrics: td_error_var is the mean of the two networks' | SAC.update_critic/compute_actor_gradient_cosine; networks.metrics | TwinCriticDefectTest TDvar/cosine; TwinCriticPathTest.test_structural_metrics_are_per_network | **CPU VERIFIED** | TDvar average of withinQ variances (not pooled variance); perQ columns; cosine min; parameter/weight/gradient norms jointL2. |
| M L542 | Limitation: with a small fixed actor, twin critics (the minimum) may add | Methodology limitation; A1/A2 suites | Source + parameter count inspection; no causal estimator-bias test | **CPU VERIFIED** | Different single/twin estimator is documented. Possible min underestimation is a limitation, not a demonstrated empirical outcome in this audit. |
| M L546 | Implementation note: on the pinned jax 0.4.34 / flax 0.8.4 the twin critic | SACClippedDoubleCritic.__call__ | TwinCriticDefectTest.test_twin_critic_initialises_with_two_independent_networks | **CPU VERIFIED** | Pinned Flax/JAX vmap input broadcasting; same per-network math/leading parameter axis; break restores pre-fix failure. |
| M L549 | Parameter-count limitation (verified 2026-10-06; separate task ratios approved | SACActor/SACCritic/SACClippedDoubleCritic init via jax.eval_shape; pinned HB task source and XML | CPU shape/count reproduction; lead approved both task ratios | **CPU VERIFIED** | Run obs51/act19, actor143782: D4W1536 perQ75654145, twin151308290 (526.17/1052.35); D4W1024 twin67317762 (468.19). Reach obs57/act19, actor144550: 75663361/151326722 (523.44/1046.88); D4W1024 twin67330050 (465.79). Online before injection; no targets/optimizer. Limitation, not causal evidence. |

Traceability inventory: 203 individually mapped source items. Each row points to a test or explicitly identifies source-only / absent validation; named tests do not imply that every mathematical/scientific statement in a limitation has been empirically established.

## Complete executed Exp 1/2 test inventory

Statuses below come from the completed final verbose log, cross-checked against unittest totals and the static module inventory. CPU-only scope and missing-HB subcases still apply.

| Module | Executed | Pass | Fail | Error | Skip |
|---|---|---|---|---|---|
| `test_exp12_diagnostics` | 10 | 10 | 0 | 0 | 0 |
| `test_exp12_exp2_analysis` | 11 | 11 | 0 | 0 | 0 |
| `test_exp12_fork` | 25 | 24 | 0 | 0 | 1 |
| `test_exp12_foundations` | 12 | 12 | 0 | 0 | 0 |
| `test_exp12_injection` | 8 | 8 | 0 | 0 | 0 |
| `test_exp12_manifest` | 9 | 9 | 0 | 0 | 0 |
| `test_exp12_phase3` | 19 | 19 | 0 | 0 | 0 |
| `test_exp12_pipeline` | 3 | 2 | 0 | 0 | 1 |
| `test_exp12_positive_control` | 15 | 15 | 0 | 0 | 0 |
| `test_exp12_probe` | 9 | 9 | 0 | 0 | 0 |
| `test_exp12_reports` | 5 | 5 | 0 | 0 | 0 |
| `test_exp12_twin_critic` | 23 | 22 | 1 | 0 | 0 |

<details>
<summary>All 149 executed test identifiers and statuses (including all 23 twin tests)</summary>

```text
PASS  test_exp12_diagnostics.DiagnosticsPrecisionTest.test_diagnostic_forward_passes_are_highest_and_training_matmuls_keep_the_run_setting
PASS  test_exp12_diagnostics.KnownAnswerTest.test_diagnostics_never_change_the_update
PASS  test_exp12_diagnostics.KnownAnswerTest.test_existing_path_has_no_new_outputs
PASS  test_exp12_diagnostics.KnownAnswerTest.test_gnorm_std_is_the_population_sd_of_the_window
PASS  test_exp12_diagnostics.KnownAnswerTest.test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old
PASS  test_exp12_diagnostics.KnownAnswerTest.test_policy_kl_near_deterministic_actor_is_reported_not_gated
PASS  test_exp12_diagnostics.KnownAnswerTest.test_saturation_is_the_fraction_of_sampled_components_beyond_the_threshold
PASS  test_exp12_diagnostics.ReferenceBatchTest.test_reference_batch_per_window_from_a_dedicated_stream
PASS  test_exp12_diagnostics.TrainingRunTest.test_metrics_logged_every_window_once_learning_starts
PASS  test_exp12_diagnostics.TrainingRunTest.test_training_is_identical_with_diagnostics_on_and_off
PASS  test_exp12_exp2_analysis.BandTest.test_constant_seeds_give_a_degenerate_band
PASS  test_exp12_exp2_analysis.BandTest.test_iqm_is_the_default_and_unknown_statistics_are_refused
PASS  test_exp12_exp2_analysis.BandTest.test_point_is_the_statistic_over_seeds_and_the_band_contains_it
PASS  test_exp12_exp2_analysis.BandTest.test_reproducible_and_iqm_resists_an_outlier
PASS  test_exp12_exp2_analysis.BandTest.test_resampled_seed_sets_are_shared_across_points
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_band_per_environment_over_complete_confirmatory_forks
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_dev_included_on_request
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_normalize_hook_is_applied_per_environment
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_paired_difference_is_injected_minus_control_mean_return
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_secondary_is_the_check2_success_subset
PASS  test_exp12_exp2_analysis.SyntheticResultsTest.test_tables_and_files
PASS  test_exp12_fork.ForkEndToEndTest.test_arm_rechecks_original_panel_for_existing_forks
PASS  test_exp12_fork.ForkEndToEndTest.test_arm_records
PASS  test_exp12_fork.ForkEndToEndTest.test_arm_refuses_bad_inputs
PASS  test_exp12_fork.ForkEndToEndTest.test_arm_stops_before_training_when_check1_fails
PASS  test_exp12_fork.ForkEndToEndTest.test_check1
PASS  test_exp12_fork.ForkEndToEndTest.test_check2_written_and_paired_with_the_exp1_check
PASS  test_exp12_fork.ForkEndToEndTest.test_control_stops_before_ready_if_restore_changes_panel
PASS  test_exp12_fork.ForkEndToEndTest.test_exp1_ledger_records_fork_and_control_completes
PASS  test_exp12_fork.ForkEndToEndTest.test_exp2_analysis_runs_on_the_real_outputs
PASS  test_exp12_fork.ForkEndToEndTest.test_fork_is_invisible_to_the_exp1_trajectory
PASS  test_exp12_fork.ForkEndToEndTest.test_fork_plan_and_files
PASS  test_exp12_fork.ForkEndToEndTest.test_identity_fork_is_bit_identical
PASS  test_exp12_fork.ForkEndToEndTest.test_injected_arm_frozen_head_unchanged_and_new_head_trained
PASS  test_exp12_fork.ForkUnitTest.test_both_arms_must_run_on_the_fork_device_model
PASS  test_exp12_fork.ForkUnitTest.test_check1_compares_values_and_action_gradients
PASS  test_exp12_fork.ForkUnitTest.test_check1_control_and_identity_must_be_bit_exact
PASS  test_exp12_fork.ForkUnitTest.test_check1_fails_when_the_injection_construction_is_broken
PASS  test_exp12_fork.ForkUnitTest.test_fork_plan
PASS  test_exp12_fork.ForkUnitTest.test_post_fork_eval_leaves_training_state_untouched
SKIP  test_exp12_fork.HumanoidBenchReachEvalSeedingTest.test_reach_eval_seeding_saves_and_restores_the_training_rng
PASS  test_exp12_fork.IdentityValidationTest.test_compare_script_passes_and_detects_a_difference
PASS  test_exp12_fork.IdentityValidationTest.test_flag_refused_outside_dev
PASS  test_exp12_fork.IdentityValidationTest.test_jobs_stop_at_the_snapshot
PASS  test_exp12_fork.KillMatrixTest.test_at_and_after_the_fork
PASS  test_exp12_fork.KillMatrixTest.test_exp1_before_the_fork
PASS  test_exp12_foundations.Angle1ParityTest.test_same_params_optimizer_state_rng_and_metrics
PASS  test_exp12_foundations.DiscountAndHorizonTest.test_gamma_per_suite
PASS  test_exp12_foundations.DiscountAndHorizonTest.test_keyturn_is_truncated_at_100_raw_steps
PASS  test_exp12_foundations.DiscountAndHorizonTest.test_pen_twirl_keeps_its_registered_50_step_limit_as_in_simba
PASS  test_exp12_foundations.DiscountAndHorizonTest.test_truncation_bootstraps_and_termination_does_not
PASS  test_exp12_foundations.EnvRestoreAcrossProcessesTest.test_every_environment_replays_bit_exactly_in_a_new_process
PASS  test_exp12_foundations.EnvRestoreBreakTest.test_restore_is_exact_and_each_component_is_necessary
PASS  test_exp12_foundations.KillAndResumeEntryPointTest.test_kill_points
PASS  test_exp12_foundations.MyoSuiteSeedingTest.test_seeded_factory_is_reproducible_and_repo_factory_is_not
PASS  test_exp12_foundations.ResumeExactnessTest.test_breaking_each_restored_component_is_detected
PASS  test_exp12_foundations.ResumeExactnessTest.test_resume_is_bit_exact_for_each_suite
PASS  test_exp12_foundations.SimbaRandomWarmupTest.test_random_until_min_length_then_policy
PASS  test_exp12_injection.InjectionInvariantsTest.test_gradients_reach_earlier_blocks_through_the_frozen_head
PASS  test_exp12_injection.InjectionInvariantsTest.test_head_boundary_per_label
PASS  test_exp12_injection.InjectionInvariantsTest.test_new_copies_identical_and_freshly_initialised
PASS  test_exp12_injection.InjectionInvariantsTest.test_optimizer_state_rules
PASS  test_exp12_injection.InjectionInvariantsTest.test_parameter_counts
PASS  test_exp12_injection.InjectionInvariantsTest.test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance
PASS  test_exp12_injection.InjectionInvariantsTest.test_target_injection_and_polyak
PASS  test_exp12_injection.InjectionInvariantsTest.test_training_keeps_frozen_heads_bit_identical_and_moves_trainable_parts
PASS  test_exp12_manifest.Exp12GridTest.test_arm_jobs_only_for_completed_forks_grouped_by_device
PASS  test_exp12_manifest.Exp12GridTest.test_cli_writes_the_manifests
PASS  test_exp12_manifest.Exp12GridTest.test_done_and_resume_classification
PASS  test_exp12_manifest.Exp12GridTest.test_every_job_composes_with_the_methodology_budget
PASS  test_exp12_manifest.Exp12GridTest.test_manifests_never_share_a_checkpoint_dir
PASS  test_exp12_manifest.Exp12GridTest.test_the_195_run_grid
PASS  test_exp12_manifest.LauncherTest.test_claim_launcher_dry_run_on_exp12_manifests
PASS  test_exp12_manifest.PreflightTest.test_fail_when_the_architecture_does_not_fork
PASS  test_exp12_manifest.PreflightTest.test_pass_with_fork_and_injected_arm
PASS  test_exp12_phase3.Exp1AnalysisTest.test_analysis_is_reproducible
PASS  test_exp12_phase3.Exp1AnalysisTest.test_dev_runs_do_not_change_the_confirmatory_result
PASS  test_exp12_phase3.Exp1AnalysisTest.test_full_analysis_writes_outputs
PASS  test_exp12_phase3.Exp1AnalysisTest.test_known_effect_is_recovered_and_null_effect_is_not
PASS  test_exp12_phase3.Exp1AnalysisTest.test_missing_run_is_refused
PASS  test_exp12_phase3.Exp1LedgerEndToEndTest.test_run_writes_ledger
PASS  test_exp12_phase3.LedgerTest.test_incomplete_and_foreign_rows_are_rejected
PASS  test_exp12_phase3.LedgerTest.test_round_trip_and_dev_excluded_by_default
PASS  test_exp12_phase3.TriggerTest.test_consecutive_checks_rule
PASS  test_exp12_phase3.TriggerTest.test_current_worse_triggers_and_reverse_never_does
PASS  test_exp12_phase3.TriggerTest.test_edge_cases
PASS  test_exp12_phase3.TriggerTest.test_f_star_first_eligible_check_only
PASS  test_exp12_phase3.TriggerTest.test_known_loss_is_detected
PASS  test_exp12_phase3.TriggerTest.test_null_false_trigger_rate_is_measured
PASS  test_exp12_phase3.TriggerTest.test_null_threshold_shifts_the_firing_line
PASS  test_exp12_phase3.TriggerTest.test_reproducible_from_seed_and_check
PASS  test_exp12_phase3.TriggerTest.test_settings_come_from_config
PASS  test_exp12_phase3.TriggerTest.test_shipped_config_is_the_current_rule
PASS  test_exp12_phase3.TriggerTest.test_statistic_is_iqm_not_mean
PASS  test_exp12_pipeline.PipelinePerSuiteTest.test_dmc
SKIP  test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench
PASS  test_exp12_pipeline.PipelinePerSuiteTest.test_myosuite
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_evaluate_chooses_m
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_evaluate_stops_on_noise
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_evaluate_stops_when_l_trigger_is_not_positive
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_loss_rounds_ordered_numerically
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_noise_exactly_at_the_threshold_stops
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_pooled_sd
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_recovery_toward_the_fresh_critic
PASS  test_exp12_positive_control.MSelectionArithmeticTest.test_select_smallest_m_within_tolerance
PASS  test_exp12_positive_control.PositiveControlEndToEndTest.test_cli_needs_no_definition_flags
PASS  test_exp12_positive_control.PositiveControlEndToEndTest.test_full_report
PASS  test_exp12_positive_control.PositiveControlEndToEndTest.test_injection_does_not_change_predictions_before_the_probe
PASS  test_exp12_positive_control.PositiveControlEndToEndTest.test_never_triggered_run_stops
PASS  test_exp12_positive_control.PositiveControlEndToEndTest.test_refuses_wrong_setting
PASS  test_exp12_positive_control.SharedOffsetTest.test_default_is_each_critics_own_offset
PASS  test_exp12_positive_control.SharedOffsetTest.test_shared_offset_modes
PASS  test_exp12_probe.ProbeArithmeticTest.test_a_critic_that_learns_scores_higher_than_one_that_cannot
PASS  test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score
PASS  test_exp12_probe.ProbeArithmeticTest.test_iqm_matches_rliable_definition
PASS  test_exp12_probe.ProbeArithmeticTest.test_non_finite_round_marks_check_invalid
PASS  test_exp12_probe.ProbeArithmeticTest.test_sign_convention_plasticity_loss_positive_when_current_is_worse
PASS  test_exp12_probe.ProbeDoesNotChangeTrainingTest.test_breaking_probe_isolation_is_detected
PASS  test_exp12_probe.ProbeDoesNotChangeTrainingTest.test_probes_on_equals_probes_off
PASS  test_exp12_probe.ProbePairingTest.test_identical_inputs_targets_and_minibatches
PASS  test_exp12_probe.ProbePairingTest.test_per_critic_offset_is_each_critics_own_mean
PASS  test_exp12_reports.RangeCriterionTest.test_a_size_below_0_9_or_missing_fails
PASS  test_exp12_reports.RangeCriterionTest.test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule
PASS  test_exp12_reports.UnittestLogTest.test_every_test_and_every_failing_subtest_is_listed_with_its_status
PASS  test_exp12_reports.UnittestLogTest.test_missing_statuses_are_flagged_not_guessed
PASS  test_exp12_reports.UnittestLogTest.test_the_block_a_shape_failures_counted_per_subtest
FAIL  test_exp12_twin_critic.TwinCheck1Test.test_panel_values
PASS  test_exp12_twin_critic.TwinCriticDefectTest.test_actor_grad_cosine_uses_min_of_the_two_networks
PASS  test_exp12_twin_critic.TwinCriticDefectTest.test_td_error_var_is_the_mean_of_the_two_networks
PASS  test_exp12_twin_critic.TwinCriticDefectTest.test_twin_critic_initialises_with_two_independent_networks
PASS  test_exp12_twin_critic.TwinCriticDefectTest.test_update_many_runs_with_twin_critics
PASS  test_exp12_twin_critic.TwinCriticPathTest.test_actor_loss_uses_min
PASS  test_exp12_twin_critic.TwinCriticPathTest.test_checkpoint_round_trip_keeps_both_networks
PASS  test_exp12_twin_critic.TwinCriticPathTest.test_critic_loss_uses_the_shared_min_target_and_each_network_its_own_term
PASS  test_exp12_twin_critic.TwinCriticPathTest.test_polyak_update_covers_both_target_networks
PASS  test_exp12_twin_critic.TwinCriticPathTest.test_structural_metrics_are_per_network
PASS  test_exp12_twin_critic.TwinDiagnosticsTest.test_diagnostics_never_change_the_update
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_check1_both_networks
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_check2_uses_the_mean_and_records_both_networks
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_identity_fork_is_bit_identical
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_injected_arm_both_networks
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_probe_records_are_the_mean_of_both_networks
PASS  test_exp12_twin_critic.TwinForkEndToEndTest.test_runs_are_twin_and_record_the_critic_count
PASS  test_exp12_twin_critic.TwinInjectionTest.test_predictions_unchanged_and_action_gradients_within_tolerance
PASS  test_exp12_twin_critic.TwinInjectionTest.test_same_head_in_both_networks_and_targets
PASS  test_exp12_twin_critic.TwinInjectionTest.test_training_keeps_frozen_heads_and_optimizer_rules
PASS  test_exp12_twin_critic.TwinProbeIsolationTest.test_probes_on_equals_probes_off
PASS  test_exp12_twin_critic.TwinProbeTest.test_combined_loss_is_the_mean_of_the_two_networks
PASS  test_exp12_twin_critic.TwinProbeTest.test_expand_gives_each_network_its_own_slice
```

</details>
