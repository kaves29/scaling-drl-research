# Experiments 1 and 2: master summary (Phase 8)

Branch `claude/eloquent-fermat-inxqlt` (not merged; nothing pushed to main).
Source of truth: `.claude/methodology-exp1-exp2.md`, with Amendments (a)–(u).
Decision log: `docs/exp12_decisions.md`. CUDA command sheet:
`docs/exp12_cuda_commands.md`.

Everything below was verified on CPU (Linux, jax/jaxlib 0.4.34) unless marked
**NEEDS CUDA VERIFICATION**. No grid run, dev run, pilot or CUDA job has been
launched or scheduled.

## 1. What was built

| File | Main functions | Methodology bullet / amendment |
|---|---|---|
| `configs/base_exp12.yaml` | probe, trigger, fork, injection, checks, diagnostics, positive_control, testing sections | R-CONFIG: critics, actor D1W128, UTD 2, lr 1e-4, batch 256, AdamW wd 1e-2, τ 0.005, action repeat 2, target entropy −\|A\|/2 (SAC sign convention for the Methodology's \|A\|/2), one Q critic, 1 train env; all trigger settings as config values |
| `configs/env/myosuite_simba.yaml`, `configs/env/humanoid_bench.yaml` | env groups | R-ENV budgets: 1M MyoSuite, 2M HumanoidBench (dmc_hard 1M and dmc_medium 500k already existed); (i), (j), (s) |
| `configs/suite_metadata/*.yaml` | data only | (g) benchmark constants, never used to normalise |
| `experiments/exp12/envs.py` | `create_envs`, `create_eval_env`, `ExactRestore`, `make_myosuite_env` (seeded), `make_humanoid_bench_env`, `restore_env` | R-ENV 13 environments; R-FORK exact env state + RNG restore at the exact check step (D1); G3 MyoSuite seeding |
| `experiments/exp12/state.py` | `new_state_dir`, `commit_state_dir` (atomic LATEST), `save_buffer`/`load_buffer` (np.savez, filled part only), `state_differences` | R-FORK / R-PERSIST complete state, no pickled big arrays; (t) |
| `experiments/exp12/trainer.py` | `Exp12Trainer.train/save/restore/inject` | R-RUN same SAC loop and fused `update_many`; SimBa random warm-up (h); complete-state save/restore incl. global numpy RNG restored, not re-seeded |
| `experiments/exp12/probe.py` | `run_probe`, `probe_round`, `_fit` (one jitted lax.scan), `summarize`, `iqm` | R-PROBE (Lyle 2023): P = b − final MSE, target a + sin(1e5·f(x;ω0)), per-critic offset (c), paired current/fresh, own RNG streams, 5 rounds × 1000 steps, IQM |
| `experiments/exp12/run_probes.py` | `RunProbes.capture_fresh/maybe_check/record_check/extend_to` | fresh critic probed and stored before training; 20 checks at k/20 of B; probes continue after the fork on the same grid (F2) |
| `experiments/exp12/trigger.py` | `bootstrap_interval`, `triggered`, `f_star`, `last_eligible_check` | R-TRIGGER: percentile bootstrap (10,000, IQM, 95%), fire when the lower bound > 0, f*_run = check completing the first run of 2 consecutive firing checks among 1..19 (l) |
| `experiments/exp12/ledger.py` | `write_run`, `load`, `load_metrics` | R-LEDGER: run.csv, checks.csv, probe_curves.npz, metrics.csv; dev runs excluded by default |
| `experiments/exp1.py` | `run` (registered `exp1`) | R-RUN; fork at f*_run for D4W1024/D6W1536; the process restarts from the fork state as the CONTROL to max(B, fork+0.25B) (D2) |
| `experiments/exp12/fork.py` | `fork_plan`, `write_fork`, `sample_panel`, `panel_q_and_grad`, `check1`, `post_fork_eval`, `matmul_precision_report`, `check_same_device` | R-FORK, R-CHECKS Check 1 (m), post-fork evaluations every 1% of B × 10 episodes (F1), device pinning (p) |
| `experiments/exp12/injection.py` | `InjectedSACCritic`, `inject`, `injected_optimizer`, `split_params`, `head_blocks` | R-INJECT (Nikishin 2023): head = last m blocks + post-LN + output; Q = old + (new − copy); trunk keeps AdamW state, new head fresh AdamW, frozen params zero updates and no decay; target injected; m ∈ {last, half, all} |
| `experiments/exp2_arm.py` | `run` (registered `exp2_arm`), `check2` | separate resumable injected arm to fork + 0.25B; Check 1 enforced, Check 2 reported (n); identity arm for validation |
| `experiments/exp12/exp2_ledger.py` | `write_arm`, `write_json` | Exp 2 records per arm: checks, raw per-episode returns/success, metrics (Check 3) |
| `experiments/exp12/diagnostics.py` | `ActorDiagnostics`, `DiagnosticPendingUpdateMetrics` | R-ACTOR-DIAGNOSTICS I1 KL churn, I2 gnorm SD, I3 saturation, I4 all Exp 1/2 runs |
| `experiments/exp12/m_selection.py` | `recovery`, `pooled_sd`, `select_m`, `evaluate` | m-selection (d), healthy reference = fresh critic, noise and stop rule (q) |
| `scripts/positive_control.py` | `run` | R-POSITIVE-CONTROL steps 1–4 on a dev run, shared-offset sensitivity (c) |
| `scripts/compare_identity_fork.py` | `compare` | identity fork bit-identity (D2 gate) |
| `scripts/probe_fresh_checks.py` | range and null modes | CUDA calibration checks with pre-specified rules |
| `scripts/profile_exp12.py` | `profile_arch`, `_fork_timing`, `_eval_cost`, `_train_rate` | section 6 measurements |
| `scripts/install_humanoid_bench.sh` | — | pinned HumanoidBench commit, `--no-deps` |
| `analysis/exp1_analysis.py` | `primary_endpoint`, `trajectories`, `f_star_table`, plots | R-EXP1-ANALYSIS: rliable stratified bootstrap of IQM(scaled) − IQM(D2W512) at check 20, trajectories, every seed, f* table |
| `analysis/exp2_analysis.py` | `paired_returns`, `bootstrap_band`, `run_analysis` | R-EXP2-ANALYSIS: paired injected − control return per env with IQM bands (r) and every seed; both arms' diagnostics and plasticity; Check tables; shared time axis; Check-2-success secondary; no scalar summary |
| `generate_manifest.py` (`--grid exp12`) | `add_exp12_grid`, `classify_exp12` | R-PERSIST: 195-run grid, absolute paths, DONE/resume; arm jobs per GPU model once m is frozen |
| `scripts/claim_launcher.py` | `--phase-files` | launcher compatibility |
| `scripts/preflight_checkpoint_check.py` | `--experiment exp1 [--with-fork]` | preflight for the new grid |

Small edits outside the Exp 1/2 paths, each with no change to existing behaviour (Angle 1 parity and the existing suite are tested):

- `scale_rl/agents/sac/sac_agent.py`, `sac_update.py` and
  `agents/wrappers/normalization.py`: the optional diagnostics arguments,
  default off.
- `utils/run_metadata.py`: humanoid_bench is added to the simulator-version
  list.
- `experiments/__init__.py`: registers exp1 and exp2_arm.
- `requirements.txt`: rliable 1.2.0 plus its pinned dependencies; no existing
  pin changed.
- `CLAUDE.md`: a pointer line only.

## 2. Decisions

All are in `docs/exp12_decisions.md`. The governing ones are the Methodology
amendments:

- (a) HumanoidBench tasks h1-reach-v0 and h1-run-v0.
- (b) The fork design.
- (c) The per-critic probe offset, plus a one-time shared-offset check.
- (d) The m rule: the smallest m within 0.10 of the best recovery.
- (e) The positive control on D6W1536 dog-run with a dev seed.
- (f) Superseded by (q).
- (g) No normalisation; graphs only.
- (h) SimBa random warm-up until 5,000 transitions.
- (i) MyoSuite max_episode_steps 100, γ 0.95.
- (j) HumanoidBench with one Q critic (limitation).
- (k) HumanoidBench limitations.
- (l) Two consecutive firing checks.
- (m) Check 1: control and identity arm bit-exact; injected arm ≤ 64 eps.
  Its TF32 stop is superseded by (v).
- (n) Check 2: paired difference with its interval; 5-round caveat.
- (o) Superseded by (q).
- (p) Both arms on the fork's GPU model.
- (q) Healthy reference = the fresh critic (L = 0). Recovery =
  (L_trigger − L_inj) / L_trigger. Noise = pooled SD of the 4 real-settings
  L series / L_trigger. Stop if L_trigger ≤ 0 or noise ≥ 0.10.
- (r) IQM bands in Exp 2.
- (s) Time limits and truncation exactly as in SimBa; PenTwirl's 50-step
  limit is a limitation.
- (t) Checkpoint retention.
- (u) One GPU model for the whole grid.
- (v) TF32 for every matmul on the GPU (`jax_default_matmul_precision =
  "tensorfloat32"`, no-op on CPU); no bf16/fp16; x64 off. Check 1 and A0's
  error measurement run under a local "highest" context. Metadata records
  the precision, GPU model and JAX/jaxlib/CUDA versions.

Routine engineering choices are listed under "Choices (routine, shown for veto)" in each phase section of the log. The main ones:

- Probe pool 25,600, batch 256, eval chunk 2,560.
- Probe streams keyed by (seed, "PROB", check, round). Bootstrap stream
  (seed, "BOOT", check).
- Fork panel of 256 pairs (stream "PANL"). Post-fork evaluations seeded by
  (seed, "PFEV", index), with the global RNGs saved and restored.
- Injection key (seed, "INJC"). m = "half" is max(1, D // 2).
- KL reference batch of 256 states per logging window (stream "KLRF"),
  normalised at the window start. gnorm SD is the population SD.
- One routine save per probe check. Checkpoint layout
  `<root>/exp1/<arch>/<env>/seed_<s>` and `<root>/exp2_arm/.../injected`.
- Injected-critic structural metrics are NaN. The identity snapshot is
  taken at 1,000 interaction steps.

## 3. Deviations and conflicts

- Deviations from the Methodology: none known. Every interpretation was
  asked about and recorded as an amendment.
- Conflicts with older documents (not edited; the Methodology wins for
  Exp 1/2):
  - CLAUDE.md and `.claude/research-methodology.md` give scaled critics
    D5W768/D7W1024, UTD 5 and a different 10-environment core set. They use
    td_error_var-based degradation and treat hopper-hop and
    myoHandPenTwirlFixed as Angle 3 held-out. Their MyoSuite tier labels
    differ, and they do not mention HumanoidBench.
  - The Exp 1/2 entry points do not call the Angle-specific held-out
    validators.

## 4. Test results

Phase 7 run, CPU, 2026-10-04: every test module in its own process, three at a time.

- **Full suite:** 375 tests. 374 pass, 1 fails, and 2 are skipped (the
  HumanoidBench ones; both pass in venv_hb).
- **Existing tests:** 258, the same set as the Phase 2 baseline; 257 pass.
- **Pre-existing failure, separated:**
  `test_angle2a_env_state_determinism_smoke.TestMyosuiteDeterminism.test_myo_baoding_p1`.
  It fails identically before any Exp 1/2 change.
- **No new failures.** Angle 1, 2A, 2B and 2C behaviour is unchanged:
  - all their tests pass;
  - Exp 1 with probes off reproduces angle_1's parameters, optimizer state,
    RNG and metrics bit for bit (Angle1ParityTest). The only difference is
    the three added diagnostic columns.
- **HumanoidBench in venv_hb:** env restore across processes, the restore
  break test, discount and horizon, and resume exactness (8 tests); the
  Reach evaluation-seeding test; the tiny pipeline. All pass.

| Exp 1/2 module | Tests | What it proves |
|---|---|---|
| test_exp12_foundations | 12 | exact env restore mid-episode for all 13 envs in a new process (HumanoidBench in venv_hb); each restored component is necessary; MyoSuite seeding; bit-exact resume per suite; kill-and-resume of exp1; SimBa warm-up; γ per suite; PenTwirl's registered 50-step limit; truncation bootstrapped, termination not; Angle 1 parity |
| test_exp12_probe | 9 | known-answer scores; current/fresh paired on identical inputs, targets and minibatches; per-critic offset; L sign; invalid checks; probes on = probes off for training |
| test_exp12_phase3 | 19 | trigger settings from config; IQM bootstrap; lower bound > 0; edge cases (ties, zeros, NaN); synthetic null rate; c = 2 consecutive checks and eligibility to check 19; ledger round trip; dev excluded; rliable analysis on known effects, reproducible |
| test_exp12_injection | 8 | Q bit-identical at injection, dQ/da within tolerance; parameter counts; head boundary; new = copy; frozen heads bit-identical over 50 steps incl. weight decay; optimizer state rules; gradients reach the trunk; target and Polyak |
| test_exp12_fork | 23 (1 skipped here, passes in venv_hb) | fork plan and files; identity arm bit-identical to the control; Check 1 (bit-exact control and identity, 64 eps injected, run under full FP32, fails on a broken injection construction, dQ/da compared); Check 2 paired difference; arm records; frozen head unchanged; fork invisible to the Exp 1 trajectory; refusals (config, m, missing fork, device model); post-fork evaluation isolation and seeding; identity-validation procedure and compare script; HumanoidBench Reach seeding; **kill matrix**: before a check, mid-interval, inside a routine save (previous checkpoint intact), inside the fork write, right after the fork, mid-post-fork control and arm, all bit-identical after resume |
| test_exp12_positive_control | 15 | recovery toward the fresh critic; m rule; pooled SD; stop rules (L_trigger ≤ 0, noise ≥ 0.10, at the boundary); shared offset modes; end to end on a forced dev run (trigger probe reproduced exactly; noise from the four real-settings series only) |
| test_exp12_diagnostics | 8 | KL equals the closed form, direction KL(π_t ‖ π_{t−1}); saturation on the sampled actions; gnorm population SD; reference batch from a dedicated stream per window; update bit-identical with diagnostics on or off; metrics present every window |
| test_exp12_exp2_analysis | 11 | bands (known answers, shared resamples, reproducible, IQM default); paired differences; incomplete forks and dev runs excluded; Check-2-success secondary; normalize hook per environment |
| test_exp12_manifest | 9 | 195 jobs, unique absolute paths, budgets and composed configs; DONE/resume; arm jobs per fork and device, config equal to the parent except the arm keys; no overlap; launcher dry run; preflight PASS and FAIL |
| test_exp12_pipeline | 3 (1 skipped here, passes in venv_hb) | tiny end-to-end per suite (DMC, MyoSuite, HumanoidBench): 20 checks, forced trigger, fork, both arms, Checks 1–2, post-fork probes and evaluations, both ledgers, both analysis scripts |
| **Total Exp 1/2** | **117** | |

**Break-and-restore evidence** (`python tests/exp12_break_checks.py`): 55
mutations. Each one makes its named test fail by an assertion (never by a
crash), and the test passes again once restored. They cover: probe pairing,
offset, baseline, sign and validity; trigger tail, strictness, eligibility,
statistic and consecutive rule; dev exclusion; SimBa warm-up; every
injection invariant; the m rule, recovery and noise; the shared offset;
Check 1 (dQ/da, bit-exact control, tolerance, device); the fork plan; post-fork
evaluation isolation; the diagnostics; the Exp 2 analysis; the manifests;
and truncation (critic target, final observation, PenTwirl limit). Two
further HumanoidBench Reach mutations were run by hand in venv_hb: both
fail the test, and the restored code passes.

**NEEDS CUDA VERIFICATION:** every test above on the Linux + CUDA stack, with
and without deterministic ops; the identity-fork gate per architecture ×
suite; Check 1 tolerances on the grid's GPU; TF32 active at the run setting (A0); all GPU performance
numbers in section 5.

## 5. Measured numbers

| Quantity | Value | Where |
|---|---|---|
| Training it/s with probes off vs the current code (angle_1) | CPU: exp1/angle_1 wall-time ratio 0.96 (D2W512, humanoid-run); the real ratio **NEEDS CUDA VERIFICATION** (CUDA B4, `ratio_exp1_over_angle1`) | |
| Probe overhead per critic size | CPU: D2W512 one check 413 s, projected 7.1% of a 500k-step run's wall-clock; all sizes **NEEDS CUDA VERIFICATION** (A3, B4). Forecast under TF32 for D6W1536 dog-run ≈ 7% (CUDA sheet, an estimate). Rule: if D6W1536 exceeds ~5%, I report and ask; the probe is never reduced | |
| Actor diagnostics overhead | CPU: within noise (−1.2%, D2W512, 60 steps); **NEEDS CUDA VERIFICATION** (`diagnostics_overhead_pct`) | |
| Fork save / restore | CPU, local disk: D6W1536 dog-run complete state at a 95%-of-B fork = 2.62 GB, save 17.4 s, restore 18.4 s. D2W512 hopper-hop: 99 MB, 1.0 s / 0.8 s. Cluster disk **NEEDS CUDA VERIFICATION** | |
| Post-fork evaluation cost (F1, 26 × 10 episodes per arm) | CPU: dog-run 41 s per evaluation (0.30 h per arm); hopper-hop 7.3 s; MyoSuite 1.7–2.8 s and h1-run 3.4 s (lower bounds: untrained policies end episodes early; full-length h1-run ≈ 49 s). As a share of arm training time: **NEEDS CUDA VERIFICATION** | |
| Peak GPU memory per size, recommended concurrency | **NEEDS CUDA VERIFICATION** (`peak_device_bytes`, `recommended_jobs_per_gpu_upper_bound`) | |
| Retained disk per run (worst case dog-run, fork at 95% of B) | D2W512 1.06 GB; D4W1024 5.03 GB (3.32 run + 1.71 arm); D6W1536 9.18 GB (6.19 + 2.99); whole-grid upper bound ≈ 734 GB. Estimate from parameter counts and measured bytes per transition, checked against the measured 2.62 GB fork state (estimate 2.75 GB) | |
| Synthetic-null false-trigger rate per check (5 rounds, one check) | 4.93% (nominal one-sided 2.5%); the fresh-pair null with real probes **NEEDS CUDA VERIFICATION** (B3) | |
| Check 1 on CPU | identity and control 0 eps; injected Q exact, dQ/da up to 7.2 eps (D6W1536) | |

## 6. Risks, limitations, open points

- The probe's dynamic range at the real critic sizes is unknown. On a small
  CPU critic, P sat near 0.008 of b ≈ 0.5. The range rule and fallback
  ladder are pre-specified (CUDA A2, B2).
- Run-level false-trigger rate. The per-check percentile bootstrap over 5
  rounds is anti-conservative (4.93% vs 2.5%). Two consecutive checks
  reduce the run-level rate, but it is measured only on CUDA (B3, null).
  Check 2's interval has the same 5-round caveat (n).
- KL churn is heavy-tailed. Near-deterministic action dimensions (σ ≈ 6e-5)
  gave 10–20 nats per update on a tiny CPU policy.
- HumanoidBench:
  - it runs on mujoco 3.6.0 instead of its pinned 3.1.6;
  - SimBa's code uses h1hand tasks while its paper matches no-hands;
  - it needs EGL;
  - it uses one Q critic instead of SimBa's two;
  - it is untested on GPU.
- MyoSuite: KeyTurn is truncated at 100 raw steps (registered 200), and
  PenTwirl at its registered 50, while γ is derived from 100. Both match
  SimBa.
- Bit-exactness on CUDA is unproven. If the identity fork passes only with
  deterministic GPU ops, the tolerance and flags are your decision.
- Precision (v): every matmul runs in TF32 on the GPU; Check 1 and A0's
  measurement run under full FP32. TF32 no longer stops anything. A0 must
  show the run setting active (error ~1e-4 to 1e-3) and highest at ~1e-7;
  NEEDS CUDA VERIFICATION. TF32 rounding changes results relative to FP32
  runs, so all Exp 1/2 runs must use the same setting (they do: one helper).
- Device pinning: every scaled run and both arms must stay on one GPU
  model (u). A control resumed on another model refuses to run.
- Disk: the grid's upper bound is ≈ 734 GB of retained state (section 5).
- The positive control may stop: the dev run may never trigger, L_trigger
  may be ≤ 0, or the noise may be ≥ 0.10. Each stops and consults by
  design; the trigger is never loosened.
- I am unsure whether the probe overhead for D6W1536 on the 500k-step tasks
  stays under ~5%; the CPU forecast suggested it may not.

## 7. Commits per phase

| Phase | Commits |
|---|---|
| 0 | `447dbc1` Methodology verbatim; `07d0e1a` audit and questions; `950c75b` final version + amendments |
| 1 | `8a82c3d` foundations: exact state save/restore, resume |
| 2 | `10fa2e1` probe, tests, break checks, profiler; `cd85a5b` gate evidence |
| 3 | `d4d9b7b` trigger, ledger, Exp 1 analysis, A9/G2/HumanoidBench; `f6a4d4e` pre-specified CUDA rules; `631ef77` trigger settings as config, null mode |
| 4 | `f9002e0` fork, injection, Checks, identity fork, positive control; `57215f8` Phase 4 decisions |
| 5 | `f4e3972` actor diagnostics and Exp 2 analysis |
| 6 | `e79a9fa` manifests, launch plumbing, profiling |
| 7 | `6063411` Phase 5–6 decisions, per-suite pipeline and kill matrix |
| 8 | `2b9ccc3` summary draft; the final commit carries the Phase 7 results |

## 8. What to run next (all on the CUDA stack, from `main/`)

The full ordered sheet is `docs/exp12_cuda_commands.md`. In order:

1. **Setup and precision** (Setup, A0). Install the requirements and
   HumanoidBench, export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl, and save
   `A0_matmul_precision.json`. It must show `matmul_precision`
   tensorfloat32 on gpu with error ~1e-4 to 1e-3, and highest at ~1e-7.
2. **CUDA tests and break checks** (A1, B1). Run them with default and with
   deterministic XLA flags:
   `python -m unittest discover -s tests -p "test_exp12_*.py" -t .` and
   `python tests/exp12_break_checks.py`.
3. **Calibration** (A2, A3, B2–B4). The fresh-critic range check (pre-specified
   10–90% of b rule), the fresh-pair null (≥ 100 pairs per size), and the
   profile per critic size and suite. Send me the JSON files; any rule that
   fails comes back to you.
4. **Identity-fork gate** (Block C). D4W1024 and D6W1536 × dog-run, myo-key-turn
   and h1-run-v0, on the grid's GPU model. `compare_identity_fork.py` must
   exit 0 for all six.
5. **Positive-control dev run** (Block D). D6W1536 dog-run, seed 102, run_role=dev,
   then `scripts/positive_control.py`. You freeze m (`injection.m`) from its
   result.
6. **Preflight** (Block E) per critic size on the grid's GPU model, with
   `--with-fork` for D4W1024 and D6W1536.
7. **Pilot** (your call; dev role, seeds outside 1–5, kept out of the
   confirmatory data). For example, dog-run with all three critics, seed 201:

   ```bash
   for A in "2 512" "4 1024" "6 1536"; do set -- $A
     python run.py --experiment exp1 --config_name base_exp12 --overrides env_name=dog-run --overrides env=dmc_hard \
       --overrides critic_num_blocks=$1 --overrides critic_hidden_dim=$2 --overrides seed=201 --overrides run_role=dev \
       --overrides results_root=/abs/pilot_results --checkpoint_dir /abs/pilot/D$1W$2 --checkpoint_interval 25000 \
       --checkpoint_start_frac 0.0
   done
   ```
8. **Main grid** (Block F). Run
   `python generate_manifest.py --grid exp12 --ckpt-root ABS --results-root ABS`
   (195 jobs), then `scripts/claim_launcher.py --phase-files
   exp12_exp1_jobs.txt`. Once m is frozen and forks exist, regenerate with
   `--injection-m <m>`, run `check_manifest_overlap.py`, then launch
   `exp2_arms_<device>.txt` on that GPU model.

**Pre-deployment checklist** (every item from the steps above):

- [ ] A0: TF32 active at the run setting, highest at FP32 level (Check 1's context).
- [ ] CUDA tests and break checks pass; any deterministic-ops dependence decided.
- [ ] Range rule passes at all three sizes; the round spread is reported.
- [ ] Null fire rate ≤ 5% per size, or your decision on the p95 threshold.
- [ ] Probe overhead for D6W1536 ≤ ~5%, or your decision.
- [ ] Identity fork is bit-identical for all six architecture × suite cells.
- [ ] Positive control: m chosen and frozen in `injection.m`.
- [ ] Preflight PASS per size on the grid's GPU model.
- [ ] All grid jobs pinned to one GPU model; disk ≥ ~734 GB free.
- [ ] Concurrency per GPU set from the measured peak memory, with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- [ ] `WANDB_API_KEY` exported. Unique joblog per launch. Overlap check OK
  before arms run alongside the grid.
