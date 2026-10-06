# Experiments 1 & 2: decision log

Audit trail for every choice not literally stated in
`.claude/methodology-exp1-exp2.md`. Each entry: date, question, options,
recommendation, answer. Answers marked PENDING have not been given yet; nothing
depending on them is implemented.

## Phase 0 audit facts (verified 2026-10-03 by running code)

These are measurements, not decisions. Machine: Linux cloud container, 4 CPUs,
no GPU, CPU-only jaxlib 0.4.34 with the repo's pinned requirements.txt. It is
not the jax-metal Mac, so every timing and memory figure for real runs needs CUDA.

- Single Q critic: `critic_use_cdq: ${env.episodic}` and every DMC/MyoSuite env
  config has `episodic: false`. Resolved configs give `critic_use_cdq=False`
  for dog-run, hopper-hop and myo-key-turn, so the critic is `SACCritic`, one Q.
- Parameter counts, from `SACCritic`/`SACActor` with the real env dims. The
  Methodology's 4.2M / 33.6M / 113M are exactly the residual blocks alone
  (4,201,472 / 33,583,104 / 113,310,720). Full critics add the input projection,
  the final LayerNorm and the output layer:

  | env | obs | act | D2W512 | D4W1024 | D6W1536 | actor D1W128 |
  |---|---|---|---|---|---|---|
  | dog-run / dog-trot | 223 | 38 | 4,337,153 | 33,854,465 | 113,717,761 | 170,700 |
  | humanoid-run/walk/stand | 67 | 21 | 4,248,577 | 33,677,313 | 113,452,033 | 146,346 |
  | swimmer-swimmer15 | 61 | 14 | 4,241,921 | 33,664,001 | 113,432,065 | 143,772 |
  | hopper-hop | 15 | 4 | 4,213,249 | 33,606,657 | 113,346,049 | 135,304 |
  | myoHandKeyTurnFixed-v0 | 93 | 39 | 4,271,105 | 33,722,369 | 113,519,617 | 154,318 |
  | myoHandPenTwirlFixed-v0 | 83 | 39 | 4,265,985 | 33,712,129 | 113,504,257 | 153,038 |
  | myoHandPoseRandom-v0 | 108 | 39 | 4,278,785 | 33,737,729 | 113,542,657 | 156,238 |
  | myoHandReachFixed-v0 | 115 | 39 | 4,282,369 | 33,744,897 | 113,553,409 | 157,134 |

- Environments: all 7 DMC and 4 MyoSuite IDs load and step. The DMC episodes
  are 1000 raw steps each. MyoSuite's registered `max_episode_steps` are
  KeyTurnFixed 200, PenTwirlFixed 50, PoseRandom 100 and ReachFixed 100. The
  repo's `TimeLimit(1000)` never binds for these, and with random actions the
  episodes terminated early (term=True) for KeyTurn, PenTwirl and Reach.
- MyoSuite seeding: `make_myosuite_env` ignores `seed`. Two fresh
  constructions with seed=1 give different reset observations for
  myoHandPoseRandom-v0, and identical ones for myoHandKeyTurnFixed-v0 and
  dog-run. This is a pre-existing reproducibility gap.
- HumanoidBench: none of the 64 commits on any branch ever contained
  HumanoidBench code, so there is nothing to restore. The upstream registry
  (carlosferrazza/humanoid-bench @ cb11890) registers `f"{robot}-{task}-v0"`.
  `h1-reach-v0` and `h1hand-reach-v0` exist, and no `shelf-place` task exists:
  the nearest are `bookshelf_simple` and `bookshelf_hard`. Upstream pins
  gymnasium==0.29.1, mujoco==3.1.6, dm_control==1.0.20 and torch==2.3.1, and
  it imports torch at import time. Installed `--no-deps` into a throwaway venv
  copy, it fails with `ModuleNotFoundError: torch`. The Reach task samples its
  goal with the global `np.random`.
- SimBa (arXiv 2410.09754, Table 7, Table 1) uses clipped double-Q on
  HumanoidBench only. MyoSuite episode length is 100 (effective 50), so the
  TD-MPC2 heuristic gives gamma=0.95 there. HumanoidBench uses action repeat 2
  and 2M env steps.
- rliable 1.2.0 installs under the pins. In a venv copy it leaves numpy 1.26.4,
  scipy 1.17.1 and pandas 2.1.4 unchanged, `pip check` is clean, and IQM plus
  `get_interval_estimates` run.
- Earlier fixes, verified by reading the code. The checks under "Tests" were
  not run because the full suite exceeds the 10-minute cap here.
  - Absolute checkpoint paths are enforced in run.py and `save_checkpoint`.
  - The DONE marker is written by angle_1 and skipped by generate_manifest.
  - Fused `lax.scan` updates exist (`update_many` → `_update_sac_networks_scan`).
  - Resume restores the numpy and python global RNG and the JAX key.
  - Resume does not restore the train or eval env state or RNG: the env is
    rebuilt and reset. It also does not restore the partial logging window.
  - The first action after a resume is uniform-random, because `timestep is None`.
  - Resume is therefore not trajectory-exact today.
- Existing actor diagnostics, already logged every run:
  - `train/actor_gnorm`: global L2 norm of the actor-loss gradient, per update,
    averaged per logging window.
  - `train/actor_action`: mean |a| of the sampled stochastic actions on the
    training batch.
  - `train/policy_churn` / `train/churn`: per update, the mean L2 norm of the
    change in the deterministic action tanh(mean) on `churn_ref_batch`, which is
    the first training batch, frozen.
  - `train/actor_grad_cosine`: every 30 updates.
- Existing behaviour not listed in the Methodology:
  - Initial temperature is 0.01.
  - Target entropy is −0.5·|A|.
  - Temperature uses weight decay 0.0. Actor and critic use AdamW with
    betas (0.9, 0.999), eps 1e-8, and weight decay on all params including
    biases and LayerNorm.
  - Replay capacity is 1,000,000. Updates start at 5,000 transitions.
  - Only interaction step 1 uses a random action; the untrained actor acts
    from step 2.
  - Observation normalisation keeps a running mean/var, updated on every
    training action.
  - Evaluation runs every 50,000 interaction steps with 5 deterministic
    episodes, and 50 episodes at the end.
  - n_step=1, reward_scale=1, actions rescaled to [−1, 1].
  - Logging runs every 2,000 interaction steps.

## Questions (PENDING unless answered)

Numbered for reference. Each lists the options, then my recommendation and why.

### Q0 Methodology version
The brief's section 2 holds two consecutive versions of the Methodology. They
differ in, for example, the m-selection rule ("no statistically detectable
deficit" vs "observed recovery, smallest m among similar") and the Exp 1
endpoint ("pooled trajectory" vs "final checkpoint, scaled minus default").
Rec: the second (later) version governs, since section 4 of the brief
matches it.

### A Probe (Lyle et al. 2023, §2.2/§3.1/§5.1/App. A.2)
Lyle defines the probe as follows:
- Plasticity is P(θ) = b − E_ℓ[ℓ(θ*)], where θ* = O(θ, ℓ).
- ℓ is the MSE to g(x) = a + sin(10^5 · f(x; ω0)). ω0 is a fresh init of the
  same architecture, and a is "the network's mean prediction".
- b is the variance of the targets.
- O is the network's own optimizer, run for 2000 steps (we use 1000).
- X is the transitions in the replay buffer, and there are 10 random target
  functions (we use 5).
- Plasticity loss is P(θt) − P(θ0).
- A1 Inputs: per round, a fixed probe pool of N (s, a) pairs sampled
  uniformly from the filled replay buffer with a dedicated probe RNG.
  Observations are normalised with the current obs_rms, exactly as the critic
  sees them. Each of the 1000 steps trains on a minibatch of 256 drawn from the
  pool, and the final loss is the MSE over the whole pool.
  Options for N: 256 (full-batch), 2,560, or 25,600.
  Rec: 25,600 (100 batches). It is closest to "E over the replay buffer"
  without memorising a tiny set, and it fits on device.
- A2 Target offset a. The Methodology says "same targets" for current and fresh.
  (i) One shared a, the current critic's mean prediction on the pool, used
  for both. The fresh critic must then also move its mean to a; with Adam at
  lr 1e-4 for 1000 steps (≈0.1 per parameter), a Q-scale offset may be
  unreachable. That penalises the fresh reference, so plasticity loss reads
  negative.
  (ii) Each critic uses its own mean prediction as a, with identical inputs,
  ω0 and sin(·) component. The targets then differ only by a constant, have
  the same b, and this is Lyle's own construction.
  (iii) a = 0 for both.
  Rec: (ii). It is a literal deviation from "same targets", so I need your call.
- A3 Target generator architecture: the probed critic's own architecture,
  which is Lyle-literal and fine within a run. The positive-control
  cross-architecture comparison is in E7. Rec: the own architecture.
- A4 Optimizer: AdamW (lr 1e-4, wd 1e-2, betas 0.9/0.999) with freshly
  initialised state for both the current and the fresh copy, or the current
  critic's training AdamW state. Rec: fresh state for both. The fresh params
  have no state of their own, so this keeps the pair symmetric.
- A5 Score and sign. Per round r, P_r = b_r − L_r, where L_r is the final pool
  MSE. The per-round paired loss is Δ_r = P_r(θ0) − P_r(θt) = L_r(θt*) − L_r(θ0*).
  Lyle writes P(θt) − P(θ0), which is negative when plasticity is lost.
  Rec: report PL = P(θ0) − P(θt), so that positive means plasticity lost.
- A6 Per-check summary: IQM over the 5 rounds of the paired Δ_r, or
  IQM(P(θ0)) − IQM(P(θt)). These differ. Rec: IQM of the paired Δ_r as the
  check's plasticity loss, which matches the paired bootstrap. Also store
  IQM P(θt) and IQM P(θ0) as the two "plasticity scores".
- A7 Learning curves: store the per-step minibatch loss (1000 × 5 rounds ×
  2 nets per check) plus the final pool loss, as compressed .npz. Rec: yes.
- A8 Probing an injected critic: train only its trainable params (the new
  trainable head and the earlier blocks), keep the frozen parts frozen, and
  use fresh AdamW state. The plasticity-loss reference for both arms after the
  fork stays the run's stored fresh params. Rec: yes.
- A9 Fresh probe timing: at the first interaction step where the buffer holds
  min_length = 5,000 transitions, immediately before the first update. Store the
  fresh params, and persist them in every checkpoint. Rec: yes.

### B Trigger
- B1 Resamples: 10,000, with a numpy Generator seeded from (run seed, check
  index). Rec: 10,000.
- B2 Statistic on each resample: IQM (= mean of the middle 3 of 5) or mean.
  Rec: IQM, consistent with the point estimate.
- B3 "Distinguishable from zero": the two-sided 95% percentile interval
  [2.5, 97.5] has a lower bound > 0, in the PL-positive convention. Equality
  does not trigger. Rec: yes.
- B4 Caveat (no change proposed). With n = 5 rounds the percentile bootstrap
  is anti-conservative. Under a symmetric null, all 5 rounds share the same sign
  3.1% of the time on one side, and 4-of-5 can also fire. Across 19 checks a
  null run can trigger with substantial probability. The Methodology already
  calls the trigger operational and lists multiple comparisons as a limitation.
  The tests will measure and report the actual false-trigger rate rather than
  claim it is nominal.
- B5 Any non-finite probe loss marks the check invalid. An invalid check never
  triggers and is recorded in the ledger. Rec: yes.

### C Experiment 1 analysis
- C1 rliable aggregate: IQM over the 65 runs. The difference is
  IQM(scaled) − IQM(D2W512), computed with rliable's stratified bootstrap
  (strata = environments) and percentile intervals. rliable's paired-algorithm
  path resamples the two architectures independently.
  Options: IQM or mean. Rec: IQM, with rliable's default 50,000 reps.
- C2 Seed pairing: unpaired (independent) or paired by (env, seed index).
  The same seed index across architectures is not a real pairing: different
  networks, and trajectories diverge after step 1. Rec: unpaired.
- C3 Probe-loss units are comparable across environments (sin targets in
  [−1, 1], b ≈ 0.5), so no per-environment normalisation is needed before
  pooling. Rec: no normalisation.

### D Fork mechanics
- D1 Fork exactly at the check step. The complete state is saved, including
  the train/eval env physics state, the env RNGs and the wrapper and episode
  counters, so mid-episode restore is exact. The alternative is to fork at the
  next episode boundary, which still needs the env RNGs. Rec: exact check step.
  If any env cannot be restored bit-exactly, I come back to you.
- D2 Arms.
  (a) The Exp 1 run continues untouched to 100% of B. At f*_run it writes the
  fork state, and two new jobs (control, injected) restore from it. This is
  symmetric: both arms go through the identical restore path.
  (b) The control is the original run continuing to max(100% B, fork + 25% B).
  Only the injected arm is a new job. This saves up to ~17% compute but is
  asymmetric.
  Rec: (a).
- D3 Follows from D2. Under (a) every Exp 1 run ends at 100% of B, and the
  arms run from the fork to fork + 25% of B (≤ 120%).
- D4 Arms are separate resumable jobs launched from the saved state. Rec: yes.
- D5 Identity fork: tests here, plus a CUDA validation command for each
  architecture × suite before the main grid. It is not run on every real
  fork. Rec: yes.
- D6 Only scaled-architecture runs (D4W1024, D6W1536) fork. D2W512 runs still
  record f*_run in the ledger. Rec: yes.
- D7 Injection RNG: the new-head init key comes from a dedicated stream,
  fold_in(PRNGKey(seed), constant), and never consumes the training key.
  Rec: yes.

### E Injection
- E1 How m is expressed: as the rule label {last, half, all}. For D4 that is
  1/2/4 blocks, for D6 1/3/6, and for D2 last = half = 1. The alternative is
  a fixed block count. Rec: the label, so the frozen choice transfers across
  D4 and D6.
- E2 New head init: the original SimBa initialisers (he_normal block kernels,
  zero biases, LayerNorm scale 1 / bias 0, orthogonal(1) output), with
  θ'1 = θ'2 bit-identical. Rec: yes.
- E3 Target critic: its frozen old head is the target's own current head
  (lagged). Its new copies take the same θ' values as the online new copies,
  so the target prediction is also unchanged. Polyak runs over the whole
  target tree: θ'2 stays constant, θ'1 tracks the online θ'1, and the old head
  keeps tracking the frozen online old head. Rec: yes.
- E4 "Newly created optimizer state": optax adamw shares one step `count`.
  Zero moments under the old count would give the new params bias-corrected
  first steps ≈3× lr. Rec: a separate AdamW instance with its own count for the
  new trainable params, existing params keeping their state including count,
  and frozen params getting zero updates and no weight decay
  (optax.multi_transform).
- E5 "Similar recovery". Recovery_m = (PL_degraded − PL_m) / (PL_degraded − PL_healthy),
  using IQM. Options: (a) the smallest m within 0.10 of the best recovery;
  (b) the smallest m whose recovery bootstrap CI overlaps the best's.
  Rec: (a), as the simple pre-registered rule. Your call.
- E6 Positive-control architecture and environment. Rec: D6W1536 (largest,
  most likely to degrade) on one 1M DMC hard task. Your choice of task.
- E7 "Normally trained critic".
  (a) A D2W512 dev run on the same env, trained to the same step. The
  Methodology calls the default architecture "healthy".
  (b) The same dev run's last check before the trigger.
  Cross-architecture probing needs identical inputs and targets for all four
  critics, so I would generate all targets with one fixed random net of the
  scaled architecture. Rec: (a). Your call.

### F Experiment 2 outcomes
- F1 Return evaluation in the arms. Today it runs every 50,000 interaction
  steps, which gives only 1–5 points in a 25% horizon. Rec: evaluate every B/100
  steps since the fork (25 points), with 10 deterministic episodes each. The
  Exp 1 run keeps its existing schedule. Your call on the cadence and episodes.
- F2 Arm probes at fork + k·B/20, k = 0..5, where k = 0 is Check 2. These
  coincide with the absolute k/20 grid up to 120%. Rec: yes.
- F3 Pooling returns per scaled architecture across environments.
  Options: (a) per environment only; (b) each paired difference divided by that
  env's mean final eval return of the D2W512 confirmatory runs, with MyoSuite
  using success rate (SimBa's metric, already in [0, 1]); (c) SimBa-paper
  reference scores. Rec: (b) for the pooled band, plus (a) per environment.
- F4 A scalar summary endpoint: none unless you ask for one. Rec: none.

### I Actor diagnostics (Schaul 2022; Tang & Berseth 2024; Bjorck et al. ICLR 2022 / arXiv 2110.11222)
Definitions in the papers:
- Schaul: W(π, π') = E_s ½ Σ_a |π(a|s) − π'(a|s)|, a fraction of switched
  greedy actions; this is discrete-action.
- Tang & Berseth: churn on a reference batch drawn from replay, between the
  current network and the past N updates every 1k updates. For SAC they use
  KL(π_t(·|s) ‖ π_−(·|s)); for TD3 the L1 action difference.
- Bjorck: "actor gradient norm ‖∇ℓ‖" and "average absolute value of actions |a|"
  over training.
- I1 Policy churn.
  (a) Keep the existing metric (per-update L2 change of tanh(mean) on the fixed
  first-batch reference). It is cheap and already logged.
  (b) Add Tang's SAC definition: a per-update KL(π_t ‖ π_{t−1}). For
  tanh-Gaussians this equals the closed-form KL of the pre-tanh diagonal
  Gaussians, averaged over a 256-state reference batch from replay. The batch
  is refreshed each logging window by a dedicated RNG that never touches the
  training stream, and the KL is computed inside the scan.
  Rec: (b) in addition to (a). Caveat: (b) adds an output to the compiled
  update, so I will gate it with a static flag and test bit-identity of the
  params with it on and off.
- I2 Actor gradient norm: keep `train/actor_gnorm` (Bjorck's ‖∇ℓ‖) and add the
  per-window std, for "stability", from the per-update values already on the
  host. Rec: yes.
- I3 Saturation: keep `train/actor_action` (Bjorck's mean |a|, on sampled
  actions) and add the fraction of action components with |a| > 0.99. Rec: yes,
  threshold 0.99. Your call on the threshold.
- I4 Diagnostics are needed in the Exp 1 scaled runs before the fork too, for
  the shared-time-axis plot. They are logged in every Exp 1/2 run. Rec: yes.

### N Naming (routine; shown for veto)
- `experiments/exp1.py` (registered "exp1")
- `experiments/exp2_arm.py` ("exp2_arm")
- `experiments/exp12/` (probe, trigger, injection, fork_state, diagnostics, ledger)
- `configs/base_exp12.yaml`
- `analysis/exp1_analysis.py`, `analysis/exp2_analysis.py`
- `scripts/positive_control.py`
- generate_manifest `--grid exp12`
- Confirmatory seeds 1–5; dev seeds 1001+.

### G Hyperparameters and settings not in the Methodology
- G1 Keep all existing settings listed in the audit facts above, and set UTD
  to 2 in a new config. Rec: keep all of them. Specifically:
  - temp0 = 0.01, matching SimBa.
  - Target entropy −|A|/2. SimBa writes |A|/2, but the SAC sign convention is
    negative.
  - Temperature weight decay 0.
  - AdamW betas (0.9, 0.999) with weight decay on all params.
  - Replay buffer 1M, min_length 5,000.
  - Observation normalisation as it is now.
  - Single random first action.
- G2 Discount heuristic episode length.
  (a) Keep max_episode_steps = 1000 → gamma 0.99 everywhere (current repo).
  (b) Use each task's registered horizon / 2, as SimBa does. MyoSuite would get
  KeyTurn 200 → 0.975, PenTwirl 50 → 0.95, Pose 100 → 0.95 and Reach 100 → 0.95;
  DMC stays at 0.99.
  Rec: (b), since the Methodology says "as used in SimBa".
- G3 Fix MyoSuite seeding (pass `seed` to MyoSuite) for the Exp 1/2 code path
  only, leaving the Angle paths untouched. Without it, myoHandPoseRandom-v0
  runs are not reproducible from their seed. Rec: yes.

### H HumanoidBench
- H1 There is no historical code, so it would be new code. Dependencies:
  (a) install HumanoidBench `--no-deps` plus CPU torch into the pinned env, and
  verify it steps and is deterministic;
  (b) a separate environment with HumanoidBench's own pins (gymnasium 0.29.1,
  mujoco 3.1.6), which differ in simulator versions from DMC/MyoSuite;
  (c) drop HumanoidBench.
  Rec: try (a) in a throwaway venv first. It adds torch, so I need your OK.
- H2 "h1-shelf-place" does not exist in HumanoidBench. What should replace it?
  Candidates: h1hand-bookshelf_simple-v0, h1hand-bookshelf_hard-v0, or one of
  SimBa's 14 H1 locomotion tasks. Also, h1-reach-v0 (no hands, SimBa's) or
  h1hand-reach-v0? Rec: h1-reach-v0. Your call on the shelf task.
- H3 Clipped double-Q: SimBa uses it on HumanoidBench, while the Methodology
  says single Q. Rec: single Q everywhere, as the Methodology says.
- H4 Action repeat 2 for HumanoidBench matches SimBa Table 1. Rec: keep.
- H5 HumanoidBench Reach uses the global `np.random` for goals, so it shares
  the stream with replay sampling. The fork already saves that state, and the
  probe never uses the global RNG. No change. FYI.

### R Runtime and process
- R1 The full existing test suite exceeds 10 minutes here (killed at 600 s).
  May I run it in full? Estimate 15–40 min on 4 CPUs.
- R2 Probe overhead forecast: 20 checks × 5 rounds × 2 nets × 1000 steps
  = 200k critic forward/backward steps. Training is ≈ 2 × (B/2) updates, each
  ≈ 2.5 critic forward/backward-equivalents. Against update FLOPs that is
  ≈ 8% for 1M budgets, ≈ 16% for 500k and ≈ 4% for 2M, before env-step time
  dilutes it. This likely exceeds your 5% threshold for D6W1536 on the medium
  tasks. Real numbers NEED CUDA. I will measure what I can and ask before doing
  anything.
- R3 Conflicts with older docs. The Methodology wins for Exp 1/2, and I will
  not edit these files.
  - CLAUDE.md and research-methodology.md list scaled critics D5W768/D7W1024,
    UTD 5, a different 10-env core set, and td_error_var-based degradation.
  - hopper-hop and myoHandPenTwirlFixed are "Angle 3 held-out" there, and
    `validate_dmc_not_heldout` / `validate_myosuite_core4` reject them. The
    Exp 1/2 entry points will not call those validators.
  - The MyoSuite tier labels differ.
  - HumanoidBench is absent from both documents.

## Answers from the project lead (received 2026-10-03)

The pending questions above are resolved as follows. Where an answer differs
from my recommendation, the answer governs.

- Q0: the later version is final. Only that version is saved verbatim in
  `.claude/methodology-exp1-exp2.md`, followed by an "Amendments from the Phase 0
  Q&A" section. Future changes arrive only as explicit messages. CLAUDE.md gets
  a pointer; no other old docs are edited.
- A1, A3, A4, A6, A7, A8: approved as proposed.
- A2: per-critic mean-prediction offset, identical inputs and identical g(·)
  shape. Paper check before implementing: see "Lyle et al. 2023 quotes" below.
  Limitations (i) and (ii) are recorded in the amendments. A one-time
  shared-offset sensitivity check runs on the development run only.
- A5: P is the plasticity score (higher = more plastic) and L = P(fresh) −
  P(current), positive when plasticity is lost. Tests required: current worse
  than fresh gives L > 0 and triggers; the reverse never triggers.
- A9: confirm that 5,000 transitions matches the repo's initial-random-steps
  setting, and ask if it does not. Outcome: it does NOT match, see Q-A9 below
  (2026-10-03).
- B1–B3: 10,000 resamples, IQM, trigger when the 95% lower bound of L > 0.
  B4: measure the false-trigger rate on healthy-run data and report it; never
  call it nominal. B5: a non-finite check is invalid and never triggers.
- C: rliable IQM of (scaled − D2W512) L at the final check, runs unpaired, no
  per-environment normalisation.
- D1: fork at the exact check step with full env state and RNG. Test that a
  mid-episode restore reproduces the original trajectory per suite. If that
  fails for a suite, STOP and ask; never fall back to an episode boundary.
- D2: the original process restarts from the saved state as the CONTROL arm and
  runs to max(100% B, fork + 25% B). The INJECTED arm is a separate resumable
  job from the same save, running to fork + 25% B. Both use the same restore
  path. GATE: the identity fork must pass on CUDA per architecture × suite
  before any Exp 1 grid launch; the project lead runs those commands.
- D3–D7: as proposed. Only D4W1024 and D6W1536 fork.
- E1–E4: approved.
- E5: recovery = (L_trigger − L_injected) / (L_trigger − L_healthy); choose the
  smallest m within 0.10 of the best. Measure probe noise (the spread across the
  5 rounds); if it exceeds 0.10, STOP and ask.
- E6: D6W1536 on dog-run, with a dev seed outside 1–5. If it never triggers,
  STOP and ask; never loosen the trigger.
- E7: the healthy reference is the same critic's own earlier checks in the same
  dev run, before the trigger.
- F1: evaluate every 1% of B since the fork, 10 episodes each; cost it in
  Phase 6. F2: approved.
- F3/F4: no normalisation and no scalar summary. Graphs only: per-environment
  paired differences in raw return with percentile bootstrap bands and every
  seed's line. Save raw per-episode returns (and per-episode success where
  provided). Analysis takes normalize(env, values), defaulting to identity.
  One metadata file per suite holds benchmark constants as data only.
- I1–I4: approved. Logging stays at the existing cadence with no new host syncs.
- G1: keep the existing values with UTD = 2. G2: read SimBa's released code,
  report its per-suite discount, and match it (see below). G3: fix MyoSuite
  seeding in the Exp 1/2 code path only.
- H1: try HumanoidBench plus torch in a throwaway env without changing the main
  pins; STOP if it cannot coexist. Outcome: it cannot (see below). H2: tasks
  are h1-reach-v0 and h1-run-v0. H3: one Q critic, listed as a limitation, with
  a quote of where SimBa uses two.
- R1: run the full existing suite once in the background. R2: accepted; measure
  on CUDA and never reduce the probe.
- N: proposed names approved. Dev seeds lie outside 1–5.

## Lyle et al. 2023 quotes (arXiv 2303.01486, checked 2026-10-03 against the PDF text)

- Offset a, Section 3.1, verbatim: "Given some offset a ∈ R, we will apply the
  transformation g(x) = a + sin(10^5 f(x; ω0)), with ω0 sampled from the same
  distribution as θ0, to construct a challenging prediction objective which
  measures the ability of the network to perturb its predictions in random
  directions sampled effectively uniformly over the input space. Because the
  mean prediction output by a deep RL network tends to evolve away from zero
  over time as the policy improves and the reward propagates through the value
  function, we will set a to be equal to the network's mean prediction in order
  not to bias the objective in favour of random initializations, which have mean
  much closer to zero."
- Appendix A.2 adds nothing about a. It reads, verbatim: "we draw 10 randomly
  sampled target functions generated by the procedure described in Section 2.2,
  and for each run the network's optimizer from the current parameters to
  minimize the loss with respect to these new targets for 2000 steps."
- Verdict: the paper sets a to "the network's mean prediction", where "the
  network" is the one being probed. That matches the per-critic mean, so there is
  nothing to stop on. The paper does not say over which inputs the mean is
  taken. Stated choice: the mean prediction over the round's probe pool,
  computed on the critic before probe training.
- Baseline b, verbatim (Section 3.1, eq. 5): "we set a baseline value b to be
  the loss obtained by some baseline function (e.g. if ℓ is a regression loss on
  some set of targets, we set b to be the variance of the targets), and then
  define plasticity to be the difference between the baseline and the
  expectation of the final loss obtained by this optimization process after
  starting from an initial parameter value θt and optimizing a sampled loss
  function ℓ subtracted from the baseline b. P(θt) = b − E_ℓ∼L[ℓ(θ*_t)] where
  θ*_t = O(θt, ℓ)". Also: "We then define the loss of plasticity over the course
  of a trajectory (θt) as the difference P(θt) − P(θ0). We note that this
  definition of plasticity loss is independent of the value of the baseline b".
- Correction to the brief's belief: b is subtracted, not divided. There is no
  ratio normalisation. With per-critic offsets the target variance is identical
  for current and fresh, so b cancels in L exactly as the paper notes.
  Implemented as P = b − final_loss.

## SimBa released code (github.com/SonyResearch/simba @ 7d0358b, read 2026-10-03)

- Discount: `configs/base.yaml` computes gamma = max(min((L/5 − 1)/(L/5), 0.995),
  0.95), with L = max_episode_steps / action_repeat.
  - DMC: `configs/env/dmc_hard.yaml` and `dmc_em.yaml` set max_episode_steps
    1000, so gamma = 0.99.
  - MyoSuite: `configs/env/myosuite.yaml` sets max_episode_steps 100 for every
    MyoSuite task, so gamma = 0.95.
  - HumanoidBench: `configs/env/hb_locomotion.yaml` sets max_episode_steps 1000,
    so gamma = 0.99.
- Two Q critics on HumanoidBench (for H3). `configs/env/hb_locomotion.yaml` sets
  `episodic: true`, and `configs/agent/sac_simba.yaml` sets
  `critic_use_cdq: ${env.episodic}`. The paper's Table 7 says "Clipped Double
  Q: HumanoidBench: True, Other Envs: False". Exp 1/2 keeps one Q critic, as the
  Methodology says. This is a listed limitation.
- Random actions: SimBa's `run.py` overwrites actions with
  `train_env.action_space.sample()` while `buffer.can_sample() is False`, i.e.
  for the first min_length = 5,000 transitions. It still calls
  `agent.sample_actions` first so obs_rms statistics are updated. This repo's
  `experiments/angle_1.py` takes a random action only on interaction step 1.
- HumanoidBench task IDs: SimBa's `scale_rl/envs/humanoid_bench.py` lists the
  `h1hand-*` variants (e.g. "h1hand-run-v0", "h1hand-reach-v0"). The paper's
  Table 5, however, gives action dim 19, which is the H1 body without hands.

## HumanoidBench coexistence test (H1, 2026-10-03)

Setup: a throwaway copy of the venv with the pins unchanged, HumanoidBench @
cb11890 installed `--no-deps` (editable), torch 2.3.1+cpu, and OSMesa for
headless rendering.

Result: `gym.make` fails for h1-reach-v0, h1-run-v0, h1hand-reach-v0 and
h1hand-run-v0 with `AttributeError: 'mujoco._structs.MjModel' object has no
attribute 'flex_xvert0'`. The failure is in
`humanoid_bench/dmc_deps/dmc_index.py:630`, a vendored copy of dm_control's
indexer written for mujoco 3.1.6. It cannot coexist with the pinned
mujoco 3.6.0. STOPPED; see Q-H1b.

## New questions (2026-10-03), PENDING

- Q-A9 (initial random actions). The repo takes a uniform-random action only on
  interaction step 1, and the untrained actor acts from step 2. SimBa's released
  code takes uniform-random actions until the buffer holds 5,000 transitions.
  The fresh probe "after the initial replay buffer has been populated" happens
  at 5,000 transitions either way. What differs is who filled the buffer.
  Options:
  (a) keep the repo behaviour (G1: "keep all existing values");
  (b) match SimBa's random actions for the first 5,000 transitions, in the
      Exp 1/2 path only.
  Rec: (b), because the brief says the hyperparameters come from SimBa and
  (a) looks like an accidental change. Your call.
- Q-G2 (MyoSuite horizon). SimBa gets gamma = 0.95 by setting
  max_episode_steps = 100, which also applies TimeLimit(100). For
  myoHandKeyTurnFixed-v0, whose registered horizon is 200, that truncates
  episodes at 100 raw steps. Pen (50), Pose (100) and Reach (100) are
  unaffected because their own limit binds first.
  Options:
  (a) match SimBa exactly with max_episode_steps = 100 for MyoSuite, giving
      gamma 0.95 and TimeLimit 100 for all four tasks;
  (b) gamma 0.95 but keep the TimeLimit at 1000, so KeyTurn runs to 200.
  Rec: (a), an exact SimBa match.
- Q-H1b (HumanoidBench cannot coexist). Options:
  (a) patch HumanoidBench's vendored `dmc_deps` for mujoco 3.6 and verify
      physics against the reference;
  (b) a separate environment with HumanoidBench's pins (mujoco 3.1.6,
      gymnasium 0.29.1, dm_control 1.0.20), with the repo code verified under
      it;
  (c) drop HumanoidBench from Exp 1/2.
  No recommendation without your input. (a) modifies third-party code.
  (b) runs different simulator versions per suite. (c) changes the env set.
- Q-H2b (no-hands premise). SimBa's released code lists `h1hand-*` IDs, but its
  paper's Table 5 dims (act 19) match the no-hands H1. Keep h1-reach-v0 and
  h1-run-v0 as decided? Rec: keep your decision; FYI only.

## Phase 1–2 engineering choices (routine; cannot change results) and findings (2026-10-03/04)

- The exp1 loop (`experiments/exp12/trainer.py`) repeats angle_1's per-step body in
  the same RNG-consumption order. The Angle1ParityTest shows identical params,
  optimizer state, JAX key, obs_rms and every logged metric row.
- Exact env restore: `ExactRestore` records the simulator RNG state just before
  each reset plus the actions taken since. A restore resets with that RNG and
  replays the actions. Verified bit-exact in a new process for all 11
  DMC/MyoSuite envs, across episode resets.
- Resume also persists SACAgent's per-window host buffers (actor_loss / entropy /
  churn). These are missing from angle_1's resume path (a pre-existing gap, left
  untouched there). Without them, the logged window containing the resume point
  differs.
- State directories are named `step_<n>_<uid>`. A directory name is never
  reused, because tensorstore/Orbax caches by path within a process; reusing a
  name caused NOT_FOUND restores in tests. `LATEST` is repointed atomically, and
  older or orphaned states are deleted after the commit.
- The fresh critic is saved once per run to `<run_dir>/fresh_critic` (Orbax),
  not into every state, because it is 454 MB for D6W1536.
- Probe RNG: the JAX key is fold_in(PRNGKey(seed), "PROB", check, round). The
  pool sampler is np.random.default_rng([seed, "PROB", check, round]). Neither
  consumes any training stream.
- Probe rounds run as a Python loop over one jitted fit, compiled once per
  (critic architecture, optimizer). Current and fresh are separate calls of the
  same compiled function, sharing the pool, target function and minibatch
  indices, with one host transfer per check. Whether rounds should instead be
  vmapped for speed is a CUDA measurement question (NEEDS CUDA).
- Pool and final-loss evaluation are chunked (2,560 rows per chunk) to bound
  activation memory for D6W1536.
- A guard rejects configs whose first check falls at or before
  buffer.min_length (impossible in real budgets: 12,500 > 5,000).
- Existing test suite (R1): the single run was OOM-killed by the 15 GB container
  memory cgroup after 126 tests, all passing. One process reached 13.8 GB while
  running concurrently with the Exp 1/2 tests. It will be re-run alone, one
  module per process.

## Phase 2 gate evidence (2026-10-04, CPU only)

- Existing suite, re-run alone with one module per process: 258 tests, 257 OK.
  The single failure is the known pre-existing
  `test_angle2a_env_state_determinism_smoke.TestMyosuiteDeterminism.test_myo_baoding_p1`.
  No new failures.
- Exp 1/2 tests: 7 Phase 1 and 9 Phase 2 tests pass. The break checks
  (`tests/exp12_break_checks.py`) give 6 out of 6 mutations detected and
  passing again once restored.
- Tiny-run demo: hopper-hop, critic D1W64, 20k interaction steps, real probe
  settings (5 rounds × 1000 steps, pool 25,600). All 21 probes ran, and one check
  took a median of 17 s on CPU.
  - Finding for the lead (no change made): by check 20 both critics' probe
    losses plateau near b ≈ 0.5, so P ≈ 0.008–0.009. The probe's dynamic range
    is compressed for this small critic. Its range at the real critic sizes must
    be checked on CUDA.
  - L stayed negative at every check (current more plastic than fresh), so the
    trigger would never have fired in this run.
- Profiler (CPU, D2W512, humanoid-run): 4.4 training it/s, 413 s per probe
  check, projected probe overhead 7.1% of wall-clock over a 500k-step run, and
  exp1/angle_1 wall-time ratio 0.96 with probes off. All of these NEED CUDA
  numbers; the CPU ratio is not representative.

## Answers to the Phase 2 gate (received 2026-10-04)

- 1, probe overhead: proceed with the probe unchanged, and measure it on CUDA
  (docs/exp12_cuda_commands.md §2).
- 2, dynamic range: add a fresh-critic range check to the CUDA plan, covering
  D2W512, D4W1024 and D6W1536 with real settings. Report the fresh score, the
  round spread, and the score at pools {1,600, 6,400, 25,600}, using fresh
  critics only. Nothing changes until the lead decides. Implemented as
  `scripts/probe_fresh_checks.py --mode range` (§3).
- 3, A9: match SimBa (random actions until 5,000 transitions) in the Exp 1/2
  code only. Implemented in `Exp12Trainer.train`; amendment (h).
- 4, G2: match SimBa exactly (MyoSuite max_episode_steps = 100, so gamma 0.95),
  with KeyTurn's truncation added to the limitations. Implemented as
  `configs/env/myosuite_simba.yaml`; amendment (i).
- 5, HumanoidBench: time-boxed. First try a patch so that one stack serves all
  suites; if that fails, try one shared mujoco version in a throwaway env.
  Report before creating a second env, and never drop HumanoidBench without
  asking.

## HumanoidBench outcome (2026-10-04): one stack serves all suites, no second environment

- Approach: no HumanoidBench file is edited. `experiments/exp12/envs.py` imports
  HumanoidBench with two `sys.modules` entries.
  - Its bundled `dmc_deps/dmc_index.py` is a copy of dm_control's
    `mujoco/index.py` whose size table is frozen at mujoco 3.1.6. It is
    pointed at the installed `dm_control.mujoco.index` (dm_control 1.0.38),
    the same module with a current table.
  - The torch-only `mjx/flax_to_torch` module, used only by HumanoidBench's
    hierarchical policy wrappers and not by h1-reach / h1-run, is replaced
    by a stub that raises if called. So torch is not needed.
- Install: `scripts/install_humanoid_bench.sh`, which installs carlosferrazza/humanoid-bench @
  cb1189039151c8aadaaa987b442da54383c87fab with `--no-deps -e`. An editable
  install is required because `dmc_deps/` and the assets are not packaged. No
  pinned dependency changes.
- Verified here (CPU, throwaway copy of the pinned venv plus HumanoidBench):
  - Both tasks register and step. h1-reach-v0 has obs 57 and act 19;
    h1-run-v0 has obs 51 and act 19. Both match SimBa's paper Table 5, which
    confirms the no-hands H1 matches the paper. SimBa's released code lists the
    `h1hand-*` variants instead.
  - Phase 1 restore tests pass with HumanoidBench included: the cross-process
    mid-episode restore is bit-exact for both tasks, and the trainer's
    save/restore/resume on h1-reach-v0 is bit-exact.
  - Break evidence covers four cases: env RNG not restored, global np.random
    not restored (the Reach goal), and actions not replayed are all detected,
    and the intact restore passes.
- Randomness: HumanoidBench uses the env's own `np_random` for the initial-state
  noise, and Reach draws its goal from the global `np.random` at reset. That
  global stream is shared with replay sampling, which is SimBa's behaviour too.
  ExactRestore records both RNG states before each reset. Exp12Trainer restores
  the envs before the global RNG, so the order is correct.
- Rendering: HumanoidBench builds an offscreen renderer at construction, so runs
  need `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` on GPU nodes (osmesa on CPU). The
  factory raises a clear error otherwise.
- Limitation: the physics run on mujoco 3.6.0 rather than HumanoidBench's pinned
  3.1.6, the same engine version as the DMC and MyoSuite runs. Results are
  therefore not bit-comparable to SimBa's HumanoidBench numbers.

## Phase 3 findings (2026-10-04)

- rliable 1.2.0 defect: `StratifiedIndependentBootstrap._get_indices` draws with
  the global `np.random.choice` and ignores `random_state`. The paired-algorithm
  intervals were therefore not reproducible: the same input twice gave different
  CIs in the 3rd decimal place. Fix in `analysis/exp1_analysis.py`: the global
  stream is seeded immediately before every rliable call. This is offline
  analysis, and a test now checks reproducibility.
- Trigger false-trigger rate under a synthetic null (B4, measured, not
  nominal). Five rounds of symmetric N(0, 1) per-round losses and 10,000
  resamples give a per-check rate of 4.93% over 4,000 null checks, about twice
  the one-sided nominal 2.5%. That is the anti-conservatism of a percentile
  bootstrap with n = 5. If the 19 eligible checks were independent, a null run
  would trigger at least once with probability ≈ 0.62. Real checks are
  correlated (the same critic evolves), so the run-level rate is lower than
  that, but likely substantial. The empirical rate on fresh-vs-fresh critics
  with real probes is §4 of the CUDA plan. No change made.
- Statistics written down (as decided): the bootstrap uses np.random.default_rng([seed, "BOOT",
  check]), 10,000 resamples and the IQM (scipy trim_mean 0.25), with
  numpy-percentile 2.5/97.5. A check triggers on a lower bound > 0 strictly.
  f*_run is the first triggering check among checks 1..19.
- Exp 1 primary endpoint: rliable `get_interval_estimates` with a tuple input
  (scaled, default), meaning stratified (by environment) independent resampling,
  percentile method, 50,000 reps, statistic IQM(scaled) − IQM(D2W512) of the
  final-check L. A missing (seed, environment) cell is refused rather than
  imputed.
- Ledger layout (routine): one directory per run under
  `results/exp12/exp1/runs/<run_key>/` holding run.csv, checks.csv and
  probe_curves.npz. Every save rewrites it atomically, so concurrent jobs never
  share a file.
- Dependencies: rliable==1.2.0 with its transitive deps pinned in requirements.txt
  (arch 7.2.0, statsmodels 0.15.0, seaborn 0.13.2, patsy 1.0.3, formulaic 1.2.2,
  interface_meta 2.0.1, narwhals 2.26.0). No existing pin changes, and
  `pip check` is clean.

## Pre-specified rules for the CUDA calibration checks (logged 2026-10-04, BEFORE any CUDA result exists)

Decided by the project lead with the Phase 3 approval. This entry is committed
before any range, null or profile result exists, and must not be edited after
results arrive. Later changes are appended as new, dated entries.

- RANGE rule (`scripts/probe_fresh_checks.py --mode range`, real settings,
  pool 25,600). PASS if, at all three critic sizes (D2W512, D4W1024, D6W1536),
  the fresh critic's score P lies between 10% and 90% of the target baseline b,
  i.e. 0.1·b ≤ P ≤ 0.9·b, using the IQM over the 5 rounds of P and of b. The
  round-to-round spread (std and range of P over the 5 rounds) is always
  reported. The script prints the PASS/FAIL verdict per size.
- RANGE fallback ladder if the rule fails. Step 1 is a smaller probe pool (the
  range run already reports 6,400 and 1,600). Step 2 is more probe steps. Each
  step is proposed to the lead, not applied. Anything else, such as a different
  learning rate or target, needs asking first.
- NULL rule (`--mode null`, fresh-pair null, at least 100 pairs per size). If the
  per-check fire rate on fresh pairs exceeds 5% at any size, the lead will
  decide whether to adopt a null-calibrated threshold: the 95th percentile of L
  between two independent fresh critics at that size. Claude does NOT adopt it.
  The script only reports the rate and the would-be threshold, and saves every
  per-round loss, IQM and interval so alternative rules can be evaluated
  offline.
- CONSECUTIVE checks (require 2 consecutive firing checks): UNDECIDED. The
  lead's message carried the unresolved placeholder "[YES/NO]". The current
  rule (one firing check) stays until the lead confirms. The count is a config
  value (`trigger.consecutive_checks`, currently 1). If it is ever set to c > 1,
  f*_run is the check that completes the first run of c consecutive firing
  checks, and that check must itself be at or before 95% of the budget.
- All trigger settings are config values in `configs/base_exp12.yaml` under
  `trigger:`, with the current values unchanged: resamples 10,000, confidence
  0.95, consecutive_checks 1, null_threshold 0.0 (fire when the lower bound is
  > null_threshold) and eligible_fraction 0.95.

## Phase 4 (2026-10-04): fork, injection, Checks 1-3, identity fork, positive control

### What was built
- `experiments/exp12/injection.py`. Head = last m blocks + post-LN + output.
  Q = old(z) + (new(z) − copy(z)). The frozen old/copy heads get
  stop_gradient on their parameters only, so gradients still reach the trunk
  through them. The trunk keeps its AdamW state and count; the new head gets
  a fresh AdamW state and its own count; frozen parameters get zero updates
  and no weight decay (E3/E4). The target critic is injected the same way.
  m labels: last = 1, half = max(1, D // 2), all = D. half = 1 for D2, 2 for
  D4, 3 for D6.
- `experiments/exp12/fork.py`, the fork in `experiments/exp1.py`, and
  `experiments/exp2_arm.py` (D2). At f*_run the complete state is saved. The
  original process restarts from that saved state as the control and runs to
  max(N, fork + 0.25N). The injected arm is a separate, resumable `exp2_arm`
  job running to fork + 0.25N. Both go through the same restore path.
- Post-fork evaluations every 1% of B (26 points × 10 episodes, F1) run on a
  fresh env seeded from (seed, eval index) with a dedicated JAX key, and
  restore the agent key and the numpy and Python global RNG states
  afterwards. Tests show they cannot change training (the control equals a
  fork-disabled run bit for bit). Post-fork probes continue on the same k/20
  grid past check 20.
- Records: `results/exp12/exp2/<run_key>/{fork,check1_<arm>,check2}.json`
  and `arm_<arm>/{checks,eval_episodes,metrics}.csv`, with raw per-episode
  return, length and success (F3/F4).
- `scripts/compare_identity_fork.py` and `testing.stop_after_identity_snapshot`
  (validation only; dev only).
- `experiments/exp12/m_selection.py` and `scripts/positive_control.py`, for
  amendments (c)–(f).

### Choices made (routine, shown for veto)
- Identity fork "1,000 training steps" is read as 1,000 interaction steps
  (= 2,000 gradient updates at UTD 2), via `fork.identity_snapshot_steps=1000`.
- The structural metrics of an injected critic (the encoder intermediates
  in `sac_update.get_critic_with_metrics`) are NaN in the injected arm's
  metrics.csv. The injected critic has no single encoder output, and the
  normal path is unchanged. Q values, TD loss and actor metrics are
  unaffected. (The I1–I4 actor diagnostics are Phase 5.)
- The fork panel is 256 replay (s, a) pairs from a dedicated stream, drawn
  once at the fork and shared by Check 1 for all arms.
- The positive control's injected critics use the same injection key as the
  injected arm (`injection_key(seed)`), so they are the critics the arm
  would start from. Its probes reuse the trigger check's own pool, targets
  and minibatch order, so the degraded critic reproduces the recorded
  trigger probe exactly (asserted on CPU: max |diff| = 0).
- The shared-offset sensitivity check (amendment (c)) reports three versions
  of the single offset: the degraded critic's mean, the fresh critic's mean,
  and their average. For each it reports whether the sign of L agrees with
  the per-critic-offset L. It is reported only and changes nothing.

### Open questions for the project lead (nothing below is applied)
- P4-1 Check 1 tolerance. Q is bit-identical after injection. dQ/da is not:
  reverse mode adds the three heads' cotangents in a different order. The
  measured gap on CPU is up to 7.2 float32 eps × max|dQ/da| at D6W1536, and
  0.83 eps in the tiny smoke run. Proposed:
  pass if both |ΔQ| and |Δ dQ/da| ≤ 64 · eps_f32 · the pre-injection max
  magnitude (`checks.check1_tolerance_eps: 64`). The control and identity
  arms must still be bit-identical; this tolerance applies only to the
  injected arm. Approve or change it.
- P4-2 What counts as "Check 2 passed". Proposed: the 95% percentile
  bootstrap interval of IQM over rounds of P(injected) − P(control), on the
  fork check's own pool, lies above 0. It is reported in check2.json and
  never excludes a fork. Approve or change it.
- P4-3 Which L is the healthy reference. "The same critic's own earlier
  checks before it triggers" (f) can mean (a) `last_pre_trigger`: L at check
  f* − 1; or (b) `iqm_pre_trigger`: the IQM of all per-round L of checks
  1..f* − 1. The script requires `--healthy_reference`, has no default, and
  stops if f* is check 1.
- P4-4 How to measure the probe's own noise against 0.10. Implemented as the
  spread over the 5 rounds of each probed critic's L (the degraded critic and
  the three injected ones), divided by L_trigger − L_healthy so that it is in
  recovery units. The largest of these is compared to 0.10. Spread is either
  (a) `range`: max − min, or (b) `std`: sample std. The script requires
  `--noise_statistic`, has no default. Also confirm that normalising by
  L_trigger − L_healthy is the intended reading of "exceeds 0.10".
- P4-5 Identity-fork coverage. The CUDA commands cover D4W1024 and D6W1536 ×
  one environment per suite (dog-run, myo-key-turn, h1-run-v0). The CPU
  restore tests already covered all 13 environments. Is one environment per
  suite enough?
- Still open from Phase 3: consecutive checks (YES/NO).

### Bugs found and fixed by the Phase 4 tests (2026-10-04)
- Resume right after the fork. The kill test failed when the control was
  killed at step 101, after the fork and before its first save. The resumed
  control read its plan back from fork.json, which also holds two directory
  paths, so its `extra_state["fork"]` differed from an uninterrupted run.
  `fork.read_fork` now returns the plan only. All three kill points and the
  arm kill point resume bit-identically.
- Post-fork evaluation pairing on HumanoidBench. Reach draws its goal from
  the global `np.random` at reset, so a post-fork evaluation's goals depended
  on the training process's global RNG state at that moment. Training was
  never affected (the state was restored), but injected and control
  evaluations were not guaranteed to face the same goals. The global numpy
  and Python RNGs are now seeded from (seed, eval index) inside the
  evaluation and restored afterwards. Checked on h1-reach-v0 (venv_hb): two
  different training RNG states give identical eval returns, and the
  training state is unchanged.

## Answers to Phase 4 (received 2026-10-04) and how they are implemented

1. Check 1. A 64 eps relative tolerance for the injected arm only. The
   control must match bit-exactly. Report the observed maximum and the GPU
   matmul precision. If TF32 is on, or any GPU deviation exceeds 64 eps,
   stop and ask.
   Implemented in `fork.check1(..., injected, precision)`:
   - pre vs control: tolerance 0. Pre vs after and after vs control: 64 eps
     only when the arm is injected; an identity arm also needs 0.
   - `max_eps_units` records the observed maximum.
   - `matmul_precision_report()` records the configured precision
     (jax_default_matmul_precision, NVIDIA_TF32_OVERRIDE, XLA_FLAGS), the
     device, and a measured float32 matmul error. TF32 is "detected" when the
     relative error exceeds 1e-4 (float32 gives about 1e-6 here, TF32 about
     1e-3).
   - On failure, the arm writes check1_<arm>.json and stops (Check1Failed)
     before its first save.
2. Check 2. Approved. check2.json reports the paired difference
   P(injected) − P(control) per round, its IQM and its 95% interval, and pass
   = interval above 0. The 5-round bootstrap caveat is amendment (n).
3. Healthy reference. The IQM of all pre-trigger checks. The last
   pre-trigger check is reported as a sensitivity only. Exit 3 if
   L_trigger − L_healthy ≤ 0 or below the probe noise.
4. Probe noise. The pooled SD of per-round L over all probe evaluations in
   the script, divided by L_trigger − L_healthy. Stop if ≥ 0.10.
   Implemented: the pooled within-series SD over the 7 L series the script
   probes (degraded, injected last/half/all, and the 3 shared-offset repeats
   of the degraded critic). `noise_sd_main_probe_only` (the 4 main series)
   is also reported.
5. Identity fork. D4W1024 and D6W1536 in each suite, on the GPU model the
   grid will use. In Phase 6, both arms of a fork go on the same GPU model.
   Implemented now: fork.json records the device model. The control (on
   resume) and the arm jobs refuse to run on a different model
   (`fork.check_same_device`), and every launch records its device in
   run_metadata.json.
6. Consecutive checks: 2. This is amendment (l), with
   `trigger.consecutive_checks: 2`. The test-only hook
   `testing.force_trigger_check=k` now forces f*_run = k: checks k−1 and k
   fire, and k must be ≥ 2. The CUDA identity commands force check 2.
7. HumanoidBench Reach evaluation seeding.
   `HumanoidBenchReachEvalSeedingTest` runs on a real h1-reach-v0 env in
   venv_hb. It asserts that Reach reads np.random at reset (so the test is
   not vacuous). It then checks that two different training RNG states give
   identical evaluation returns, and that the numpy and Python global RNGs
   and the agent key are restored. Break and restore (run by hand in
   venv_hb): removing the per-evaluation seeding fails the test, removing
   the numpy restore fails the test, and the restored code passes.

### Questions raised by these answers (PENDING; my working reading is in place until you answer)
- P5-Q1. With 2 consecutive checks, check f*−1 has itself fired. "All
  pre-trigger checks" can mean (a) checks 1..f*−1, which is the literal
  reading and is implemented, so the healthy reference includes a firing
  check; or (b) checks 1..f*−2, which leaves out both checks of the
  triggering pair. Which do you want?
- P5-Q2. "All probe evaluations in the script" is implemented to include the
  3 shared-offset repeats of the degraded critic. These use a different
  offset by design. Pooling only the 4 main series is the alternative, and
  is reported alongside. Which should decide the stop?

## Phase 5 (2026-10-04): actor diagnostics I1–I4 and the Experiment 2 analysis

### Actor diagnostics (experiments/exp12/diagnostics.py; approved definitions I1–I4)
- I1, `train/policy_kl`. The per-update KL(π_t ‖ π_{t−1}) of the pre-tanh
  diagonal Gaussians (equal to the tanh-squashed KL, since tanh is a
  bijection), averaged over a 256-state reference batch from replay. It is
  computed inside the fused scan and logged as the window mean. The
  existing `train/policy_churn` / `train/churn` are kept unchanged.
- I2. `train/actor_gnorm` is kept. `train/actor_gnorm_std` is the
  population SD (ddof 0) of the per-update norms in the logging window. The
  values come from the host copy that the existing flush already makes, so
  there is no new device sync.
- I3. `train/actor_action` is kept. `train/actor_saturation` is the fraction
  of the actor-update sampled action components with |a| > 0.99. It uses the
  same sampled actions, so no RNG is consumed.
- I4. On in every Exp 1/2 run (`diagnostics.enabled: true`). The Angle 1 path
  passes neither argument and is unchanged. Tests: the update is bit-identical
  with the diagnostics on or off; Angle 1 parity still holds bit for bit, and
  exp1 adds exactly the three columns.

Routine choices (shown for veto):
- The KL reference batch is redrawn at the start of each logging window
  from a dedicated stream (seed, "KLRF", window). It is normalised with the
  obs statistics at that moment and then held fixed, so the KL reflects
  parameter updates only, not the normaliser's drift.
- The reference batch and the open window's gnorm values are saved with
  the training state, so resume stays bit-exact (verified by the resume and
  kill tests).
- The Exp 1 per-run ledger gains `metrics.csv`, the training metrics per
  logging window, for the shared-time-axis plot. Existing files are
  unchanged.

Observation (CPU, tiny D1W8 actor). `train/policy_kl` came out at about
10–20 nats per update, while the L2 churn was about 1e-3. A hand-computed
closed form confirms the value (11.0385 vs 11.0377 logged). It is large
because some action dimensions have σ ≈ 6e-5 (log_std_min = −10), and KL is
very sensitive to near-deterministic dimensions. Expect heavy-tailed values;
it is reported as specified.

### Experiment 2 analysis (analysis/exp2_analysis.py)
- Primary: per architecture and environment, the paired difference
  (injected − control) of each evaluation's mean raw return, aligned by steps
  since fork. Every seed's line is drawn, with a 95% percentile bootstrap
  band over seeds; the same resampled seed sets are used at every point.
  Forks enter only when both arms have all 26 evaluations.
- Also produced:
  - both arms' post-fork plasticity loss and actor diagnostics;
  - the Check 1 and Check 2 tables;
  - the shared-time-axis plot of each scaled run (L, KL, gnorm, gnorm SD,
    saturation, |a|, eval return; f*_run dashed);
  - the SECONDARY graphs restricted to forks whose Check 2 passed.
- No normalisation: `normalize(env, values)` defaults to the identity and
  is applied per environment to per-episode returns. There is no scalar
  summary. Dev runs and the identity arm are excluded.
- `configs/suite_metadata/{dmc,myosuite,humanoid_bench}.yaml` record
  benchmark constants as data only, each with its source:
  - DMC returns lie in [0, 1000].
  - HumanoidBench success bars: h1-reach 12000, h1-run 700 (Run inherits
    Walk's bar).
  - MyoSuite's registered horizons, and that it defines no return bound.

### Questions raised in Phase 5 (PENDING)
- P5-Q3. Which statistic over seeds goes in the Exp 2 bands, mean or IQM?
  With at most 5 seeds per environment, the IQM averages the middle values
  only. `--statistic` is a required argument with no default until you
  decide.
- P5-Q4. Observed while recording the MyoSuite horizons: the registered
  horizon of myoHandPenTwirlFixed-v0 is 50 raw steps, below our
  max_episode_steps of 100. The registry's own TimeLimit therefore ends
  PenTwirl episodes at 50 raw steps (25 interaction steps), while γ = 0.95
  is derived from 100. Verified: the env's wrapper chain has the registry's
  TimeLimit(50) inside our TimeLimit(100). Amendment (i) noted only KeyTurn's
  truncation. Should
  this go into the limitations, or should something change?

## Phase 6 (2026-10-04): manifests, launch plumbing, compute profiling

### What was built
- `generate_manifest.py --grid exp12 --ckpt-root ABS --results-root ABS
  [--injection-m m]`. The Angle 1 default is unchanged.
  - `exp12_exp1_jobs.txt`: the 195 Exp 1 jobs (3 critics × 13 envs × seeds
    1–5), absolute checkpoint and log paths.
  - `exp2_arms_<device>.txt`: one injected-arm job per completed fork of a
    D4W1024/D6W1536 run, grouped by the GPU model recorded in fork.json.
    Arm jobs are written only once `--injection-m` is given, i.e. after m is
    frozen; until then they are listed as waiting.
  - Re-running is safe: DONE runs are skipped, and every other started run
    resumes.
- `scripts/claim_launcher.py --phase-files ...`: optional. The default (the
  Angle 1 phase files) is unchanged.
- `scripts/preflight_checkpoint_check.py --experiment exp1 [--with-fork]`:
  a dev-role smoke run (tiny probe settings, 4,000 env steps). It must leave
  a complete state and a ledger. With --with-fork it must also fork, run the
  injected arm to DONE, and pass Check 1 on that device.
- `scripts/profile_exp12.py`, new options:
  - `--diagnostics_off`: the I1–I4 overhead;
  - `--fork_timing`: complete-state save and restore with the buffer at a
    95%-of-budget fork;
  - `--eval_cost`: the F1 cost;
  - also `recommended_jobs_per_gpu_upper_bound`.
- Tests (`tests/test_exp12_manifest.py`): the 195-run grid with unique,
  absolute paths, every job's composed config and budget, DONE/resume
  classification, arm jobs (forks only, per device, config identical to
  the parent except the 3 arm keys), no checkpoint overlap, the claim
  launcher dry run, and the preflight PASS and FAIL cases.

### Choices (routine, shown for veto)
- Checkpoint interval = N/20 interaction steps, one save right after each
  probe check: 12,500 for swimmer and hopper, 25,000 for 1M-step tasks,
  50,000 for HumanoidBench. A crash loses at most 5% of a run.
  `checkpoint_start_frac 0.0`.
- Layout: `<ckpt-root>/exp1/<arch>/<env>/seed_<s>`,
  `<ckpt-root>/exp2_arm/<arch>/<env>/seed_<s>/injected`, and
  `<ckpt-root>/logs/`.
- The scaled Exp 1 jobs must stay on the grid's GPU model: after their fork
  they are the control arm, and a control resumed on another model refuses
  to run (amendment (p)).

### Measured here (CPU only; NEEDS CUDA VERIFICATION for the GPU numbers)
- Complete state at fork size, D6W1536 on dog-run: 2.62 GB with a
  475,000-transition buffer; save 17.4 s, restore 18.4 s (local disk).
  D2W512 on hopper-hop: 99 MB, 1.0 s / 0.8 s.
- One post-fork evaluation (10 episodes, actor D1W128, CPU):
  - dog-run 41.1 s, i.e. 0.30 h per arm for 26 evaluations;
  - hopper-hop 7.3 s;
  - myo-key-turn 2.8 s and myo-pen-twirl 1.7 s;
  - h1-run-v0 3.4 s.
  The untrained policy ends MyoSuite and HumanoidBench episodes early, so
  those three are lower bounds; full-length h1-run episodes would take about
  49 s. As a share of the arm's training time it needs the GPU it/s from
  section 4.
- Diagnostics overhead on CPU, D2W512, 60 steps: within noise (−1.2%). The
  real number comes from section 4.

## Answers to the Phase 5-6 questions (received 2026-10-04)

All are recorded as Methodology amendments (q)–(u).
- P5-Q1 (replaces amendment (f), the healthy-reference part of (o), and
  decision E7). The healthy reference is check 0, the fresh critic. L_healthy
  = 0, and recovery = (L_trigger − L_injected) / L_trigger. The pre-trigger
  IQM reference, its last-check sensitivity, and all f−1 / f−2 logic are
  removed from `m_selection.py`, `scripts/positive_control.py`, their tests
  and the break-and-restore mutations. Exit 3 if L_trigger ≤ 0, or if the
  noise in recovery units is ≥ 0.10. The positive-control text now reads
  "recovers toward the fresh critic's level".
- P5-Q2. Noise = the pooled SD of per-round L over the four real-settings
  evaluations (degraded and injected last/half/all), divided by L_trigger.
  The shared-offset repeats are excluded from the noise and the stop, and
  are reported separately. The "4-series-only" report is dropped.
- P5-Q3. `analysis/exp2_analysis.py --statistic` defaults to `iqm`
  (amendment (r)).
- P5-Q4. Compared against SimBa's released code (quotes in amendment (s)).
  - The MyoSuite time limit and discount match: the registered limit stays
    inside the TimeLimit(100), and gamma is 0.95.
  - Truncation handling matches: the final observation is stored, and the
    target uses (1 − terminated) only, so truncation is bootstrapped.
  Nothing changed; our buffer `add`, `RepeatAction`, wrapper order and
  critic target match SimBa line for line. The 50-step PenTwirl behaviour is
  recorded as a limitation. New tests
  (`DiscountAndHorizonTest.test_pen_twirl_keeps_its_registered_50_step_limit_as_in_simba`
  and `test_truncation_bootstraps_and_termination_does_not`) each have
  break-and-restore mutations.
- Checkpoints. One save per probe check, as before.
  - Already retained: only the latest routine state (`commit_state_dir`
    repoints LATEST atomically, then deletes older `step_*` directories),
    plus the fork state, plus the fresh critic. Nothing changed.
  - New kill-matrix case: a crash in the middle of a routine save, after the
    new state is written and before LATEST is repointed, leaves the previous
    state as LATEST and loadable, and the run resumes bit-exactly.
  - Retained size per run (worst case: dog-run, the largest observation;
    the fork at 95% of B; buffer capped at 1M transitions). The trained agent
    ≈ 16 bytes per critic parameter (critic, target, two Adam moments; fresh
    agents compress to less). Buffers are float32, 1,948 B per dog-run
    transition. Check: the D6W1536 dog-run fork state is estimated at
    2.75 GB and measured at 2.62 GB.
    - D2W512: 1.06 GB (no fork).
    - D4W1024: 3.32 GB run directory (final state + fork state + fresh
      critic) + 1.71 GB injected-arm directory = 5.03 GB.
    - D6W1536: 6.19 GB + 2.99 GB = 9.18 GB.
    - Upper bound for the whole grid if every scaled run forks: 36 GB (D2) +
      214 GB (D4) + 484 GB (D6) ≈ 734 GB.
- Limitation recorded (amendment (u)): the whole grid, including both arms
  of every fork, runs on one GPU model.

Working rules from this message:
- Scope: no features, scripts, tests or safeguards beyond what the
  Methodology requires. Fine-print choices that change no measured
  quantity, rule or visible result are taken as the simplest option and
  logged here. The protections stay (probe isolation, exact restore,
  identity fork, Check 1).
- Branch: stay on `claude/eloquent-fermat-inxqlt`. No merge or push to main
  until asked.

## Matmul precision (decided 2026-10-04): FP32 everywhere, no TF32 (amendment (v))

**SUPERSEDED the same day by the TF32 decision below.**

- **The gap:** the Methodology says nothing about matmul precision, and the
  code never sets `jax_default_matmul_precision`. On an A100, JAX's backend
  default is believed to be TF32 for float32 matmuls (step A0 measures it),
  and under amendment (m) that would stop every injected arm at Check 1.
- **The options:** (a) FP32 everywhere, (b) TF32 everywhere (amendment (m)
  would have to change), (c) mixed.
- **Your decision: (a).** Every job exports
  `JAX_DEFAULT_MATMUL_PRECISION=highest`; JAX honours the variable, so no
  code change is needed. `docs/exp12_cuda_commands.md` sets it in Setup and
  requires it for the grid.
- **What checks it:** A0 must show `tf32_detected: false` with the setting.
  Check 1 still stops any arm that runs with TF32. Exp 1 training itself has
  no such check, so the variable must be exported for every job.
- **Estimated cost:** ~3× slower training and probes for D6W1536 than TF32
  (~15 vs ~50 it/s on dog-run; not yet measured on a GPU).

## Matmul precision revised (2026-10-04): TF32 everywhere possible (amendment (v) rewritten)

Your instruction replaces "FP32 everywhere". Implemented as follows.

- **The setting.** `experiments/exp12/precision.py:set_matmul_precision()`
  sets `jax_default_matmul_precision = "tensorfloat32"` when
  `jax.default_backend() == "gpu"`, and does nothing otherwise. It is called
  first in `exp1.run`, `exp2_arm.run`, and at import in
  `scripts/probe_fresh_checks.py`, `scripts/profile_exp12.py` and
  `scripts/positive_control.py` (right after `configure_hardware_env()`).
  The preflight goes through `exp1.run`/`exp2_arm.run`.
- **Why "tensorfloat32" and not "high" or the default.** In jax 0.4.34,
  `"tensorfloat32"` is an alias of `Precision.HIGH` (the two are identical).
  The `Precision` docstring says DEFAULT and HIGH both use "tensorfloat32
  where available, otherwise float32" on GPU, HIGHEST uses float32, and
  precision "has no impact on CPU backends". Leaving the default would
  probably also give TF32, but whether DEFAULT is TF32 depends on the
  XLA/cuBLAS defaults of the installed jaxlib. "tensorfloat32" is the one
  value whose meaning is TF32 by name, so it is set explicitly. Verified on
  CPU: the 17 dots of Check 1's Q-and-gradient function lower to
  `precision = [HIGH, HIGH]` under the setting and to `[HIGHEST, HIGHEST]`
  inside Check 1's context. Whether HIGH really runs TF32 kernels on the A100 is
  **NEEDS CUDA VERIFICATION** (A0). If A0 shows FP32-level error at the run
  setting, I will send it to you and propose `"default"`.
- **What it covers.** Every float32 dot and convolution in Exp 1/2 jobs:
  training, action selection, probes, post-fork evaluation and diagnostics.
  Dtypes stay float32. There is no bf16 or fp16, `mixed_precision: false` in
  both agent configs, and x64 is off.
- **Check 1 and A0.** `fork.panel_q_and_grad` runs inside
  `jax.default_matmul_precision("highest")` (`fork.CHECK1_PRECISION`).
  Check 1's tolerances are unchanged (0 for control and identity, 64 eps for
  the injected arm). The TF32 stop and the `precision` argument of `check1`
  are removed, and `check1_<arm>.json` records `matmul_precision: highest`.
  The A0 command in the CUDA sheet runs `matmul_precision_report()` at the
  run setting and inside a "highest" context. Verified on CPU: the report
  records `tensorfloat32` and `highest` respectively (errors are equal on
  CPU, about 6e-7, as documented).
- **Metadata.** `run_metadata.json` launches now carry
  `runtime = {jax, jaxlib, platform, device_kind, backend_platform_version
  (the CUDA version on GPU), jax_cuda12_plugin, jax_cuda12_pjrt,
  matmul_precision}` next to `device`. There is no new refusal logic.
- **Search (whole repo, every text file).** Every hit for
  `JAX_DEFAULT_MATMUL_PRECISION`, `jax_default_matmul_precision`,
  `precision=`, `Precision.HIGHEST`, `NVIDIA_TF32_OVERRIDE`, tf32, XLA flags
  and `jax_enable_x64`:
  - `experiments/angle_1.py:80` `jax.config.update("jax_enable_x64", False)`.
    Kept: x64 is off, confirmed at runtime after importing `exp2_arm`
    (`jax.config.jax_enable_x64 == False`).
  - `generate_manifest.py:48`, a commented CPU thread flag
    (`--xla_cpu_multi_thread_eigen=false`). It does not touch GPU precision.
    Kept.
  - `docs/exp12_cuda_commands.md:187` `--xla_gpu_deterministic_ops=true` (the
    B1 deterministic pass). It is believed not to disable TF32 (it governs
    reduction and scatter determinism and autotuning), but this is not
    verified. Kept; only the tests run under it.
  - The old FP32 decision: `JAX_DEFAULT_MATMUL_PRECISION=highest` in the CUDA
    sheet's Setup and grid notes, and `NVIDIA_TF32_OVERRIDE`/`XLA_FLAGS` in
    `matmul_precision_report()`. Removed. The sheet now says not to set
    either variable.
  - The old `fork.TF32_REL_ERROR`, `tf32_detected` (fork, exp2 analysis,
    preflight output, tests) and the TF32 break-check mutation. Removed or
    replaced (below).
  - The new code: `precision.py`, `fork.py:116` (the Check 1 context), the
    preflight line `check1_precision=...`, and its test regex.
  - No `precision=` argument and no `Precision.*` exists anywhere in model or
    agent code (scale_rl, experiments), and no launch script sets a
    precision or TF32 variable.
- **SimBa's released code** (`/tmp/claude-0/simba` @ 7d0358b). It sets no
  matmul precision, TF32, XLA or x64 flag; only `mixed_precision: false`. It
  therefore ran at JAX's default precision (believed to be TF32 on Ampere,
  per the docstring; not verified), so no question arises.
- **Tests.**
  - `ForkUnitTest.test_check1_tf32_stops` is replaced by
    `test_check1_fails_when_the_injection_construction_is_broken`. It
    patches `InjectedSACCritic.__call__` to drop the frozen copy
    (old + new instead of old + (new − copy)). Check 1 must then fail on
    pre-vs-after while pre-vs-control still passes. The correct
    construction passes.
  - Break check: "Check 1 ignores TF32" is replaced by "Check 1 tolerance
    effectively infinite" (1e12 eps), which this test catches.
  - `test_arm_stops_before_training_when_check1_fails` now forces the
    failure with a one-ulp change in Q after the restore, not by injecting
    TF32.
- **Docs.** Amendment (v) is rewritten (with a note on (m)). The CUDA sheet:
  - the Setup export is removed;
  - A0 measures both settings;
  - A3 profiles TF32 only;
  - all estimates are rescaled to TF32 (~3× faster matmul-bound steps than
    FP32, an estimate);
  - Block A is ~20 min.
  The master summary's precision lines and its section references to the
  sheet (§0–§8 → Setup, A0–A3, B1–B4, C, D, E, F) are refreshed.
- **Verified on CPU** (this machine; GPU behaviour is NEEDS CUDA VERIFICATION):
  - all 10 Exp 1/2 test modules, 117 tests: fork 23, injection 8,
    positive_control 15, diagnostics 8, exp2_analysis 11, manifest 9 (the
    preflight with fork prints `check1_precision=highest`), probe 9, phase3
    19, foundations 12 (Angle 1 parity, bit-exact resume), pipeline 3;
  - the two HumanoidBench-only tests in venv_hb (pipeline h1-reach, Reach
    evaluation seeding) pass;
  - `tests/exp12_break_checks.py`: 55 of 55 mutations fail their test by
    assertion and pass again when restored, including the new Check 1 one.

## Hot-path efficiency scan (2026-10-05)

**How it was measured.** CPU only (4-core Xeon, 2.8 GHz). Scratch profilers
outside the repo, nothing committed:
- one tiny run per suite at D2W512 (dog-run, myo-key-turn, h1-run-v0 in
  venv_hb), 150–300 steps after warm-up, each component timed;
- sync mode isolates device compute from host work; cProfile shows the
  Python frames;
- separate timings of the probe check, post-fork evaluation, env build,
  save, logging flush and window metrics;
- compile counts per phase with `jax_log_compiles` and
  `jax_explain_cache_misses` over a tiny exp1 → fork → injected arm;
- compile cost per critic size (D2/D4/D6).

GPU shares below combine the measured host costs with the FLOP-based TF32
device estimates of the CUDA sheet. A training step is ≈ 12.5 ms (D2W512) to
23.7 ms (D6W1536) on dog-run. They are **estimates**, and A4 checks them.

| # | Location | Finding | Measured share (dog-run unless noted) | Proposed fix | Tier | Risk |
|---|---|---|---|---|---|---|
| 1 | env step (dm_control, MyoSuite, HumanoidBench) | Simulator plus dm_control's Python (MuJoCo 4.4 of 8.1 ms) | 8.1 ms/step: ~65% (D2), ~34% (D6); MyoSuite 7.7 ms, h1-run 4.2 ms | none (third-party simulator code) | — | — |
| 2 | `Exp12Trainer.train`: action selection waits for the previous update | Host and GPU run in series | GPU time ~12% (D2) to ~54% (D6) of a step | overlap by acting with stale parameters | 3 (changes the algorithm's order) | not allowed |
| 3 | `fork.post_fork_eval` (F1: 26 × 10 episodes per arm) | Env-bound; env built per evaluation (0.35 s) | 43.7 s per evaluation; ~27% of a D6 arm, ~37% of a D4 arm | parallel or vectorised evaluation envs | 2 (changes episode streams and batching) | changes a measured quantity's sampling |
| 4 | train loop host work (`buffer.sample` ×2, `np.stack`, `_normalize`, host-to-device copies, `buffer.add`, read-back) | ~2.7 ms/step in total. Bit-identical pieces: the second copy in `buffer.sample` 0.05 ms, the KL-reference re-upload every step 0.08 ms | ~22% (D2), ~11% (D6) in total; bit-identical pieces < 1% each | normalise inside the jit or sample 512 indices at once (not bit-identical); remove the redundant copy and cache the KL reference on device (bit-identical) | 2 / 1 | Tier 1 parts are below the 3% threshold, so not applied |
| 5 | probe check (`run_probe`, Methodology) | One compiled `_fit` per critic structure (one more after a restore, from argument sharding); pool sampling 0.15 s per round | ~6% of a D6 run (forecast) | run current and fresh in one vmapped launch | 2 (changes operation order) | changes FP results |
| 6 | regular evaluations (every 50k steps, Methodology) | Env-bound | ~3% (D6) to ~6% (D2) | none | — | — |
| 7 | `_update_sac_networks_scan` compiled 3× per process start, +1 after every restore | Weak-typed scalars in the initial state turn strong after the first updates; restored arrays are committed | CPU: +20 s (D2), +26 s (D4), +44 s (D6) per process start; < 1% of a run | strong-typed initial scalars in `scale_rl` init (bit-identical in float32) | 1 | below threshold and in shared code, so not applied |
| 8 | checkpoint save (20 per run) and restore | `np.savez` plus Orbax; restore reads the agent checkpoint twice (`load_checkpoint`'s unconstrained restore for `churn_ref_batch`) | save 17.4 s at D6 fork size, ~1–2% of a run; double read < 0.1% | asynchronous saves; a single read | 2 (async) / 1 (double read) | async touches the crash-safety protection; double read below threshold |
| 9 | logging window (`PendingUpdateMetrics.flush`, `get_metrics`) | 2,000 × 19 small device arrays fetched one by one; eager metrics with SVD | CPU 0.18 + 0.16 ms/step (~1.4% + 1.3% of a D2 step) | stack on device and fetch once (bit-identical) | 1 | below threshold on CPU; GPU per-transfer latency may raise it (A4) |
| 10 | small recompiles | `_sample_sac_actions` twice after a restore; Check 1's `jit(lambda)` per call | seconds per process | none | — | — |
| 11 | JAX persistent compilation cache | Not enabled | saves the compiles above per process start/resume (~1–3 min per D6 process, a GPU estimate) ≈ 0.5–1% | enable `jax_compilation_cache_dir` | 2 (JAX config) | none on outputs; a launch change |
| 12 | thread settings | OMP/MKL/OPENBLAS=1 appear only in a comment in `generate_manifest.py`; the Exp 1/2 launch does not set them | the hot path uses no BLAS threads (mean/var only; MuJoCo single-threaded); the risk is oversubscription at 4 jobs per GPU | export the three at 1 in Block F | 2 (launch) | expected no output change |
| 13 | update scan buffer donation | None, so the old and new training states coexist during an update | memory, not time (≈ 2× training state transiently) | donate the actor, critic and target inputs | 2 (memory affects jobs per GPU) | probes and Check 1 snapshots hold parameter references |
| 14 | D2W512 update on GPU | Probably kernel-launch-bound (an estimate) | ~1.5 ms of 12.5 ms | XLA command buffers (CUDA graphs) flag | 2 (XLA flag) | verify in A4 first |

**Result: nothing applied.** No Tier 1 candidate reaches ~3% of
end-to-end wall-clock for any critic size or suite, and none is a defect
that affects outputs. Per the threshold rule, no code was changed, so there
are no before/after numbers. The scan's main consequence is the runtime
estimate: host overhead was measured at ≈ 11 ms per dog-run step, not the
assumed 6 ms. The CUDA sheet is refreshed (D2W512 ~80, D4W1024 ~68,
D6W1536 ~42 it/s; whole-run table; probe share ~6%).

**Tier 2 recommendations (awaiting your approval; nothing done):**
- (11) Persistent compilation cache: enable for the grid only if A3/B4 show
  ≥ ~1 min of compile per process and jobs restart often; otherwise skip.
- (12) Export `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` in
  Block F: recommended (cheap; no output change expected). Confirm with the
  identity fork under the same environment.
- (8) Asynchronous checkpointing: not recommended (1–2% against the
  crash-safety protection).
- (13) Donation: not recommended unless B4's peak memory limits jobs per GPU.
- (3) Parallel post-fork evaluation: not recommended (changes the episodes
  the F1 quantity is measured on).
- (4) Normalisation inside the jit, or fused sampling: not recommended
  (changes FP results or the random stream for ≤ ~4% of the step).
- (5) Vmapped current + fresh probe: not recommended (changes operation
  order for ~5% of the probe's 6%).
- (14) XLA command buffers: only if A4 shows D2W512 is launch-bound and you
  want the gain; it changes XLA flags.

**CPU-only findings:** every number above is from this CPU. The jit
dispatch time seen here (2.6–3 ms) is the CPU backend starting execution,
not Python overhead (cProfile shows no Python frames), so it does not
transfer to GPU.

**Re-verify on the GPU:**
- A0: TF32 active at the run setting, highest at FP32 level.
- A4 trace:
  - GPU busy time against wall time per step;
  - whether D2W512's update is launch-bound;
  - the host-to-device copies;
  - the cost of the window flush (item 9) when GPU transfer latency applies.
- A3/B4:
  - it/s per size (forecast 80/68/42 on dog-run) and the probe share
    (forecast ~6%);
  - compile time per process (decides item 11);
  - peak memory (decides item 13);
  - post-fork evaluation cost per suite.
- B1 and Block C: all parity and identity tests under TF32. Angle 1 parity,
  probes-on vs off identity, resume bit-exactness and the identity fork are
  unchanged by the scan (no code changed), but they have not yet run on
  CUDA.

## Block A result and follow-up (2026-10-05; job 22667743, A100-SXM4-40GB, commit 7812b17)

### Numbers (from the job's summary.txt)
- **Stack:** jax 0.4.34, jaxlib 0.4.34, PJRT C API, CUDA 12030.
- **A0:** float32 matmul max relative error 3.05e-4 at the run setting
  (tensorfloat32) and 2.16e-7 at highest.
- **A3, D6W1536 dog-run, TF32, probes off:** 34.6 it/s; one two-critic probe
  check 50.0 s; probe overhead 6.77% of a run (accepted, R2; probe
  unchanged); peak GPU memory 4.01 GiB.
- **A4:** 31.8 it/s under the profiler.
- **Wall times:** A0 24 s, A2 238 s, A3 181 s, A1 300 s, A4 87 s.
- **Runtime table:** refreshed in docs/exp12_cuda_commands.md from this
  point. The device took ~17.9 ms per step (not 12.7 ms) and a probe check
  1.4× the forecast; the unmeasured sizes are scaled by these factors.
- **Still unmeasured:** D2W512 and D4W1024 speeds, MyoSuite, HumanoidBench,
  deterministic ops, Delta's host overhead, and the packing slowdown.

### A1: why the summary showed 3 failures and 30 tests when unittest reported 8 and 34
- **Subtests (5 of the 8):** `test_predictions_unchanged_..._within_dtype_tolerance`
  failed in 5 of its 6 (blocks, m) subtests. unittest counts each failing
  subtest; the old parser kept only the first status of that test.
- **Parser (4 tests lost):** when a test writes output while it runs
  (warnings, progress lines), unittest's status lands on a later line. The
  old parser only accepted "... ok" on the same line, so those tests were
  dropped:
  - `test_exp12_probe.ProbeDoesNotChangeTrainingTest.test_breaking_probe_isolation_is_detected` (passed);
  - `test_exp12_probe.ProbeDoesNotChangeTrainingTest.test_probes_on_equals_probes_off` (passed);
  - `test_exp12_diagnostics.TrainingRunTest.test_training_is_identical_with_diagnostics_on_and_off` (**failed**);
  - `test_exp12_fork.IdentityValidationTest.test_compare_script_passes_and_detects_a_difference` (passed).

  These statuses follow from the totals: 34 ran, 8 failures, 0 errors, 0
  skips, and the 8 FAIL headers name exactly 5 subtests plus 3 tests.
- **Fix:**
  - The report code moved to `scripts/exp12_reports.py` (no jax), used by
    the Block A runner and by Block B.
  - Every test is listed; a status on a later line is attributed to the
    pending test, and failing subtests come from the FAIL/ERROR headers.
  - A test with no status is inferred to have passed only when the totals
    leave no alternative; otherwise it is UNKNOWN with a warning.
  - The runner also sets `PYTHONUNBUFFERED=1`.
- **New test:** `tests/test_exp12_reports.py` runs a real `unittest -v`
  subprocess with every failure mode seen on Delta and asserts that all 8
  tests and both failing subtests are listed. The old parser listed 6 of the
  8 on that log (checked by hand).
- **Test-order effect found on the way:** `ProbeDoesNotChangeTrainingTest`
  calls `exp1.run`, which sets the matmul precision process-wide. Every A1
  test after it ran under the explicit tensorfloat32 setting.

### A1 failures: details and classification
Evidence used:
- **(E1)** A CPU emulation of TF32 (scratch, not committed): every
  `lax.dot_general` rounds its inputs to a 10-bit mantissa (forward and both
  backward matmuls) and accumulates in float32, unless the precision is
  "highest". Calibrated on A0's measurement: emulated 3.0517e-4 against the
  A100's 3.05e-4.
- **(E2)** Same-executable determinism on the GPU:
  `test_probes_on_equals_probes_off` passed in A1, under TF32. Two exp1 runs
  in one process with the same compiled training program gave a
  bit-identical full state.
- **(E3)** `ForkUnitTest.test_check1_fails_when_the_injection_construction_is_broken`
  passed on the GPU. Its correct-construction half requires dQ/da within 64
  eps under "highest" for the same `_critic(2)`/`_inject` construction.

1–5. **`test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance`**,
   line 77: `assertLessEqual(max|g0 − g1|, 64·eps·max|g0|)`.
   - **Discrepancy per subtest** (deviation relative to max|dQ/da|; TF32 u = 2^-11):

     | Subtest | Observed | Bound | × bound | eps units | TF32 u |
     |---|---|---|---|---|---|
     | blocks=2, last | 4.8846e-5 | 1.1937e-5 | 4.09 | 262 | 0.064 |
     | blocks=2, half | 4.8846e-5 | 1.1937e-5 | 4.09 | 262 | 0.064 |
     | blocks=2, all | 3.9428e-5 | 1.2644e-5 | 3.12 | 200 | 0.049 |
     | blocks=4, half | 1.1277e-4 | 1.6483e-5 | 6.84 | 438 | 0.107 |
     | blocks=4, all | 2.5719e-5 | 2.1663e-5 | 1.19 | 76 | 0.019 |

     blocks=4, last passed. With 2 blocks, "last" and "half" are the same
     construction (one head block), hence identical numbers.
   - **Class (i):** a test written for CPU float32, run under TF32 without
     Check 1's local "highest". Evidence:
     - The run setting was tensorfloat32 (test-order effect above).
     - Under E1, the unmodified test fails in 5 subtests with 3.5–6.5e-5,
       and passes with the emulation off.
     - E3: the same construction passes within 64 eps under "highest" on
       the same GPU.
     - Mechanism: (a + b) − b differs from a by one float32 ULP in the summed
       cotangents. TF32 rounding of that input in the downstream backward
       matmuls turns it into TF32-level noise (0.02–0.11 u).
   - **Task 1c:**
     - Q before and after injection is **bit-identical under TF32** on the
       GPU for every pair that was compared. The Q assertion precedes the
       failing line and never failed; in the 5 failing subtests the target
       pair was not reached.
     - dQ/da deviation: under TF32, 76–438 eps (table). Under "highest" on
       the GPU, only the bound ≤ 64 eps is known (E3); the exact size is
       measured in Block B's A1 follow-up step. On CPU FP32 it was ≤ 7.2 eps
       (Phase 4, D6W1536).
   - **Fix (test only, allowed by the scope rule):** `_q_and_grad` computes
     Q and dQ/da inside `jax.default_matmul_precision("highest")`, as Check 1
     does.
     - Why this is precision-independent: new and copy have identical
       parameters and inputs, so new(z) − copy(z) = 0 under any
       deterministic precision (seen bit-identical even under TF32). The
       64-eps tolerance concerns float32 summation order only.
     - Before: fails on the A100, and fails under E1. After: passes under E1
       and on plain CPU. The A100 result after the fix is **NEEDS CUDA
       VERIFICATION** (Block B lane GPU 2).
     - Still able to fail:
       - existing break check: dropping the frozen copy → fails on Q;
       - a value-preserving mutation applied by hand (`old(stop_gradient(z))`:
         Q unchanged, dQ/da wrong) → fails at 1.57 against a bound of
         1.2e-5, and passes again once restored.

6. **`test_diagnostics_never_change_the_update`**, line 92:
   `assert_array_equal` on the agent state after 4 updates, diagnostics off
   vs on.
   - **Discrepancy:** the first differing leaf (8 elements, actor width 8)
     differs in all 8 elements. Max abs 1.17e-6; max element-wise relative
     2.7% (on an element of 4.2e-5). Relative to the leaf's max |value|
     (3.93e-4) that is 3.0e-3: about 6 TF32 u, not float32-ULP level.
   - **Class (iv):** other — GPU program-dependent rounding.
     - Not (iii): the update math is identical, and the state is
       bit-identical on CPU both with and without E1. So TF32 rounding alone
       does not cause it.
     - Not (ii) as far as can be shown: E2, the same executable gives
       identical results.
     - The two calls compile different programs (the diagnostics add inputs
       and outputs), and XLA's GPU backend can fuse and choose kernels
       differently for each.
   - **Task 1d:**
     - On the GPU, parameters and optimizer state after N = 4 updates
       **differ** between diagnostics on and off.
     - Max deviation 1.17e-6 absolute (3.0e-3 of the leaf's max): TF32
       level, not float32-ULP level.
     - Whether it shrinks to float32 level under "highest", or disappears
       with deterministic ops, is measured in Block B.
     - The diagnostics code is unchanged.

7. **`test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old`**, line 58:
   `assert_allclose(diagnostic KL, closed form, rtol=1e-4)`.
   - **Discrepancy:** 403.7256 vs 387.5590; abs 16.17, rel 4.17%.
   - **Class (iv):** program-dependent GPU numerics, amplified by an
     ill-conditioned quantity.
     - The test passes on CPU, and under E1 too, because consistent TF32
       rounding in both paths cancels.
     - The KL is ~390 nats because the fresh tiny actor has σ down to
       4.8e-5. KL ∝ (Δμ)²/σ² then amplifies any difference between μ computed
       inside the update scan and μ recomputed eagerly by the test.
     - On the GPU these are different programs.
     - Whether "highest" restores rtol 1e-4 is measured in Block B (I expect
       it to be borderline).
   - Not changed: a "highest" context is not shown to be sufficient here.

8. **`test_training_is_identical_with_diagnostics_on_and_off`**, line 161:
   - **Discrepancy:** after a 400-step tiny exp1 run, 103 state components
     differ between diagnostics on and off. The first is the actor's Adam
     first moment of `encoder/Dense_0/bias`.
   - **Class (iv):** same mechanism as 6, compounded over 400 steps. The
     same-executable run pair (E2) is bit-identical, and the CPU is
     bit-identical with and without E1.

### Decisions needed from the lead (nothing changed)
- **D1 — tests 6 and 8 (bitwise diagnostics-isolation protections) on GPU:**
  - (A) Keep them as CPU gates (CPU proves logical isolation: no RNG use,
    no state mutation) and report the GPU deviation as information.
  - (B) Gate on the GPU only under deterministic ops + "highest", if Block B
    shows bit-identity there.
  - (C) Code change: compute the diagnostics in a separate compiled call, so
    the update program does not depend on them.

  Recommendation: (A) now, revisit with Block B's numbers. Every Exp 1/2 run
  has diagnostics on, so compared runs are never confounded by this.
- **D2 — test 7:** keep it as is until Block B shows whether it passes under
  "highest". If it does, add the local context (test only).
- **D3 — production effect on diagnostic I1 (KL):** under E1, TF32 changed
  the KL computed for the same parameter pair by 9.6% (test actor 1×8) and
  0.7% (actor 1×128, dog-run shapes) against FP32. Both actors were freshly
  initialised with σ as small as 4.6e-5. The existing churn metric has the
  same, weaker sensitivity.
  - (a) Accept, and record it as a limitation of amendment (v).
  - (b) Compute the diagnostics' actor forward passes (KL, churn reference)
    under a local "highest" context. Training is unchanged and the cost is
    negligible.

  Recommendation: (b). It changes the computation of a Methodology
  diagnostic, so it needs your approval.

### A2 rule replaced (amendment (w), the lead's decision)
- Recorded in .claude/methodology-exp1-exp2.md as (w). The original rule and
  result are kept there and above (the 2026-10-04 entry is not edited).
- Runner: A2's exit code follows P/b ≥ 0.9 at the configured pool at every
  size (`scripts/exp12_reports.py range-check`); the old rule is printed as
  information.
- On Block A's numbers, (w) passes: P/b 0.990, 0.997 and 0.993.
- Probe settings are unchanged.
- **Per-round final-loss spread** (from A2.log; the quantity that matters for
  L, since b cancels):

  | Critic | Pool | Final loss IQM | SD | Range | P SD (for comparison) |
  |---|---|---|---|---|---|
  | D2W512 | 1,600 | 0.00296 | 0.00446 | 0.0109 | 0.00947 |
  | D2W512 | 6,400 | 0.0145 | 0.0144 | 0.0335 | 0.0167 |
  | D2W512 | 25,600 | 0.00346 | 0.00778 | 0.0189 | 0.00764 |
  | D4W1024 | 1,600 | 4.95e-08 | 8.19e-09 | 1.94e-08 | 0.00823 |
  | D4W1024 | 6,400 | 2.18e-06 | 4.27e-06 | 1.03e-05 | 0.00138 |
  | D4W1024 | 25,600 | 0.00133 | 0.00226 | 0.00485 | 0.00133 |
  | D6W1536 | 1,600 | 3.49e-08 | 7.74e-09 | 1.86e-08 | 0.00975 |
  | D6W1536 | 6,400 | 0.00019 | 5.93e-05 | 0.000135 | 0.0104 |
  | D6W1536 | 25,600 | 0.00384 | 0.00157 | 0.00416 | 0.00614 |

  At the small pools, the larger critics drive the final loss to ~1e-8, so
  P's round-to-round SD there is almost entirely b's variation between
  rounds, which cancels in L.

### Block B: the 4-GPU job (built and CPU-tested, not submitted)
- **Files:**
  - `scripts/sbatch_exp12_blockB.sh`: the Slurm wrapper;
  - `scripts/exp12_blockB.sh`: the driver, with `--dry-run`;
  - `scripts/collect_report.sh`: one file to paste (report + failing logs and
    tracebacks);
  - the report logic in `scripts/exp12_reports.py`.
- **Layout, as decided:**
  - GPU 0: dev run → positive control → preflight;
  - GPU 1: injected-arm watcher;
  - GPU 2: packing → A1 follow-up measurements → test suite (default, then
    deterministic, with the break checks between) → hopper-hop range →
    identity forks (reduced budget);
  - GPU 3: fresh-pair null.

  It replaces the old B1–B4 sequence. The old B4 compute profile is not in
  the layout and stays in the sheet as "not scheduled".
- **Fine-print choices (simplest option, no effect on any measured quantity
  or rule):**
  - The arm's m comes from `ARM_M` (default `pc`: wait for this job's
    positive control and use its m). A fixed value starts the arm at the
    fork. This is listed for your decision.
  - GPUs are addressed by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. CPUs are
    the allocation's own set, split into 4 equal groups; packing jobs take 4
    consecutive cores of GPU 2's group each.
  - Packing jobs:
    - use seeds 990 + job index;
    - set `num_eval_episodes=1`, which only shortens the start-up
      evaluation, outside the timed window;
    - have a 600 s start-barrier timeout;
    - get an overlap flag at ≥ 90%, descriptive only.
  - `MUJOCO_GL=egl` is set only for HumanoidBench steps, and only when
    HumanoidBench imports and EGL loads; otherwise `disable`.
  - Internal deadline 9.5 h (5 min kept for the report); the report is also
    written on SIGTERM.
  - The A1 follow-up measurement is one heredoc run under default and
    deterministic ops. It reuses the test modules' helpers.
- **Time and resources:** derived in the sheet's Block B section from
  Block A's measured point. The critical path is ~7.6 h with `ARM_M=pc` and a
  late fork; the request is 10 h, 4 GPUs, 64 CPUs and 128G.

## Block B decisions (received 2026-10-05) and how they are implemented

None of these changes the Methodology except (x), which the lead approved.
Block B is a development job, not confirmatory.

1. **Injected arm: `ARM_M=half`** (the driver's default). The positive
   control chooses m separately, and the lead freezes it. The arm starts at
   the fork. `ARM_M=pc` remains available.
2. **Diagnostics-isolation tests on the GPU** (`test_diagnostics_never_change_the_update`,
   `test_training_is_identical_with_diagnostics_on_and_off`):
   - CPU stays bit-exact, unchanged.
   - On the GPU they follow PyTorch's TF32 on/off pattern
     (`tests/exp12_helpers.py`):
     - in a deterministic-ops process (`XLA_FLAGS` has
       `--xla_gpu_deterministic_ops=true`), both sides run under a local
       "highest", at a tight tolerance;
     - otherwise the diagnostics-on side runs at the run setting (TF32)
       against a diagnostics-off reference under "highest", at a loose
       tolerance.
   - The reference side never runs under TF32.
   - Deviation: max over state leaves (and update outputs) of
     |x − ref| / max|ref|.
   - **Tolerances are set only from measured GPU deviations.** Rule (fine
     print): 10× the largest measured value, one significant digit. None
     has been measured, so all four are `None`. On the GPU each test then
     records its deviation (`NUMERICS` line and
     `$EXP12_NUMERICS_OUT`) and skips. Block B's report lists each deviation
     next to its tolerance; the tolerances get set from those numbers in a
     follow-up commit.
   - The diagnostics code is not changed for this.
   - Note: the 400-step training comparison includes environment feedback,
     so any per-step difference can grow along the trajectory. If the
     measured deviation turns out large, a tolerance there would be
     uninformative, and I will say so with the numbers.
3. **KL closed-form test (test only).**
   - A well-conditioned actor: log-std head bias 0.55, kernel × 0.1, σ
     0.13–0.88, actor learning rate 1e-3. That gives a per-update KL of
     0.0192, large enough that the float32 KL is not dominated by
     cancellation (at lr 1e-4 the KL is 1.2e-4 and float32 cancellation
     alone gives 2.5e-4 relative).
   - The reference is the float64 numpy closed form on FP32 ("highest")
     forward passes; the gate is rtol 1e-4. Measured on CPU: relative
     difference 3.0e-6. A direction check requires |KL − reverse KL| / KL >
     10 × rtol (measured 3.9e-3).
   - The near-deterministic actor (σ ≈ 5e-5) is a separate test that
     reports and does not gate (CPU: relative difference 5.6e-6).
   - **Break-and-restore:** the direction swap (re-anchored) and a wrong KL
     formula (the mean term only, no variance terms) each fail the test and
     pass again once restored. Production KL code is unchanged.
4. **Diagnostics precision, amendment (x)** (approved).
   - In `scale_rl/agents/sac/sac_agent.py:_sac_update`, the churn reference
     forward passes (before and after the update) and the KL's two forward
     passes run under `jax.default_matmul_precision("highest")`, but only
     when the Exp 1/2 diagnostics are on (`kl_ref_observations` given).
     Otherwise the code uses a no-op context.
   - `jax.default_matmul_precision(None)` was rejected: inside a TF32 job it
     resets to DEFAULT (seen in the lowered HLO).
   - Training matmuls are unchanged. On CPU the context is a no-op, so CPU
     results are bit-identical.
   - **Test:** `DiagnosticsPrecisionTest` lowers the scanned update under
     the TF32 run setting and requires:
     - exactly 2·n_mean + 2·n_full matmuls at HIGHEST (the churn forward
       passes use 4 matmuls, the compiler dropping the unused log-std head;
       the KL's use 5);
     - none with diagnostics off;
     - the HIGH (TF32) count lower by exactly the churn passes.
   - **Break check:** returning a no-op context instead of "highest" fails
     the test.
5. **Old B4 dropped.** Kept:
   - per-suite training speed (myo-key-turn, h1-run-v0 × D2W512, D4W1024,
     D6W1536; one job, 4 cores, probes off, 600 timed steps) on GPU 3 after
     the null; dog-run comes from the packing test's 1-job runs;
   - the dev run's fork save and restore times and the wall time of every
     post-fork evaluation, now printed by `exp1.py`, `exp2_arm.py` and
     `fork.post_fork_eval` (print only) and collected in report.txt.
6. **HumanoidBench through `HB_ENV`.**
   - The HumanoidBench steps (identity cells, h1-run speed, the two
     HumanoidBench tests) run with `$HB_ENV/bin` first on PATH and
     `MUJOCO_GL=egl`, only if HumanoidBench and EGL load there.
   - Unset, they are recorded `SKIPPED_UNAVAILABLE`, which does not fail
     OVERALL.
   - The main environment is never used for HumanoidBench. The clone
     commands and the unchanged-main-environment check are in the sheet's
     Setup.
7. **Request:** 10 h, 4 GPUs, 64 CPUs, 128G. The node check (`sinfo`,
   `scontrol show node`) is in the sheet. The critical path is now GPU 0's
   ~6.4 h, since the arm starts at the fork.
8. **Cross-process numerics.**
   - **(a) Persistent compilation cache:**
     `experiments/exp12/precision.py:configure_compilation_cache()`.
     - One directory per GPU model: `main/jax_cache/<device kind>`
       (gitignored; `EXP12_JAX_CACHE_ROOT` or `EXP12_JAX_CACHE_DIR` override
       it, and `off` disables it).
     - `jax_persistent_cache_min_compile_time_secs=0` and
       `min_entry_size_bytes=0`, so small compiles are cached too.
     - It is called at the start of every Exp 1/2 entry point and in the
       packing jobs; the path goes into `run_metadata.json`, and each
       process prints its cache hits and misses at exit.
     - Found on the way: in jax 0.4.34 the cache is fixed at a process's
       first compile, so a later configuration is ignored (0 files written).
       The helper therefore calls `compilation_cache.reset_cache()` after
       configuring. That is a private API of the pinned jax; with it, a late
       configuration works (checked on CPU).
   - **(b) Identity forks run twice:** cold (separate empty caches for the
     parent and the arm) and warm (one shared cache). Both comparisons are
     reported.
   - **(c) Start-up effect:** the arm's process wall time, cold vs warm,
     plus its cache hits and misses, in report.txt.
   - **(d) Flag names:** both names are accepted by jaxlib 0.4.34 on CPU; an
     unknown one aborts. Help text: `xla_gpu_deterministic_ops` "Guarantees
     run-to-run determinism on GPU"; `xla_gpu_exclude_nondeterministic_ops`
     "Excludes non-deterministic ops from compiled executables". The sheet's
     flag is the right one; it is NEEDS CUDA VERIFICATION on the GPU.
9. **GPU memory flags.**
   - Grep of launch scripts, sbatch files, `claim_launcher.py`, the sheet
     and the old Angle 1 scripts:
     - `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` was only in
       `scripts/run_angle1_a100x8.sh` and `scripts/run_angle1_a40x4.sh`;
       both now unset it, with a comment;
     - `PREALLOCATE=false` stays everywhere it was (both Angle 1 scripts,
       both Block sbatch files, both drivers' defaults, the sheet);
     - `claim_launcher.py` sets neither; no `XLA_CLIENT_MEM_FRACTION`
       anywhere.
   - No fraction is kept: the packing test measures per-job peaks.
   - Not verified here (the CUDA plugin is not installed on this machine):
     whether the fraction caps memory when preallocation is off. I believe
     XLA sizes its GPU allocator as fraction × free memory either way.
   - Editing the scripts does not affect jobs already submitted (Slurm
     copies the script at submission).

## Critic sizes: D4W1536 replaces D6W1536 (received 2026-10-06)

**Decision (yours).** The critic grid is D2W512, D4W1024, D4W1536 (depth ×
width, per Q network); the actor stays D1W128; D6W1536 is dropped
everywhere. Reason: the largest critic was too large relative to the actor,
and the change cuts cost. Nothing else changes (probe, trigger, injection,
UTD, checkpoints, precision). Recorded as Methodology amendment (y).

**Where it changed** (grep of the repo for `D6W1536`, `1536` and `6 blocks`,
plus `critic_num_blocks=6`, `6:1536` and `D6`):
- configs: `configs/base_exp12.yaml` (`fork.architectures`,
  `positive_control.architecture`);
- grid manifest: `generate_manifest.py` (`EXP12_ARCHS`, `EXP12_FORKING`);
- analysis: `analysis/exp1_analysis.py` (`SCALED`, `ARCH_COLORS`),
  `analysis/exp2_analysis.py` (`SCALED`);
- scripts: `positive_control.py` (docstring, help), `probe_fresh_checks.py`
  and `profile_exp12.py` (`ARCHS`, usage), `preflight_checkpoint_check.py`
  (usage), `exp12_blockA.sh` (A2 archs, A3, A4 trace, range check),
  `exp12_blockB.sh` (dev run, packing, range, null, identity, speed,
  preflight, the packing size table), `sbatch_exp12_blockB.sh` (time and
  justification), `exp12_reports.py` (Block A file names and section
  title, the null's size list);
- tests: `test_exp12_phase3.py`, `test_exp12_positive_control.py`,
  `test_exp12_exp2_analysis.py`, `test_exp12_manifest.py` (labels only);
- docs: the Methodology (body size list, (e), (p), new (y)), the CUDA
  sheet, the master summary, this log.

**Deliberately unchanged** (records of what was measured, which name the
size measured then):
- this log's earlier entries and the Block A numbers (A2's P/b 0.993 and
  A3's 34.6 it/s, 50.0 s, 6.77%, 4.01 GiB were measured on D6W1536);
- the comment in `tests/test_exp12_injection.py` (7.2 eps measured at
  D6W1536 on CPU);
- `tests/test_exp12_reports.py` `RangeCriterionTest.BLOCK_A` (Block A's
  measured P/b values, keyed by the sizes measured; `range_criterion` takes
  the size list as an argument, so nothing depends on the key);
- the memory comments in `scripts/run_angle1_a100x8.sh` and
  `run_angle1_a40x4.sh` (D6W1536's measured 4.01 GiB peak);
- `scripts/sbatch_exp12_blockA.sh` and `scripts/collect_report.sh`
  contain no critic size.

**Fine-print choices (logged, not asked):**
- The A3 output file is now `A3_profile_dog_run_D4W1536.json` (was `_D6`),
  in `exp12_blockA.sh`, `exp12_reports.py` and the sheet together.
- Block E's manual preflight writes `E_preflight_D<d>W<w>` (was `D<d>`),
  because D4W1024 and D4W1536 share the depth.
- Block B timeouts: dev run 7.5 h → 6 h (≈ 26% over the 4.8 h estimate, as
  the old 7.5 h was over 5.9 h); preflight 30 → 20 min per size; the
  driver's deadline 9.5 → 7.5 h.

**Parameter counts** (CPU, `jax.eval_shape` of the real `SACCritic` and
`SACActor`, with the observation and action dimensions read from the built
environments: dog-run 223/38, myo-key-turn 93/39, h1-run-v0 51/19; per Q
network, and the total critic is one network because every suite is single
critic now):

| | dog-run | myo-key-turn | h1-run-v0 |
|---|---|---|---|
| actor D1W128 | 170,700 | 154,318 | 143,782 |
| D2W512 | 4,337,153 (×25) | 4,271,105 (×28) | 4,239,361 (×29) |
| D4W1024 | 33,854,465 (×198) | 33,722,369 (×219) | 33,658,881 (×234) |
| D4W1536 | 75,947,521 (×445) | 75,749,377 (×491) | 75,654,145 (×526) |
| D6W1536 (dropped) | 113,717,761 (×666) | 113,519,617 (×736) | 113,424,385 (×789) |

(×n = critic/actor parameter ratio.)

**Runtime (all D4W1536 figures are unmeasured estimates).** Anchor: Block
A's measured D6W1536 point (34.6 it/s, 50.0 s per probe check, 6.77% probe
overhead, 4.01 GiB peak). D4W1536 has D6W1536's layer shapes with 4 blocks
instead of 6, so device work is scaled by the parameter ratio 0.668: device
time 17.9 → 12.0 ms per step, plus the measured ~11 ms of host overhead →
~43.6 it/s; probe check ~33.4 s; overhead ~5.7%; peak ~2.7 GiB. Block B:
GPU 0 critical path ~5.1 h (dev run to 120% of B after a late fork ~4.8 h),
GPU 2 ~3.9 h, GPU 3 ~2.0 h; the request drops from 10 h to 8 h (~57% over
the critical path, as 10 h was ~55% over 6.4 h). 4 GPUs, 64 CPUs, 128G
unchanged.
