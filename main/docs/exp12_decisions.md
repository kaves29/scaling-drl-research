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
