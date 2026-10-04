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
