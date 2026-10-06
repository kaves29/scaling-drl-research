# Methodology: Experiments 1 and 2 (source of truth)

The text between the two horizontal rules is the Methodology (final, later
version from the task brief), saved verbatim. Amendments decided by the project
lead after the Phase 0 Q&A follow below it; the verbatim text is never edited.
Changes arrive only as explicit messages from the project lead.

---

Statistical Practices:

* MUST USE PERCENTILE BOOT STRAPS → rliable
* Use interquartile-mean if you have distributions with low amounts of samples → reduces sensitivity to outliers and computes a robust mean-like aggregate
* Using the hyper parameters from other papers in field is justified because we are not aiming to compare different architectures, but rather understand. Furthermore, the hyper parameters were tested on a larger number of environments which supports their generalizability since we are only testing on a subset of what they tested on
* Include baselines to give the reader an understanding of your results → healthy/default architecture in Experiment 1 and paired no-injection control in Experiment 2
* Development/calibration runs used to validate the injection and select the injection head are kept separate from the confirmatory main runs
* “Another common approach is to isolate the precise modification made in the proposed
algorithm. Perhaps the proposed algorithm introduces a new sampling strategy from replay
buffers. Instead of comparing against a suite of different learning rules, we can instead make
a small number of pairwise comparison between similar algorithms. For instance, we can
endow DQN and A3C with our proposed replay buffer and perform two pairwise comparisons
against the original DQN and A3C respectively. We do not care if DQN with the novel replay
buffer outperforms A3C with the old replay buffer, as many design variables change between
these populations. By eliminating several such pairwise comparisons, we can significantly
decrease the probability of false negatives and number of samples to detect differences.” → seemed very relevant to our research design but I could not pinpoint exactly how

Configurations:
Seeds:
5 seeds throughout the study
Critic Size:
Where actor architecture is held constant at depth=1 and width=128

1. depth=2 & width=512 → 4.2 million parameters
2. depth=4 & width=1024 → 33.6 million parameters
3. depth=6 & width=1536 → 113 million parameters

UTD Ratio:

* Holding constant across experiments at 2 to isolate the effects of scaling → default used by the research papers
* Could possibly change in ablations but that is later

Other hyperparameters to hold constant:

* Learning rate: $0.0001$
* Batch Size: 256
* Discount Factor: Heuristic → TD-MPC2 heuristic (Hansen et al. 2023), as used in SimBa
* Target Soft Update: 0.005
* Action Repeat: 2
* Entropy: abs(A)/2
* Temperature Learning Rate: 1e-4
* Weight Decay: 1e-2
* Optimizer: AdamW

Experiment 1:
General:
Scaling → critic degradation/development of optimization pathologies
What is the definition of degrade → loss of plasticity and ability to fit to new targets → using an adapted version of the plasticity probe introduced by Lyle et al. 2023
Environments:
DMC:

1. dog-run - hard
2. dog-trot - hard
3. humanoid-run - hard
4. humanoid-walk - hard
5. humanoid-stand - hard
6. swimmer-swimmer15 - medium
7. hopper-hop - medium

Myosuite:

1. myoHandKeyTurnFixed-v0 - hard
2. myoHandPenTwirlFixed-v0 - hard
3. myoHandPoseRandom-v0- close to hard
4. myoHandReachFixed-v0 - medium

Humanoid Bench:

1. h1-shelf-place - hard
2. h1-reach - hard

Environment Steps: 500k for medium DMC and 2 million for Humanoid Bench, everything else is 1 million
Methodology:

* Train one of each of the 3 unique critics within 3 separate agents
* Before critic training, after the initial replay buffer has been populated, run the plasticity probe on the fresh, untrained critic to establish the run's initial plasticity score. Store the fresh critic parameters so the same fresh reference can be re-probed later
* Periodically check for plasticity loss for all of them as introduced by Lyle et al. 2023. For each period check, we will perform 5 different rounds of evaluation with 1000 step-probe each. Periodically means each run must have 20 probe tests scattered evenly across the training time at 1/20, 2/20, ... 20/20 of the run's training budget
* The probe is performed on a copy of the critic so that the probe does not modify the training critic
* For each period check, use the same probe inputs and targets for the current critic and the stored fresh critic parameters so the comparison measures parameter plasticity change rather than changes in the probe input/target distribution
* Each probe round evaluates the critic's ability to fit a new target, following the procedure introduced by Lyle et al. 2023, adapted to the SAC critic setting
* Take the IQM across the probes of each period check to get one plasticity score per agent for that period
* For each run, compute the loss of plasticity relative to the same run's fresh, untrained parameters using the plasticity-loss formulation from Lyle et al. 2023 → handles that every network can lose some plasticity over training and that critics of different sizes fit the probe differently from the start
* Use percentile bootstrap confidence intervals to determine whether the observed plasticity loss is statistically distinguishable from zero at the 95% level → this is used as an operational fork trigger, not as a population-level scientific claim about an individual run
* The bootstrap is performed on the paired probe-level plasticity-loss observations for that check
* Use the probe learning curves descriptively to determine whether the plasticity loss manifests as the slower/shallow new-target learning described by Lyle et al. 2023 → do not introduce a new slope formula
* The first check where an individual run's measured plasticity loss is statistically distinguishable from zero is called $f^*_{run}$ and is used as the operational fork point in Experiment 2
* $f^*_{run}$ is an operational eligibility trigger and is separate from the architecture-level statistical comparison used to determine whether scaling produces excess plasticity loss
* For the main Experiment 1 scientific comparison, compare the seed-level plasticity-loss trajectories between each scaled architecture and the default architecture
* Use the final scheduled checkpoint as the primary pre-specified endpoint for the overall scaling-related plasticity-loss comparison → difference = scaled plasticity loss − default plasticity loss
* Pool the final-checkpoint comparison across environments using rliable's stratified bootstrap (65 runs per architecture) → this produces the primary architecture-level estimate and confidence interval for scaling-related plasticity loss
* The full plasticity-loss trajectories are also shown descriptively with the time axis in fractions of the training budget
* Every seed in every environment is shown descriptively, and the individual-run $f^*_{run}$ values are retained in the structured ledger
* If a run has not reached the operational plasticity-loss trigger by 95% of the nominal training budget, $f^*_{run}$ is null and the run is reported as a null result for Experiment 2
* The architecture-level result and the individual-run trigger are kept separate → a scaled architecture can show greater average plasticity loss than D2/W512 even when some individual runs never trigger

Limitations:

* Every network is expected to lose some plasticity over training (Lyle et al. 2023) → the fresh same-architecture reference measures the change within each run, while the separate scaled-vs-default comparison determines whether scaling is associated with greater plasticity loss
* The individual-run trigger is an operational rule rather than a population-level statistical test → it is used only to determine when a run is eligible for the Experiment 2 intervention
* Twenty scheduled checks create repeated opportunities for an operational trigger to occur → the first detected crossing should therefore be interpreted as an intervention point rather than an exact estimate of the biological/optimization onset of degradation
* With few probe rounds and 5 seeds, bootstrap intervals can be wide or unstable → all seed-level results and probe distributions are retained and shown descriptively
* Individual runs can reach the operational trigger at different points in training → Experiment 2 therefore aligns the post-fork analysis by steps since fork rather than absolute training step
* Some runs may never reach the trigger within 95% of the nominal budget → this is a valid null result and means there is no eligible fork for that run
* 5 seeds matches SimBa's ablations, but SimBa used 10 seeds for its main SAC comparisons → confidence intervals will be wide

Experiment 2:
General:
Must establish: Critic pathology propagates from the critic to the actor, producing downstream actor performance degradation and measurable changes in actor-side optimization/behavioral diagnostics
Environments:
DMC:

1. dog-run - hard
2. dog-trot - hard
3. humanoid-run - hard
4. humanoid-walk - hard
5. humanoid-stand - hard
6. swimmer-swimmer15 - medium
7. hopper-hop - medium

Myosuite:

1. myoHandKeyTurnFixed-v0 - hard
2. myoHandPenTwirlFixed-v0 - hard
3. myoHandPoseRandom-v0- close to hard
4. myoHandReachFixed-v0 - medium

Humanoid Bench:

1. h1-shelf-place - hard
2. h1-reach - hard

Environment Steps: 500k for medium DMC and 2 million for Humanoid Bench, everything else is 1 million
Methodology:
Positive control (run before the main experiment, and before the main runs of Experiment 1):

1. Probe a fresh critic
2. Probe a normally trained critic
3. Obtain a naturally degraded scaled critic checkpoint from a preliminary SAC/SimBa development run using the Experiment 1 degradation criterion and probe it → the score should show clear plasticity loss
4. Apply plasticity injection to the naturally degraded critic and probe it again → the score should recover toward the fresh critic's level

* Use one scaled critic architecture and one environment for the positive-control calibration
* This validates the plasticity-injection intervention in SAC critics, SimBa, and continuous control; Lyle et al. 2023 establishes the plasticity probe in their value-learning settings, not ours
* Step 4 is also the first test of the injection
* No separate artificial damage procedure is used; the degraded critic comes from the same SAC/SimBa setting being studied
* The preliminary development run and all results used to validate the injection or select $m$ are excluded from the confirmatory main Experiment 1 and Experiment 2 analyses

m-selection (part of the positive control):

* Head = last m residual blocks + post-layer-norm + output layer
* Since SimBa has no dedicated encoder, this head boundary is defined explicitly for the injection
* Test m = last block only, half of the total residual blocks, and all residual blocks
* For each m, apply the plasticity injection to the naturally degraded critic and run the plasticity probe
* Select m based on the observed recovery toward the healthy reference, with the effect size/recovery reported for every candidate → if multiple candidates produce similar recovery, keep the smallest m
* Record which m was chosen and the evidence (probe scores and effect sizes for each candidate) in a results file so the decision can be audited later
* Once selected, m is frozen and is not retuned per architecture, environment, or main-run seed

Main experiment:

* Record whether critic pathology precedes actor-side degradation as a descriptive result → measured visually and not deep empirically
* Create a scaled critic architecture agent
* When an individual run reaches $f^*_{run}$, fork the complete training state into two identical agents. The fork is performed once per run
* If an individual run has not reached the operational plasticity-loss trigger by 95% of the nominal training budget, no fork is performed → reported as a normal null result
* After the fork, continue both agents for a fixed post-fork horizon of 25% of the nominal training budget for that environment, measured from the fork point → ensures every run has the same amount of time to respond to the intervention
* The maximum total training budget is 120% of the nominal training budget → a fork must therefore occur by 95% to receive the full 25% post-fork horizon
* To isolate whether the critic pathology is propagating to the actor, perform plasticity injection (Nikishin et al. 2023) to the critic of one of the agents and not the other
* The two branches begin from the same complete training state at the fork → model parameters, optimizer state, replay buffer, RNG state, reproducible environment state, and relevant training counters are duplicated before the intervention
* Plasticity injection: freeze the critic's head, and add two newly initialized copies of it, one trainable and one frozen. The head's output = frozen old head + trainable new head − frozen copy of the new head. The two new copies start identical, so predictions are unchanged at the moment of injection. Earlier residual blocks remain trainable and receive gradients through the injected head construction. Trainable parameter count is unchanged and total parameter count grows by the size of the two new head copies (Nikishin et al. 2023)
* SimBa has no dedicated encoder, so the head is defined by choice → head = [last m residual blocks + post-layer-norm + output layer]
* The target critic is also injected using the same construction, and the normal SAC Polyak target update continues from the post-injection state
* New trainable injection parameters continue with newly created optimizer state while existing trainable parameters keep their existing AdamW state; frozen parameters are excluded from the optimizer and do not receive weight-decay updates
* Inject the single Q critic used by this SAC implementation
* Use a fixed panel of 256 state-action pairs sampled once from the replay buffer at the fork and fixed thereafter for the injection correctness checks
* All injected/control forks remain in the primary analysis regardless of whether the immediate plasticity-rescue check succeeds

Check 1, at the fork:

* Compute Q-values and $\nabla_aQ$ on the fixed panel for (1) the critic just before injection, (2) the injected critic immediately after injection, and (3) the control critic immediately after the fork → all three must match within an implementation-appropriate numerical tolerance determined from the actual dtype

Check 2, just after the fork:

* Run the plasticity probe on a copy of each critic → the injected critic should show improved plasticity relative to the control
* This is a validation of the intended plasticity-rescue effect, not something to force through implementation changes
* Whether Check 2 succeeds or fails does not determine inclusion in the primary analysis → a success-only subset may be reported separately as a secondary analysis

Check 3, later:

* Q-values, actor updates, and returns are expected to diverge → this is the result being measured, not something to force

Identity fork:

* Fork without injecting either branch → both must remain numerically identical within an implementation-appropriate tolerance over 1000 further training steps
* This tests fork fidelity (RNG, replay buffer, optimizer state, environment state/reproducibility, and relevant training state) separately from injection
* We will sustain the plasticity loss checks within both critics after the fork → verifies whether the injection changes the critic's plasticity trajectory and checks that the critic itself is not silently drifting in a way that confounds interpretation
* Observe the performance of the two agents thereafter, and measure for the development of actor-side degradation through observing overall agent performance and recording the following actor-side pathology/optimization diagnostics:
   * Policy churn → measures rapid changes in actor behavior and policy updates (Schaul et al. 2022; Tang & Berseth 2024)
   * Actor gradient norm → measures changes in the magnitude and stability of the actor optimization signal (Bjorck et al. 2021)
   * Action saturation / average absolute action → measures whether the actor's outputs become abnormally concentrated near the action bounds (Bjorck et al. 2021)
* The primary actor outcome is the return/performance trajectory; the actor-side pathology/optimization diagnostics are supporting diagnostics rather than separate primary endpoints
* Actor_grad_cosine is retained as a mechanism diagnostic for Experiment 3 rather than being treated as one of the primary Experiment 2 actor-side diagnostics

Limitations:

* Plasticity, exploration, and performance interact in RL (Nikishin et al. 2023) → a change in the injected agent's reward after the fork could partly come through changes in the actor/environment interaction, which is why return alone is not treated as the explanation for propagation
* The injection intervention tests whether rescuing critic plasticity after degradation changes the downstream actor trajectory, but it does not necessarily remove the original reason the scaled critic became degraded → if the critic later redegrades, this is an important result rather than an implementation failure
* With 5 seeds, a visible gap in the injected vs. control trajectories can still be affected by seed variation → plot percentile bootstrap confidence bands over paired seed-level differences
* Individual runs fork at different absolute training steps → all post-fork comparisons are therefore aligned by steps since fork, with the same 25% post-fork horizon
* If degradation occurs after 95% of the nominal training budget, the run is reported as a null result rather than receiving a shorter post-fork window
* The 25% post-fork horizon may not capture effects that emerge only after a longer period of continued training → conclusions are limited to the fixed post-fork observation window
* The actor-side pathology/optimization diagnostics are supporting measurements and are not individually treated as proof of propagation; the causal conclusion comes from the paired intervention and the downstream trajectory
* Policy churn and action saturation require definitions appropriate to continuous stochastic SAC rather than being copied directly from discrete-action or different actor-critic settings; the implementation should follow the relevant literature while preserving the meaning of the metrics
* Development/calibration runs are excluded from the confirmatory analysis → the selected injection configuration is frozen before the main Experiment 2 runs

---

## Amendments from the Phase 0 Q&A

Decided by the project lead on 2026-10-03, in reply to the Phase 0 questions. The
rationale and the full Q&A are in docs/exp12_decisions.md.

(a) HumanoidBench tasks. HumanoidBench uses h1-reach-v0 and h1-run-v0, the
no-hands H1 versions. h1-shelf-place is removed from both experiments. Both IDs
must be verified as registered and stepping. Status: blocked, see
docs/exp12_decisions.md H1/H2 (2026-10-03).

(b) Fork design. At the trigger the complete training state is saved. The
original process then restarts from that saved state and continues as the
CONTROL arm, running to max(100% of B, fork + 25% of B). The INJECTED arm is a
separate resumable job restored from the same save, running from the fork to
fork + 25% of B. Both arms use the same restore code path. Experiment 1's data
after the fork comes from the restored control. The identity fork must pass on
CUDA for every architecture and suite before any Experiment 1 grid run is
launched. The fork happens at the exact check step, with the full environment
state and its random state restored.

(c) Probe target offset. In each probe round both the current critic and the
stored fresh critic fit g(x) = a + sin(1e5 * f(x; w0)) on identical inputs with
an identical random function f(.; w0). Each critic's offset a is its own mean
prediction. Plasticity score P = b - (final probe loss), where b is the variance
of the targets. Plasticity loss L = P(fresh) - P(current), which is positive when
plasticity has been lost.
Limitations:
  (i) the current and fresh critics' targets differ by a constant shift;
  (ii) the probe budget (5 rounds x 1000 steps) is smaller than Lyle et al.'s
       (10 targets x 2000 steps).
On the development run only, the probe is also computed once with a single
shared offset, and the report states whether the sign of the plasticity loss
agrees. This is a one-time sensitivity check.

(d) m-selection rule, fixed before any result is seen.
  recovery(m) = (L at trigger - L after injection with m) / (L at trigger - L at the healthy reference)
0 means no change and 1 means fully back to healthy. Choose the smallest m whose
recovery is within 0.10 of the best recovery. Report recovery for all three
candidates (last block, half of the residual blocks, all residual blocks). m is
expressed by these labels so it transfers across architectures. Measure the
probe's own noise (the spread across the 5 rounds on the same critic). If it
exceeds 0.10, stop and consult the project lead.

(e) Positive control. D6W1536 on dog-run (DMC hard, 1M environment steps), with
a development seed outside 1-5. If it never triggers, stop and consult. The
trigger is never loosened.

(f) [SUPERSEDED by (q), 2026-10-04] Healthy reference. The "normally trained
critic" is the same critic's own earlier checks in the same development run,
before it triggers. No separate D2W512 reference run is used.

(g) No normalization in Experiment 2. No returns are normalized anywhere, and no
scalar summary is reported. Experiment 2 outputs graphs only: per-environment
paired differences (injected minus control) in raw return units, aligned by
steps since fork, with percentile bootstrap bands and every seed's line shown.
For post-hoc normalization later:
  - raw per-episode returns (and per-episode success where the environment
    provides it) are saved for every evaluation;
  - analysis scripts take a normalize(env, values) function that defaults to
    identity;
  - one metadata file per suite records benchmark-defined constants as data only.

## Amendments from the Phase 2 approval (2026-10-04)

(h) Initial random actions. As in SimBa's released code, Experiments 1 and 2
take uniform random actions until the replay buffer holds 5,000 transitions,
then use the policy. The agent still sees each observation, so obs_rms is
updated during the warm-up. The fresh-critic probe happens at that point,
before the first update.

(i) MyoSuite horizon and discount. Matching SimBa exactly, every MyoSuite task
uses max_episode_steps = 100, which gives gamma = 0.95 via the TD-MPC2 heuristic.
DMC and HumanoidBench keep 1000, which gives gamma = 0.99.
Limitation: the resulting TimeLimit(100) truncates myoHandKeyTurnFixed-v0 at
100 raw steps (50 interaction steps), half of its registered 200-step horizon.

(j) HumanoidBench single Q. SimBa's released code and paper use clipped
double-Q on HumanoidBench (paper Table 7: "Clipped Double Q: HumanoidBench:
True, Other Envs: False"; configs/env/hb_locomotion.yaml sets episodic: true,
and sac_simba.yaml sets critic_use_cdq: ${env.episodic}). The Methodology's
single Q critic is used everywhere. Limitation: on HumanoidBench this departs
from SimBa's configuration.

## Amendments from the Phase 3 approval (2026-10-04)

(k) HumanoidBench limitations (h1-reach-v0, h1-run-v0):
  - Physics runs on mujoco 3.6.0, the repo's pinned version shared with DMC and
    MyoSuite, rather than HumanoidBench's pinned 3.1.6. Results are not
    bit-comparable to SimBa's HumanoidBench numbers.
  - SimBa's released code lists the with-hands `h1hand-*` tasks, while its
    paper's Table 5 dimensions (action dim 19) match the no-hands H1 used here.
  - HumanoidBench needs an EGL offscreen context on GPU nodes
    (MUJOCO_GL=egl PYOPENGL_PLATFORM=egl).
  - The HumanoidBench integration is verified on CPU only; it is untested on GPU.

## Amendments from the Phase 4 approval (2026-10-04)

(l) Trigger: two consecutive firing checks. A check fires when the lower bound
of its 95% percentile bootstrap interval is above 0. f*_run is the check that
completes the first run of 2 consecutive firing checks (k-1 and k). It must
itself be at or before 95% of the budget, so the earliest f*_run is check 2
and the latest is check 19. Config: trigger.consecutive_checks = 2.

(m) Check 1 tolerance. The control arm, and the identity arm used for
validation, must reproduce the pre-fork critic's Q and dQ/da on the panel bit
for bit. The injected arm may differ from it by at most 64 float32 eps times
the pre-injection maximum magnitude, separately for Q and for dQ/da (the
reverse-mode summation order of dQ/da changes). Check 1 records the observed
maximum in eps units, the float32 matmul precision in use (configured and
measured) and the device model. If TF32 matmuls are detected, or any
deviation exceeds its tolerance, the arm stops before training and the
project lead decides. [TF32 part SUPERSEDED by (v), 2026-10-04: Check 1 runs
in full FP32 and no longer stops on TF32. Any deviation beyond its tolerance
still stops the arm.]

(n) Check 2 is reported as the paired difference P(injected) - P(control) on
the fork check's own pool, with its IQM and 95% percentile bootstrap interval.
It passes when the interval lies above 0. It is reported only and never
excludes a fork.
Limitation: Check 2's interval, like the trigger's, is a bootstrap over only
5 probe rounds. With 5 values the bootstrap distribution of the IQM is
coarse (few distinct values), and percentile intervals from so few
observations can be too narrow, so the nominal 95% coverage is not
guaranteed.

(o) Positive control. [SUPERSEDED by (q), 2026-10-04]
  - Healthy reference: the IQM of the per-round L pooled over all checks
    before f*_run. The last check before f*_run is reported as a sensitivity
    only and is never used to choose m.
  - Probe noise: the pooled SD of the per-round L over every probe evaluation
    in the positive-control script, divided by L_trigger - L_healthy (recovery
    units).
  - Stop and consult (exit 3) if L_trigger - L_healthy is non-positive, if it
    is below the pooled noise SD, or if the noise is 0.10 or more.

(p) Device model. Both arms of a fork run on the GPU model that produced the
fork state. The fork records it, and the control (on resume) and the arm jobs
refuse to run on a different model. The identity-fork validation runs for
D4W1024 and D6W1536 in each suite on the GPU model the grid will use.

## Amendments from the Phase 5-6 decisions (2026-10-04)

(q) Positive control: healthy reference, noise and stop rule (replaces (f), (o)
and decision E7). The healthy reference is check 0, the fresh critic at
initialization, before any critic training. Plasticity loss L is measured
against the fresh critic, so L_healthy = 0 by definition and
  recovery(m) = (L_trigger - L_after_injection(m)) / L_trigger.
Steps 1 and 2 of the positive control therefore probe the same critic.
Probe noise: the pooled SD of the per-round L over the script's evaluations
made with the experiment's real settings (per-critic offsets: the degraded
critic and the three injected critics), divided by L_trigger (recovery units).
The shared-offset repeats of (c) are reported separately and play no part in
the noise or the stop. Stop and consult (exit 3) if L_trigger <= 0, or if the
noise in recovery units is 0.10 or more.

(r) Experiment 2 bands. The per-environment bands of the paired return
differences use the IQM over seeds (percentile bootstrap over seeds), and
every seed's line stays visible.

(s) Time limits and truncation, matching SimBa's released code
(github.com/SonyResearch/simba @ 7d0358b).
  (a) MyoSuite builds the registered task, keeping its own registered time
      limit, and then applies one more TimeLimit with max_episode_steps = 100
      raw steps, before action repeat:
        scale_rl/envs/myosuite.py:  env = myo_gym.make(MYOSUITE_TASKS_DICT[env_name])
        scale_rl/envs/__init__.py:  # limit max_steps before action_repeat.
                                    env = TimeLimit(env, max_episode_steps)
        configs/env/myosuite.yaml:  max_episode_steps: 100
        configs/base.yaml:          gamma: max(min((eff_episode_len / 5 - 1) / (eff_episode_len / 5), 0.995), 0.95)
      eff_episode_len = max_episode_steps / action_repeat = 50, so gamma = 0.95
      for every MyoSuite task, myo-pen-twirl included. Our implementation does
      the same; nothing changed.
  (b) Time-limit truncation is bootstrapped and termination is not. The
      buffer stores the final observation on termination or truncation
      (run.py: "if terminateds[env_idx] or truncateds[env_idx]:
      next_buffer_observations[env_idx] = env_infos["final_observation"][env_idx]"),
      and the critic target uses (1 - terminated) only (sac_update.py:
      "target_q = batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * next_q").
      Our implementation is identical; nothing changed.
  Limitation: myoHandPenTwirlFixed-v0 is registered with a 50-raw-step time
  limit, which stays in effect inside the 100-step limit. PenTwirl episodes
  therefore end at 50 raw steps (25 interaction steps), while gamma = 0.95 is
  derived from 100. myoHandKeyTurnFixed-v0 (registered at 200) is truncated
  at 100 raw steps, as in (i).

(t) Checkpoints. One routine save per probe check (every N/20 interaction
steps). Each save is written to a new directory, and only then is the LATEST
pointer atomically repointed and older routine states deleted. A run
therefore keeps only its latest routine state, plus the complete fork state
saved at the trigger, plus the stored fresh critic. A crash mid-save leaves the
previous state as LATEST.

(u) Limitation: the whole grid, including both arms of every fork, must run
on one GPU model.

## Amendment from the precision decision (2026-10-04, revised the same day)

(v) Matmul precision (replaces the earlier FP32-everywhere version of (v)).
- Every float32 matrix multiplication in Experiment 1 and 2 jobs uses TF32
  on GPUs that support it. This covers training updates, action selection,
  probes, post-fork evaluation and diagnostics.
- It is set explicitly at the start of every GPU entry point:
  `jax_default_matmul_precision = "tensorfloat32"`. Nothing else forces a
  higher precision, and the setting is a no-op on CPU.
- No other mixed precision is used (no bf16 or fp16). Everything that is not
  a matmul stays float32, and x64 is off.
- The one exception is Check 1 (and the A0 error measurement), which
  computes Q and dQ/da in full FP32 inside a local
  `jax.default_matmul_precision("highest")` context. Its tolerances are
  unchanged: injected arm within 64 eps, control and identity arms
  bit-exact.
- The earlier rule that Check 1 stops an arm when TF32 is detected is
  removed. Every launch records its matmul precision setting, GPU model, and
  JAX, jaxlib and CUDA versions in run_metadata.json.

## Amendment from the Block A result (2026-10-05)

(w) Fresh-critic range check (A2 and B2): acceptance rule replaced.
- Original rule (pre-specified 2026-10-04, kept here for the record): at
  the configured pool (25,600), the fresh critic's P must lie within 10–90%
  of b, at every critic size.
- Original result (Block A, job 22667743, A100-SXM4-40GB, dog-run): FAILED
  at all three sizes. P/b was 0.990 (D2W512), 0.997 (D4W1024) and 0.993
  (D6W1536).
- Why it is replaced:
  - Lyle et al. (2023) chose the probe budget so that networks from random
    initialisation can solve the task. A fresh critic that fits the probe is
    therefore the intended design.
  - The rule and its fallback ladder (smaller pool, then more steps) assumed
    a floor effect. They would push the probe further into the ceiling.
  - The replacement was decided without any degradation data.
- New criterion: at the configured pool, P/b ≥ 0.9 at every critic size.
  The old rule is still reported, as information only. The step's exit code
  follows the new criterion.
- The probe settings are unchanged (pool, steps, rounds, offsets).
- What this check can and cannot show: it cannot establish sensitivity.
  Sensitivity comes from the positive control and the development run, where
  a critic that has lost plasticity must score clearly lower. The fresh-pair
  null measures noise.
- The spread of the per-round final loss is reported per size and pool. L =
  P(fresh) − P(current) is a difference of final losses on identical
  targets, so b cancels.

## Amendment on diagnostic precision (2026-10-05)

(x) Precision of the actor diagnostics. In every Experiment 1 and 2 job, the
actor forward passes behind the policy-churn and policy-KL diagnostics run in
full FP32, inside a local `jax.default_matmul_precision("highest")` context.
These are the churn reference before and after each update, and the KL's new
and old policies.
- Everything else keeps amendment (v)'s TF32: training updates, action
  selection, probes and evaluations.
- Training is unchanged. The Angle 1 path, which has no diagnostics, is
  unchanged.
- Reason: a per-update parameter change is often below TF32's resolution, so
  TF32 rounding biases these diagnostics. Under a CPU emulation of TF32,
  calibrated to the A100's measured matmul error, the KL computed for the
  same parameter pair changed by 9.6% (a 1×8 test actor) and by 0.7% (the
  1×128 actor) against FP32.
- Checked by `DiagnosticsPrecisionTest`: under the TF32 run setting the
  compiled update has exactly the diagnostics' forward passes at "highest",
  and its training matmuls stay TF32. A break-and-restore check removes the
  context.
