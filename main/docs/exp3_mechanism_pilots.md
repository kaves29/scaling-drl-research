# Experiment 3 mechanism pilots

The pilot asks which part of the critic-to-actor pipeline merits a controlled
intervention. Its outcome is a shortlist, with null and contradictory results
retained. It does not choose the full Experiment 3 intervention.

## Evidence and source priority

The November 23 Experiment 1 snapshot found lower median final-probe plasticity
loss for D4W1536 than D2W512 in all ten environments with matched pairs (79 pairs).
Latest actor saturation was higher in those environments, but the actor metrics
were not matched by training step. Saturation is not equivalent to lower return.
The Experiment 2 hypothesis is worse policy improvement with larger critics
trained on shared replay. Treat that prediction as unverified until paired pilot
returns are audited. Record the experiment and source for every result.

1. Start with the verified final **Experiment 1** parents. Pair critics by
   environment and seed. At each of the two reference actors, bring both critics
   onto the same raw observations with their native observation normalization.
   Reuse the reference actor, optimizer, temperature and saved replay. These
   comparisons include the two original training histories; they cannot attribute
   differences to capacity alone. These are the historical parents approved for
   replay reuse, not a claim to cover the entire official Experiment 1 cohort.
2. For attribution, repeat on retained **Experiment 2 replay-pilot** states from
   the same reference parent. The saved parent identity, config hash, source SHA,
   protocol and serialized critic payload must agree. The frozen reference actor
   and its optimizer are loaded from Experiment 1. The result records the exact
   Exp2 update checkpoint. Compare those results separately from Exp1 screens.
3. Generate short rollouts for evaluation and a short new experience stream only
   for pilot 4. Use an available matched Exp2 checkpoint for pilot 4. A missing
   checkpoint is reported as missing and does not trigger a full training run.

## Six questions and what would advance each one

| Pilot | Manipulation and measurement | Interpretation and intervention lead |
| --- | --- | --- |
| 1. Policy step size | With both critics frozen, apply 0.25, 0.5, 1 or 2 times one fixed actor displacement per critic. Log held-out policy KL, return change and fraction of near-boundary sampled actions. | A direction that improves return at small steps but loses it at ordinary size motivates an adaptive actor step limit. Comparable changes at small sizes help separate step size from guidance. |
| 2. Local guidance | Match small and large update proposals at held-out KL targets 0.001 and 0.005. Separately fork the same exact simulator state and try opposite small perturbations along each critic direction. Follow the unchanged actor afterward and compare both critics' rankings on the **same** action pairs against paired discounted soft rollout returns. | A gap that survives KL matching motivates inspecting local slopes. Disagreement with soft rollouts motivates local gradient calibration. Finite horizons and using a common continuation actor limit interpretation, especially for Exp1 critics trained with different actors. |
| 3. Action saturation | Compare standard updates with a squared penalty on sampled action magnitude above 0.95. Keep critic and temperature fixed and match KL at 0.001. Report saturation above 0.99. | If the penalty reduces saturation and selectively improves large-critic returns, test it on held-out cases. The penalty also changes the update direction, so this is intervention evidence rather than mediation proof. |
| 4. Experience feedback | With paired Exp2 shadow critics and the same starting actor, cross size with shared or own experience. Shared transitions come from the fixed reference policy. Each arm uses the same update count, random indices, and mixture of old replay and newly observed transitions. Record post-training returns. Relate the interaction to the existing Exp2 frozen and learning branch curves. | A gap that grows mainly with own experience motivates a data-collection control. A gap on shared new experience motivates examining the critic update itself. A 256-interaction window is a screen rather than replication of the longer Exp2 branches. |
| 5. Entropy versus critic guidance | Measure the pre-optimizer critic and entropy gradient norms and alignment using the same states and actor samples. With the large critic frozen, change only the actor entropy coefficient to match the small critic's norm ratio. Compare standard and adjusted proposals at matched KL. | Selective recovery of return with reduced saturation supports balancing the two signals as an intervention. Matching norms does not match gradient directions. |
| 6. Actor capacity | Cross critic size with original and doubled actor width. Widen with a function-preserving initialization and verify the action distribution. Reset Adam state for both widths. Measure return at matched KL and critic gradient cosine between disjoint state batches. | A selective widening benefit at large critic size motivates a controlled actor/critic co-scaling study. Width also changes optimization. Gradient cosine is a descriptive interference proxy. |

The baseline is the unchanged actor on the same evaluation seeds. Store raw
episode returns and paired changes, not just means. An unreachable KL is labeled
and omitted from comparisons that require matching. Seeds remain the independent
units; the two reference regimes, multiple action pairs, and repeated evaluation
episodes are correlated measurements. Start with seed 1 in dog-run and
myo-pen-twirl, each in both reference regimes. Reserve seed 2 in both environments
for confirmation of selected leads. The default horizon is 500 policy
interactions (episode termination may shorten it). The rank probe uses three
seeded starting states and four paired continuations per state and direction.
Neither a short pilot nor a saturation improvement alone establishes a scaling
mechanism. A later fixed-depth width series with an intervention and control
should test whether a rescue grows with critic width while preserving plasticity.

Relevant literature motivates the contrasts, without establishing the cause in
these runs: [TRPO](https://proceedings.mlr.press/v37/schulman15.html) for controlling
policy change, [Compatible Gradient Approximations](https://arxiv.org/abs/2409.01477)
for local critic guidance, [SAC](https://arxiv.org/abs/1812.05905) for the entropy
term, [Nauman et al.](https://proceedings.mlr.press/v235/nauman24a.html) for competing
actor-critic pathologies, and [SimBa](https://arxiv.org/abs/2410.09754) for scaling
and architecture. Avoid claiming a first result on these general topics.

## Delta qualification and launch

Use the published source SHA in a clean checkout and the existing Python/CUDA
environment. Use the verified 20-run parent manifest on Delta. The launcher runs
one case per array task on the A100 40 GB partition. It writes an atomic JSON
result, refuses to overwrite an existing one, and logs peak process RSS via
`/usr/bin/time -v`. No W&B credentials or Modal transfer are needed for
the Exp1 screen. Keep qualification and production outputs separate.

Array mapping: 0/1 dog-run seed 1, 2/3 dog-run seed 2, 4/5 myo-pen-twirl seed 1,
6/7 myo-pen-twirl seed 2. Even tasks use the small reference actor; odd tasks
use the large reference actor. First qualify the heavier Myo case with task 5
and a 30-minute limit. Sample GPU memory through the job, record peak RSS,
runtime and result completeness. If the run times out, use its timing logs and
the 2-hour cap for a second qualification, without widening concurrency.
A fixed 32 GiB host request and one worker per GPU are the launch defaults.
Two array tasks may run concurrently on separate GPUs. Six workers sharing an
A100 was a target for Experiment 2 and is not qualified for these diagnostics.

From the previous Exp2 smoke, the 4,625 MiB VRAM and 11.9 GiB process RSS were
single-worker observations on an A100 40 GB. They are a starting reference,
without coverage of wider actors, full rollout counts or feedback compilation.
One worker per GPU is the only initial operational limit. Increase concurrency
only if a simultaneous two-worker measurement covers both environments and
worst observed phases, with enough GPU and Slurm cgroup RAM headroom. This
launcher deliberately uses separate allocated devices instead of co-location.

Budget ceilings: one 30-minute qualification costs at most 0.5 physical
GPU-hours. Four screening cases at two hours each can use up to 8 GPU-hours
and at most four hours of allocated wall time if two slots remain available
throughout, excluding queue waits. Four held-out cases add up to 8 GPU-hours.
A separate eight-case feedback array adds up to 16 GPU-hours. These are request
ceilings, not speed predictions. Use measured qualification time before
estimating completion. Submit screening tasks `0,1,4,5%2`; preserve
`2,3,6,7%2` for follow-up. Use the same optional Exp2 input for paired
feedback cases only when the retained checkpoints are present and valid.

With all five Exp1 diagnostics enabled, the maximum per case is approximately
74,000 evaluation interactions before early episode endings (up to 50,000 for
actor variants and 24,000 for ranking rollouts), plus short state collection.
Controls shared by pilots 2, 3, and 5 are evaluated once. The other Exp1
critic is restored without loading a second replay buffer. Use the recorded
wall time on both environments to refine the estimate.

CPU engineering checks include stock SAC update parity, widened actor
distribution parity, reward/entropy gradient ratio matching, n-step episode
boundaries, simulator state replay, and refusal of wrong-source or corrupt
Exp2 checkpoints. A synthetic fixture exercises all six paths. Real dog-run
and myo-pen-twirl resets reproduced the seeded initial observation and first
transition on CPU. These checks do not validate the contents of the actual
Delta checkpoints or qualify A100 memory.
