# Experiment 2 matched-data framework — proposal, 6 November 2026

This is an opt-in design and CPU data-admission prototype, **not** the approved
Experiment 2 intervention in `.claude/methodology-exp1-exp2.md`. It does not
change or launch Experiment 1, replace an active job's source, choose an Exp2
sample, or authorize a GPU run. Its question is whether critic size changes the
actor updates and resulting policy when the transition data are held fixed.

## Exp1 inputs we actually have

The official population is 11 environments, seeds 1–10, and D2W512,
D4W1024, D4W1536 critics; its actor is D1W128
(`experiments/exp12/official_exp1.py`, `configs/base_exp1_official.yaml`). The
selected source is `22e1666f0ff680699b06ded1ce70ec0962f9c103`; the owner
reported that the queued small job uses this SHA and the running large job uses
the separately frozen no-fork source
`d12ae8bad5273c4216997dcc3189cf15bbb54455`. No current completion is
inferred from that report. Historical accepted Exp1 cells may carry other
source SHAs and need separate provenance review.

Each **completed** Exp1 run retains `run_metadata.json`, `DONE`, and the
`state/LATEST` complete state. That state contains `buffer.npz` with raw
observation, action, reward, termination, truncation, and next-observation
arrays; `obs_rms.pkl` with the normalizer; `agent_ckpt/` with actor, critic,
target critic, temperature, optimizer state, and JAX RNG; and `meta.pkl` with
other restoration state (`experiments/exp12/state.py:83`,
`scale_rl/agents/sac/sac_agent.py:687`,
`scale_rl/agents/wrappers/normalization.py:111`,
`experiments/exp12/trainer.py:206`). The configured replay capacity is one
million transitions (`configs/buffer/numpy_uniform.yaml`); verify each saved
count rather than assuming the final buffer covers every interaction. Separate
Exp1 policies collected separate data even when environment and seed match.

`experiments/exp12/exp2_shared_data.py` is the current runnable framework:
it checks two *completed* small/large parents, checks their matching environment,
seed, protocol, replay shapes/dtypes and official probe setting, hashes the
replay and RMS bytes, and writes a non-executable JSON plan. It never copies or
edits the large source states. Both future arms can call `paired_indices` for
the same balanced, deterministic source/row selections at each update. The
source SHA and config hash for **each** parent remain explicit. This proves an
equal replay-index stream, not equal learned Bellman targets or equal initial
critic parameters.

Example, **after both chosen parents finish**:

```bash
cd main
python -m experiments.exp12.exp2_shared_data \
  --small-run /absolute/path/to/small/exp1/run \
  --large-run /absolute/path/to/large/exp1/run \
  --sampling-seed 17 --batch-size 256 \
  --out /absolute/new/output/exp2_shared_data_plan.json
```

The replay mix is provisionally half from each parent. This balances collector
origin within a batch but can still favor the two source policies over other
policies. Selecting the parent pair, pooling rule, seed, data-support test, and
holdout scheme requires scientific review before an actual study. The plan
contains no Exp2 training command.

## Proposed comparison to review

1. Use the same immutable replay plan and the same actor starting parameters,
   actor optimizer state, observation normalizer, minibatch sequence, learner
   update count, and evaluation episode seeds in both arms. Vary critic
   architecture D2W512 versus D4W1536. **Do not simply restore the two Exp1
   agents and call them matched:** their actors and training replays differ.
   Decide whether critics start fresh on this common data or retain their
   Exp1 weights. The latter tests trained critics but carries prior-data
   confounding; it is a different estimand.
2. Use the existing SAC update implementation, target networks, entropy rule,
   actor architecture, and optimizer settings. Offline-only replay may cause
   action-support/extrapolation problems not present in online Exp1, so qualify
   a tiny CPU lifecycle and a bounded GPU feasibility pilot before treating
   return differences as evidence of update quality. Define an update budget
   in **optimizer steps**, not extra environment interactions. The previously
   discussed 200,000-interaction continuation is not silently adopted here.
3. At predeclared update milestones, freeze both actors for evaluation only.
   Log per-episode return/success, actor loss, entropy, action changes, critic
   Q distributions and selected actor-gradient/update diagnostics. For a more
   direct comparison, start cloned actors from *one identical* actor state,
   apply each critic's proposed actor update on the same batch, and compare
   their policies using the same evaluation episode seeds. This is an
   additional paired diagnostic, not proof that any proxy is causal.
4. Use the **existing** Exp1 probe implementation
   (`experiments/exp12/probe.py`, `experiments/exp12/run_probes.py`): ten
   randomized sine targets, 2,000 continuous AdamW updates per target,
   architecture-specific teacher, own-mean offset, paired current/fresh
   reporting, and raw per-round learning curves. No new target or score
   definition. A fixed-data study can have 50 evenly spaced *learner-update*
   checks; Exp1's 50 checks are on *interaction* steps, so the time axes must
   be labeled separately. The same probe code would operate on the common
   replay, not the original arm-specific Exp1 replay; results are not
   numerically interchangeable with Exp1.

## Decisions and implementation still required

The owner has **not yet frozen** the parent/data-source selection, whether the
estimand is fresh-on-common-data versus restored Exp1 critics, common actor
origin, treatment of normalizer statistics, offline update budget and check
grid, held-out data/support diagnostics, evaluation seeds/episodes, or the
paired one-update outcome measure. Those choices affect scientific results.
Until selected, do not wire a training launch or promote this prototype as the
approved Experiment 2. Once selected, the minimum engineering continuation is
an opt-in runner using the existing SAC agent and probe functions, with exact
parent/plan hashes, resume tests, and CPU checks that both arms see identical
raw transitions and normalized inputs. The separate historical injection Exp2
source remains intact.

The fixed-dataset critic-depth precedent is *Simple Actors and Deep Critics
for Scalable Reinforcement Learning* (2026), Section 5.3, Figure 2,
<https://arxiv.org/abs/2608.26659>. Its offline actor objective and OGBench
data differ from SimBa-SAC; it does not establish identical minibatch streams
or our policy-update mechanism.
