# Proposed online reference-policy Experiment 2 prototype

Status: opt-in CPU prototype, **not a frozen scientific protocol or GPU launch**.
Source starting reference: `22e1666f0ff680699b06ded1ce70ec0962f9c103` (official Exp1 candidate). The actual small and large Exp1 jobs may have distinct source SHAs. This prototype does not alter the historical injection/fork Exp2 or the immutable-replay PR26 design.

## Question and design

An independent reference SimBa-SAC agent owns online collection, replay, observation normalization, actor, critic, target, temperature, and training RNG. Fresh D2W512 and D4W1536 shadow critics learn from each newly sampled reference replay batch. Neither shadow has a live actor or writes reference state. At selected checkpoints, copies of the reference actor and its optimizer state take one SAC actor step against each frozen shadow critic, then the unchanged actor and both copies are evaluated on matched episode seeds. The copies are discarded. This measures a local actor-guidance contrast under one reference trajectory; it does not measure eventual return after replacing the live critic.

The reference critic architecture, tasks/seeds, interaction budget, checkpoints, episode count/mode, analysis aggregate, and resource cap are deliberately null in `configs/base_online_exp2.yaml`. `validate()` rejects missing values. Only tiny `run_role=dev` fixtures may override the shadow architecture/probe size. Production retains the official probe constants and fails if they differ. No runner submits a job.

## Audited source symbols

| Concern | Source symbols and role |
|---|---|
| Actor and sampling | `SACActor`, `SACAgent.sample_actions`, `_sample_sac_actions`; training samples policy actions, evaluation uses temperature 0 in the source path. |
| Critic/targets | `SACCritic` or `SACClippedDoubleCritic`, `sac_update.update_critic`, `update_target_network`; separate target Q, source `(1-terminated)` bootstrap mask, and Polyak update. |
| Optimizers/temperature | `_init_sac_networks` builds AdamW actor/critic and temperature trainers; `_sac_update` orders actor, temperature, critic and target updates. `Trainer.apply_gradient` owns moments and parameter updates. |
| Replay/normalization | `NpyUniformBuffer.sample` selects uniform indices; `ObservationNormalizer.update_many` normalizes both observation fields from one RMS snapshot. The optional `return_indices` path records IDs without changing draws. |
| Environment/evaluation | `exp12.envs.create_envs`, `create_eval_env`, `ExactRestore`; `Exp12Trainer.evaluate_episodes` supplies the ordinary reference evaluation. Prototype evaluations use separate instances and matched seeds. |
| State | `Exp12Trainer.save/restore` and `exp12.state.commit_state_dir` persist the reference agent, replay, RMS, RNG and environment. The new `before_commit` hook stores shadow state in the same uncommitted directory. |
| Probe | `probe.run_probe`, `probe_round`, `_fit`, `summarize` through `run_probe_networks`; each shadow uses its own critic-shaped teacher and exact saved fresh parameters. |

The reference trainer samples all batches for a UTD block, then the shadow callback reads the same batches and a single reference actor/temperature/normalizer snapshot at that block boundary. It draws one next action and log probability per batch, updates both shadows with their own target Q, then the ordinary scanned reference update runs. The callback uses independent JAX/NumPy diagnostic streams. A controlled CPU fixture compares the resulting reference actor, critic, RNG and replay with the ordinary trainer. Shadow updates retain the source critic MSE, termination mask, AdamW and target update rule. Truncation remains bootstrap-eligible as in source.

## Outputs and restart

`online_exp2.jsonl` contains update indices, batch and target-policy hashes, shadow losses/Q-target summaries, episode-level returns and contrasts, intervention metrics, and probe artifact references. `interventions/step_*.npz` retains raw flattened actor gradients, realized parameter deltas, action panels and their observation indices. `probes/{small,large}_*.npz` retains per-round scores, full-pool losses and curves. `analysis.json` contains one equal-checkpoint run summary. The existing trainer logs retain the reference return trajectory. Each committed state includes `online_exp2.msgpack` and a SHA-256 manifest beside the complete reference state; restore rejects a mismatched configuration, source bytes or damaged shadow payload. Completed checkpoint diagnostics are represented in the saved ledger and are not rerun on resume.

Both shadow probes use the same replay-pool seed and indexing. Each architecture retains its own teacher and its trained/fresh pairing, as in the existing assay.

Run-level analysis should first average `advantage_large` over matched episodes at each checkpoint, then weight scheduled checkpoints equally within the independent reference-training run. Keep environments separate and retain every raw episode/check/round. Gradient cosine is undefined when either norm is zero. These outcomes are not proof of gradient correctness, Q truth or a causal plasticity mechanism.

## CPU use and limits

The entry point is `python run.py --experiment online_exp2_prototype --config_name base_online_exp2 --checkpoint_dir /absolute/new/path --overrides ...`. It fails on unset scientific fields. The test fixture uses small explicit settings and a deterministic synthetic environment. It exercises online updates, fresh probe, actor intervention, matched evaluation, saved artifacts and restore. Before any GPU pilot: freeze the null fields, verify source-bound full-size probe parity and real environment reset/seed behavior, run long checkpoint/restart identity, inspect per-process memory/throughput and storage, then obtain a separate resource/launch decision. No resource cost has been measured by this CPU fixture. The literature named in the owner brief informs the design but does not replace pinned source semantics or establish this prototype's correctness.
