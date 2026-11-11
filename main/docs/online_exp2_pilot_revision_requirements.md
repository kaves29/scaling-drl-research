# Experiment 2 pilot: revised requirements

The revised design supersedes the diagnostic horizon and evaluation schedule in
`online_exp2_revision_spec_20261110.md`. It is a requirements record, not a
claim that these changes have been implemented or qualified.

- Twenty independent reference runs: dog-run and humanoid-run each have three
  matched seeds; myo-key-turn and myo-pen-twirl each have two matched seeds.
  For every environment and seed, train one small-reference and one
  large-reference agent. Both shadow architectures learn from that run's
  shared reference data. Use the Experiment 1 raw environment-step budget:
  one million steps per run, action repeat two.
- At 15 approximately evenly spaced reference-training checkpoints, pause
  reference training and separately run the established plasticity probe and
  four policy branches. Run the tests and branches sequentially. Every test
  starts from an isolated copy of the same reference checkpoint; no branch or
  probe may change the reference run or another test's starting state.
- The policy branches cross small versus large shadow critic with frozen
  critic versus learning critic. Actors learn in all four branches. In each
  learning-critic branch, start with an independent copy of reference replay
  and add that branch's own experience. Preserve critic, target, optimizer,
  replay, actor, temperature and random-state semantics explicitly.
- Each branch runs for 10,000 **raw environment steps**, counting actual
  executed action repeats, approximately 5,000 actor interactions at repeat
  two. Capture ordinary training episode returns and established actor and
  critic diagnostics; mark 5,000 and 10,000 raw steps on graphs. Do not run
  separate evaluation episodes solely for these marks.
- Keep probe, branch condition, shadow architecture and reference regime
  identifiable and separate. Record environment, seed, checkpoint, run ID,
  architecture, source/config identity and raw counters. Enable W&B logging
  under the requested Experiment 2 Pilot Testing project, using a valid W&B
  project identifier and the established entity where available.
- Target Kaggle. Schedule no more than two independent reference runs per GPU;
  detect actual devices and validate memory before enabling concurrent jobs.
  The requested concurrency is a ceiling, not evidence it fits. Do not assume
  that two visible GPUs imply multi-GPU training of one run.
- Preserve durable restart and test checkpoint isolation, replay isolation,
  raw-step accounting including early termination, logging, and scheduling.
  Qualify the full revised path and concurrency on actual Kaggle hardware
  before presenting the pilot as ready. GPU hours are not a launch criterion.
