---
description: Protect research compute, experiment outputs, checkpoints, and long-running jobs from accidental waste or corruption.
---

# Compute and Data Safety

This repository contains expensive and sometimes irrecoverable research runs. Before consuming substantial compute or modifying research outputs, prioritize correctness, recoverability, and clear experiment separation.

## 1. Compute Before Launching

Never launch a large or long-running experiment just to discover whether the code path works.

Before expensive runs:

1. Verify the configuration.
2. Verify the experiment name.
3. Verify architecture.
4. Verify environment.
5. Verify seed.
6. Verify checkpoint arguments.
7. Verify output paths.
8. Verify GPU/device assignment.
9. Run the smallest practical smoke test.

A short successful smoke test is required before scaling to many seeds or long horizons when the code path is new or recently modified.

Do not launch multiple expensive jobs merely because the first one has not yet been inspected.

## 2. Seed and Process Separation

Before launching multiple seeds, verify that every process has a distinct:

- seed
- output path
- checkpoint path
- log path
- WandB identity
- GPU/device assignment

For multi-seed launches, inspect the final commands before submission.

Do not assume copy-pasted commands differ correctly.

Especially verify:

- `--overrides seed=...`
- `CUDA_VISIBLE_DEVICES`
- checkpoint directories
- log filenames
- output directories

A duplicate seed or duplicate output path can invalidate an otherwise expensive run.

## 3. Expensive Compute Is Evidence-Producing Work

Treat large experiments as scientific data collection, not disposable computation.

Before consuming substantial compute, ask:

- What scientific question does this run answer?
- Is the configuration already validated?
- Is this run necessary?
- Could a shorter run validate the implementation first?
- Could an existing artifact answer the question instead?

Do not spend large amounts of compute to test a hypothesis that can be checked with a small deterministic test.

## 4. Never Destroy Research Outputs

Do not delete:

- checkpoints
- replay-buffer artifacts
- metrics
- logs
- WandB runs
- probe datasets
- onset ledgers
- baseline caches
- result directories
- analysis outputs

unless explicitly instructed.

Do not overwrite a previous experiment with a new experiment using the same path.

When semantics or configuration change, prefer a new run identity/path.

Treat completed experimental outputs as immutable evidence.

## 5. Checkpoint Safety

Before launching a long run with checkpointing:

- verify the checkpoint directory is unique;
- verify checkpoint filenames encode the relevant run identity;
- verify checkpoints are actually being written;
- verify a short save-and-resume test when checkpoint behavior has changed.

Never assume that because checkpoint code executed, the checkpoint is usable.

A checkpoint must not silently belong to another:

- seed
- architecture
- environment
- experiment
- configuration

If a requested checkpoint cannot be loaded, fail clearly rather than silently starting a fresh run.

## 6. Long-Running Jobs

For long jobs:

- prefer persistent storage;
- verify that outputs are written during training;
- do not rely on ephemeral session storage;
- make sure a runtime interruption does not destroy the only copy of the results.

For remote environments such as Kaggle, Colab, or HPC systems, confirm that the storage location persists beyond the current session or allocation.

## 7. HPC / Delta Jobs

For Delta or other HPC systems:

- verify the requested GPU count;
- verify CPU allocation;
- verify that each process is assigned to the intended GPU;
- verify Slurm task/process separation;
- verify output paths before submission;
- avoid requesting substantially more resources than the workload requires.

For independent seeds, prefer separate processes/jobs rather than modifying scientific training code to create artificial cross-seed coupling.

The MacBook used to submit a remote job should not participate in the computation unless explicitly intended.

## 8. GPU Utilization and Benchmarking

Before estimating the cost or duration of a large campaign, benchmark a representative run.

Measure, when practical:

- interaction steps/sec
- wall-clock time
- GPU utilization
- CPU utilization
- memory usage

Do not estimate speedup from hardware specifications alone.

Do not redesign the research implementation solely for performance before measuring the actual bottleneck.

If a run appears CPU-bound, determine whether environment stepping, replay-buffer operations, Python orchestration, or synchronization is responsible before changing the experimental configuration.

## 9. Vectorization and Performance Changes

Do not increase:

- `num_train_envs`
- vectorized environments
- batch sizes
- update frequencies
- parallel workers

solely because they appear faster.

A performance change may alter:

- RNG behavior
- interaction-step semantics
- replay-buffer ordering
- episode handling
- logging frequency
- training dynamics

If a performance optimization could affect experimental semantics, treat it as a methodological change and stop for review.

## 10. WandB Safety

Before launching a large sweep:

- verify the project name;
- verify run naming;
- verify seed and architecture metadata;
- verify the run is actually online if online logging is expected;
- verify a short test run appears correctly in WandB.

Do not trust a clean exit code as proof that WandB logging succeeded.

WandB should not be the only copy of raw experimental evidence when persistent local storage is available.

## 11. Output Naming

Every long-running experiment should have an output identity that makes the following recoverable:

- experiment
- environment
- architecture
- seed
- relevant configuration/version

Never allow two experimental conditions to silently write into the same output location.

If a configuration change would make two results scientifically incomparable, use a new run identity.

## 12. Failure Handling

If a job fails:

1. determine whether the failure is deterministic;
2. inspect the actual traceback/log;
3. identify whether outputs are partial;
4. preserve useful artifacts;
5. fix the cause before launching many more copies.

Do not repeatedly relaunch an obviously deterministic failure.

Do not delete the failed run merely to make the result directory look clean.

Failed runs are useful debugging evidence.

## 13. Before Multi-Seed Campaigns

Before launching a full seed sweep, verify:

- one seed works;
- the output is correct;
- metrics are sane;
- WandB metadata is correct;
- checkpoints are written if enabled;
- seed separation works;
- the output paths are unique;
- the environment/configuration is the intended one.

Only then scale to multiple seeds.

## 14. Research Campaign Changes

If a code or configuration change could invalidate an already-running experiment:

**stop rather than silently continuing under the modified implementation.**

Do not mix results produced by materially different code/configuration versions under the same experimental identity.

## 15. Priority

When choosing between:

- saving compute but risking experimental validity,
- or spending a small amount of additional compute to validate correctness,

prefer the latter.

A small amount of wasted compute is preferable to a large amount of scientifically invalid data.