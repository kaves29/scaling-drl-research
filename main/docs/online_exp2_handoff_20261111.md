# Experiment 2 revised pilot: continuation handoff

Status as of 12 November 2026: **GPU engineering qualification passed; the
scientific pilot is not running**. The private Kaggle pilot notebook
`shouryakaveti/experiment-2-actual-pilot` version 1 stopped during setup because
Kaggle could not provide `WANDB_API_KEY`; no pilot training began. The Cloud
W&B credential authenticates and can access `Experiment-2-Pilot-Testing`, but
that credential is not automatically passed into Kaggle. Enable the existing
W&B secret for this Kaggle notebook, then rerun it. The owner's latest
requirements are in `online_exp2_pilot_revision_requirements.md`.
This work belongs on branch `codex/exp2-online-prototype-20261107`. Read the
branch's current HEAD, preserve all pre-existing uncommitted files, and use the
root/project Agent-Context research instructions and relevant skills.

## Approved pilot design

Twenty independent reference runs: dog-run and humanoid-run, seeds 1–3;
myo-key-turn and myo-pen-twirl, seeds 1–2; each environment/seed has one
small-reference and one large-reference run. Each runs one million raw
environment steps with action repeat two and trains both shadow critics from
the reference replay. At 15 approximately evenly spaced checkpoints, run the
plasticity probe and four policy branches sequentially, isolated from each
other and the reference. Branch conditions cross small/large shadow critic
with frozen/learning critic. In every policy branch the actor learns; in the
learning-critic branches copy reference replay independently and append branch
experience, and use the source SAC update. Each branch runs 10,000 **actual
raw environment steps**, not 10,000 actions; handle early termination in action
repeat and avoid overshooting. Use normal branch-training return graphs, with
5k and 10k raw-step markers, without extra evaluation episodes. Preserve the
existing actor and critic diagnostic definitions. Label outputs by reference
regime, architecture, environment, seed, checkpoint, condition and run ID.
Enable W&B under a valid project identifier representing `Experiment to Pilot
Testing`, using the established entity if accessible. The owner wants Kaggle,
at most two independent runs per GPU; verify actual VRAM and reduce concurrency
if needed. The earlier qualification saw two T4s. Do not assume a GPU model or
that a full experiment fits Kaggle session/time limits.

## Implemented versus remaining

The old pilot (`online_exp2_pilot.py`) supports **only frozen critics**, stops
after 10,000 actor interactions, runs separate evaluation episodes, and has
resource-cap and analysis assumptions for the old protocol. It is not the
approved new pilot. `revised_exp2_population.py` supplies the population and
nominal 15-checkpoint interaction schedule; it is not integrated. Actual
reference raw-step tracking needs careful work because the trainer currently
uses interaction count times action repeat. `copy_reference_replay` and
`learning_branch_step` in `online_exp2_diagnostic.py` are tested primitives,
not a complete branch. The replay copy clears an unfinished n-step queue to
avoid joining different episodes; verify this decision and memory cost.
`learning_branch_step` calls the source `_sac_update`; the parity test is a
consistency check, not independent validation. Wire durable isolated state,
resume, main-run preservation, branch-local replay and normalizer/temperature
semantics, raw-step caps, metrics and logging. Change config validation and
analysis so they do not silently enforce old evaluation semantics. Avoid
unbounded logging or artifacts, preserve raw evidence, and run meaningful
CPU and GPU integration/restart tests before the scientific pilot.

The last local diagnostic primitive suite had 13 passing tests. Population
test was run directly because the research venv lacks pytest. Neither test
validates the full revised runner. Existing uncommitted notebook
`main/notebooks/Experiment 2 Testing.ipynb`, untracked notebook folders and
`exp2-online-prototype-20261107.bundle` predate this handoff: preserve them.

## Kaggle access and verified prior run

First check the current task's available Kaggle connector/CLI and whether an
appropriate Kaggle credential is present **without printing secrets**. A prior
Cloud task authenticated using an existing `KAGGLE_API_TOKEN` and the official
Kaggle CLI with `KAGGLE_CONFIG_DIR=/tmp/kaggle-cli-config`; ephemeral paths,
tools and secrets may differ in another task. Do not ask the owner to paste a
credential into chat. Check status/logs before rerunning or overwriting.

The prior private notebook `shouryakaveti/experiment-2-revised-pilot-gpu-qualification`
version 4 finished successfully and reported both visible T4s plus short
full-size branch checks and restart. It pinned source `9b3aad9ea1823e8fa4ae341bfef84b5a2c898ec7`,
used 20 interactions per branch, and skipped full plasticity; it did **not**
validate the new unfrozen-critic design, 10,000 raw steps, concurrency or the
scientific pilot. Finalization produced a 52 MB notebook and 49 MB HTML, so
limit captured logs. Qualification URL:
https://www.kaggle.com/code/shouryakaveti/experiment-2-revised-pilot-gpu-qualification

The owner authorized implementation, GPU qualification, and Kaggle pilot launch.
The latest GPU qualification is the private notebook
`shouryakaveti/experiment-2-resume-qualification`, version 6, completed
successfully. Its `combined_qualification.json` confirms source
`a7b229335b9737ab1ea1fe1b9f428bbc3ea26a6a`, small and large regimes, all four
10,000-raw-step branches, plasticity probes and restart. The receipt SHA-256 is
`e42829e20049bfa75545f96cafc42d57184f85c2e4345171468b02036db63d7a`. It
qualifies one process per GPU only; it explicitly does not qualify two jobs on
one GPU or the one-million-step reference budget.

The real pilot notebook is `main/notebooks/exp2_actual_pilot/` and Kaggle
version 1 is at
https://www.kaggle.com/code/shouryakaveti/experiment-2-actual-pilot. Its first
API-launched execution ended at the Kaggle Secrets lookup before installing
dependencies or starting GPU work. After enabling `WANDB_API_KEY` for this
notebook in Kaggle Add-ons > Secrets, rerun the notebook. W&B must remain online
for the scientific pilot; do not silently switch pilot logging to offline mode.
The notebook requests two T4 GPUs and asks the scheduler for a maximum of two
jobs/GPU, but the source-bound admission check will use one/GPU until same-GPU
concurrency is qualified. Do not launch on Delta or affect Experiment 1 jobs.
