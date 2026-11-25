# Held-out replay diagnostics (experimental)

The opt-in runner `python -m experiments.exp12.replay_exp2_heldout` accepts
`--parent-manifest`, `--output`, `--identity` and `--checkpoint-count` (15 or 20).
It writes a separate, source-bound output and rejects old pilot directories.
It collects 5,000 raw steps once from the restored reference actor and uses
the fixed data only for diagnostics. The offline critics train on the original
parent replay; each of the four actor branches runs 5,000 raw steps.

The dataset retains transitions, episode boundaries, and termination/truncation.
The reward-to-go is interaction-discounted and excludes SAC's entropy term;
truncated and unfinished trajectories are censored. Comparisons to Q are
descriptive, not a ground-truth estimate of the soft value. Train and held-out
TD residuals are model-dependent bootstrapped errors, not proof of overfitting.
Gradient and Adam parameter-update comparisons are computed on the same states,
actor and random action key. Return is still measured in environment branches.

Use the separate `replay_exp2_qualification` module with `--heldout` and the
existing two-phase `--phase interrupt` / `--phase resume` protocol for a GPU
qualification. `--a100-smoke` is a shortened engineering check, not full-horizon
qualification. No GPU qualification, full-size parent restore, W&B online
delivery or production run has been verified for this branch. Do not use this
branch for production until those checks pass and the scientific outputs are
reviewed. Older pilot results are not resumable under this protocol.

## Delta two-hour pilot allocation

`infra/delta/heldout_pilot_2h.sbatch` requests two A100 40 GB GPUs, 16 CPUs,
128 GB host memory, and a two-hour wall-time limit. It starts the held-out
protocol with 15 checkpoints and up to six workers per GPU; the launcher checks
free VRAM before starting workers and lowers the per-GPU worker ceiling after
an out-of-memory failure. A Slurm warning 120 seconds before the limit stops
new work, signals active workers to save resumable state, and returns status 75.

The source manifest must contain ten matched Exp1 parent pairs (small and large
reference regimes) across `dog-run` seeds 1–3, `humanoid-run` seeds 1–3,
`myo-key-turn` seeds 1–2, and `myo-pen-twirl` seeds 1–2. This is 20 Exp2
parent-specific runs, not ten individual processes. The two-hour request is a
bounded allocation slice; it does not guarantee that all 20 runs finish in
that slice. To resume a stopped slice, resubmit the same script with the same
manifest and results paths and mode `resume`. For the first slice, use mode
`fresh` and a new results directory. Never use an old pilot output directory.
