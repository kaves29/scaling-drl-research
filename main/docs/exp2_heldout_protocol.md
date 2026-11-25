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
