# Development m-control pilot — implementation and Delta handoff

Owner confirmed this five-condition plan on November 1. It does not adopt
whole-critic FIRE injection. Base: PR #21 at
`38e5abb7325a949e23440d6197b5d7d9ecd57c43`, based on PR #20. This successor
is a review candidate, not production promotion or launch approval.

## Fixed protocol

`base_exp12_m_control_ten` inherits the approved Method C producer on this
candidate: dog-run D4W1536, development seed9, 500,000-interaction budget,
10 sine targets, 2,000 continuous AdamW updates, 50 scheduled checks.
First1,000 draws and next1,000 draws retain the original streams and continuous
optimizer state. All existing teacher, normalization, own-offset, optimizer,
precision, trigger and eligibility settings remain inherited unchanged.
Development seed9 is excluded from official seeds1–8.

Use a saved eligible development checkpoint if one exists. The runner rejects
wrong settings, forced triggers and inconsistent natural-fork evidence. Otherwise
run one development parent with the same trajectory, ending once the first
eligible natural fork is fully saved, restored and its control identity check and
initial evaluation have completed. It does not continue post-fork SAC training.
If no eligible fork occurs, stop; do not force damage, alter thresholds, switch
environment or sweep seeds. Checkpoint files from interrupted generation remain
available but do not authorize an automatic retry.

The existing positive control evaluates five critics on the same per-round pool,
teacher and minibatch streams: exact saved fresh, unchanged degraded, injected
last/half/all (1/2/4 residual blocks, final LayerNorm and output head). The input
projection remains in the trunk. New/frozen heads retain existing initialization,
optimizer and gradient rules. These are five evaluated states of one parent,
not five independent training seeds. Fits run serially in one process.

All three fresh/degraded shared-offset comparisons remain, using degraded mean,
fresh mean and their average. Main100,000 + sensitivity120,000 = **220,000 probe
updates**. This is workload accounting, not a runtime estimate. The smallest m
within0.10 of best recovery is reported using the existing pooled-noise stop.
Ranking is not proof of rescue: report every raw loss, recovery, noise and
unrescued-minus-injected contrast. The owner freezes m after reviewing the result.

## One-GPU budget and execution

No Slurm script or submission is included. Delta Codex must use this PR's exact
reviewed commit, a clean checkout, the pinned environment and exactly one allocated
A100. Verify allocation accounting, UUID-to-child binding, CUDA startup, CPU/RAM,
durable storage and available capacity. No memory or runtime qualification is
claimed by CPU tests. Do not launch additional qualification allocations outside
the spending ceiling. Existing source-independent startup checks may run inside
the startup reservation; full-size calibration memory remains unmeasured.

`python -m scripts.m_control_pilot` enforces one visible CUDA GPU, explicit CUDA
(no CPU fallback), a clean exact SHA, a process lock and persistent reservations:
startup900s, optional generation21,600s, calibration13,500s. Maximum36,000 physical
GPU seconds, **10 total GPU-hours**. Omitting generation reserves14,400s (4hours).
Reservations are not refunded, even after interruption. Failure/timeout blocks
automatic retries. Partial logs/round arrays remain. Configure compilation cache
outside the source checkout. Preserve all outputs; never delete data to fit.

From the checkout's `main/`, after Delta-specific preflight and owner launch approval:

```bash
export JAX_PLATFORMS=cuda
export EXP12_JAX_CACHE_DIR=/absolute/durable/cache
# Delta must set CUDA_VISIBLE_DEVICES to the one allocated physical device.
python -m scripts.m_control_pilot --root /absolute/pilot --source-sha REVIEWED_SHA --stage startup
python -m scripts.m_control_pilot --root /absolute/pilot --source-sha REVIEWED_SHA --stage generate
python -m scripts.m_control_pilot --root /absolute/pilot --source-sha REVIEWED_SHA --stage calibrate
```

With an existing checkpoint, omit generate and pass the identical
`--run-dir /absolute/existing_dev` to both commands. Delta must use allocation
wall limits that also bound startup/cleanup/idle allocation time: one GPU for at
most10hours if generating, at most4hours if reusing. Stage timers alone cannot
measure scheduler-held idle GPU time. Never leave a failed allocation waiting.
Do not allocate a second GPU. No automatic resubmission or hidden retry budget.

## Evidence and limits

Retrieve the entire root: budget.json, stage logs, development run metadata,
fresh critic, replay/RMS/agent/optimizer/RNG fork state, fork plan and identity
receipts, historical probe arrays and CSVs, calibration JSON, main and sensitivity
curves, and per-round `partial_*` arrays. Retained rounds survive later failures;
an in-progress fit may have no completed round artifact. Preserve distinct output
roots and hash the complete artifact inventory before transfer; verify checksums
after retrieval. Logs and arrays are local artifacts, not automatically uploaded
to W&B. The inherited training logger follows its normal W&B configuration.

CPU tests validate protocol admission, ten-round selection, actual tiny2,000-step
fits, callback equality, stage reservations, development stopping, existing
injection/Method C behavior and regression coverage. They do not establish GPU
runtime/VRAM, natural positive-control rescue or general DMC–Myo effects.
Existing confirmatory configurations and injection definitions are unchanged.
The confirmatory ten-target source/certifier integration is a separate readiness
item; this development pilot does not grant official Exp2 admission.
