# PR #11 integration finalization

Scope: continue `b0617c60cfff4b33bc267773f888d6059e9cbc17` on its existing
`codex/echocritic-batch2-20261024` branch. Preserve the #8 → #9 → #10 ancestry;
PR #12 remains a sibling and is not merged. No GPU job, PR merge, scientific
setting, acceptance threshold or archived output is changed.

## Mutation investigation

The unchanged b061 suite passed all 74 mutation/restoration pairs in this Cloud
environment, including the twin metadata pair. Its mutated case raises the
expected missing-`critic_count` KeyError; restored setup and assertion pass.
Per-case baseline logs retain both outcomes. This does not reproduce the prior
Mac failure and is not evidence of its definitive root cause.

A confirmed fixture defect was reproduced using the original b061 twin fixture:
failed `setUpClass` leaves W&B patches, `wandb.run` and its temporary directory
behind because unittest does not call `tearDownClass` after setup fails. The
new failing-setup regression verifies cleanup of all three. Class cleanups now
register resources immediately, and the W&B helper restores `wandb.run` as well
as its patched functions. The mutation's original metadata builder is retained
in the harness rather than installed as an extra global in the production
Exp1 module.

The earlier task history also shows a commit while its full mutation run was
in progress. Exp2 deliberately rejects source/dirty state differing from its
parent, including during the twin restored-state fixture. A controlled test
reproduces that source mismatch failure. The mutation harness now fingerprints
HEAD and tracked file contents around each run and stops with an explicit
source-changed error. It retains restored-failure tracebacks rather than
silently discarding them in `/dev/null` and leaking file descriptors.

The original failure traceback was discarded by the old harness. Source
instability and failed-setup pollution are demonstrated hazards, but attribution
of the historical 73/74 result remains unverified. Do not present either as a
proven historical root cause. Deterministic stable-source reruns, rather than
tolerance changes or disabled checks, are required for current qualification.

## Instrumentation

See [runner usage and semantics](echocritic_batch2.md#runner-integration-and-finalization-24-october-2026).
Observers are attached only at runner/lifecycle and existing metric publication
boundaries. SAC kernels, RNG use, optimizer construction and numerical guards
remain unchanged. Fixed-task runner GPU admission gates remain in force;
entrypoint lifecycle tests simulate admission on CPU and do not execute jobs.
Original runner failures survive secondary terminal-sidecar I/O failures.

## Real campaign archive

BLOCKED: no complete W&B metadata/history ZIP is available in this Cloud task's
uploads or artifacts. The only accessible archives contain prior engineering
evidence. No 103-record ingestion, pilot/canonical attribution, seed independence,
crash/missing-metric inventory or raw-value comparison against real histories is
claimed. Existing synthetic tests remain synthetic.

Once supplied, run from `main/` in the pinned CPU environment:

```bash
python -m analysis.campaign.pipeline /absolute/path/real-export.zip \
  --out /absolute/new/real-analysis --bootstrap-reps 10000
```

Before accepting aggregation, review the validation report against raw metadata
for every ID, nominal condition, source, campaign declaration, seed and crash;
reconcile duplicates and compare every metric row directly to the ZIP. Unknown
membership/provenance stays excluded rather than inferred from names. Real
schema adaptation, if required, must preserve raw evidence and receive its own
fixtures. The command alone is not that validation.

## Validation

Final validation receipts are recorded after stable-source execution. Artifacts
are outside Git in `/workspace/artifacts/pr11-finalization-20261024`.
CPU scope does not qualify CUDA/A100 numerical equivalence. The historical A100
segmentation failure remains unresolved and no GPU work is authorized here.
