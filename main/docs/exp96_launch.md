# Method C ten-target 96-parent study, version 1

**Current deployment note (17 October 2026):** existing pending allocation
`22945337` is pinned to `bd568d482c261f067add0b1d44bb57a6aceb52b1`
and its `/work/hdd/biqc/skaveti1/exp96/` queue. Engineering smoke `22952754`
is pinned to `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` and writes under
`/projects/biqc/skaveti1/exp96/`. The commands below describe the source
revision in which they were written; they are not instructions to resubmit or
retarget either pending job. The unchanged allocation gate will reject a smoke
receipt bound to the other source SHA.

This study is isolated from the archived five-target pilot and 195-parent Exp1/2
campaign. It is **not authorized to launch** by this document. The source commit,
smoke evidence, scientific calibration, resource profiles, analysis plan, and
storage approval must be frozen by the owner before campaign preparation.
For a normal pending Slurm reservation before those approvals exist, use the
separate [fail-closed staged-queue procedure](exp96_pending_queue.md). It
creates immutable run identities but cannot start training without every gate.

## Exact parent population and settings

`scripts/exp96_campaign.py` creates 4 environments × 3 critics × 8 seeds = 96
distinct parents. The environments are `dog-run`, `humanoid-run`, `myo-key-turn`
(`myoHandKeyTurnFixed-v0`), and `myo-pen-twirl`
(`myoHandPenTwirlFixed-v0`). The latter ID was checked against the installed
MyoSuite registry in the CPU environment, as well as the repository mapping.
Critics are D2W512, D4W1024 and D4W1536; seeds are integers 1–8 in each cell.
Every run uses `base_exp12_ten`, 500,000 **agent interactions**, action repeat 2,
50 scheduled checks at 10,000-interaction increments, ten distinct target draws
per check, 2,000 fit steps per target, IQM10, 10,000 paired bootstrap resamples,
95% intervals, zero lower-bound threshold, and two consecutive scheduled checks.
The fresh baseline at replay warmup is an additional check 0. The complete-state
checkpoint is saved every 10,000 interactions. The historical configuration and
its results remain separate.

Fork eligibility ends at check 47 (470,000 interactions). A natural trigger at
that boundary yields 125,000-interaction control and injected horizons ending
at 595,000, inside the 600,000 maximum. A new first trigger at check 48 would
end at 605,000 and is ineligible. Exp2 injected arms are **not** in the 96
parents and need a separate scientific approval and allocation. The Exp1 parent
preserves the exact trigger state and continues its untreated control extension.

## Durable scheduling and approvals

The queue reuses `exp12_dynamic_scheduler.Ledger`: per-run identity hashes
environment, architecture, seed, resolved configuration and source SHA. Atomic
claims prevent two allocations admitting the same row. A completed row has an
immutable receipt bound to DONE, run metadata, state pointer and validation
receipt hashes. Stale claims require Slurm-confirmed owner termination;
unrecorded failures block for source-bound manual review. An unfinished parent
resumes its complete checkpoint, or restarts deterministically before its first
checkpoint. Failed attempts/logs are retained. Workers admit another compatible
pending run as soon as one finishes. Weight limits are 1/6 small, 1/4 medium,
1/2 large per A100; CPU/RAM/VRAM, wall-time and measured throughput checks also
apply. Starting dense packing before the first two checkpoints requires a
source-bound passing concurrency smoke and explicit allocation approval.

Study approval JSON must identify `study_version: method-c-ten-96-v1`,
`expected_commit`, `parent_count: 96`, `approved: true`, `decision_reference`,
and APPROVED gates with decision references for `ten_function_method`,
`scientific_calibration`, `gpu_smoke`, `allocation_and_storage`, and
`cross_seed_analysis`. The smoke gate must contain an `evidence` object with
absolute `path` and `sha256` of a PASS smoke JSON for the **same source commit**.
Do not fill those fields until the actual approvals and evidence exist.
`configs/exp96_study_approval.template.json` and
`configs/exp96_allocation_approval.template.json` are deliberately unapproved
starting points; copy them outside the checkout before completing them.

The separate allocation approval must contain the same source SHA, a SHA-256 of
the immutable queue JSON, an explicit owner decision reference, one GPU,
positive CPUs and RAM, at most 48 hours, and `admission.packing: ten_target`
with approved per-architecture CPU, host RAM, VRAM, runtime and throughput
profiles, safety margins and `validated_concurrency: true`. Slurm metadata and
the live CUDA/CPU backends are checked by the worker. Each allocation is one
GPU; multiple approved allocations can point at the same durable queue.

After approval, on a clean detached checkout at `$SHA`:

```bash
python scripts/exp96_campaign.py prepare \
  --root /work/hdd/biqc/skaveti1/exp96/campaign_v1 \
  --commit "$SHA" --approval /work/hdd/biqc/skaveti1/exp96/study_approval.json \
  --queue /work/hdd/biqc/skaveti1/exp96/queue_v1
python scripts/exp96_campaign.py allocation-command \
  --queue /work/hdd/biqc/skaveti1/exp96/queue_v1 \
  --approval /work/hdd/biqc/skaveti1/exp96/allocation_approval.json
```

The second command **prints** the reviewable sbatch command; it does not
submit. Preserve the exact approval, plan, queue, Slurm output/error, attempts,
receipts, checkpoints, probe NPZ/CSV, local metrics/evaluation CSV and W&B run
IDs. The completion validator requires 50 scheduled ten-round checks (and all
post-fork control checks if naturally triggered), continuous 2,000-step curves,
finite saved agent/optimizer state, replay/normalization restoration, training
and evaluation records, and the declared endpoint. Confirmatory results remain
subject to scientific certification.

## One-A100 bounded engineering smoke

The smoke runner starts small, medium and large **development** critics on one
A100, then admits a replacement small critic at the first completion. Each has
20,000 interactions and full ten-function 2,000-step fitting at scheduled
checks 10,000 and 20,000, plus the fresh baseline. The medium run is terminated
once after a committed 10,000-interaction checkpoint and restarted from it.
W&B runs offline so the logging path is exercised without networking. The
runner records attempt logs, endpoint/curve artifacts, peak GPU memory, process
RSS, restart status and a source-bound PASS/INCOMPLETE JSON. The 85-minute
workload cap leaves five minutes in a 90-minute Slurm allocation to preserve
artifacts. A timeout is INCOMPLETE; it does not qualify packing. Actual runtime
of full probes on three shared processes is unknown, so 90 minutes is a cap,
not a predicted completion time.

```bash
SHA=<final-committed-source-SHA>
ROOT=/work/hdd/biqc/skaveti1/exp96/smoke_${SHA}
mkdir -p /work/hdd/biqc/skaveti1/exp96/slurm
env EXPECTED_COMMIT="$SHA" EXP96_SMOKE_ROOT="$ROOT" \
  sbatch --parsable --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=16 --mem=64G --time=90 \
  --output=/work/hdd/biqc/skaveti1/exp96/slurm/smoke_%j.out \
  --error=/work/hdd/biqc/skaveti1/exp96/slurm/smoke_%j.err \
  scripts/sbatch_exp96_gpu_smoke.sh
```

Maximum request: 1.5 A100 GPU-hours, 16 CPUs, 64 GiB host RAM. This is a
proposed qualification, **not submitted**. Passing requires all four dev runs
reach their 20,000-interaction endpoints, both scheduled probes and fresh
baseline exist for each, the medium checkpoint restart succeeds, and no process
fails. Peak VRAM and throughput must then be reviewed against the proposed
production admission profiles; the script does not choose those profiles.

## Fresh/fresh control and scientific analysis

The prior tiny CPU controls yielded 1/4 on synthetic data and 2/4 on saved
tiny replay. Each denominator counts **one independently initialized pair at
one check**. Neither result applies the two-consecutive longitudinal trigger.
The exact 95% Clopper–Pearson intervals are approximately 0.63–80.59% and
6.76–93.24%, respectively. Positive loss is `P(fresh)-P(current)`; swapping
pair order negates each paired per-function loss, and a reversed CI exchanges
its signed endpoints. The controls have no trained critic, only four pairs,
tiny models and short fits. They cannot establish a full-width false-trigger
rate or detector sensitivity. A bounded next calibration would collect, for
one preselected environment/architecture, independent fresh pairs at two
scheduled replay/check positions under full ten-target 2,000-step fitting,
with pair-order reversal on the same targets. Start with at most 40 pairs and
a fixed four-hour, one-A100 cap (at most 4 GPU-hours); report per-check and
two-consecutive rates separately. This is a proposed upper bound, not a measured
runtime or authorization to submit it.
The owner must approve its source cell and compute cap. The sample remains
exploratory and cannot prove a ≤5% longitudinal rate.

Before official launch, freeze seed-level outcomes, missing/censored-run
handling, environment × architecture contrasts, confidence intervals and any
multiplicity rules. A reproducible analysis should retain all eight seeds per
cell, report evaluation trajectories and final returns, probe IQM/CI trajectories,
first eligible natural trigger time or censoring, and Exp2 arms separately.
Every result must carry this version, source SHA, resolved config hash, seed,
run identity and immutable completion receipt. Analysis choices are pending
owner approval; no official inferential result is inferred from the CPU tests.

GPU-hour totals for 96 parents cannot be credible until the smoke measures
per-architecture probe time, peak VRAM and throughput under sharing. The
workload contains 96 × 50 × 10 × 2 = 96,000 critic fits of 2,000 steps each,
plus fresh baselines and any post-fork control probes. One parent has 500,000
agent interactions; measured single-process training speed from another
configuration does not estimate the ten-target probe overhead or packing gain.
For planning, there are 32 runs of each size, so the nominal weights sum to
32 × (1/6 + 1/4 + 1/2) = 29⅓ full-GPU slot equivalents **per common run
duration**. This is a packing quantity, not a GPU-hour forecast; wall time and
throughput can change when processes share a GPU.

## Completion-validator correction (2026-10-18)

The A100 engineering smoke at source `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574`
was reported PASS. Its four cases use `dmc_hard` or `myosuite_simba`, whose
`episodic: false` configuration resolves `agent.critic_use_cdq: false`. The
probe therefore saves `fresh_losses` and, at later checks,
`current_losses`, each with ten 2,000-step curves. The earlier 96-parent
completion validator incorrectly required twin-only `q1/q2` curve names.
The consolidation-branch correction selects the required curves from the
resolved, source-hashed critic configuration, retains the full check schedule
and score/curve shape checks, and rejects wrong-schema or nonfinite evidence.
It does not change training or probe computation. The original smoke PASS is
evidence for its own source SHA; source-bound campaign approval for the
corrected revision must name that revision explicitly.
