# Consolidated exploratory D1/D2 A100 diagnostic — authorization review

Base: `777c579c8e02f51313c000d81ad142bec74ede51`. This branch changes only the exploratory diagnostic, its Slurm wrapper, tests, and this report. Production probe settings, trigger rules, range threshold, and trained parents are untouched. No GPU work has been run here.

## Included in one bounded allocation

| Stage | Work and scientific interpretation | Expected elapsed; fixed stage cap |
|---|---|---:|
| D2 historical pairing | D4W1536 hopper-hop seed 990, exact archived 25,600-item teacher pools, initial fresh critic, 1,000 then 2,000 contiguous AdamW steps. Compare step-1,000 curves and full-pool MSE to job 22817406. Any mismatch blocks attribution of the 2,000-step change to budget alone. | 5–10 min estimate; 15 min cap |
| D2 independent repeats | D4W1536 hopper-hop seeds 992 and 993, fresh trainer/replay/initialization/teacher per seed; five rounds each, paired 1,000/2,000 endpoints. These are separate random seeds, not five additional rounds of seed 990. | 5–10 min each estimate; 15 min each cap |
| D1 null pairing | D4W1024 dog-run seed 991, 100 fresh/current critic pairs, each with five shared teacher/replay/minibatch rounds and contiguous 1,000/2,000-step fits. Report A (`ci_low > 0` at 1,000), B (`ci_low > 0.0009080017606417335` at 1,000), and C (`ci_low > 0` at 2,000), each with observed count and exact binomial interval. B's threshold remains fixed from the **two historical** 100-pair datasets; their SHA-256s and linear p95 calculation are in [the D1/D2 comparison](d1_d2_exploratory_comparison.md). | 1.5–3 h tentative; 200 min cap |

Stage caps sum to **245 minutes (4 h 5 min)**; the Slurm wall limit is **4 h 30 min**, leaving 25 minutes for startup, integrity checks, logging and teardown. Maximum allocation cost is **4.5 A100-hours**. Expected total elapsed is approximately **2–3.5 h**, with material uncertainty because Block B reported 6,042 s for a multi-lane 300-pair suite but no per-size timing. A timeout preserves completed stage outputs and cannot be called a qualified 100-pair rate. No automatic resubmission occurs. One A100, 16 CPUs, 64 GiB host RAM remains a plausible request: prior D4W1536 diagnostics fit an A100; this staged fit processes one critic at a time with GPU preallocation disabled. Peak memory of this **new combined workload is unmeasured**, so telemetry and OOM are explicit outcomes, not assumed away.

The driver records per-fit and per-round wall/CPU seconds, optimizer updates per wall second, process peak RSS, source/runtime provenance, five per-step curves, round endpoints, seeded bootstrap intervals, and stage summaries. The wrapper samples GPU used/total and process memory, utilization, and host memory every five seconds. Individual pair JSON files and D2 round files are atomically published; `summary.json` is updated after each completed pair and labels partial sample sizes. A selected historical D2 round saves its step-2,000 parameter/AdamW pytree leaves with a checksum. Other incomplete rounds are reproducibly rerunnable from pinned seeds rather than treated as completed. Slurm logs, stage exit codes, overall exit code and output checksums survive failure. A large optimizer snapshot for every null pair would consume substantial storage without improving completed-pair recovery, so it is deliberately omitted.

## Scientific limits and excluded additions

**Trained-critic comparison is blocked by unavailable evidence.** The known pilot 22800749 path is `/work/hdd/biqc/skaveti1/exp12_dev/dog_run_D4W1536_seed102_7d21eaa_pilot01`, but no pilot checkpoint, replay/normalization snapshot, natural `probe_checks.csv`, or per-check NPZ is present in this workspace. The pilot uses D4W1536; B's available p95 threshold was calibrated for D4W1024, so applying it to this pilot would conflate architecture populations. A D4W1536 natural-check CSV, if supplied, can support offline replay of its approved A rule. A 2,000-step C comparison requires exact matched trained critic/replay/normalization/teacher states at each chosen check; approximate reconstruction is unacceptable. This allocation has **no trained-critic stage** and makes no Exp2 trigger-eligibility claim. Inspect pilot artifacts read-only before designing one; do not add an unreviewed GPU workload to this submission.

**No automatic 3,000-step D2 stage.** Its possible value depends on the as-yet-unobserved 2,000-step trajectories. Choosing a slope magnitude or acceptance criterion now would introduce an arbitrary exploratory decision, and preserving all five live AdamW states through a post-hoc conditional stage adds memory and storage cost. The 2,000-step output retains full curves and a selected optimizer snapshot; after review, the owner can decide whether a separate bounded 3,000-step experiment is justified. No 3,000-step endpoint can be inferred from the current data.

The D1 100-pair result estimates **per-check fresh-null firing** for one seed; it does not establish a production two-consecutive-check false-trigger rate, positive sensitivity, or training-time trigger timing. Five teacher rounds per pair and pair keys reused from historical seed 991 mean the 100 pairs are not independent *replication of the prior seed*. Exact binomial intervals are descriptive under an independence assumption. D2 range quality is separate from both null calibration and Exp2 eligibility. A 2,000-step range improvement could indicate a fitting-budget limitation; it neither authorizes changing the approved 1,000-step rule nor establishes trained-critic plasticity.

The diagnostic measures first- and second-stage wall times directly, so its observed incremental fitting cost can replace the prior conditional forecast of roughly 22.38 extra A100-hours for 195 parents at 20 checks. That extrapolation must include architecture and twin-critic mix, compilation, checkpoint, evaluation, and environment differences before use as a campaign forecast.

## Delta preflight and exact command after authorization

First place **copies** of the existing verified hopper archive and seed-991 null JSONL at the input paths below. Their SHA-256s must be `e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868` and `d6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e`. Their original Delta paths are unknown here. The null input lets the diagnostic compare the new first-1,000 results with each historical pair. Use the full published commit hash provided with this report; never point this job at pilot 22800749's checkout or output.

```bash
set -euo pipefail
REV=REPLACE_WITH_PUBLISHED_40_CHARACTER_CONSOLIDATED_COMMIT
CHECKOUT=/work/hdd/biqc/skaveti1/exp12_exploration_checkouts/$REV
HOPPER_CAPTURE_ARCHIVE=/work/hdd/biqc/skaveti1/exp12_exploration_inputs/hopper_capture_22817406.tar.gz
NULL_991_JSONL=/work/hdd/biqc/skaveti1/exp12_exploration_inputs/null_pairs_D4W1024_seed991.jsonl
PROBE_DIAGNOSTIC_ROOT=/work/hdd/biqc/skaveti1/exp12_exploration_results/${REV}_consolidated_d1_d2
test -f "$HOPPER_CAPTURE_ARCHIVE" && test -f "$NULL_991_JSONL" && test ! -e "$CHECKOUT" && test ! -e "$PROBE_DIAGNOSTIC_ROOT"
echo 'e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868  '"$HOPPER_CAPTURE_ARCHIVE" | sha256sum -c -
echo 'd6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e  '"$NULL_991_JSONL" | sha256sum -c -
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$REV"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
export EXPECTED_COMMIT="$REV" HOPPER_CAPTURE_ARCHIVE NULL_991_JSONL PROBE_DIAGNOSTIC_ROOT
mkdir -p "$(dirname "$PROBE_DIAGNOSTIC_ROOT")"
sbatch --export=ALL \
  --output="${PROBE_DIAGNOSTIC_ROOT}.slurm-%j.out" \
  --error="${PROBE_DIAGNOSTIC_ROOT}.slurm-%j.err" \
  scripts/sbatch_exp12_paired_probe_budget.sh
```

Review `hopper_990/hopper.json` first: compare its historical first endpoint, then the 2,000-step ratio and curves. Review `hopper_992`/`hopper_993` separately for between-seed behavior. Review `null_991/summary.json`: require `complete=true`, 100 unique pair files, finite bounds, stage exit zero, and original first-1,000 behavior before interpreting C's firing rate. B remains exploratory even if it suppresses all null firings. Any scientific adoption of B, C, 3,000 steps, a changed range threshold, or an exclusion requires explicit owner approval.

CPU verification: focused tests cover first-stage production equivalence, continuous optimizer state, A/B/C partial-summary separation, strict threshold replay, and two-check logic. Shell syntax and whitespace checks pass. No GPU validation is claimed.
