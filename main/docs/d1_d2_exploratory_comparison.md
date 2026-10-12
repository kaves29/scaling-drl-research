# D1/D2 exploratory comparison (12 October 2026)

This report changes no approved detector, range criterion, or production configuration. The original null files are the uploaded `null_pairs_D4W1024_original.jsonl` (SHA-256 `89846b78306e84b337a6919ae5a1b20809913f7b0bc10e1d32109c6ca8137b5d`) and `null_pairs_D4W1024_seed991.jsonl` (SHA-256 `d6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e`). All 200 rows passed `null_followup.validate`, including float32 paired subtraction, five-round IQM, seeded 10,000-resample percentile-bootstrap bounds, firing flags, pair/check indices, and keys. The original file lacks embedded source provenance; the seed-991 file reports clean source `9d6a82d4aef81299901177865e383640439cde2a`. The reported Block B source for the original is `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`.

## D1: strict lower confidence bound > threshold

Production `experiments/exp12/trigger.py` fires a valid check only when its 95% percentile-bootstrap **lower bound**, from five paired rounds, is strictly greater than the threshold. `f_star` requires the first two **consecutive** fired checks, completing at k=2..19 of 20. Each fresh-null pair is one check, so these data estimate per-check firing, not the frequency of production two-check triggers. Exact binomial intervals below assume independent pairs; shared warmup/replay and reused pair initialization keys weaken that interpretation.

| Method | Threshold and fit budget | seed 990 | seed 991 | What is established |
|---|---|---:|---:|---|
| A, approved | 0; 1,000 steps | 13/100 (95% Clopper–Pearson 7.11–21.20%) | 6/100 (2.23–12.60%) | Both exceed the approved point-estimate limit of 5/100. |
| B, exploratory pooled | 0.0009080017606417335; 1,000 steps | 0/100 (0–3.62%) | 0/100 (0–3.62%) | **In-sample** for both datasets; not independently qualified. |
| B, cross-replicate | seed-991 threshold 0.0005196526646614074 applied to 990; seed-990 threshold 0.0010963221391042065 applied to 991 | 0/100 | 0/100 | Out-of-sample with respect to each threshold's seed, but the two runs reused pair initialization keys; rule selection is exploratory. |
| C, exploratory | 0; 2,000 steps | Unmeasured | Unmeasured | Original 1,000-step endpoints cannot predict 2,000-step bounds. |

The pooled p95 uses `numpy.percentile(..., method="linear")` on the 200 saved IQM(L) values, not the confidence bounds. It interpolates 5% of the way between sorted zero-based positions 189 (`0.0009072224299112955`) and 190 (`0.0009228090445200602`). The pooled B replay has 0/200 in-sample firings (nominal exact 95% interval 0–1.83%). A positive threshold suppresses weak positive findings; these fresh-pair data cannot establish sensitivity to real plasticity loss. Neither dataset contains independently labeled positive cases. Because the pair keys are reused and each pair is isolated, neither cross-replicate result establishes production trigger timing or a two-consecutive-check false-trigger rate.

No full-length pilot `probe_checks.csv` was present in the available uploaded artifacts. The full-width gate evidence is a forced engineering validation, and local scratch CSVs are CPU test fixtures; neither is a natural training trajectory. `scripts/sci_investigation/compare_probe_decisions.py` can replay an authentic, unforced CSV prefix for A and B when supplied. Until then, changes in real trigger eligibility or timing are **unverified**.

C must recompute the complete five-round fit for every current and fresh critic (both heads in twin critics), including new 1,000-step optimizer continuation with live AdamW moments, original teacher, replay pool, initial parameters, offsets and first-1,000 minibatch sequence. It then recomputes FP32 paired round losses, IQM, seeded percentile intervals, and the two-check sequence. Existing null JSONL files save endpoints only, not fitted weights or optimizer states. C changes the statistic's interpretation: fit-after-2,000-step capacity rather than fit-after-1,000-step capacity. It is not a runtime-only alteration.

## D2: hopper-hop D4W1536 configured pool

The uploaded job-22817406 archive (SHA-256 `e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868`) contains all five 1,000-step curves, 25,600-row teacher pools, replay warmup and source/runtime provenance. Its saved checksums passed. The approved 25,600-pool, five-round, 1,000-step result is `IQM(P)/IQM(b)=0.6023976348282202` against required `>=0.9`.

| Round | P/b | Final full-pool MSE | Required MSE if that round alone were at P/b=0.9 |
|---:|---:|---:|---:|
| 0 | 0.558653 | 0.219331 | 0.049696 |
| 1 | 0.613904 | 0.198245 | 0.051346 |
| 2 | 0.649450 | 0.176255 | 0.050280 |
| 3 | 0.586019 | 0.206364 | 0.049849 |
| 4 | 0.601805 | 0.195323 | 0.049052 |

Mean round P/b=0.601966, sample SD=0.033644, observed range=0.558653–0.649450. Every round lies at least 0.25055 below 0.9. The last-100-step mean minibatch losses are 0.2117–0.2525, below the preceding-100 means 0.2643–0.3095; none demonstrates a plateau. Minibatch loss is not full-pool MSE, so it cannot be substituted for P/b. Five potentially dependent target rounds do not support a precise population uncertainty interval, but their observed variability alone does not plausibly bridge the 0.2976 IQM-ratio gap. The evidence supports insufficient fitting at the approved budget as a plausible mechanism, not a proven root cause; an untrained fresh critic failing its measurement range is not evidence of trained-critic plasticity degradation.

Saved curves and final MSEs cannot be continued from step 1,000: the archive has no fitted parameters or AdamW state. A valid extension must redo the first 1,000 steps with the captured teacher/pool and matching initial critic, retain its optimizer state in memory, then take 1,000 more steps. `paired_probe_budget.py` does this with the original first-1,000 minibatch stream and a separate continuation stream. It records bitwise and maximum absolute differences between its new 1,000-step result and the historical capture; any mismatch must be investigated before interpreting the 2,000-step contrast. It does **not** change the 0.9 rule or production fit budget.

A longer budget could make the fresh range gate pass while also changing measured current-vs-fresh plasticity loss and trigger timing across all architectures. Lowering 0.9 weakens the original measurement-quality criterion. Excluding hopper-hop D4W1536 drops five approved parents and changes architecture/environment coverage. All three are scientific owner decisions; none is adopted here.

## Smallest paired GPU experiment awaiting authorization

One A100 allocation runs D2 first, then one **100-pair seed-991 D4W1024 fresh-null** comparison. Each fit records exactly matched first-1,000 and continued 2,000-step endpoints and loss curves, fixed architecture, seeds, replay/teacher, initial critics and optimizer. The hopper stage consumes the captured exact pools and teacher targets. The null stage reconstructs seed-991 inputs on the pinned source and saves both endpoints; it is a new matched sample, not a claim that the old JSONL can be resumed. A genuine trained-checkpoint sensitivity comparison is desirable but cannot be executed without a source-compatible, complete trained checkpoint and its replay/normalization state; none was supplied here. If later supplied, run it as a distinct, labeled stage with the same paired optimizer scheme and real natural check sequence.

**Interpretation:** D2 supports a fitting-budget explanation only if its recreated step-1,000 endpoint agrees with the historic capture and step-2,000 endpoint changes materially. A step-2,000 ratio below 0.9 still fails the unchanged range rule. A ratio above 0.9 identifies a budget-dependent fresh-fit gate, but does not itself authorize C for production. D1 compares fresh-null per-check bounds at both budgets; a 100-pair sample can test the approved <=5/100 point rule for C but cannot establish true-positive sensitivity or production two-check timing. A/B sensitivity and timing still require a natural trained trajectory with saved check rounds. D1 and D2 may share one allocation and code, but their different architecture/environment/sample populations must remain separate analyses.

The diagnostic requests one A100, 16 CPUs and 64 GiB. Wall limit is six hours (maximum allocation cost **6 A100-hours**); expected elapsed time is tentatively **2–4 hours**, driven by the 100 D4W1024 paired 2,000-step pairs and compilation. This is an estimate, not a measurement: Block B's three-size/300-pair null suite took 6,042 s in a multi-lane setting, with no per-size timing. The hopper-only five-round stage is expected to be minutes to tens of minutes. A successful 1,000-step stage does not guarantee 2,000 steps fit within the limit; preserve partial results and report timeout if not. Peak device/host memory for this new workload is unmeasured; preallocation is disabled and the output retains per-pair progress.

Extending every check doubles critic fitting updates: 10,000 extra per single-critic check, 20,000 per twin check. If all 195 parents execute all 20 checks, that is **45 million additional critic optimizer steps** (including the 30 twin-critic parents). The historical, unverified linear model estimated 22.38 A100-hours for parent probes at 1,000 steps, so another approximately 22.38 A100-hours is a **conditional fit-dominated estimate**, excluding altered fork/control schedules and fixed compilation/evaluation costs. It is not a measured campaign forecast.

### Exact Delta preparation and submission commands (execute only after approval)

Stage the exact uploaded archive on Delta first, preserving its checksum, at `HOPPER_CAPTURE_ARCHIVE` below. This copy does not alter the original evidence. The command assumes that file is already accessible there; the original Delta archive location was not recorded in the supplied evidence.

```bash
set -euo pipefail
REV=REPLACE_WITH_PUBLISHED_40_CHARACTER_EXPLORATION_COMMIT
CHECKOUT=/work/hdd/biqc/skaveti1/exp12_exploration_checkouts/$REV
HOPPER_CAPTURE_ARCHIVE=/work/hdd/biqc/skaveti1/exp12_exploration_inputs/hopper_capture_22817406.tar.gz
ROOT=/work/hdd/biqc/skaveti1/exp12_exploration_results/${REV}_paired_d1_d2
test ! -e "$CHECKOUT" && test ! -e "$ROOT" && test -f "$HOPPER_CAPTURE_ARCHIVE"
echo 'e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868  '"$HOPPER_CAPTURE_ARCHIVE" | sha256sum -c -
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$REV"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
export EXPECTED_COMMIT="$REV" PROBE_DIAGNOSTIC_ROOT="$ROOT" HOPPER_CAPTURE_ARCHIVE
mkdir -p "$(dirname "$ROOT")"
sbatch --export=ALL --output="${ROOT}.slurm-%j.out" --error="${ROOT}.slurm-%j.err" \
  scripts/sbatch_exp12_paired_probe_budget.sh
```

Expected outputs: external Slurm stdout/stderr, `ROOT/exit_status.txt`, `ROOT/SHA256SUMS`, `ROOT/hopper.log`, `ROOT/hopper/hopper.json` with five per-step curves and historical comparison, `ROOT/null.log`, `ROOT/null_seed991/provenance.json` and `pair_000.json` through `pair_099.json`. Stop interpretation for nonzero exit, source/runtime mismatch, invalid capture checksum, absent pairs, nonfinite values, or failed step-1,000 reproduction. Never reclassify a scientific FAIL as an engineering PASS.

CPU verification: four focused tests pass; `staged_fit` first stage matches production `_fit` bitwise in a deterministic CPU fixture. `bash -n` and `git diff --check` pass. No A100 result is claimed.
