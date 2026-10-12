# Exp1/Exp2 launch runbook (2026-10-12)

Branch `claude/exp12-production-launch`, based on `codex/exp12-production-readiness` @ `1ea37e4`.

**Status:**
- Production remains **NO-GO** until the owner decisions in §2 are recorded.
- No GPU job was submitted. No amendment was approved, no threshold was changed, and nothing was merged.
- Pilot 22800749 was not touched.

**Inputs:**
- [exp12_gate_decision_memo.md](https://github.com/kaves29/scaling-drl-research/blob/claude/exp12-gate-decision-memo/main/docs/exp12_gate_decision_memo.md) and
  `exp12_proposed_amendment.md` (both @ `ec74dee`, and both **PROPOSED**);
- Codex's `exp12_oct12_execution_handoff.md` (@ `1ea37e4`).

## 1. Verification of the gate review against source and protocol
| Claim | Verdict | Evidence |
|---|---|---|
| The NULL rule permits adopting the 95th percentile | **Confirmed.** The rule is unchanged since 2026-10-04 (`docs/exp12_decisions.md` l.699–705): "the lead will decide whether to adopt a null-calibrated threshold: the 95th percentile of L between two independent fresh critics at that size". No later entry supersedes it. | The rule text |
| … computed from the existing failed samples | **Rule-consistent, with one clarification needed.** The rule computes the percentile from *the* null at that size, and the two samples are that null. The rule does not say how to combine two samples, so "pooled" is an owner clarification. Codex's handoff says "no value may be selected from the failed datasets". That is a stricter *policy proposal*, not the pre-registered rule; the owner decides. | — |
| Pilot and production share the science paths | **Confirmed by a recomputable certificate.** 128 of 129 in-scope files are byte-identical between `7d0883c` and `1ea37e4`. In-scope means `experiments/`, `scale_rl/`, `utils/`, `configs/` (except `launch_readiness/`), `run.py`, `generate_manifest.py`, `requirements.txt` and `positive_control.py`. The only difference is the 22-line source guard in `experiments/exp2_arm.py`, which the positive control does not exercise. | `scripts/exp12_source_compatibility.py` |
| Each parent's check 0 equals the range probe | **Confirmed.** `RunProbes.capture_fresh` runs `run_probe` at `check_index` 0 on the untrained critic at min replay 5,000, the same call as `probe_fresh_checks.py --mode range`. P, b and the curves go to `check_00.npz` and `probe_curves.npz`. | Code |
| **Incompatibility (fixed):** the pilot fork could not qualify m | The positive-control wrapper and the Exp2 approval required the fork's source commit to equal the campaign commit, so the pilot's fork could never qualify m for a later SHA. Both now accept an earlier-commit fork **only** with a hash-pinned, git-recomputed compatibility certificate. Using it is owner decision D4. | — |
| **Defect (fixed):** medium DMC tasks crashed the scheduler | hopper-hop and swimmer are in group `dmc_medium` (30 parents). The approval accepted runtime estimates only for {dmc_hard, myosuite_simba, humanoid_bench}, so `can_admit` would raise a `KeyError` on the first medium-DMC row and kill the allocation worker. | — |

## 2. Owner decisions
Each decision is an owner-signed JSON file. A file publishes nothing until it is passed to `exp12_dynamic_scheduler.py release`, which validates it and publishes it write-once. Every `release` decision carries the same header:

```json
{"approved": true, "expected_commit": "<FINAL_SHA>", "queue_sha256": "<sha256 of $QUEUE/queue.json>",
 "decision_reference": "<your dated statement>", "release": { ... }}
```

**Prerequisite for every decision below.** The parent-plan approval must set `"release_mode": "per_run"` and mark
every gate decided by release as `"status": "DEFERRED_TO_RELEASE"`.

| ID | Decision | Recommended | Exact `release` body | Consequence |
|---|---|---|---|---|
| D1 | **Null threshold for D4W1024** | Adopt the pooled p95 (all 200 pairs). State pooling in writing **before** running the computation. | `{"type":"fresh_null","decision":"adopt_threshold","selection":"pooled","override":"<report.override>","computation":{"path":"<report>","sha256":"<sha>"}}` | Only the 65 D4W1024 rows get `trigger.null_threshold=<value>`, which changes their config hash. The code SHA is unchanged, so no requalification. D4W1024 sensitivity falls (possibly few forks); D2W512 and D4W1536 are unchanged. |
| D1′ | Alternative to D1 | Keep 0, with the stated limitation | `{"type":"fresh_null","decision":"keep_zero"}` | Some D4W1024 forks on critics that have not lost plasticity, diluting Exp2 at that size. |
| D2 | **Hopper-hop D4W1536** | Approve A1 → `release`. If A1 is rejected: `exclude` (H3), or publish nothing (H2 hold). | `{"type":"hopper_range","decision":"release"}` | `release`: the 5 parents run as specified. A1's per-cell check-0 classification is analysis-only and is computed from saved data. `exclude`: they never start. |
| D3 | **A2 staged identity** | Approve if the HumanoidBench identity is likely to lag | `{"type":"a2_staged_identity","amendment_reference":"A2 approved <date>"}` | D2W512 starts after the positive control and the global gates. Each suite's scaled parents and arms start once that suite's two identity certificates are released. Without D3, all six cases are needed before any parent. |
| D4 | **Pilot-source positive control and m** | Approve (the science code is identical) | Gate release `{"type":"gate","gate":"positive_control","positive_control":{"path":"<pc>/result/positive_control.json","sha256":"…","source_compatibility":{"path":"<cert>","sha256":"…"}}}`; later `{"type":"exp2","injection_m":"<chosen_m>","positive_control":{…same…}}` | Avoids a new final-source development run, which needs a natural trigger: more than 6 h, and not guaranteed. m is the positive control's mechanical choice (d). |
| D5 | **GPU resource model** | Allocations of 3 or 4 nodes (12–16 A100s), 48 h, one process per GPU; `idle_wait_minutes` 0–60 | Allocation approval JSON (`dynamic_allocation_approval.template.json`; any allocation name other than the legacy A/B) | Packing stays at 1 per GPU until a packing profile is approved. Throughput estimates are in §5. |
| D6 | Global gates already in Codex's plan | Approve each with its evidence | `{"type":"gate","gate":"<suite_runtime_storage \| grid_allocation \| cuda_numeric_bounds \| study_methodology_freeze>"}` | `cuda_numeric_bounds`: the 4 `GPU_TOLERANCES` are still `None`. Approve values from Block B's `NUMERICS` records under the logged 10× rule, or reword the gate; those records are not in this checkout. |

**Exact computation for D1** (CPU, minutes, run on Delta where the files are). It recomputes every row and writes `COMPUTED_NOT_ACTIVATED`:

```bash
python scripts/exp12_null_threshold.py \
  --sample s990=/abs/blockB/null_pairs_D4W1024.jsonl=89846b78306e84b337a6919ae5a1b20809913f7b0bc10e1d32109c6ca8137b5d \
  --sample s991=/abs/seed991/null_pairs_D4W1024.jsonl=d6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e \
  --selection pooled --out /abs/decisions/d4w1024_null_threshold.json
```

## 3. What was built (CPU only)
- **Per-run releases** (`exp12_dynamic_scheduler.py`):
  - one immutable 195-row queue;
  - gates can be deferred to write-once owner releases;
  - before **every** start, `run_blockers` checks every prerequisite that applies to that run;
  - blocked runs stay pending, never failed;
  - running allocations pick up new releases without restarting, Exp2 included;
  - commands `release` and `status` (a read-only census with blockers per run).
- **Array path refused for per-run plans:** the fixed-array and batch paths refuse per-run plans, so nothing can bypass the eligibility checks.
- **Allocations** of any name: 1–4 nodes, ≤ 48 h; A and B keep their exact shapes.
  - Arms are scheduled first, then the longest remaining work first.
  - Admission uses the unfinished fraction of resumable runs, so the tail of an allocation is used.
  - An optional `idle_wait_minutes` keeps an allocation waiting for releases instead of exiting.
- **New CPU tools:**
  - `exp12_null_threshold.py`;
  - `exp12_source_compatibility.py`;
  - `exp12_elapsed_estimate.py`;
  - the compatibility option on `sbatch_exp12_positive_control.sh`.
- **Tests:** focused runs only. 63 tests across the new `test_exp12_staged_release.py` (20) and the existing scheduler, launch-preparation and Delta-preflight suites all pass. They include:
  - an end-to-end census on a real 195-config plan and queue;
  - a run-loop test proving that only released runs start.

## 4. GPU qualification commands (prepared; do not submit until authorized)
**Setup.** `FINAL_SHA` is the commit of this branch that you approve. Delta setup is as in Codex's handoff, run from a fresh clean detached checkout:

```bash
set -euo pipefail
FINAL_SHA=<this branch's reviewed 40-char SHA>
FINAL_CHECKOUT=/work/hdd/biqc/skaveti1/exp12_final_checkouts/$FINAL_SHA
QUAL_ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$FINAL_CHECKOUT"
git -C "$FINAL_CHECKOUT" checkout --detach "$FINAL_SHA"; cd "$FINAL_CHECKOUT/main"
mkdir -p "$QUAL_ROOT/slurm" "$QUAL_ROOT/identity" /work/hdd/biqc/skaveti1/exp12_decisions
```

**Q1. Six identity cases.** Submit the HumanoidBench cases first: they are the slowest and gate the longest jobs.

- Resources per case: 1 A100, 16 CPUs, 64 GB, ≤ 4 h. The six cases total ≤ 24 A100-h and take about 4 h of elapsed time if run in parallel.
- Cold cache only. A warm-cache case is not required by the methodology.

```bash
for case_id in D4W1536_h1-run-v0 D4W1024_h1-run-v0 D4W1536_dog-run D4W1024_dog-run \
               D4W1536_myo-key-turn D4W1024_myo-key-turn; do
  r="$QUAL_ROOT/identity/${case_id}_cold_01"
  EXPECTED_COMMIT="$FINAL_SHA" IDENTITY_CASE="$case_id" IDENTITY_CACHE_MODE=cold \
  IDENTITY_ROOT="$r" HB_ENV="$APPROVED_HB_ENV" \
    sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=04:00:00 \
      --output="$QUAL_ROOT/slurm/identity_%j.out" --error="$QUAL_ROOT/slurm/identity_%j.err" \
      scripts/sbatch_exp12_identity_case.sh
done
```

**Q2. Positive control on the pilot's natural fork.** Run only if pilot 22800749 reached a natural trigger (`fork/FORK_READY`). The pilot's directory is only read. First the CPU certificate, then the GPU job:

```bash
PILOT_RUN=<pilot root>/checkpoints/exp1/D4W1536/dog-run/seed_102
CERT=/work/hdd/biqc/skaveti1/exp12_decisions/pilot_source_compat.json
python scripts/exp12_source_compatibility.py certify --source 7d0883cfa99d27fed062d77a68cbbfa27e560061 \
  --target "$FINAL_SHA" --allow-path main/experiments/exp2_arm.py --out "$CERT"     # must print COMPATIBLE
EXPECTED_COMMIT="$FINAL_SHA" PC_RUN_DIR="$PILOT_RUN" PC_OUTPUT_ROOT="$QUAL_ROOT/positive_control_01" \
PC_SOURCE_COMPATIBILITY="$CERT" PC_SOURCE_COMPATIBILITY_SHA256="$(sha256sum "$CERT" | cut -d' ' -f1)" \
  sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=01:00:00 \
    --output="$QUAL_ROOT/slurm/positive_control_%j.out" --error="$QUAL_ROOT/slurm/positive_control_%j.err" \
    scripts/sbatch_exp12_positive_control.sh
```

- Exit 0 means `status: m_chosen`.
- Exit 3 means **stop and consult** [M].
- The workload is 55,000 probe-fitting updates. At Block A's measured ~33 s per 10,000-update D4W1536 probe check, that is about 3 min plus loading. The 1 h cap is a proposal.
- If the pilot has **no natural fork**, the methodology says stop and consult. Do not force a trigger.

**Q3. Delta shared-mount and binding preflight.** Needed before any multi-node allocation (G12/G16), because cross-node claims rely on atomic `mkdir`/`link` on the shared mount. It is unchanged from Codex: 2 nodes × 4 A100s, 10 min, ≤ 1.33 A100-h, and it may run alongside Q1 if quota allows.

```bash
EXPECTED_COMMIT="$FINAL_SHA" EXP12_PREFLIGHT_ROOT="$QUAL_ROOT/fsbind_01" \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=16 --mem=64G \
    --time=00:10:00 --output="$QUAL_ROOT/slurm/fsbind_%j.out" --error="$QUAL_ROOT/slurm/fsbind_%j.err" \
    scripts/sbatch_exp12_dynamic_delta_preflight.sh
```

**Injection qualification.** No separate GPU job is indispensable:
- m comes from Q2;
- injected Check 1 (≤ 64 float32 eps × the pre-injection maximum; Q and dQ/da) runs inside **every** arm and stops it on failure;
- the source guard refuses an arm whose code differs from its parent's.

The first natural arm acts as the live canary. `status` shows it.

## 5. Production launch (prepared; owner approvals are filled from the templates)
```bash
CAMPAIGN_ROOT=/work/hdd/biqc/skaveti1/exp12_campaign_$FINAL_SHA; QUEUE=$CAMPAIGN_ROOT.queue
D=/work/hdd/biqc/skaveti1/exp12_decisions
# 1. Plan: release_mode "per_run". Gates not yet evidenced are "DEFERRED_TO_RELEASE".
python scripts/exp12_launch_plan.py prepare --stage parents --root "$CAMPAIGN_ROOT" --commit "$FINAL_SHA" \
  --approval $D/parents_approval.json --out $D/parent_plan.json
python scripts/exp12_dynamic_scheduler.py prepare --plan $D/parent_plan.json --queue "$QUEUE"
# 2. Publish each owner decision as it is made (write-once); census at any time.
for f in $D/release_*.json; do python scripts/exp12_dynamic_scheduler.py release --queue "$QUEUE" --decision "$f"; done
python scripts/exp12_dynamic_scheduler.py status --queue "$QUEUE"
# 3. Print (not submit) each approved allocation; submit the printed line after authorization.
mkdir -p "$QUEUE/submissions/G16a"
python scripts/exp12_dynamic_scheduler.py allocation-command --queue "$QUEUE" \
  --approval $D/allocation_G16a.json --allocation G16a
```

The same queue serves all three launch tranches. Each run starts automatically once its own prerequisites are released.

| Tranche | Starts when | Rows |
|---|---|---|
| First eligible Exp1 parents | positive control + D6 gates + identity (all six; or under A2, none for D2W512 and per suite for scaled sizes) | Under A2 with only the DMC cases: 65 D2W512 + 30 DMC D4W1536 (hopper held). Without A2: 125 rows once all six pass. |
| Remaining Exp1 parents | Each further release: an identity suite, `fresh_null` (65 D4W1024), `hopper_range` (5) | Started on the next scheduler poll (≤ 15 s) of any running allocation |
| Exp2 arms | `exp2` release (m) + the fork's suite identity | Discovered from `FORK_READY` while parents run, and scheduled ahead of parents |

**Fastest order.** Scheduling is longest-first automatically.
- HumanoidBench D4W1536 (~10–16 h each) and D4W1024 (~6 h) come first, then MyoSuite and DMC D4W1536 (~4 h), D4W1024 (~3 h) and D2W512.
- Arms come ahead of new parents.
- Submit the HumanoidBench identity cases first. Under A2, D2W512 HumanoidBench parents can start immediately.

## 6. Estimates
**Inputs and assumptions:**
- The measured A100 training rates come from Block B (dog-run and myo-key-turn): 112.5/65.8/46.3 and 117.5/66.1/45.1 interactions/s. They are scaled by 42.44/46.3 to match the newer D4W1536 measurement.
- HumanoidBench uses the **unmeasured** twin-critic estimates (88/56/32).
- Overhead × 1.20 covers probes, evaluations, checkpoints and compilation.
- Arms are 0.25 B each; forks at 60% of the parent; one process per GPU.
- Run `scripts/exp12_elapsed_estimate.py` to reproduce or vary the assumptions.
- Elapsed hours exclude Slurm queue waits, gaps between allocations and qualification.

| Scenario | Exp1 GPU-h | Exp2 GPU-h | Total GPU-h | Elapsed at 8 / 12 / 16 A100 |
|---|---|---|---|---|
| Base (all 130 scaled parents fork) | 642 | 125 | 767 | 96 / 64 / 48 h |
| Half of the scaled parents fork | 630 | 62 | 691 | 87 / 58 / 43 h |
| Pessimistic (HB rate × 0.75, overhead × 1.35, half fork) | 786 | 77 | 863 | 108 / 72 / 54 h |
| Pessimistic, all fork, controls +20% | 895 | 156 | 1,051 | 132 / 88 / 66 h |

**Reading the table:**
- The longest single job, HumanoidBench D4W1536, is 11–19 h, so the critical path is set by total work divided by GPUs. Launching it first is what keeps that true.
- 900 A100-h covers every scenario except the last.
- **For an October 18 finish:**
  - 12–16 concurrent A100s finish in 2–4 days of compute after launch;
  - 8 A100s need 4–5.5 days and only fit if launch is no later than about October 13.
- Allocations are limited to ≤ 48 h each. Every scenario needs 2–3 consecutive or overlapping allocations, and the shared queue resumes runs across them automatically.
- Larger concurrency is useful only if Delta grants it. Nothing here assumes it.
