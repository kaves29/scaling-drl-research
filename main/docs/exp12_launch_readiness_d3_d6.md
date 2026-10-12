# Exp1/Exp2 launch readiness: D3–D6 (2026-10-12)

Branch `claude/exp12-production-launch`; this round builds on `f045cc5`.
- **Decisions in force:** D3 is selected (A2 staged identity) as the *intended* configuration. No release has been published, nothing is active, and no job was submitted.
- **Unchanged:** D1 and D2, and the scientific methodology.

**Access limit.** This work ran in a cloud container with **no access to Delta**: no `/work` mount, no Slurm, no SSH. Nothing here was observed on Delta. Facts about Delta come from NCSA's public documentation (cited). The pilot's own artifacts could not be read, so §1 gives a verdict procedure, not a verdict.

## 1. Pilot 22800749: trigger and fork verdict
**Determined here:** nothing about the pilot's contents. **Indications** from the facts you supplied:

| Fact (owner-supplied) | What it implies (inference, not proof) |
|---|---|
| Wall time 03:24:44, exit 0:0 | Consistent with the full 500,000-interaction dog-run horizon. That is 40.7 interactions/s end-to-end, against 42.44/s training-only measured earlier. |
| Root ≈ 2.2 GB | One complete D4W1536 dog-run state is about 2.19 GB with a full replay (historical estimate). A fork keeps a **second** complete state (`fork/state`, ~1.1–2.2 GB depending on fork time). 2.2 GB therefore points to **no fork state**. |
| grep found no trigger/fork matches | A fork prints `[exp1] fork state saved at interaction_step N` and later `post-fork evaluation` lines to the run's training log. If the grep covered `attempts/*/training.log`, this also points to no fork. The trigger itself is not printed. |

**Deciding procedure.** It is read-only and CPU-only, minutes on a Delta login node: `scripts/exp12_pilot_inspect.py`. It never writes inside the root, and its tests show the run tree is byte-identical before and after. What it does:
- recomputes every check with the production bootstrap and the f*_run rule;
- reports the horizon, fork artifacts, natural-fork validity and the positive control's input requirements;
- generates the source-compatibility certificate;
- records the SHA-256 of every evidence file.

```bash
# Lightweight CPU environment (no JAX/GPU stack): numpy, scipy, omegaconf, hydra-core, plus git.
python3 -m venv ~/exp12_inspect_env && ~/exp12_inspect_env/bin/pip install -r "$FINAL_CHECKOUT/main/requirements-inspect.txt"
cd "$FINAL_CHECKOUT/main"   # clean checkout of the reviewed branch commit (full history: contains 7d0883c)
PILOT=/work/hdd/biqc/skaveti1/exp12_pilot_runs/7d0883cfa99d27fed062d77a68cbbfa27e560061_dog_run_D4W1536_seed102
D=/work/hdd/biqc/skaveti1/exp12_decisions; mkdir -p "$D"
~/exp12_inspect_env/bin/python scripts/exp12_pilot_inspect.py --root "$PILOT" --out "$D/pilot_inspection.json" \
  --final-sha "$(git rev-parse HEAD)" --compat-out "$D/pilot_source_compat.json"
```

Evidence it reads, in the layout written by `exp1.py`, `run_probes.py`, `fork.py` and `ledger.py`:
- `<run>/run_metadata.json` (source commit, dirty flag, resolved config, launches with GPU);
- `<run>/DONE` and `<run>/state/LATEST` with its `meta.pkl` (horizon and recorded `f_star`);
- `<run>/probes/probe_checks.csv` and `check_00..20.npz` (per-round scores, losses, recorded bounds);
- `<run>/fork/{fork.json,FORK_READY,state/}`;
- `<results_root>/exp12/exp1/runs/<run_key>/{run.csv,checks.csv}`;
- every `.log`, `.out` and `.err` under the root.

`<run>` is the single directory found with an exp1 `run_metadata.json`. For the prepared pilot launcher it is `checkpoints/exp1/D4W1536/dog-run/seed_102`.

| Verdict | Meaning | Protocol-prescribed next step |
|---|---|---|
| `NO_NATURAL_TRIGGER` | Complete run, no two consecutive firing checks in 1–19, no fork | Methodology (e): **"If it never triggers, stop and consult. The trigger is never loosened."** No positive control is possible, so **no Exp1 parent may start, D2W512 under A2 included.** Do not force a fork. The report's per-check L and lower bounds are the evidence for the consultation. |
| `NATURAL_FORK_AVAILABLE` | Natural f*_run, a complete fork, `_validate_natural_fork` passes | Technically eligible input. Run the positive control (Q2) with the certificate. Its exit status, not this report, decides L_trigger > 0, noise < 0.10 and m. |
| `FORCED_FORK_NOT_ELIGIBLE` / `INCOMPLETE_RUN` / `INCONSISTENT_EVIDENCE` | — | Not usable; preserve the evidence and review. |

**Compatibility (technical determination).** It holds whatever the verdict. Pilot source `7d0883c` and the final source share 128 of 129 in-scope scientific files byte-for-byte; the only difference is the `exp2_arm.py` source guard, which the positive control does not execute. Reuse is therefore **technically valid if and only if** the verdict is `NATURAL_FORK_AVAILABLE`.

**Recommendation:** reuse in that case. In every other case, reject the pilot as positive-control input.

## 2. D3 (A2) as the launch configuration, and containment
**The configuration:**
- the parent plan uses `release_mode: per_run`, with every gate below deferred to releases;
- the A2 release body is `{"type":"a2_staged_identity","amendment_reference":"A2 selected <date>"}`.

**Per-run guarantee.** `run_blockers` runs before **every** start; tests cover the live loop.

| Run | Must hold before it starts |
|---|---|
| Every parent | positive-control gate release; the 4 other global gates (storage/runtime, allocation, CUDA numeric bounds, methodology freeze); allocation approval |
| D2W512 parent | the above only (A2) |
| D4W1024 / D4W1536 parent in suite S | the above, plus **both** S identity certificates |
| D4W1024 parent | additionally the `fresh_null` release (D1) |
| hopper-hop D4W1536 parent | additionally `hopper_range` = `release` (D2) |
| Exp2 arm | its parent's prerequisites, plus the `exp2` release (m) |

**Containment of a later shared-code defect** (new: `<queue>/HOLD.json`):
1. **Stop.** Write `{"reason":"…","terminate_active":false}` to `$QUEUE/HOLD.json`. Every worker stops admitting parents and arms within one poll (≤ 15 s), and running jobs finish and are validated.
   - Use `"terminate_active": true` if running jobs must stop. They get SIGTERM, stop at their last committed checkpoint, and their rows are blocked for an explicit `review-retry`.
   - A malformed `HOLD.json` also counts as a hold.
2. **Scope.** All runs share one source commit (the queue is commit-bound). `status --rows`, plus the ledger's `completed.json` receipts and each attempt's `start.json`, list exactly which runs started, completed or were running on the affected code.
3. **Classify** the affected path:
   - probe/trigger: Exp1 endpoint and fork placement;
   - fork/injection/restore: Exp2 and post-fork Exp1;
   - environment/agent: all data;
   - logging/IO only: often none.
4. **Fix on a new commit.** This creates a new `FINAL_SHA`, which needs a new plan and queue; the old queue can never run new code.
   - Rerun the affected runs under the new queue.
   - Keep an already-completed run only by owner decision, backed by `exp12_source_compatibility.py certify --allow-path <fixed file>` showing that run's path is untouched.
   - Requalify identity and the positive control if the certificate shows their paths changed.
5. **Resume.** Delete `HOLD.json` only after that review.

## 3. Delta hardware: A100 and A40
**Observed in NCSA documentation:**

| Partition | Nodes | GPUs/node | GPU memory | Cores/node | RAM/node | Max wall | Max nodes/job |
|---|---|---|---|---|---|---|---|
| gpuA100x4 | 100 | 4 × A100 SXM | 40 GB | 64 | 256 GB | 48 h | all |
| gpuA100x8 | 6 | 8 × A100 SXM | 40 GB | 128 | 2,048 GB | 48 h | all |
| gpuA40x4 | 100 | 4 × A40 PCIe | 48 GB | 64 | 256 GB | 48 h | all |

**Billing (also observed):**
- On 4-way nodes, 1 SU is roughly one GPU-hour, with 16 cores and 62.5 GB of host memory per GPU; a node-exclusive 4-GPU hour costs 4 SU.
- On 8-way A100 nodes, a GPU's share is 16 cores and 250 GB.
- A40 is discounted by an unpublished factor (ACCESS SU converter).
- Other per-job GPU or user caps are "TBD" in the docs.
- Check before submitting: `accounts` (balances), `sinfo -p gpuA100x4,gpuA100x8,gpuA40x4 -o "%P %a %l %D %t"` and `sacctmgr -nP show assoc where user=$USER format=account,partition,qos`. Partition access for `biqc-delta-gpu` is **unverified**.

**Answers:**
1. **Allocation B (8 A100s):**
   - **2 × gpuA100x4** is supported now, drawn from a 100-node pool.
   - **1 × gpuA100x8** is now also supported by the scheduler (`partition: gpuA100x8`, 8 GPUs/node). The partition has only 6 nodes and per-GPU host-memory billing, and access is unverified, so treat it as the alternative.
   - Both are the same GPU model (A100-SXM 40 GB). The scheduler checks `nvidia-smi` names and fails closed on a mismatch.
2. **A40 for confirmatory training: not compatible with the current protocol or code.** **Recommendation: do not use A40 before October 18.**
   - Methodology (p) requires identity on *the grid's GPU model*, and arms on the fork's model.
   - The code pins `NVIDIA A100-SXM4-40GB` throughout (scheduler, identity certification, positive control, launch plan).
   - A mixed grid also needs a balanced hardware assignment, for example by seed across all architectures, so that hardware is not confounded with architecture.
   - Minimum to use it:
     1. an owner amendment to (p) and the grid GPU model;
     2. code generalization of the GPU pin, with arms pinned to their fork's model;
     3. six identity cases plus numeric bounds on A40;
     4. one throughput profile.
   - That is about 1–2 days and roughly 30 A40-h.
3. **A40 speed. Unmeasured; vendor datasheet estimates only:**
   - A40: TF32 74.8 TFLOPS dense, 696 GB/s.
   - A100 SXM: 156 TFLOPS, 1,555 GB/s.
   - Our TF32 small-batch MLP updates suggest A40 is ≤ A100 per GPU for D4W1536, and similar for host-bound D2W512.
   - It could shorten the calendar only if A100 queue waits exceed the 1–2 days of A40 qualification. It is not worth it on current evidence. Your earlier L4 benchmarks are not used.

## 4. Prepared allocation commands (DO NOT SUBMIT)
**Shapes:**
- A (4 × 35 h) uses the generic name `A35`.
- B uses `B48`, which is 2 × gpuA100x4. The alternative is `B48x8`, one gpuA100x8 node.
- The legacy names A and B are fixed at 30 h and 2 × 4/48 h, so they are not used.

**Resources.** Per-node CPUs and memory follow Delta's billing shares: 64 cores and 240 GB on 4-way nodes (4 SU/h); 128 cores and 1,900 GB on the 8-way node (8 SU/h).

**How to produce the commands.** Run `python scripts/exp12_dynamic_scheduler.py allocation-command --queue "$QUEUE" --approval $D/allocation_<NAME>.json --allocation <NAME>` after `mkdir -p "$QUEUE/submissions/<NAME>"`. With real paths and approval digests it prints exactly:

```bash
env EXPECTED_COMMIT=$FINAL_SHA EXP12_QUEUE=$QUEUE EXP12_ALLOCATION_APPROVAL=$D/allocation_A35.json \
  EXP12_APPROVAL_SHA256=<sha256 of that file> EXP12_ALLOCATION=A35 \
  sbatch --parsable --export=ALL --nodes=1 --partition=gpuA100x4 --ntasks-per-node=1 --gpus-per-node=4 \
  --cpus-per-task=64 --mem=240G --time=2100 --output=$QUEUE/slurm/%x_%j.out --error=$QUEUE/slurm/%x_%j.err \
  scripts/sbatch_exp12_dynamic_allocation.sh > $QUEUE/submissions/A35/job-id.txt 2> $QUEUE/submissions/A35/submission.stderr

env EXPECTED_COMMIT=$FINAL_SHA EXP12_QUEUE=$QUEUE EXP12_ALLOCATION_APPROVAL=$D/allocation_B48.json \
  EXP12_APPROVAL_SHA256=<sha256> EXP12_ALLOCATION=B48 \
  sbatch --parsable --export=ALL --nodes=2 --partition=gpuA100x4 --ntasks-per-node=1 --gpus-per-node=4 \
  --cpus-per-task=64 --mem=240G --time=2880 --output=$QUEUE/slurm/%x_%j.out --error=$QUEUE/slurm/%x_%j.err \
  scripts/sbatch_exp12_dynamic_allocation.sh > $QUEUE/submissions/B48/job-id.txt 2> $QUEUE/submissions/B48/submission.stderr

# Alternative B on one 8-way node (access unverified):
env EXPECTED_COMMIT=$FINAL_SHA EXP12_QUEUE=$QUEUE EXP12_ALLOCATION_APPROVAL=$D/allocation_B48x8.json \
  EXP12_APPROVAL_SHA256=<sha256> EXP12_ALLOCATION=B48x8 \
  sbatch --parsable --export=ALL --nodes=1 --partition=gpuA100x8 --ntasks-per-node=1 --gpus-per-node=8 \
  --cpus-per-task=128 --mem=1900G --time=2880 --output=$QUEUE/slurm/%x_%j.out --error=$QUEUE/slurm/%x_%j.err \
  scripts/sbatch_exp12_dynamic_allocation.sh > $QUEUE/submissions/B48x8/job-id.txt 2> $QUEUE/submissions/B48x8/submission.stderr
```

**Approval JSON values** (`dynamic_allocation_approval.template.json`). These are proposals; you approve them.
- Resources:
  - `partition` and `nodes` as above;
  - `gpus_per_node` 4 (8 for B48x8);
  - `cpus_per_node` 64 (128);
  - `memory_gb_per_node` 240 (1900);
  - `wall_minutes` 2100 for A35, 2880 for B.
- Admission:
  - `max_processes_per_gpu` 1;
  - `gpu_headroom_gb` 4 and `host_headroom_gb` 16;
  - `stop_margin_minutes` 20;
  - `idle_wait_minutes` 60.
- Per-architecture profile: 16 CPUs (on the 8-way node, 8 × 16 = 128 fills it exactly), 48 GB host RAM, 8 GB GPU memory, `min_interactions_per_second` 1. The interactions floor only matters for packing.
- Runtime estimates in minutes, conservative (1.2× overhead). They only decide whether a run still fits before the allocation ends; runs resume across allocations.

| Group | D2W512 | D4W1024 | D4W1536 | Arm D4W1024 | Arm D4W1536 |
|---|---|---|---|---|---|
| dmc_hard | 96 | 168 | 234 | 42 | 60 |
| dmc_medium | 48 | 84 | 120 | 21 | 30 |
| myosuite_simba | 90 | 168 | 240 | 42 | 60 |
| humanoid_bench | 228 | 360 | 624 | 90 | 156 |

**Scheduler behavior.** There is one shared queue and atomic claims; whichever allocation starts first takes the longest eligible work.
- If A starts first, its 4 GPUs take HumanoidBench D4W1536 and D4W1024 first. B then picks up the next-longest rows as soon as it starts, with no duplicate claims.
- If B starts first, the same order applies.
- When an allocation nears its end, it admits only runs that fit, using the unfinished fraction of resumable runs; anything cut off resumes from its last checkpoint under the next allocation.
- Under A2, runs blocked by a missing release stay pending. With `idle_wait_minutes` > 0, an allocation waits for releases instead of exiting.

**Budget and overlap.** Simulated with `scripts/exp12_elapsed_estimate.py --windows`. A+B = 140 + 384 = **524 A100-h**. Every overlap order completes about 515 GPU-h. Start order changes the finish time, not the amount of work done.

| Scenario (campaign GPU-h) | A+B together at t=0 | B 12 h late | B 24 h late | A 24 h late | Left after A+B |
|---|---|---|---|---|---|
| Pilot-calibrated (1.05× overhead), half the scaled parents fork (605) | ends 48 h | 60 h | 72 h | 59 h | **89 GPU-h** (64 short parents, 6 arms) |
| Pilot-calibrated, all fork (671) | 48 h | — | — | — | 156 GPU-h |
| Base (1.2×), all fork (767) | 48 h | 60 h | 72 h | 59 h | 252 GPU-h |
| Pessimistic: HB × 0.75, 1.35×, half fork (863) | 48 h | 60 h | 72 h | 59 h | 343 GPU-h |
| Worst: HB × 0.75, 1.35×, all fork, controls +20% (1,051) | 48 h | — | — | — | 533 GPU-h |

**A + B alone cannot complete the campaign under any scenario.** The leftover is mostly short D2W512 runs, because the longest work goes first. A third allocation is needed:
- C48 = 2 × gpuA100x4 × 48 h (384 GPU-h), making 908 GPU-h in total, about your 900.
- Submitted alongside A and B, everything finishes by about 58–61 h after start in the calibrated and base cases. The pessimistic case fits; the worst case does not.
- Its command is identical to B48's with name C48.
- Alternatively, D2W512 packing would save about 55 GPU-h (Block B measured a 2.3× aggregate gain with 4 D2 jobs per GPU), but it needs your approval of `max_processes_per_gpu` > 1.

## 5. D6: is a short production preflight needed, and five or ten minutes?
| Measurement | A ~5-min job can measure it reliably? | Evidence that covers it instead |
|---|---|---|
| Cross-node atomic claims, stale-claim recovery, GPU/CPU binding | Yes. It typically takes 2–5 min (Codex); **keep the 10-min cap**. Slurm bills elapsed time, so a cap that ends at 3 min costs nothing extra, while a 5-min cap risks a spurious TIMEOUT on a cold Lustre `conda`/JAX import, which by rule blocks launch. | **Q3 `fsbind` (required)**: 2 × gpuA100x4, ≤ 1.33 A100-h. Needed in every A+B plan, because A and B are different nodes sharing claims. |
| Per-architecture throughput | No. Compile, 5,000 random warm-up steps and JIT dominate the first minutes. | DMC/MyoSuite: Block B single-job rates plus the pilot's 3.41 h end-to-end (dog D4W1536). HumanoidBench and full-width memory: the **six identity cases**, now instrumented with 5-s `nvidia-smi` samples, GNU-time max RSS and per-phase wall time, over ~10% of a run plus 1,000 steps at full settings. |
| GPU memory and utilization | Partly | Identity samples; earlier D4W1536 dog peak 3.76 GB. Every architecture is far below 40 GB at one process per GPU. |
| Host CPU/RAM | No: the replay grows over the whole run | Identity GNU-time RSS (or `sacct -j <job> --format=JobID,MaxRSS,Elapsed`); dog final state ≈ 2.19 GB including the replay; admission at 48 GB per process |
| Safe processes per GPU | No | Block B packing: D2 ×4 → 2.3× aggregate; D4W1536 ×2 → 1.08×. Recommend 1 per GPU, with D2 packing as an optional owner decision. |
| Scheduler pickup and duplicate prevention | Claims: yes (Q3). Full dispatch: no. | CPU tests (concurrent-claim, run-loop and gating tests); live canary: the first 15 min of the first allocation (`status`, attempt `start.json`). |
| Checkpoint and resume | No | GPU resume probe PASS (job 22783981, same restore code) and the identity cases' fork restores |

**Conclusion.** No new five-minute production preflight is needed. The only preflight is Q3 with its existing 10-minute cap; the instrumented identity cases supply the remaining measurements at no extra cost.

## 6. Missing technical qualification evidence
1. **Pilot verdict:** run `exp12_pilot_inspect.py` (§1). If it returns `NO_NATURAL_TRIGGER`, the protocol requires stop and consult, and **production cannot start**.
2. **Positive control:** only with a natural fork (Q2). m is recorded only from its result.
3. **Identity:** **all six** cases on the final source are still required; no final-source receipt exists. They are D4W1024 and D4W1536 × {dog-run, myo-key-turn, h1-run-v0}. A2 lets each suite release independently.
4. **CUDA numeric bounds.** Block B records are not in the repository.
   - Expected files: `<Block B checkout>/main/logs/blockB_22706349/gpu2/numerics_{default,deterministic}.jsonl` (`step_tests` in `exp12_blockB.sh`). Find them with `find /work/hdd/biqc/skaveti1 -path '*blockB_22706349/gpu2/numerics_*.jsonl'`.
   - Propose bounds: `python scripts/exp12_numeric_bounds.py --numerics <files> --out $D/bounds_blockB.json`.
   - Block B ran at `b4a90cb`. The review notes its records are calibration *candidates*. The fallback is `scripts/sbatch_exp12_numeric_bounds.sh`, which runs the three isolation tests in both modes on the final source and proposes bounds. It needs 1 A100, about 10–20 min (`--time=00:30:00`).
   - The values stay `None` until you approve them.
5. **Delta preflight Q3:** a PASS on the actual mount.
6. **Partition access and balance:** `accounts` and `sacctmgr` (§3).

## 7. Execution order for the first authorized GPU jobs
1. **CPU, login node (no authorization needed beyond running read-only tools):** pilot inspection (§1), plus the numeric-bounds search (§6.4).
2. **If `NATURAL_FORK_AVAILABLE`,** submit together, in this order:
   - Q2, the positive control (1 A100, ≤ 1 h);
   - Q1, the identity cases, HumanoidBench first (6 × 1 A100, ≤ 4 h each);
   - Q3, `fsbind` (2 nodes, ≤ 10 min);
   - only if the Block B records are unusable: the numeric-bounds job (1 A100, ≤ 30 min).

   **If `NO_NATURAL_TRIGGER`:** stop and consult. Identity, Q3 and the numeric bounds may still run to save time later, but no parent can start.
3. **Publish releases** as their evidence arrives:
   - the global gates;
   - the positive-control gate, then `a2_staged_identity`;
   - identity per suite;
   - D1 and D2 when decided;
   - `exp2` (m).
4. **Submit A35, B48 and C48 together.** The queue gives every allocation the longest eligible work. The first 15 minutes are the live canary: watch `status`.
