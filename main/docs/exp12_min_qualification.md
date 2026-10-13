# Reduced-queue qualification tooling and scheduler fixes (for review)

Branch `claude/exp12-min-qualification`. It was based on
`6eb397d0c485a5db202806df8a0308e48e46b05d` and has since been merged with `main` at
`dfe196d012fba3be59ba507d3a68da0c0448485a`, a README-only change.

**Scope.**
- No approval gate, methodology text or scientific setting is changed. The parent gate
  still requires all six CUDA identity certificates (`scripts/exp12_launch_plan.py:approve`),
  and the two-node preflight remains a documented gate until the owner decides otherwise.
  Amendment (ac) is not implemented.
- No GPU job was submitted.
- **This branch does change production scheduler code**: `scripts/exp12_dynamic_scheduler.py`
  (ledger completion, finish retries, evidence binding and the allocation loop). It also
  changes one test helper that GPU measurement jobs use (`tests/exp12_helpers.record_numerics`).
  Both are described in [Production scheduler changes](#production-scheduler-changes).

## Files

| File | Status | Purpose |
|---|---|---|
| `scripts/exp12_dynamic_scheduler.py` | **changed (production)** | Bounded `finish` retry; evidence bound to validation; outcomes that cannot be recorded never keep an allocation idle |
| `scripts/exp12_shared_fs_claim_check.py` | new | Two-host shared-filesystem check (no JAX, no GPU); read-only, repeatable `verify` |
| `scripts/sbatch_exp12_shared_fs_claim_check.sh` | new | The same check as a two-node **CPU-partition** job |
| `scripts/run_exp12_hb_cpu_restore_checks.sh` | new | HumanoidBench CPU restoration checks with the reviewed HumanoidBench interpreter |
| `scripts/exp12_qualification_commands.py` | new | Writes, and never runs, an idempotent submission script with a reserve/record/reconcile protocol |
| `scripts/exp12_numeric_bounds.py`, `scripts/sbatch_exp12_numeric_bounds.sh` | ported from `claude/exp12-production-launch` @ `a092101`, **then changed** | Source-bound, validated bound proposals. The 10× rule is unchanged. |
| `tests/exp12_helpers.py` | **changed** | `record_numerics` adds commit, backend, device and mode to every record |
| `scripts/exp12_resume_perturbation_study.py` | new | CPU study of non-bit-identical resumes (limited; see below) |
| `tests/test_exp12_hb_cpu_restore.py`, `tests/test_exp12_min_qualification.py`, `tests/test_exp12_ledger_contention.py`, `tests/test_exp12_review_blockers.py` | new | Tests |
| `tests/test_exp12_dynamic_scheduler.py` | changed | `finish` with a receipt now passes its validated evidence |
| `docs/exp12_resume_perturbation_results.json` | new | Study rows and summaries |

## Production scheduler changes

All of these are engineering changes. Atomic claims, exactly-once completion and
fail-closed behaviour are preserved and tested.

1. **Guard contention in `Ledger.finish`.**
   - **Before:** `finish` raised "transition guard is occupied" whenever another worker's
     claim attempt held the row's guard. The allocation loop re-raised it through
     `future.result()`, ending the worker while its training processes ran.
   - **Now:** `finish` waits for the guard with doubling sleeps (0.05 s to 2 s), for at
     most `FINISH_GUARD_TIMEOUT_SECONDS` (120 s). It never removes a guard. On timeout it
     raises `LedgerContention` having written nothing, and the claim stays.
2. **Evidence bound to its validation.**
   - **Before:** `completed.json` recorded the evidence and receipt digests as they were
     at *publication*. If the run's `DONE`, metadata, `LATEST` or receipt changed after
     `validate_done` (for example while a finish waited on contention), unvalidated
     evidence was certified.
   - **Now, in the worker:** `_finish` hashes the evidence before and after
     `validate_done`. A difference blocks the row ("changed during validation"). The
     validated digests and receipt hash travel with the outcome.
   - **Now, in `finish`:** a receipt is refused unless it comes with those digests. Under
     the guard, `finish` recomputes them. Any difference writes `blocked.json` ("changed
     after validation", recording both digest sets) and never `completed.json`.
   - **Reconciliation:** the path inside `claim` applies the same before/after check.
   - A blocked row needs the existing source-bound `review-retry`.
3. **Same outcome on retry.** `_finish` decides the outcome once (exit code, `exit.json`,
   validation and digests). Retries record that same outcome. A second `finish` is still
   refused ("claim owner changed"), and `completed.json` is published with an exclusive
   link.
4. **Outcomes that cannot be recorded never idle the allocation.** See
   [GPU-hour waste](#gpu-hour-waste-from-unrecorded-outcomes).
5. **Startup failures.** A startup failure that cannot be recorded no longer stops the
   worker. Other errors still stop it, as before.

## GPU-hour waste from unrecorded outcomes

**Investigation of `9e4ae46`.** A job whose outcome stayed unrecorded was kept in an
`unrecorded` list, and that list held the allocation open in two ways:
- **Queue drained early.** If the queue drained before `stop_at`, the loop's idle exit
  required `unrecorded` to be empty, so it polled until `stop_at`.
- **After `stop_at`.** With nothing active, it then waited until `deadline − 60 s` before
  abandoning.

An orphaned guard (a worker killed mid-transition, which is never cleared automatically)
causes exactly this. For example, if allocation B (8 A100s, 48 h) drained its queue at
hour 10, it could have idled for about 38 h, roughly 300 A100-hours.

**Now:**
- Each job gets at most `FINISH_MAX_ATTEMPTS` (5) contended attempts, 60 s apart and each
  waiting up to 120 s. That is at most 15 minutes per job, during which the worker keeps
  scheduling other work; no GPU is held by the retry.
- After the last attempt, or as soon as the allocation is idle (`_idle`: nothing active,
  pending or validating), or at `stop_at`, the outcome is written to
  `attempts/.../finish_unrecorded.json` and the claim is left in place. The allocation
  exits as soon as it has nothing else to do.

Consequences that remain:
- A left claim belongs to this allocation's Slurm job. Another allocation can reconcile it
  only once that job has ended. In allocation B, that means after both nodes' workers stop.
- A **successful** run is then reconciled from its evidence and receipt.
- A **failed** run (non-zero exit) has no `DONE`, so reconciliation re-claims and resumes it
  instead of recording `blocked.json`. Its `finish_unrecorded.json` and `exit.json` keep
  the original outcome for review.
- While the allocation is busy, a validator thread can spend up to 120 s per attempt
  waiting for a guard. That delays validation, not training.

## Shared-filesystem claim check

Two workers on two different hosts share one new directory on the production filesystem.
1. **Race.** 500 rows: exactly one `Ledger.claim` winner and one hard-link winner per row,
   the winner owns the published bytes and inode, no leftover guard or temporary file, and
   the race loops must overlap.
2. **Handoff.** 100 rows: each host finishes its own (bound receipts), while the other
   keeps trying to claim them. A stale cross-host view that claims a finished row is a FAIL.
   The worker counts how often a single-attempt `finish` found the other host holding the
   guard.
3. **Recovery.** Rank 1 claims a row before the start. After the handoff, rank 0, on the
   other host, recovers it as if rank 1's allocation had ended. The old claim must be kept
   under `retired_claims/`.

**`verify` is read-only and repeatable.** It never writes, renames or claims under the
evidence directory. Repeated runs give identical reports and leave the tree's bytes
unchanged. `--out` writes the verdict to a new file and refuses any path inside the
evidence directory. A run on one host is always FAIL.

The CPU job writes `<root>.verdict.json`, `<root>.submission.json` and
`<root>.exit_status.txt` next to the root. From two **different** login nodes:

```bash
ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA/claim_check_login_01
mkdir -p "$ROOT"
# login node A:  python scripts/exp12_shared_fs_claim_check.py worker --root "$ROOT" --rank 0
# login node B:  python scripts/exp12_shared_fs_claim_check.py worker --root "$ROOT" --rank 1
python scripts/exp12_shared_fs_claim_check.py verify --root "$ROOT" --out "$ROOT.verdict.json"
```

## HumanoidBench CPU restoration

```bash
EXPECTED_COMMIT=$FINAL_SHA HB_ENV=$APPROVED_HB_ENV \
HB_CHECK_ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA/hb_cpu_01 \
  bash scripts/run_exp12_hb_cpu_restore_checks.sh
```

- Requires a clean checkout at `EXPECTED_COMMIT`.
- Uses `MUJOCO_GL=osmesa` by default; use `egl` only on a GPU node.
- Runs the simulator restore check for all 65 cells (the ten HumanoidBench cells must
  PASS), `test_exp12_hb_cpu_restore`, and the existing HumanoidBench-dependent foundation,
  pipeline, fork and twin-critic tests.
- Fails on any skip.
- Covers environment and trainer restore fidelity on CPU, not CUDA numerics.

## Qualification submissions

`exp12_qualification_commands.py --final-sha ... --checkout ... --qual-root ... --hb-env ...
--out submit.sh` writes the script; it submits nothing itself. The script submits:
- the six cold identity cases (1 A100, 16 CPUs, 64 GB each; caps 1:30, or 3:00 for
  h1-run-v0);
- the numeric-bounds job (0:30);
- optionally, the CPU claim check.

**Each submission:**
1. is reserved atomically: `mkdir QUAL_ROOT/submissions/<name>` plus `reserved.json`;
2. is submitted with `sbatch --parsable` under the job name `exp12q_<sha12>_<name>`;
3. has its job id recorded durably (temporary file, fsync, rename to `job_id`).

**Rerunning the script:**
- skips every recorded submission;
- **stops (exit 3) at any reservation without a job id**, for example if `sbatch` failed,
  the script was interrupted, or another run is mid-submission;
- concurrent runs submit each job once.

**Resolving a partial reservation.** After checking Slurm
(`sacct --name exp12q_<sha12>_<name>`), resolve it explicitly:

```bash
python scripts/exp12_qualification_commands.py status --qual-root "$QUAL_ROOT"
python scripts/exp12_qualification_commands.py reconcile --qual-root "$QUAL_ROOT" \
  --name identity_D4W1536_dog-run --job-id 1234567 --reason "found in sacct"   # or:
python scripts/exp12_qualification_commands.py reconcile --qual-root "$QUAL_ROOT" \
  --name identity_D4W1536_dog-run --release --reason "sacct shows no job"
```

- A released reservation is kept under `submissions/.released/`.
- Only `PARTIAL` entries can be reconciled.

**Runtime estimates** (unchanged):

| Case | Cap | Estimate | Basis |
|---|---|---|---|
| D4W1536 dog-run (priority) | 1:30 | 25–30 min | measured: 42.4 interactions/s and about 3.5 s per 1,000-step fit (job 22765261); pilot 22800749 |
| D4W1536 h1-run-v0 (priority) | 3:00 | 60–75 min | earlier, unverified 32 interactions/s |
| D4W1024 dog-run | 1:30 | 15–20 min | earlier, unverified 61 interactions/s |
| D4W1024 / D4W1536 myo-key-turn | 1:30 | 15–35 min | MyoSuite A100 rate not measured |
| D4W1024 h1-run-v0 | 3:00 | 35–45 min | earlier, unverified 56 interactions/s |
| Numeric bounds | 0:30 | 10–20 min | earlier estimate |

## Numeric bounds: source-bound proposals

**The job** (`sbatch_exp12_numeric_bounds.sh`) still runs the same three isolation tests
in TF32 mode and in deterministic mode. It now:
- exports `EXP12_NUMERICS_COMMIT`, so every record names its source;
- calls `exp12_numeric_bounds.py --root <job dir> --expected-commit <sha>`.

**`record_numerics`** now adds `commit`, `backend`, `device_kind` and `mode` to each record.

**`propose`** returns `INVALID`, with no proposed bound at all, if any of these fails:
- the job's `submission.json` names the expected commit, and so does every record (mixed
  commits are rejected);
- every record is from the `gpu` backend on the A100;
- each record's mode matches its file (`numerics_tf32.jsonl` holds `tf32`;
  `numerics_deterministic.jsonl` holds `highest_deterministic`);
- per mode there are exactly 2 update records (one single-critic, one twin-critic) and 1
  training record;
- every deviation is a finite, nonnegative number, not a boolean or a string.

The 10× rule and `GPU_TOLERANCES = None` are unchanged; the owner approves values.

## Non-bit-identical resumes and Exp1: argument, evidence and limits

### Argument

After a resume, every saved component is restored. On CPU across processes this is
bit-exact, as the kill matrix, the 55 DMC/MyoSuite simulator cells and the HumanoidBench
tests show. A CUDA process can then differ only through rounding (kernel or
reduction-order choice).

1. Reduction-order rounding has no preferred sign and does not depend on architecture or
   environment.
2. Amendment (v) already runs every training matmul in TF32 (a 10-bit mantissa, relative
   rounding of order 1e-3 to 1e-4 per product). Float32 reduction-order differences are
   much smaller per operation.
3. Exp1's estimands are expectations over seeds. A perturbation of order |δ| changes the
   *expected* continuation by O(|δ|) if E[Y | s] is smooth in the state, even when single
   trajectories are chaotic. It adds variance rather than bias.

This is an argument, not a proof. The smoothness of E[Y | s] is assumed, not shown.

### Empirical check (CPU) and its limits

`scripts/exp12_resume_perturbation_study.py`, 20 seeds (1001–1020) per strength:
- **Setup:** myo-reach, D1W16 actor and critic, 600 interactions, killed at 300.
- **Perturbation:** at the resume, each float32 leaf of the actor, critic, target and
  temperature states (parameters and AdamW moments) is moved by one ULP, or by a relative
  1e-3, in a random direction.

| Outcome (after the resume) | Between-seed SD | 1 ULP: mean paired difference [95% bootstrap CI] | 1e-3: mean paired difference [95% bootstrap CI] |
|---|---|---|---|
| Evaluation return | 41.0 | 0.0007 [−0.000001, 0.0022] | −6.0 [−18.5, 0.6] |
| Probe plasticity loss (IQM per check) | 2.60 | 0.005 [−0.009, 0.023] | −0.006 [−0.063, 0.046] |
| Number of triggering checks, change per seed | — | 0 in all 20 | 0 in 18; +1 in one, −1 in one |
| Critic distance from the uninterrupted run (relative, median [range]) | — | 1.8e-7 [1.0e-7, 1.9e-4] | 2.2e-3 [1.1e-3, 1.8e-2] |

- **1e-3 return.** The negative mean comes from one seed (−121). The other 19 differences
  lie in [−3.3, 6.2], split 10 up and 10 down, with a median of −0.0002 (Wilcoxon p = 0.37).

**Limitations, read before relying on the study:**
- **Scale.** CPU only: tiny networks, one MyoSuite task, a 600-interaction horizon. Nothing
  production-sized: no D4W1024 or D4W1536, no other suites, no 500k–2M budgets, no
  Method C probe at full settings.
- **What it emulates.** The perturbation is applied **once**, at the resume. Real
  cross-process nondeterminism would add rounding differences at **every** later step. The
  1-ULP arm emulates the size of a single difference, not their accumulation. The 1e-3 arm
  is a crude stand-in for large accumulated drift, not an emulation of reduction-order
  changes.
- **What it perturbs.** Only network training states. Observation-normalisation
  statistics, replay, RNG keys and environment state are restored exactly, as a real
  resume would leave them.
- **Incomplete comparisons.**
  - Outcomes are means after the resume: evaluation return over 3-episode evaluations and
    the probe-loss IQM over checks.
  - Triggering is compared only as the **count** of triggering checks after the resume.
    The check-by-check sequence, f*_run's identity and timing, and any fork plan are
    **not** compared.
  - Full-state equality is not compared. Only the critic's parameter distance is reported.
  - The "exact resume" sanity check (3 seeds) compared the critic distance (0) and the
    mean evaluation return, not every state component. Bit-exact full-state resume is
    covered by the kill-matrix and HumanoidBench tests instead.
  - Return at the full budget is not measured.
- **Power.** With 20 seeds, shifts smaller than about half a between-seed SD at 1e-3
  cannot be excluded. One extreme seed dominates the 1e-3 return mean.
- **Provenance.**
  - Run in this CPU sandbox: Python 3.11.15 with CPU JAX 0.4.34, not the pinned Delta
    stack (Python 3.12.13, CUDA).
  - Seeds 1001–1013 (1-ULP) and 1001–1014 (1e-3) were started from the working tree
    before commit `a2c87f8`. One unused import was removed from the script while they ran;
    behaviour is identical.
  - A 30-minute tool limit stopped both runs. The remaining seeds were run from the
    committed script, and the JSON was assembled from the logs of both parts.
  - The results file does not itself record a commit or environment.
  - It is supporting evidence only, not a qualification.

### Where the argument does not apply

- **An incorrect restore** (a correctness defect; covered by restore tests and the
  certificates).
- **A precision-policy difference between processes** (guarded by the recorded and
  certified `matmul_precision` and the unset precision variables in every wrapper).
- **Bit-reproducibility of resumed runs**, which is lost. That is a reporting limitation.

## Risks of replacing the two-node GPU preflight with the CPU check

| Risk | What the CPU check misses | Consequence | Mitigation |
|---|---|---|---|
| GPU nodes mount `/work/hdd` with different client settings than login/CPU nodes | Cross-host visibility is measured only on the hosts it runs on | A stale `completed.json` view could let a second allocation claim a finished row. Directory and hard-link creation stay atomic, so a double claim raises and the worker backs off. | Run the CPU job on compute nodes; compare the reported mount options; start allocation A alone first |
| Two-node `srun` placement for allocation B | Task placement, per-task GPUs, `scontrol` parsing (`_actual_allocation`) | Fail-closed at B's startup, before any claim | Accept, or treat B's first start as its preflight |
| GPU and CPU binding | Not exercised | Already checked fail-closed at every allocation start | — |
| Clock skew between GPU nodes | Only CPU hosts measured | Affects only the stale-guard warning | Reported offsets |
| Contention level | 2 racers at a high rate | Atomicity does not depend on the number of racers | — |

## CPU test results

See the branch report for the current numbers. The suites are:
`test_exp12_review_blockers`, `test_exp12_ledger_contention`,
`test_exp12_min_qualification`, `test_exp12_dynamic_scheduler`,
`test_exp12_review_fixes`, `test_exp12_dynamic_delta_preflight`,
`test_exp12_launch_preparation`, the CPU diagnostics and twin-critic suites, and the
HumanoidBench runner.
