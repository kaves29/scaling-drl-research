# Reduced-queue qualification tooling (proposal; no gate changed)

Branch `claude/exp12-min-qualification`, based on `6eb397d0c485a5db202806df8a0308e48e46b05d`.
This branch adds tools and evidence only. **No approval gate, methodology text, scientific
setting or production code path is changed**: the parent gate still requires all six CUDA
identity certificates (`scripts/exp12_launch_plan.py:approve`), and the two-node preflight
remains a documented gate until the owner decides otherwise. Amendment (ac) is not
implemented. No GPU job was submitted.

## What is added

| File | Purpose |
|---|---|
| `scripts/exp12_shared_fs_claim_check.py` | Two-host shared-filesystem check without JAX or a GPU. It runs the real `Ledger.claim`/`Ledger.finish`/`_write_new` and bare hard links. |
| `scripts/sbatch_exp12_shared_fs_claim_check.sh` | The same check as a two-node **CPU-partition** job. Account and partition are given on the command line. |
| `scripts/run_exp12_hb_cpu_restore_checks.sh` | HumanoidBench CPU restoration checks with the reviewed HumanoidBench interpreter. It fails on any skip or any HumanoidBench cell that is not PASS. |
| `tests/test_exp12_hb_cpu_restore.py` | Real H1 environments and twin critic, killed and resumed in a new process, plus a cross-process identity arm. |
| `scripts/exp12_qualification_commands.py` | Writes, and never runs, the six cold identity commands with sized walltimes, plus the numeric-bounds job and optionally the CPU claim check. |
| `scripts/exp12_numeric_bounds.py`, `scripts/sbatch_exp12_numeric_bounds.sh` | Ported unchanged from `claude/exp12-production-launch` @ `a092101`. |
| `scripts/exp12_resume_perturbation_study.py` | CPU evidence on whether non-bit-identical resumes shift Exp1 outcomes. |
| `tests/test_exp12_min_qualification.py` | Tests for all of the above. |

### Shared-filesystem claim check

Two workers on two different hosts share one new directory on the production filesystem.
Each worker runs three phases:

1. **Race.** Both workers race on 500 disposable rows. Every row must end with exactly one
   `Ledger.claim` winner and exactly one hard-link winner, and the winner must own the
   published bytes and inode. No guard or temporary file may be left behind. Both workers
   start at a time agreed through the filesystem, and their race loops must overlap.
2. **Handoff.** There are 100 handoff rows. Each host claims its own rows before the start,
   then finishes them by writing `completed.json` and retiring its claim. Meanwhile the
   other host keeps trying to claim them. A stale cross-host view of `completed.json` or
   `claim/` would let it claim a finished row. That is a FAIL.
3. **Recovery.** One row's owner is reported as ended. Reclaiming it must preserve the old
   claim under `retired_claims/`.

`verify` also reports, without a pass/fail threshold:
- each host's mount (device, type, options); different filesystems are a FAIL;
- the offset between each host's clock and the filesystem timestamps;
- how often `Ledger.finish` found the other host holding the row's guard.

A run on one host is always FAIL.

On Delta, either submit the CPU job (`exp12_qualification_commands.py --cpu-partition ...
--cpu-account ...`) or use two **different** login nodes:

```bash
ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA/claim_check_login_01
mkdir -p "$ROOT"              # once, from either host
# login node A:  python scripts/exp12_shared_fs_claim_check.py worker --root "$ROOT" --rank 0
# login node B:  python scripts/exp12_shared_fs_claim_check.py worker --root "$ROOT" --rank 1
python scripts/exp12_shared_fs_claim_check.py verify --root "$ROOT"   # exit 0 only on PASS
```

The workers need only Python 3.10 or later and this checkout; JAX is never imported. Start
both within ten minutes of each other.

### HumanoidBench CPU restoration

```bash
EXPECTED_COMMIT=$FINAL_SHA HB_ENV=$APPROVED_HB_ENV \
HB_CHECK_ROOT=/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA/hb_cpu_01 \
  bash scripts/run_exp12_hb_cpu_restore_checks.sh
```

It requires a clean checkout at `EXPECTED_COMMIT` and uses `MUJOCO_GL=osmesa` by default.
In this container EGL is unavailable without a GPU and osmesa works; if osmesa is missing
on a Delta CPU host, run the script on a GPU node with `MUJOCO_GL=egl`, without using the
GPU. It runs, with the real H1 environments:
- the simulator restore check for all 65 task/seed cells (all ten HumanoidBench cells must
  be PASS);
- the new `test_exp12_hb_cpu_restore`;
- the existing HumanoidBench-dependent foundation, pipeline, fork and twin-critic tests.

These checks cover environment and trainer restore fidelity. They do not qualify CUDA
numerics.

### Identity walltimes

Every case keeps 1 A100, 16 CPUs and 64 GB. Only the cap changes, from four hours:

| Case | Cap | Estimate | Basis |
|---|---|---|---|
| D4W1536 dog-run (priority) | 1:30 | 25–30 min | measured: 42.4 interactions/s and about 3.5 s per 1,000-step fit (job 22765261); pilot 22800749 |
| D4W1536 h1-run-v0 (priority) | 3:00 | 60–75 min | earlier, unverified 32 interactions/s; twin probes |
| D4W1024 dog-run | 1:30 | 15–20 min | earlier, unverified 61 interactions/s |
| D4W1024 / D4W1536 myo-key-turn | 1:30 | 15–35 min | MyoSuite A100 rate not measured |
| D4W1024 h1-run-v0 | 3:00 | 35–45 min | earlier, unverified 56 interactions/s |
| Numeric bounds | 0:30 | 10–20 min | earlier estimate |

The total cap is 12.5 A100-hours, against the earlier plan's 24 plus 1.33. Expected use is
about 3–3.5 A100-hours.

Per case, the work is:
- 50,000 interactions to the fork at check 2 (100,000 for HumanoidBench);
- 2 × 1,000 continuation steps;
- three Method C probe checks;
- startup and compilation in two processes.

A timeout stops that case only. Inspect it before any retry.

### Numeric bounds port

Both files are byte-identical to `a092101`. Their dependencies are also identical at this
base: `tests/exp12_helpers.py`, `tests/test_exp12_diagnostics.py` and
`tests/test_exp12_twin_critic.py`. The job runs the same three isolation tests in TF32 mode
and in deterministic mode. It proposes bounds with the logged 10× rule (decisions log,
2026-10-05, item 2) and applies nothing. `GPU_TOLERANCES` stays `None` until the owner
approves values.

## Evidence: non-bit-identical resumes do not bias Exp1

### What can differ

A resume restores the complete state:
- network parameters and AdamW moments;
- the JAX key and the global NumPy and Python RNGs;
- the replay buffer and observation normalisation;
- both environments mid-episode;
- counters, probe records and the fork plan.

This is exact on CPU across processes. That is shown by the kill matrix, the 55
DMC/MyoSuite simulator cells, and the HumanoidBench tests on this branch, which compare
bit for bit.

On CUDA, a second process can differ only through rounding. XLA may pick a different
kernel or reduction order for the same mathematics. Exact restore of the state is a
separate correctness property. Each fork guards it in production with a bit-exact Check 1,
and the identity certificates qualify it on the GPU.

### Argument

1. **Rounding has no direction.** A different reduction order changes which rounding
   errors occur, with no preferred sign and no dependence on architecture or environment.
   Precision policy itself cannot drift between processes: every launch records it, and
   the certifier refuses a mismatch.
2. **TF32 already dwarfs it.** Amendment (v) runs every training matmul in TF32, a 10-bit
   mantissa with relative rounding up to about 4.9e-4 per product. Float32 reduction-order
   differences are about 1e-7 relative, roughly 10^3 to 10^4 times smaller. The registered
   design already accepts, in every step of every run, rounding far larger than a
   cross-process resume adds.
3. **Effect on an expectation.** Exp1's estimands are expectations over seeds: return at
   the budget, f*_run, and the probe-loss trajectories, summarised by IQM and bootstrap
   over 5 seeds per cell. A resume moves the state s to s + δ with |δ| of order 1e-7
   relative.
   - Individual trajectories are chaotic, so the realised continuation can differ
     completely.
   - The expected outcome E[Y | s] is averaged over the continuation's environment and
     policy randomness, so it changes by O(|δ|).
   - The resume therefore draws a different realisation from essentially the same
     conditional distribution. That adds variance and no bias.
4. **Resume frequency doesn't matter.** Larger cells, such as D4W1536 HumanoidBench, will
   resume more often than small ones. Because each resume's effect has mean zero,
   different resume rates across cells cannot produce a systematic difference between
   cells.

### Empirical check (CPU)

`scripts/exp12_resume_perturbation_study.py`:
- **Setup:** myo-reach, a D1W16 actor and critic, 600 interactions, killed at 300.
- **Perturbation:** on resume, every float32 leaf of the actor, critic, target and
  temperature states (parameters and AdamW moments) is moved in a random direction drawn
  from an RNG independent of training. Two strengths: one ULP (resume-scale), and a
  relative 1e-3 (TF32-scale, which does make trajectories diverge).
- **Comparison:** perturbed resume minus uninterrupted run, paired by seed, over 20 seeds.
- **Sanity check:** the exact resumes in the 1-ULP study are bit-identical to the
  uninterrupted runs.

Results: 20 seeds per strength (1001–1020). Raw rows and summaries are in
`docs/exp12_resume_perturbation_results.json`.

| Outcome (after the resume) | Between-seed SD | 1 ULP: mean paired difference [95% bootstrap CI] | 1e-3: mean paired difference [95% bootstrap CI] |
|---|---|---|---|
| Evaluation return | 41.0 | 0.0007 [−0.000001, 0.0022] | −6.0 [−18.5, 0.6] |
| Probe plasticity loss (IQM per check) | 2.60 | 0.005 [−0.009, 0.023] | −0.006 [−0.063, 0.046] |
| Triggering checks, change per seed | — | 0 in all 20 seeds | 0 in 18 seeds; +1 in one, −1 in one |
| Critic distance from the uninterrupted run (relative, median [range]) | — | 1.8e-7 [1.0e-7, 1.9e-4] | 2.2e-3 [1.1e-3, 1.8e-2] |

- **1 ULP.** This is the size of difference a resume introduces.
  - The effect stays at rounding level: return differences are about 10^4 times smaller
    than the between-seed SD.
  - No trigger decision changed.
  - The three exact resumes were bit-identical to the uninterrupted runs.
- **1e-3.** This is TF32-scale, roughly 10^4 times larger than a resume.
  - The perturbation does make trajectories diverge: the paired SD of the return is 27,
    about two thirds of the between-seed SD.
  - Yet the mean shift is consistent with zero for every outcome.
  - The return's negative point estimate comes from a single seed (−121). The other 19
    differences lie in [−3.3, 6.2], split 10 up and 10 down, with a median of −0.0002
    (Wilcoxon signed-rank p = 0.37).
  - This is what reselecting a realisation looks like: it adds variance, not a shift.
- **Power.** With 20 seeds the study can exclude only shifts of more than about half a
  between-seed SD at the 1e-3 strength. It supports the argument but does not replace it.

### Where the argument does **not** apply

- **An incorrect restore.** A lost or wrong state component is a correctness defect,
  not rounding. The restore tests and certificates exist for this.
- **A precision-policy difference between processes.** For example, TF32 in one process
  and full FP32 in another. This would be systematic. It is guarded by the recorded and
  certified `matmul_precision` and the unset `XLA_FLAGS`, `NVIDIA_TF32_OVERRIDE` and
  related variables in every wrapper.
- **Reproducibility.** A resumed run will not be bit-reproducible on rerun. That is a
  reporting limitation, not bias.
- **The study's limits.** It uses tiny CPU networks over a short horizon. It supports the
  mechanism and does not measure the production effect size.

## CPU test results on this branch

| Run | Result |
|---|---|
| `tests.test_exp12_min_qualification` | 22/22 OK |
| `tests.test_exp12_hb_cpu_restore` (venv with humanoid_bench, osmesa) | 3/3 OK (833 s) |
| `run_exp12_hb_cpu_restore_checks.sh` end to end, clean detached checkout of `a2c87f8`, `HB_ENV=venv_hb` | PASS, exit 0: 65/65 environment cells; 33 tests OK, none skipped (1,938 s) |
| `tests.test_exp12_dynamic_scheduler`, `test_exp12_launch_preparation`, `test_exp12_review_fixes` | 55/55 OK |
| Claim check, two local processes (one host) | Every row-level check passes; the verdict is correctly FAIL because both workers ran on one host |

## Risks of replacing the two-node GPU preflight with the CPU check

| Risk | What the CPU check misses | Consequence | Mitigation |
|---|---|---|---|
| GPU nodes mount `/work/hdd` with different client settings or versions than login/CPU nodes, such as attribute or lookup caching | The handoff phase measures cross-host visibility only on the hosts it runs on | Stale negative lookups could let a second allocation claim a finished or claimed row. Directory and hard-link creation stay atomic at the server, so a double claim raises, and the worker backs off. A stale `completed.json` is the real exposure. | Run the CPU job on CPU compute nodes rather than login nodes; record and compare mount options (`verify` reports them); launch allocation A alone first, so that cross-node claiming starts only when B joins |
| Two-node `srun` placement for allocation B | One task per node, four GPUs per task, `CUDA_VISIBLE_DEVICES`, and `scontrol` parsing for two nodes (`_actual_allocation`) | Fail-closed at B's startup before any claim. B's queue wait is lost; no run is affected. | Accept, or let B's first start serve as its own preflight |
| GPU and CPU binding | Not exercised | Already checked fail-closed at every allocation start: 4 visible GPUs, `SLURM_GPUS_ON_NODE`, node count, CPU affinity, pinned stack, HumanoidBench interpreter | None needed |
| Clock skew between GPU nodes | Only CPU hosts measured | Affects only the stale-guard *warning* (600 s), never an automatic action | Reported offsets |
| Contention level | 2 racers at high rate rather than up to 3 workers at 15 s polling | Atomicity does not depend on the number of racers | — |

**Defect found by the handoff phase, now fixed.** `Ledger.finish` used to raise "transition
guard is occupied" if another worker held the row's guard at that moment. The worker loop
re-raised it through `future.result()`, which ended the allocation's worker while its training
children ran. Locally this happened 0–2 times per 200-row handoff under deliberate contention.

How it is fixed (scheduler only; no scientific setting or gate changed):
- **Bounded wait.** `finish` now waits for the guard with doubling sleeps from 0.05 s up to
  2 s, for at most `FINISH_GUARD_TIMEOUT_SECONDS` (120 s). A guard is never removed or
  overridden.
- **Fail closed on timeout.** If the wait runs out, `finish` raises `LedgerContention` (a
  `ValueError`) having written nothing. The claim stays, so no other worker can start the row.
- **Same outcome on retry.** `_finish` decides the outcome once: exit code, `exit.json` and
  `validate_done`. Later attempts record that same outcome. Completion stays exactly-once,
  because a second `finish` is refused ("claim owner changed") and `completed.json` is
  published with an exclusive link.
- **Worker loop.** It keeps a contended job and retries every `FINISH_RETRY_SECONDS` (60 s).
  If the outcome is still unrecorded near the allocation's end, it writes
  `finish_unrecorded.json` in the attempt directory and leaves the claim. Once the allocation
  ends, a later allocation reconciles the row from its evidence.
- **Startup failures.** A failure that cannot be recorded no longer stops the worker.
- **Other errors still stop the worker**, as before.
- **The claim check** uses a single-attempt `finish` so it can still count contention.
