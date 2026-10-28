# Method C consolidated engineering review (13 October 2026)

Base: `c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b`. Reviewed Claude commits:
`d65f9bb` and `b225577c2ba5a18b2ad6d14c47b34a00b5443e05`. The
consolidated candidate preserves their history; resolve its final SHA from Git.
No scientific configuration, probe threshold, RNG rule, architecture, training
budget, resource cap, or statistical definition changed in those commits.

| Finding against Method C base | Review outcome |
|---|---|
| Late-fork control resume omitted post-budget checks | Confirmed and fixed: restored controls extend the probe grid to `control_end_step`. Real late-fork kill/resume compares the complete state and all checks. |
| Stale Slurm claim could crash worker | Confirmed and fixed: terminal/unknown Slurm states are distinguished; unknown lookup backs off that row without reassignment or worker exit. An ended owner can be retired under the atomic claim guard. |
| `dmc_medium` absent from runtime profiles | Confirmed and fixed in approval template and admission validation, including Exp2 arm profiles. It remains an unfilled owner-reviewed estimate, not an invented runtime. |
| No initial checkpoint prevented retry | Confirmed and fixed for Exp1 only: source/config metadata is checked, then the run restarts from scratch after an explicit reviewed retry; unfinished Exp2 arms still require a valid checkpoint. Aborted fresh-critic artifacts are retained separately. |
| Previous checkpoint removed before new state was durable | Confirmed risk; files and directories of the new state are now flushed before the atomic LATEST update, and the containing directory is flushed before old states are removed. Delta filesystem durability and I/O cost remain unmeasured. |
| `srun` lacked explicit CPU count | Confirmed and fixed in allocation and preflight wrappers. |
| Method C null diagnostic used different minibatch sampling | Not confirmed. Diagnostic `paired_probe_budget.py` at exploratory commit `18c76da7` and production `probe.py` both draw the original 1,000 indices from `idx_key`, then the next 1,000 from `fold_in(idx_key, 1)`. The diagnostic uses two compiled fitting stages; production uses one continuous scan. GPU numerical identity of those implementations is unverified. |
| Ordinary evaluations consume the SAC action RNG | Confirmed existing behavior: `sample_actions(training=False)` splits the agent key. Post-fork evaluations separately save/restore it as approved. No ordinary-evaluation RNG change was made because it would alter training trajectories. |
| `check1_control.npz` replacement was non-atomic | Confirmed and fixed with a flushed temporary file and atomic replace. |
| Orphaned transition guards can strand a row | Still possible by design: old guards are reported once and remain fail-closed for manual owner review; they are never removed on age alone. |
| Missing submission directories | Confirmed and fixed during queue preparation. |

All 195 canonical parent configurations resolve Method C: 2,000 steps, five
rounds, zero threshold, 95% confidence interval, and two scheduled firing
checks. Environment groups are 75 `dmc_hard`, 30 `dmc_medium`, 60 MyoSuite,
and 30 HumanoidBench. Amendment (aa) in the authoritative methodology and
the current settings table in `EXPERIMENT_STATUS.md` record Method C.

## CPU verification

- Complete source-bound driver: 684 tests across 73 modules, no failures,
  `source_unchanged: true`; one evicting-cache test skipped because this CPU
  environment lacks `filelock`. Receipts are under
  `/workspace/scratch/method-c-consolidated-full-cpu/` for reviewed executable
  source `b225577c2ba5a18b2ad6d14c47b34a00b5443e05`.
- Focused adversarial tests: 40/40, including real late-fork and pre-first-
  checkpoint restart, stale Slurm claims, four runtime groups, durability,
  atomic Check 1 publication, Method C continuation and historical capture.
- Mutation checks: 74/74; environment/seed restoration: 65/65; methodology
  CPU reference: 195 parent commands and 39 architecture/environment cells.
- Both revised Slurm wrappers pass `bash -n`. The final review commit adds
  documentation only; executable/configuration bytes match the tested source.

## Remaining gates

The owner must record the scientific disposition of the exploratory Method C
fresh-null sample (5/100 single checks, wide uncertainty) and Hopper-Hop
range evidence (three ratios above 0.90), without relabeling either result as
trained-critic sensitivity. The final source still needs six canonical CUDA
fork-identity cases, one live two-node shared-queue/GPU-binding preflight, and
review of checkpoint-flush overhead and suite-specific runtime/storage. The
Exp2 injected arms additionally need a natural eligible final-source fork,
source-matched positive control, owner-frozen injection magnitude, and their
own approval. CPU checks cannot certify CUDA determinism or the Delta mount.
The current parent approval template also requires positive control before
the 195-parent queue, so Exp1 cannot start first to create that natural fork
under the present gate. A final-source natural development fork or an
explicitly approved separation of the Exp1 and Exp2 approval gates is needed;
the latter has not been implemented here.

The shortest current path is to record the owner disposition of the existing
Method C null/range evidence and resource budget, run the six source-matched
CUDA identity cases concurrently within an authorized GPU cap, then run one
two-node Delta queue/binding preflight. A source-matched natural development
fork can proceed in parallel with identity work if separately authorized.
If it naturally qualifies, its positive control and owner-frozen `m` can then
unlock the current parent gate and, with a separate Exp2 approval, injected
arms. Packing above one process per GPU, repeated null/range diagnostics and
warm-cache repetitions are
optional unless a specific failed gate makes them necessary. CPU tests can
verify code, identities, manifests, recovery, and admission; they cannot
verify A100 placement, CUDA identity, the live shared mount or positive-control
behavior at full network width.

The guarded checkout, qualification, parent-plan and allocation command
sequences remain in `exp12_oct12_execution_handoff.md` and
`method_c_exp12_readiness.md`; replace their old source placeholder with the
final published consolidated SHA. These commands require genuine evidence
and approvals. No job was submitted during this review.
