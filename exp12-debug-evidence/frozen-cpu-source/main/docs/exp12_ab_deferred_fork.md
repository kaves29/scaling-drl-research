# Amendment (ab): deferred fork compatibility and separate launch gates

Source base: `957be5c63a48b0b90dd28ee4b361e3c6d373ab05`. The exact final
qualification source is the reviewed tip of `codex/exp12-deferred-fork-ab`.

## Compatibility verdict

**PASS on CPU for the existing control-plus-deferred-injected-arm design.** At
the natural trigger, `exp1.py` saves the complete trainer state under the
parent's separate `fork/state` root, then restores that state to continue the
Exp1 observational trajectory as the Exp2 control. The injected arm's
`exp2_arm.py` entry point later restores the same immutable fork checkpoint,
then performs Check 1, injection and post-fork continuation. A separate
later-launched control job is not part of the approved implementation; the
parent process is the control. Both arms' interaction and evaluation schedules
derive from the saved `fork_step`, not wall-clock submission time.

The saved state contains online/target networks and optimizer, replay plus
metadata, observation normalization, global/JAX RNG, environment restore
state and counters. Routine checkpoint cleanup is scoped to `parent/state`;
the fork lives under `parent/fork/state` and is retained. The parent can
recover from the fork if killed before its first control save, and later from
its control checkpoint. The injected arm can recover from its own checkpoint.
Fail-closed natural eligibility and source checks run again at delayed arm
discovery. A delayed approval requires a **new** SHA-bound allocation
approval; an allocation begun with Exp2 disabled cannot gain arms by editing
its active approval file.

Focused CPU tests run the complete tiny Exp1 parent before the identity and
injected arms, check retained fork files, compare complete control/identity
state including optimizer, replay, RNG, normalization and environment state,
and check source-step/evaluation alignment. Crash/restart tests cover fork
write, early control, late control and injected arm. Scheduler tests cover
idempotent delayed discovery and the separate Exp2 positive-control approval.
These tests establish workflow mechanics; they do not qualify A100 identity
or a naturally firing full-width parent.

## GO/NO-GO conditions

| Experiment | GO requires | Currently missing |
|---|---|---|
| Exp1 | Approved Method C and recorded fresh-null/range dispositions; six final-source scaled CUDA identity certificates; reviewed CUDA numerical bounds; suite runtime/storage and 195-cell plan; live shared-mount/GPU-binding preflight; owner-approved parent and allocation resources | Final-source GPU certificates, preflight, recorded scientific dispositions, reviewed resource/precision evidence and explicit approval |
| Exp2 | Naturally eligible same-source fork; source-bound positive control and owner-frozen m; Check 1/2 and identity requirements; complete-state intervention restore; separate Exp2 allocation approval | Natural positive control, m and separate approval; Exp1 GPU gates also apply to the shared source |

Exp1 may be approved while Exp2 remains blocked. No failed null/range result
is relabeled as passed, and no arm is forced or launched by this amendment.

## Minimum final-source GPU qualification for Exp1

Use the published final commit and a clean, pinned Delta checkout. Keep the
existing six **cold** identity cases, one for each scaled architecture and
suite, as required by the unchanged parent gate. They can run concurrently
within an approved eight-GPU limit. Each requests one A100, 16 CPUs, 64 GiB
and a four-hour *cap* (at most 24 A100-hours across six cases). Follow with
the single existing two-node, eight-A100, ten-minute shared-mount/binding
preflight (at most 1.33 A100-hours), without overlap under the eight-GPU cap.
Each case must produce an exit-zero `certificate.json` with PASS, exact source
SHA, full state identity and correct suite/architecture coverage. The
preflight needs an exit-zero PASS verdict and exactly one atomic claim winner
on the real shared mount. Any failure, timeout or wrong SHA stops approval;
inspect retained logs before deciding on a retry. Historical PASS evidence
remains historical and cannot be relabeled for the new source.

The ready-to-copy guarded Slurm commands and artifact layout are in
[`exp12_oct12_execution_handoff.md`](exp12_oct12_execution_handoff.md),
sections “Six independent cold identity cases” and “shared-mount/binding
preflight”. Set `FINAL_SHA` to the final published commit before executing
them, verify `git rev-parse HEAD` and a clean tree, and obtain explicit job
authorization first. The natural positive control is **not** in this Exp1
GPU sequence; it remains mandatory before Exp2 injected arms.
