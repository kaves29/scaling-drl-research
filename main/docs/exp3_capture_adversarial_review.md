# Exp3 capture: independent adversarial review

Review date: 2026-10-10. Owner-authorized engineering review only; no merge, Delta
access, GPU execution, source-cell selection, or scientific protocol change.

## Recommendation and provenance

**Do not integrate Claude's capture implementation unchanged.** C1's causal
binding is correct, but C3 remains incomplete at the reviewed revision: publishing
arrivals before publishing their checkpoint can make a crashed source impossible
to resume. The isolated correction is a CPU-tested integration candidate, not GPU
or scientific qualification. Review the correction before merging; capture on
production sources additionally requires the GPU checks and owner decisions below.

| Reference | Exact revision |
|---|---|
| Original Codex Exp3 implementation | `142a7fc92ec8d19ff58e6022c6c60af84bc9ed4d` |
| Reviewed Claude implementation | `5764bd09d13c8c22fd6c0a81ead237f2f2ed2fdf` |
| Claude C1 review | `ba57ab4613e9d78f9ac4ada93bf85480a324ae02` |
| Claude capture/C3/harness changes | `9dec0b844ef93df40fb9ceaa595424a9cc74cb25` |
| Exp1/Exp2 production baseline | `844e3c0cc24998b25c3d1e667bcbfe056b1b336c` |
| Isolated correction branch | `codex/exp3-capture-fixes`, based directly on the reviewed Claude head |
| Engineering correction commit | `b74797535f9fedadb677aadbce53f7bb9b91b41d` |

During review, the live Claude branch advanced to
`0b70602bbb5313a25879a526642cf9d5cb933e31`, through
`16db4951b6298c0e2632f783bbbdb0d371f3cabb`. The additional diff is documentation
only (`exp3_methodology_proposed.md`, `exp3_owner_decisions.md`, and
`exp3_pilot_review.md`); no implementation changed, so the confirmed code findings
still apply. These later proposals are not incorporated or adopted by this branch.
Their population/specification caveats should be reconciled with the recovered
original request during handoff. There is no production-file overlap with these
two newer commits.

The original owner request has been recovered verbatim in
[exp3_original_user_request.md](exp3_original_user_request.md), SHA-256
`3a444073cbbeae02e810256121ad218115060f17f32e019b276432d55e10ab10`.
It is historical authority, not authorization to execute experiments now. Claude's
S9 statement that this text was unavailable is superseded by this recovery. The
request specifies three Pilot 1 actors, pi_U, pi_I, and pi_F; that point does not
require choosing a new primary actor. Numerical protocol choices remain unfrozen.
The request establishes design constraints, not a complete frozen scientific
protocol with hypotheses, estimands, statistics and acceptance thresholds.

Exp1/Exp2 authority remains `.claude/methodology-exp1-exp2.md` and
`docs/exp12_decisions.md`. Exp3 investigation recommendations do not override it.
All 337 files of the Exp1/Exp2 baseline remain byte-identical, and all 195 resolved
parent configurations match. The review modifies only Exp3 instrumentation,
adapters, tests, isolated reference scripts, and review documentation. There is no
production-loop, optimizer, RNG-generation, precision, configuration, threshold,
budget, or tolerance modification. Opt-in instrumentation changes I/O cost;
trajectory parity is tested separately and is not inferred from unchanged code.

## Confirmed findings and corrections

| Finding | Evidence at Claude head | Correction / remaining qualification |
|---|---|---|
| C1: Stage B injection mismatch | `runner.make_passive` rejects wrong source arm, m, and seed; adversarial tests exercise wrong m/seed | Guard preserved. Stream provenance carries the actual injection. Stage A's lead-frozen m and seed convention still require the approved protocol inputs. |
| C3: stream ahead of source LATEST | `recording.py` flushes before the original save; a save failure leaves committed arrivals without their corresponding restorable source state | Stage chunks and a publication journal, call the unchanged source save, then publish one manifest at the matching cursor. Recovery accepts only the old or new checkpoint cursor. |
| Partial interval publication | Original `StreamWriter.flush` publishes each chunk separately | One manifest publishes the entire interval. Unique chunk names preserve orphan evidence after failures rather than replacing referenced bytes. |
| Unsaved tail accepted | Scope sealing can flush pending arrivals without a matching routine checkpoint | Recorder refuses this tail; only existing routine saves publish source arrivals. Generic standalone writer sealing remains supported. |
| Identical retention falsely conflicts | Stored modes `retained`/`omitted` were compared to arguments `retain`/`omit` | Correct mode comparison; differing bytes still create a conflict marker and adapters still refuse conflicted snapshots. |
| Restart launch history breaks binding | Raw metadata digest includes an appended `launches` record | Bind all immutable metadata with only `launches` excluded; retain original raw hashes and archived metadata for provenance. |
| Comparison is not bitwise | Harness converts to float64 before equality; repeat comparison accepts dtype/signed-zero differences | Compare original shapes, dtypes, values and bytes before calculating descriptive float64 errors. Repeated results also require dtype/byte equality, preserving the existing identical-NaN sentinel policy. |
| Captured key silently narrowed | Writer casts extras to uint32 | Require the source key already be uint32 with shape `(2,)`; reject altered dtype instead of truncating it. |
| Completed source cannot reopen to seal | Death after DONE but before stream sealing leaves an incomplete stream; normal source entrypoint refuses DONE | Add artifact-only `exp3_finalize_capture.py`; verify immutable source binding, DONE/LATEST endpoint, arm/injection, journal, checksums and cursor. No trainer or update is executed. |
| Concurrent capture writers | Process-local patches do not prevent another process opening the same stream | Persistent adjacent lock inode, nonblocking `flock`; never unlink it. Shared-filesystem semantics still need qualification. |
| Retention can be pruned with routine saves | A retention root under mutable `state/` is accepted | Reject roots inside the source state or its mutable LATEST root. |
| Natural fork time is unknown before launch | Only a numeric absolute retention endpoint can be supplied | Add optional explicit `--retain-until-arm-end`, deriving the window from the existing actual fork plan. No default or fork timing is changed. |

Five independent adversarial tests all failed on the untouched Claude revision:
false retention conflict, save-before-publication failure, partial multi-chunk
publication, dtype/signed-zero comparison, and captured-key narrowing. Existing
Claude tests nevertheless passed 64/64. Passing that suite was therefore not
evidence that C3 was fully resolved.

The journal provides recoverability across two publications, not a single atomic
transaction spanning checkpoint and stream. Before LATEST commits, staged chunks
are not arrivals. After it commits, recovery publishes the matching interval. A
different cursor, altered provenance, corrupt staged bytes, or missing tail is
rejected. Orphan chunks remain evidence and consume space. CPU tests cover injected
failures around these boundaries; they do not prove durability under power loss,
Lustre/NFS locking, or a killed Orbax process. Old streams lacking the new immutable
bindings cannot automatically be resumed: never invent keys or provenance.

Retention copies or hard-links the source's existing committed bytes; it makes no
extra source saves. A death after LATEST publication but before retention is
repaired from that LATEST on restart/finalization. Already lost older snapshots
cannot be reconstructed by this mechanism. U and I require separate retention
roots; sharing a step-named root correctly produces conflicts rather than matched
snapshots.

## Matched data and restart correctness

| Pilot | Required historical artifacts | Verified on CPU / limitation |
|---|---|---|
| 1 | Complete common fork; U/I agent states at equal steps, including actual I-at-fork; common panel and matched noise/actor/Adam/alpha | Real entrypoints retain I-at-fork and post-fork matched lean states; Pilot 1 runs on both. Original request specifies U/I/F actors. Numerical protocols remain explicit inputs. |
| 2 Stage A | Fork replay/state and ordered U arrivals, update counts, normalization, replay/sample/noise bookkeeping | Captured uint32 keys are the source key after action sampling and before updates. Giving those keys to the CPU fidelity diagnostic reproduces active U bitwise. The primary passive runner does not silently adopt this diagnostic protocol. |
| 2 optional Stage B | Same prerequisites plus ordered I stream and matching actual injection provenance | C1 enforces m/seed/arm; wrong provenance fails. Independent passive keys remain matched between passive learners, but do not by themselves reproduce active SAC's action-key consumption. |
| 3 optional | Matched retained agent states, complete common fork replay, frozen actors/evaluator/alpha, matched batches and noise | New end-to-end test runs Pilot 3 using real retained lean U/I states. Passive and target-training restart tests compare resumed and uninterrupted results. Lean snapshots deliberately reject replay use. |

Recording parity now compares the complete saved extra state, including probes,
fresh-critic state, fork plan and evaluation history, as well as agent, replay,
normalization, environment/RNG, counters and saved state. Backend tests check
floating parameters and optimizer moments after restore and inspect the saved
Orbax sharding metadata. They currently establish CPU placement/FP32 only; when
run with the harness on an actual A100 they require GPU save/restore placement.

This proves executable reconstruction on small CPU fixtures, not qualification of
every production cell or a full-width pilot. No fixture's forced fork or reduced
budget is adopted for production.

## Which parent cells must be planned before production?

The original request names dog-run, humanoid-walk, five seeds, and "both existing
MyoSuite environments." The repository now has four Myo tasks and two possible
scaled architectures. Neither the pair nor the source architecture is selected.
An exact approved source-cell list therefore cannot honestly be supplied yet.

The unresolved preservation envelope is:

```text
{D4W1024, D4W1536}
  x {dog-run, humanoid-walk, myo-key-turn, myo-pen-twirl,
     myo-pose-hard, myo-reach}
  x {1, 2, 3, 4, 5} = 60 potentially required parents
```

This follows the existing `base_exp12.yaml` fork architecture list (D4W1024 and
D4W1536 only) and `generate_manifest.EXP12_SEEDS`; it does not add D2 forks. The
seed-102 development run is outside this five-seed population unless the owner
separately designates it; engineering capture qualification using it would not
automatically make it an Exp3 scientific source.

Selecting two Myo tasks and both widths would leave 40 cells; selecting those two
tasks and one width would leave 20. These counts are alternatives, not agent
decisions. Hold potentially required cells, or explicitly authorize their opt-in
capture after engineering qualification, until the owner freezes the scope.

For a required eligible parent, U recording must begin at its actual natural fork
before its control continuation. Preserve the common fork and matched routine U
saves. Later I-arm execution must preserve its post-injection fork save, matched
post-fork saves, and I arrivals if Stage B is required. No natural eligible fork
means no actual matched case; report that absence rather than manufacture a fork.

The other 135 parent cells (all 65 D2W512 cells, plus scaled cells in the seven
other tasks) have no scientifically necessary Exp3 data identified under the
current pilot request. They can run without Exp3 capture for this scope, subject
to their own Exp1/Exp2 gates. This is not a promise about future research questions.

Final replay is not an ordered trajectory. Running a selected parent unrecorded
can irreversibly lose matched intermediate U states and arrivals; later retained
I states cannot repair that loss. A separately authorized continuation from the
fork would be new evidence, not proof of the historical trajectory's identity.

## Storage, RAM and overhead

The isolated reference scripts inventory actual environment shapes and recording
dtypes, then use `jax.eval_shape` for parameter counts. They neither train nor
select source cells or m. For twelve states (common fork + five U offsets + I-at-
fork + five I offsets), an ordinary single-critic FP32 agent requires
`4 * (4 * critic_parameters + 3 * actor_parameters + 3)` dense bytes. An injected
agent additionally contains two head copies in each of online/target critics:
**`16 * head_parameters` extra bytes per I state**. The optimizer's trainable
parameter count does not gain those frozen copies. Six I states amplify this cost.

| Dog-run cell, fork replay + two 125k-arrival streams + 12 agent states | m=last | m=half | m=all |
|---|---:|---:|---:|
| D4W1024, decimal GB | 10.089 | 10.895 | 12.507 |
| D4W1536, decimal GB | 19.178 | 20.991 | 24.617 |

For 20 cells (five seeds, dog/humanoid and any two Myo tasks), D4W1024 estimates
range from 172.4–222.7 GB and D4W1536 from 354.0–464.7 GB across the candidate m
values and Myo pairs. Both widths require the sum. These are payload estimates,
not a measured quota requirement: exclude pilot outputs, snapshot metadata,
Orbax encoding effects, orphan evidence, and any extra U tail. U's actual source
continues to `control_end_step`, which can exceed `arm_end_step`; recording does
not truncate it. Retaining replay in every post-fork snapshot costs substantially
more. Replay omission remains an explicit storage option, not an adopted protocol.

| Task | Dense arrival bytes | One 125k-arrival NPZ stream, GB | Conservative Python pending memory at 25k arrivals, MB |
|---|---:|---:|---:|
| dog-run | 7,330 | 0.917 | 239.4 |
| humanoid-walk | 2,270 | 0.284 | 112.9 |
| myo-key-turn | 1,686 | 0.211 | 98.3 |
| myo-pen-twirl | 1,526 | 0.191 | 94.3 |
| myo-pose-hard | 1,926 | 0.241 | 104.3 |
| myo-reach | 2,038 | 0.255 | 107.1 |

The Python estimate counts per-arrival objects and arrays and conservatively
recounts shared keys; it is not measured RSS. Stacking, chunk arrays, JAX buffers,
replay, and transient checkpoint serialization add memory. Pending arrivals are
bounded by the source save interval, not by the chosen chunk size. Stream scalar
dtypes are preserved, including DMC float64 observations/statistics.

Pilot 1 stores four gradient families per actor/state. For dog-run's 170,700 actor
parameters that alone is roughly 2.73 MB/state/actor, or 8.19 GB for 1,000 states
and three actors, before critics, activations and autodiff temporaries. Chunking
reduces peak working memory; it does not remove total output storage. Passive
learner memory also depends on the injected head, not just two ordinary agents.

CPU append/flush microbenchmarks are available in the receipts, but are not a GPU
recording-overhead estimate. Hashing retained checkpoints and validating a staged
stream prefix during each publication add cumulative CPU/I/O work. Claude's small
fixed-seconds overhead claim is unqualified. Measure GPU iterations/s, save/restore
latency, peak host/device memory, I/O and bytes with recording on/off before launch.

## Scientific issues left unchanged

1. Select Myo pair and source width(s); freeze m through the existing Exp2 decision
   process and bind the intended injection seed. This review makes no selection.
2. Freeze complete Pilot 1–3 protocol inputs, including panel/offsets, chunk size,
   normalization/alpha sources, intervention definitions, and budgets. Templates
   remain `approved: false` with null settings; existing scientific implementations
   and templates were not rewritten.
3. Decide Exp3 measurement precision after sensitivity evidence. Production SAC
   precision remains unchanged; CPU equality is not TF32 qualification.
4. Pilot 1's chunk-dependent noise draws require a frozen chunk size. Replacing
   them with per-state keys changes realized draws and is **not** silently treated
   as methodology-neutral noise bookkeeping.
5. Pilot 3's latent-based log density agrees with production on unsaturated draws
   but avoids the saturated inverse-tanh round trip. Mathematical equivalence does
   not establish identical finite-precision behavior at saturation. Owner approval
   is needed to resolve the intended numerical target; this review changes neither.
6. Confirm Pilot 3's replay/actor/evaluator choices and original Check 1 panel
   requirements. Lean snapshots support the currently implemented common-fork
   replay design; an arm-specific replay decision would require full retention.
7. Approve capture window/replay/storage and GPU allocation. The optional derived
   arm-end window is an interface, not adoption of a retention policy.

## Minimum remaining GPU evidence and commands

No jobs were submitted. Use one already authorized A100 at a time. Two logical
checks are necessary; they may share an allocation only if its approved budget
actually permits completion:

**A. Actual-GPU engineering exactness and sensitivity.** Run the existing harness
at the exact corrected/reviewed revision, using explicitly owner-provided precision,
architecture, task and synthetic-state count. It includes real entrypoint capture,
complete-state parity, retained lean Pilot 1/3, passive/target restart, journal
boundary tests, key fidelity and saved/restored backend checks. Record separate
oracle/TF32 measurements; these are measurements, not automatically accepted
numerical tolerances.

```bash
# In a fresh clean detached checkout of the final reviewed SHA, after the pinned
# environment is activated, on an already authorized allocated GPU. No sbatch here.
set -euo pipefail
: "${EXPECTED_COMMIT:?full reviewed SHA}"
: "${OUT:?absolute new evidence path outside checkout}"
: "${OWNER_APPROVED_PRECISIONS:?space-separated harness precisions}"
: "${OWNER_APPROVED_SENSITIVITY_ENV:?}"
: "${OWNER_APPROVED_SENSITIVITY_ARCH:?}"
: "${OWNER_APPROVED_SYNTHETIC_STATES:?}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false
unset JAX_PLATFORM_NAME
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
read -r -a precisions <<< "$OWNER_APPROVED_PRECISIONS"
mkdir -p "$(dirname "$OUT")"
cd main
if python -u scripts/exp3_gpu_validation.py \
  --expected-commit "$EXPECTED_COMMIT" --out "$OUT" \
  --exactness-precisions "${precisions[@]}" \
  --sensitivity-env "$OWNER_APPROVED_SENSITIVITY_ENV" \
  --sensitivity-arch "$OWNER_APPROVED_SENSITIVITY_ARCH" \
  --sensitivity-states "$OWNER_APPROVED_SYNTHETIC_STATES" \
  > "${OUT}.launcher.log" 2>&1; then
  status=0
else
  status=$?
fi
printf '%s\n' "$status" > "${OUT}.exit_status.txt"
test "$status" -eq 0
```

Require pinned JAX/JAXLIB 0.4.34, one actual A100, no CPU fallback, complete logs,
nonzero test count, zero failures/errors/skips, and `EXACTNESS_PASS`. A repeated
computation mismatch is `NONDETERMINISTIC`, not proof of a resume defect. Missing
results/timeout are `INCOMPLETE`; preserve evidence and stop, do not expand the
budget automatically. A pass does not approve oracle differences or precision.
Outputs include `backend.json`, `measurements.json`, `validation.json`,
`validation.partial.json`, per-precision suite logs, and adjacent launcher log and
exit-status file; hash/archive the whole evidence directory and source/environment
receipt. Inspect the report, not just process exit status.

**B. Full-width capture/resource qualification.** Use an owner-selected eligible
source/fork at the intended scaled width, preferably a compatible checkpoint from
the separately authorized Exp1/Exp2 gates. Compare two unrecorded references first,
then recorded continuation at unchanged SAC precision; check all saved state,
keys, counters and normalization bitwise. Exercise interruption during source save
and between LATEST/manifest/retention publication, recover, and compare to the
uninterrupted reference. Use actual Delta storage for lock/rename checks. Reuse
matched retained snapshots for lean Pilot 1/3 and passive fidelity where possible.
Measure memory, storage and timing against the source's existing resource budget.

The capture entrypoint is `scripts/exp3_record_source.py --experiment exp1|exp2_arm
--args <ordinary-authorized-run-args.json> --stream <new-external-path>
--chunk-size <approved-size> --authorized-source-run --benchmark`, with explicit
`--retain-root`, one of `--retain-until-step`/`--retain-until-arm-end`, and
`--retain-replay retain|omit` if retention is authorized. For restart, keep the
same capture/retention inputs and add `--resume-stream`; ordinary source resume
arguments still apply. For a verified DONE source whose only missing operation is
sealing, use `JAX_PLATFORMS=cpu python scripts/exp3_finalize_capture.py --stream
<stream> --run <source-run> --expected-commit <exact-captured-source-SHA>` instead
of rerunning training.

Acceptance is exact source-state parity and matched cursor/checksums/backend plus
completion within the **already approved** allocation. No new overhead percentage,
tolerance, source cell, fitting horizon or timeout is selected here. A parity
failure requires investigation; nondeterministic references preclude attribution.
Resource exhaustion or timeout requires owner review of allocation or staging,
not a scientific-parameter reduction.

**A 300-second cap is not established as sufficient.** The untouched CPU Exp3 suite
took 631 seconds and an intermediate corrected suite took 742 seconds. These are
not GPU timing predictions, but they make a guaranteed 300-second whole-harness
claim indefensible. Full-width Jacobians and checkpoint I/O add unmeasured costs.
Start with owner-approved staging/allocation; do not reuse an Exp1 runtime
diagnostic's allocation as implied authorization for this different workload.

## Validation receipts and handoff

Generated logs/JSON stay outside Git under `/workspace/scratch/exp3-review-*`.
The final receipt below records counts and SHA-256 hashes, not generated evidence.
CPU environment: Python 3.12.14, JAX/JAXLIB 0.4.34, Flax 0.8.4, Optax 0.2.3,
Orbax 0.5.3, NumPy 1.26.4; CPU backend and EGL software rendering. This differs
from Delta's reported Python 3.12.13 and is not a claim of GPU validation.

Final Exp3 coverage is **80/80 passing, zero skips**: 22 capture/boundary tests
(221.327 seconds) and 58 remaining review/pilot/stream/harness tests (636.701
seconds), covering the final engineering code bytes. The earlier untouched suite
passed 64/64 (630.864 seconds); the independent pre-fix boundary set failed all
five tests, with no import errors. Intermediate 74-test and 28-test runs also
passed, but are not substituted for final coverage. Black checks the twelve
formatted correction files, AST parsing passes for all fifteen changed Python
files, and `git diff --check` passes.

The final focused Exp1/Exp2 regression run also passes **50/50, zero skips**
(1,226.679 seconds), covering bitwise comparison, real fork/kill/resume entrypoints,
injection and all parent manifest checks. Environment restoration passes 65/65.
All **74/74 mutation checks** fail with their defect injected and pass again after
restoration; the runner exits zero. No scientific rule was changed to obtain this
result.

Reproduction from `main/`, using the CPU environment above:

```bash
python -B -m unittest -v tests.test_exp3_capture_boundaries tests.test_exp3_capture
python -B -m unittest -v tests.test_exp3_review tests.test_exp3_pilots \
  tests.test_exp3_streams tests.test_exp3_gpu_harness
python -B -m unittest -v tests.test_exp12_bitwise_comparison tests.test_exp12_fork \
  tests.test_exp12_injection tests.test_exp12_manifest
python -B -u -m tests.exp12_break_checks
python -B scripts/check_exp12_cpu_environments.py
python -B scripts/check_exp3_isolation.py \
  --base 844e3c0cc24998b25c3d1e667bcbfe056b1b336c --out <new-external-report.json>
python -B scripts/exp3_capture_storage_reference.py --out <new-task-inventory.json>
python -B scripts/exp3_checkpoint_storage_reference.py \
  --task-inventory <task-inventory.json> --out <new-size-report.json>
```

| Evidence basename under `/workspace/scratch/` | Result | SHA-256 |
|---|---|---|
| `exp3-review-baseline-tests.log` | Original Claude 64/64 pass | `470a800804dbf9e02fdab08a01481904f791c50876d562dc28eeafe9870ecbb8` |
| `exp3-review-adversarial-before.log` | 5/5 intentional pre-fix failures | `cb622a30cfa4c3c1a4574adc14505fc51ee7f39ec72c2fd59467ef8f7f1f3ddc` |
| `exp3-review-last-capture-tests.log` | Final 22/22 pass | `49dd0a6294900838ea4f07149af51892bd99a11d2f71d22f8a0088fbf9d73d02` |
| `exp3-review-final-complement.log` | Final 58/58 pass | `40d69d463699184f033a8fa80f208c8960ec6c57c2773031d93f7dc8f1969c36` |
| `exp3-review-exp12-regressions.log` | Final 50/50 focused Exp1/Exp2 regressions pass | `263fdd043c63b88f0431992cc327ee123a0a0f2e7c933447ab720167b44fe206` |
| `exp3-review-mutation-final.log` | 74/74 mutated failures and restored passes; exit 0 | `7c85782383b3c962c1c56b33d4735902a0d1f0a768917dea4320c00195b76fe6` |
| `exp3-review-environments-final.log` | 65/65 exact environment/seed restoration checks | `6dbcec9cfa9c2c1dd0b1bfa2752cfd6ee5e2c0f044d7b5a00cb49b60b1563db4` |
| `exp3-review-isolation-final.json` | 337 unchanged baseline files; 195/195 configs | `c8177ceac3d1889d25e5c05248e089b1bf4a2b7516a97e6d5793b9fa601838c7` |
| `exp3-review-storage.json` | Actual six-task shape/dtype inventory and CPU microbenchmarks | `a638952ed8a4dc78283305238671db21c45848a2f9a012c7edbb48d04ec15a09` |
| `exp3-review-checkpoint-size-script.json` | Shape-only estimates, all candidate widths/m | `28222209afa9d1625846a0f24c6e00977cfe4e37161cf0a184a7a7f28783cbef` |

Ownership: Codex's isolated review branch only. Claude's branch, the original Exp3
branch, integration/exp12, and the user's active Exp1/Exp2 launch-readiness branch
are not edited or merged. Next: independent review of these engineering changes,
owner scientific/scope/storage decisions, then explicitly authorized GPU capture
qualification. Do not run selected production parents unrecorded while expecting
later recovery of their historical Exp3 data.

One capture-interface limitation remains explicit: if the ordinary source
finishes without executing a fork continuation, the recorder raises
`source did not execute a training continuation`. The source's DONE/checkpoint
can still represent a valid completed Exp1 parent, but that wrapper exit is not
evidence of a failed training run or a usable Exp3 source. Unattended orchestration
must distinguish this no-source outcome from source failure; no trigger is forced
and no missing stream is accepted. This review does not redefine Exp1 eligibility.
