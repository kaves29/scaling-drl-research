# Independent audit of A100 job 22765261

The archived runtime diagnostic passes its intended **profile-mode** completion and transfer-bookkeeping checks. D4W1536 `identity_warm` is **not ready for recommendation under the unchanged 300-second internal timeout**: the recorded costs predict approximately 329 seconds for only training, two warm probe checks and two warm post-fork evaluations. Startup, the initial fresh probe and state I/O are additional. This is an empirical planning estimate, not a measured identity run or a mathematical lower bound. No budget or scientific rule was changed.

## Evidence integrity and accessible files

Original ZIP: `/workspace/attachments/bdf8bff2-f922-4c6d-aaf5-7172dcef3064/runtime_22765261.zip` (347,377 bytes).

Checksum: `/workspace/attachments/9104f34f-3983-4f32-862e-398808d5c189/runtime_22765261.zip.sha256` (87 bytes). The earlier checksum attachment has identical content. The SHA-256 matches exactly:

`ceea3feb7d1b123d6ea68bee9351eae3f890b32bcd58ee1d26bd2ba82e78c720`

There are 36 unique archive members; all pass ZIP CRC verification. All 34 manifest payload entries and 35 internal checksum entries verify, with no unlisted payload. All 24 bundled source/configuration files match immutable Git objects at `9d6a82d4aef81299901177865e383640439cde2a`. Bundled source is revision context; it is not an independently captured resolved runtime configuration.

The complete extraction is `/workspace/scratch/a100-audit/job22765261/`. Every path below is beneath its `artifacts/` directory:

| File | Bytes | Result |
|---|---:|---|
| `trace.jsonl` | 2,729,924 | Complete; 13,727 valid records |
| `profile.json` | 936 | One finite profile row |
| `validation.json` | 158 | Intended endpoint recorded |
| `backend.json` | 329 | A100, CPU/CUDA, JAX/JAXLIB 0.4.34 |
| `exit_status.txt` | 2 | `0` |
| `commit.txt` | 41 | Exact approved revision |
| `profile.log` | 4,251 | Complete results and nonfatal warnings |
| `trace.stacks.log` | 10,736 | Four periodic watchdog snapshots |
| `slurm.stdout` | 0 | Present; empty |
| `slurm.stderr` | 0 | Present; empty |

The empty Slurm files are consistent with the launcher's redirection into `profile.log`; they are not corrupt. The archive does not contain an independent `sacct` accounting record. Slurm COMPLETED/4m42s remains user-supplied evidence; archive exit status and program completion are independently confirmed.

## Transfer and endpoint

Every trace record was read, with duplicate JSON keys and nonfinite/overflowing JSON numbers rejected. Both wall and monotonic timestamps are nondecreasing. All 6,829 begin/end intervals match in nesting order; there are no unmatched begins, orphan ends, crossing intervals or recorded errors. Suppressed-begin events and synchronization phases are handled explicitly.

The populated step-6000 flush spans trace lines **10,926–12,931**, at offsets **119.928–121.116 seconds** from wrapper start:

| Measurement | Observed |
|---|---:|
| Pending groups | 1,001 |
| Array leaves | 19,019 |
| Array bytes | 152,152 |
| Completed grouped `device_get` calls | 1,001 |
| Leaves per group | 19 |
| Total transfer interval time | 0.888641 s |
| Longest transfer | 0.204618 s |
| Replay interval | 0.076509 s |
| Entire flush interval | **1.188707 s** |

All transfers finish before the single replay begins. Transfer leaf/byte totals match the enclosing pending payload; replay and flush both finish successfully. Structural metrics, logging/reset and subsequent training then proceed. This verifies bounded transfer completion and bookkeeping. Actual GPU metric values were not archived, so their individual numeric values cannot be retrospectively compared; existing CPU regression coverage checks values, ordering, shapes, dtypes and logger behavior.

The other flushes are also internally consistent: interactions 2000 and 4000 have zero pending groups; interaction 6061 transfers the remaining 61 groups (1,159 leaves, 9,272 bytes) before checkpoint saving. Each has exactly one replay, following all its transfers. All 1,062 update batches are therefore accounted for across the completed flushes. The final 61-group flush is outside the timed 60-interaction window, reinforcing the limitation of the warm training-only projection. Supplemental evidence is `all-metric-flushes.json`.

Final progress is **interaction 6061 / SAC update 2124**. Source geometry independently agrees: sampling begins at interaction 5000, giving `(6061 - 4999) * 2 = 2124` updates, or 1,062 `update_many` calls. The command warms through 6001 and times 60 subsequent interactions. Trace totals also confirm 10 probe rounds, 20 fits, two trainer constructions, one save/restore and two post-fork evaluations. All profiling stages finish, the last trace record is an error-free summary, profile/log JSON results agree, validation agrees, and diagnostic exit status is zero.

The trace wrapper spans **263.199819 seconds**. This excludes the launcher's preceding backend-metadata subprocess and shell/Slurm overhead. It is consistent with, but does not independently establish, the reported 282-second Slurm elapsed time.

## Runtime and numerical interpretation

Confirmed costs include 16.638 s for imports, 12.429 s for initial trainer construction, 8.989 s for start/initial evaluation, 84.040 s for training through 6001, and approximately 1.414 s for the timed 60-interaction segment. Structural metric calls take 15.465 s, 0.144 s and 1.567 s. Probe rounds total 78.492 s; their 20 fits total 73.005 s, with one approximately 5.647 s cold fit and subsequent fits near 3.545 s. Save and restore take 6.074 s and 2.512 s. The two ten-episode evaluations take 23.371 s and 22.834 s.

There are 305 backend compilations (29.858 s), 305 cold cache misses and 305 cache writes (15.724 s). The main SAC scan compiles three times across 1,062 update calls; the probe fit compiles once across 20 fit calls. This does not show a compile on every update/fit. Most remaining compiles are primitive programs and startup/structural work. The observed compilation and cache I/O costs are substantial but bounded.

The trace reports 62,650 synchronization calls totaling 18.204 s. This includes real device waiting and overlaps timed calls; it is not a pure instrumentation-overhead measurement. Inclusive stage costs must not be added to their child costs. There is no evidence of another unresolved metric-transfer stall in this run.

The four `Timeout (0:01:00)!` stack entries are the wrapper's repeating, nonterminating watchdog dumps. Later records close the sampled operations and complete the program. They are not four diagnostic failures. Orbax aggregate-save deprecations and the restore-sharding warning are nonfatal; the latter indicates extra restore metadata work.

All archived JSON numbers are finite, and no CUDA/OOM/exception is recorded. However, raw parameter, gradient, Q, per-metric and probe-score arrays were not packaged; the profile launcher does not comprehensively check these for numerical validity. Their absence prevents a blanket claim that training had no hidden numerical problem. The profile does not execute the real identity continuation, injected-arm qualification, fresh-null calibration or hopper range qualification.

## Credibility of projections

The rate is **42.441110 interaction steps/s**, measured over only 60 warm interactions. `500000 / rate / 3600 = 3.272508 h` exactly reproduces the training-only projection. UTD is two, so this is not 42.44 SAC updates/s. The populated step-6000 transfer occurs before the timed interval.

The 3.27-hour figure is a plausible warm training-cost estimate, not a full-parent runtime guarantee or confidence interval. It omits initial compilation, recurring metric/structural boundaries, probes, normal evaluations, checkpoint I/O and storage variability. The source's 21-times-warm-check estimate produces 0.208952 h of probes; the initial fresh-only check has different work from a regular current/fresh check. There is only one timed full-check repeat. Profile-mode's alleged `fresh` parameter tree is the already-trained current tree (`scripts/profile_exp12.py:113`), so these are fitting-cost measurements, not scientific fresh-versus-trained evidence. Production fresh-reference capture is unchanged.

The synthetic 475,000-transition save/restore tests I/O and restoration startup, not an exact resumed training trajectory. The warm ten-episode evaluation is 22.834 s; episodes and policy costs in a full run need not be identical. Peak device allocator use is **3,763,942,656 bytes = 3.763943 GB = 3.505445 GiB**. It does not establish host-memory capacity, full-run peak memory or safe multi-job packing. The reported seven-jobs heuristic is not qualified by this audit.

## D4W1536 identity-warm feasibility

The existing launcher uses one A100, four CPUs, 32 GiB host memory, seven-minute Slurm time and an internal `timeout -k 15 300`. `identity_warm` creates a fresh parent cache and shares it serially with the identity process; it does not inherit job 22765261's cache.

The unchanged existing fixture uses seed 101, dev role, nominal 120,000 interactions, forced trigger at check 2, first scheduled check 6000, fork 12000, and both snapshots 13000. It needs 8,001 parent `update_many` calls plus 1,000 identity calls: **18,002 SAC updates**. Initial fresh capture plus two regular checks require **25 full 1000-step fits**, with the unchanged 25,600-point pool, five rounds and optimizer. Initial evaluation plus the two post-fork evaluations total 25 episodes. No additional scheduled post-fork check/evaluation occurs before 13000.

| Partial planning cost | Seconds |
|---|---:|
| 9,001 update-producing interactions at recorded warm rate | 212.082 |
| Two regular checks at recorded warm full-check cost | 71.641 |
| Two ten-episode post-fork evaluations at recorded warm cost | 45.668 |
| **Subtotal, excluding additional work** | **329.391** |

The additional fresh-only check needs five fits, approximately 17.7 s at the observed warm per-fit cost, plus target work. Imports, backend initialization, random warmup, additional constructions, Check 1 compilation/panels, state writes/restores, periodic diagnostics and CPU comparison also remain. Cache reuse can reduce compilation in the second process, but cannot remove the required training, fitting or evaluation work. To fit just this subtotal into 300 s would require approximately 49.27 update-producing interactions/s, before allowing time for omitted stages; measured warm throughput is 42.44. Neither 329 s nor 49.27 is a new acceptance threshold.

**Recommendation: hold D4W1536 identity_warm.** Its successful completion under 300 s is not established and is unrealistic at the observed costs. A different validation staging plan or time budget needs the user's approval; neither was selected or implemented. Changing only the Slurm limit would not remove the existing 300-second internal cap. No exact larger budget is inferred from this profile.

## Engineering correction and qualification criteria

The approved amendment (m), retained by (v), requires control/identity panel Q and dQ/da to match bit for bit; the injected arm retains its separately approved 64-eps magnitude-scaled rule. Sources are `.claude/methodology-exp1-exp2.md:330` and `:436`. The existing `np.array_equal` comparator and zero-difference subtraction did not enforce the bit requirement: value equality accepts signed-zero differences and matching values with differing dtypes.

The isolated branch `codex/a100-exact-identity-audit`, based directly on 9d6a82d4, adds array shape/dtype/byte comparison, enforces exact numeric metadata types/representations, retains identically represented scalar diagnostic-NaN sentinels, rejects differing NaN payloads, and preserves the previous rejection of NaN arrays and identity probe/evaluation-record NaNs. Panel key sets are checked symmetrically. Check 1 adds a bit guard only to control/identity comparisons; injected comparisons retain the existing tolerance arithmetic. Regression tests cover dtype, byte order, signed zero, metadata, NaN payloads, layout, missing fields, both Check 1 modes and the 64/65-eps boundary. Mutation anchors were updated to mutate the complete corrected guard rather than an obsolete line; the scientific mutation criteria are unchanged.

No SAC updates, initialization, network definitions, probes, bootstrap rules, metric definitions, precision configuration, experiment YAML, launcher, timeout or resource declaration were changed. A stricter validation result on previously missed bit/type differences is the intended enforcement correction, not a newly selected scientific criterion.

An additional CPU-confirmed instrumentation bug would prevent the existing identity launcher from passing even without a timeout. `Exp1` deliberately raises/catches `ForkNow` to restore the control, and both arms deliberately raise/catch `ValidationDone` at their snapshots. The old trace wrapper labeled these normal transitions as `error`, while the launcher rejects any such trace record. The independent reproduction in `control-flow-before-fix.json` confirms both false errors despite successful outer summaries. This bug was not exercised by job 22765261's profile-only command.

The relevant source chain is `experiments/exp1.py:179`, `:186`, `:202` and `:206`, `experiments/exp12/fork.py:35`, `experiments/exp12/runtime_trace.py:327`, and the launcher's existing error rejection at `scripts/sbatch_exp12_runtime_diagnostic.sh:134`. Its immutable internal timeout is at `:69`; Slurm resource declarations are at the top of that unchanged file.

The isolated correction annotates these two exact exception classes as `control_flow`, re-raises the original exception unchanged for its existing owner, and retains genuine errors. An uncaught transition still fails the outer summary; a progress/synchronization failure remains an error. Four CPU regressions cover both normal signals, uncaught signals, progress failures and an actual traced tiny parent/identity continuation with the corrected byte comparator. All four passed, including matching snapshots and the unchanged launcher's all-errors-null trace rule. Production training, synchronization and callback scheduling are unchanged. Profile-mode train-exit records gain no new field when no transition occurs.

For a future approved identity run, PASS requires clean exact source revision, the reviewed A100 and JAX versions with CPU/CUDA available, exit zero within the existing internal cap, both complete error-free traces, fork at 12000 and snapshots at 13000, `identity.json` pass true with no differences, finite full-FP32 Check 1 panels and bit-identical control/identity Q and dQ/da, and exact nonignored saved-state components under the corrected comparator. Expected caught `ForkNow`/`ValidationDone` transitions remain explicit annotations, not errors. No new epsilon is selected. Missing snapshots, nonzero exit, missing/error/unfinished trace stages, differing dtype/bits or failed Check 1 are failures of that diagnostic. Exact identity does not qualify the injected arm, the other architecture/environment suites, or scientific plasticity gates.

Expected outputs are `commit.txt`, `backend.json`, `exit_status.txt`, `profile.log`, `parent_trace.jsonl`, `identity_trace.jsonl`, both stack logs, `identity.json`, `validation.json`, external Slurm stdout/stderr, parent/identity saved-state directories, fork panels, probe NPZs, run metadata and dev results. Identity mode intentionally produces neither `profile.json` nor `trace.jsonl`. The unchanged packager's profile-centric missing-required list must not be used as the identity completeness gate; retain the actual mode-specific evidence and checkpoint trees. Publication and any GPU submission remain unapproved.

Machine-readable audit: `artifact-audit.json`. CPU receipts and final branch/commit details are retained with the audit evidence. Reusable CPU startup guidance was saved in the cloud setup draft; it was not published. No Delta connection or GPU execution occurred.

An independent AST comparison against 9d6a82d4 confirms all four production modules are identical outside the approved comparison helpers, Check 1, the comparison command, the trace installation wrapper and their required imports. The scientific documents, SAC/trainer implementations, probes, target generation, optimizers, architecture/configuration files and launcher are unchanged. The existing original local branch and published integrated candidate are preserved.


## CPU validation environment and review scope

CPU validation uses Python 3.12.14 and CPU-only JAX/JAXLIB 0.4.34, NumPy 1.26.4, Flax 0.8.4, Optax 0.2.3 and Orbax 0.5.3. Delta's recorded Python is 3.12.13. Full local dependency versions are retained in `cpu-environment.txt`; ancillary package versions are not asserted identical to Delta. HumanoidBench is installed from the repository's pinned `cb1189039151c8aadaaa987b442da54383c87fab` revision using the project's compatibility adapter. CPU simulator validation is not a complete upstream dependency qualification or a CUDA test.

The final regression scope includes every `test_exp12*.py`, every `test_runtime*.py`, `test_metric_transfer` and `test_run_metadata`: 33 modules. Fresh receipts capture the base revision, changed paths, dependency versions, module logs and executable-source fingerprint. The source tree is frozen throughout the run. The independent comparator negative controls demonstrate three intended regressions fail when the old value-equality comparator is reinstated and pass when the correction is restored.

The only repository changes are:

- `experiments/exp12/state.py`: exact array and metadata comparison.
- `experiments/exp12/fork.py`: enforce existing bit-exact Check 1 cases.
- `scripts/compare_identity_fork.py`: apply exact state/panel/record comparisons.
- `experiments/exp12/runtime_trace.py`: classify expected fork/stop transitions.
- `tests/test_exp12_bitwise_comparison.py`: eight representation/comparison regressions.
- `tests/test_runtime_control_flow.py`: four transition/error/real-continuation regressions.
- `tests/exp12_break_checks.py`: update two existing mutation anchors.
- This report.

Generated evidence, raw diagnostic artifacts, environments, dependency caches and profiling output are outside the repository. The complete relevant regression suite passed **363 tests across 33 modules, zero failures, zero errors and zero skips**, including all twelve new regressions. This is the current integrated suite with the pinned HumanoidBench CPU paths available, rather than the older 187-test/dependency-skip snapshot. All module receipts and logs are in `final-verified-cpu/`. Its executable-source fingerprint remained unchanged throughout:

`2093c7e892349e8aff9178f3daeb9294aad31f804518667da21abdd2cd58ec8d`

The three independent old-comparator negative controls also pass their defect-detection checks. **73/73 mutation checks passed**: each injected defect was detected and the restored implementation passed. The current integrated harness has 73 existing mutations; no scientific mutation criterion was added or relaxed. **65/65 simulator checks passed** across all thirteen environments and five seeds, including both HumanoidBench tasks, with no unavailable tasks. These are CPU construction and next-transition/RNG restoration checks, not production training or GPU qualification. All three final validation commands exited zero. Black checks for the new tests, Pyflakes for all affected Python files, and `git diff --check` passed.


The local reviewed correction is one commit directly above 9d6a82d4 on the isolated audit branch. It is not pushed or merged. A verified incremental Git bundle, final source/test receipts, and held detached-checkout/submission commands are delivered separately. **The remaining approval-dependent decision is validation staging or diagnostic time budget; D4W1536 identity_warm is held under the unchanged limits.** Fresh-null, hopper range and injected-arm GPU qualification remain separate scientific gates. No threshold, continuation length, fitting budget or resource allocation was selected or changed.
