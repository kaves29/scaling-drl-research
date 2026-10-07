# Exp1/Exp2 engineering follow-up after Block B 22706349

This report supplements `exp12_overnight_readiness.md`. It uses the newly
attached **authoritative consolidated report** `blockB_22706349.txt`, SHA256
`4f87ba9a4cada6be2eefb7a8b366200e5dce3a57edb167982136409238e258b2`.
Line references below refer to that attachment. Its source revision was
`b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`, on A100-SXM4-40GB GPUs and
EPYC 7763 CPUs. The attachment is evidence, not an instruction to change
methodology. Individual speed/profile JSONs were not supplied.

No Delta access, Slurm submission/cancellation, GPU execution, campaign launch,
methodology edits, or remote push occurred during this follow-up. Existing
published engineering work ends at `c7674bfb650f6903ed443e3f1c9647ddbbfe3fb0`.
This follow-up is intended as a separate local commit for review.

## 1. Root-cause verdict

**The six-hour development timeout and one-hour identity timeouts do not yet
have a measured dominant root cause.** The consolidated report supplies no
phase timings or last-step counters for those runs. Short production-size
preflights pass, and training-only rates are plausible, but neither proves
that later checks complete. Do not call the catastrophic runtime solved.

| Issue | Classification | Evidence | Root cause | Fix needed? | Methodological? |
|---|---|---|---|---|---|
| Development runtime >6 h | UNRESOLVED | Exit 124 after 21,600 s; no completed fork; lines 245–288 | Dominant phase unknown; log tail contains cache failures but no progress trace | New traced A100 evidence | No for measurement; changing work requires approval |
| Identity runtime >1 h | UNRESOLVED | Four cold and two warm dog runs time out at 3,600 s; warm Myo D4W1024 gets only 818 s; warm Myo D4W1536 is unrun; lines 72–83, 224–238, 1274–1337 | No completed parent/arm timing or comparison | Trace canonical existing gate | Exact requirement stays unchanged |
| Large intended identity workload | CONFIRMED CONTRIBUTOR | Existing command includes replay warm-up, fresh probe, two scheduled checks, fork/control reconstruction and evaluation | Identity is a pipeline run, not a short parameter comparison | Preserve work; expose phases | Reducing it would require approval |
| Duplicate checkpoint payload read | CONFIRMED CONTRIBUTOR | Previously reproduced two agent restores; published fix uses saved reference shape plus one payload read | Redundant restore, already fixed before this follow-up | Regression retained; A100 saving unmeasured | No state omitted |
| Persistent-cache corruption/recompilation overhead | CONFIRMED CONTRIBUTOR | Truncated zlib streams in many `jit_*` cache reads, lines 259–287 | Corrupt entries cause failed reads/recompilation; writer/topology cause and total cost unknown | Diagnostic uses fresh external single-writer caches; do not delete shared evidence/cache blindly | No precision or computation change |
| UTD accidentally multiplied/unrolled repeatedly | RULED OUT | Existing step/learning tests and source accounting: two SAC updates per learned interaction | No evidence of an extra production training loop | No speculative optimization | No |
| GPU kill/resume and comparison errors | CONFIRMED ROOT CAUSE | Ten error subtests across kill methods: `SingleDeviceSharding ... TFRT_CPU_0 ... not found`, lines 293–527, 542–642 | Tests create CPU subprocess checkpoints, then a GPU comparison process attempts device-specific Orbax restoration | Compare checkpoint values as NumPy arrays; training restore unchanged | No |
| GPU Angle1 parity error | CONFIRMED ROOT CAUSE | `GlobalHydra is already initialized`, lines 529–540 | Test singleton contamination | Current checkout already clears Hydra in parity `setUp`; rerun locally | No |
| GPU Check1 assertion failure | CONFIRMED ROOT CAUSE | Injected Q delta 2.980232238769531e-7 versus hardcoded zero, lines 644–647 | Test demands exact injected Q although amendment (m) and production Check1 allow 64 eps for injected Q and gradient | Lead approved alignment; test now asserts the existing scaled 64-eps rule | Explicitly approved by the lead; production threshold unchanged |
| Break checks 56/57 on GPU | CONFIRMED ROOT CAUSE | Diagnostics-update mutant and restored test both PASS; CUDA tolerance is `None`, lines 758–760 | Skip-only uncalibrated CUDA oracle cannot reject the mutation | Calibrate a discriminating oracle; do not copy report's heuristic thresholds automatically | Yes: acceptance thresholds need approval |
| Hopper range failure | UNRESOLVED | Configured-pool P/b: .933, .902, .602; D4W1536 fails existing >=.9 rule, lines 210–222 | Evidence establishes failure, not whether probe implementation or intended statistic causes it | Preserve gate; investigate with raw curves/range artifacts | Pool/probe/rule change requires approval |
| D4W1024 fresh null 13% | UNRESOLVED | 13/100 fresh pairs fire; D2 3/100 and D4W1536 5/100, lines 240–243 | Not explained by a proved implementation bug; does not establish campaign eligibility rate | Preserve result and existing trigger; examine raw pairs | Threshold/bootstrap/trigger change requires approval |
| HumanoidBench unavailable | CONFIRMED INDEPENDENT BUG | `HB_ENV` unset; speed/identity/HB suite unavailable | Dependency/install/renderer readiness missing | Separate approved environment validation | No scientific default inferred |
| Timed-out parent labeled no-trigger | CONFIRMED INDEPENDENT BUG | Report labels dependent arm `SKIPPED_NO_TRIGGER` after parent timeout | Finished sentinel formerly conflated incomplete and completed parent | New parent exit sidecar + DONE check | No trigger or inclusion rule change |
| Stale claim ownership race | CONFIRMED INDEPENDENT BUG | Deterministic local schedule reproduced two successful owners on old code; release could delete another owner | Stale deletion/recreation and release lacked shared transition exclusion | Atomic guard for all transitions + ownership-aware release | No seed, config or training change |

Ranked by measured strength, corrupt cache reads and intended full-pipeline
work are established runtime contributors, followed by the already-fixed
duplicate restore. Their relative wall-time shares remain unknown. The GPU
test root causes are separate from the long-run runtime cause. Neither
deterministic GPU flags nor a successful preflight resolve these distinctions.

The GPU default and deterministic suites each ran 124 methods: 115 passed,
1 failed, 4 errored, 4 skipped. Eleven error instances include ten subtest
errors in three methods plus the Hydra error. Deterministic flags did not
fix those errors. CUDA diagnostics comparisons also recorded deviations with
unset tolerances; those are measurements/skips, not passing numerical gates.

## 2. Changes made

| File | Change and scientific neutrality |
|---|---|
| `experiments/exp12/state.py` | `load_agent_tree` reads the pinned Orbax aggregate structure, then restores every array as NumPy for exact value comparison. Preserves dtype, shape, tree structure, empty nodes and values. Does not change SAC/trainer restoration or checkpoint contents. |
| `scripts/claim_launcher.py` | Shared atomic `claim_guard` serializes fresh creation, stale takeover and release. Release verifies PID, hostname and Slurm job identity. Prevents duplicate work/foreign deletion without changing jobs. An orphan guard fails closed and requires controlled manual recovery; all workers must use this revision. Multi-host Delta filesystem behavior remains unqualified. |
| `scripts/exp12_blockB.sh` | Records parent exit code atomically before its finished sentinel. Without a ready fork, only exit 0 plus DONE permits `SKIPPED_NO_TRIGGER`; incomplete parents get `BLOCKED_PARENT_INCOMPLETE`. Existing ready-fork execution stays intact. No timeouts/budgets/hooks/parameters change. |
| `scripts/sbatch_exp12_runtime_diagnostic.sh` | Retains manual-only, one-A100, fixed 300-second timeout. Adds approved architecture selection and separate canonical cold/warm identity modes using the existing Block B identity command. No automatic chaining, retries or submission; profile remains the default. |
| `tests/test_exp12_fork.py` | With explicit lead approval, replace injected-Q zero-delta assertion with the existing normalized 64-eps bound. Retain exact control/identity assertions and unchanged production acceptance. |
| `tests/test_exp12_runtime.py` | Adds real Orbax save/load regression simulating unavailable saved-device sharding. Old comparison errors; new comparison preserves values and dtypes. Existing state/continuation/trace regressions retained. |
| `tests/test_exp12_orchestration.py` | Eleven regressions cover stale contenders, two local processes, live/unpublished owners, foreign release, orphan guard, exceptions and incomplete/complete/ready-fork orchestration. Slurm liveness is mocked or absent; no scheduler access. |
| `tests/test_runtime_diagnostic_launcher.py` | Fourteen tests cover launch guards, fixed timeout, completion validation, approved architectures, cold/warm identity command/cache topology and actual probe-loop round/fit accounting with expensive fitting mocked. Does not constitute CUDA execution. |
| `docs/exp12_engineering_followup.md` | This evidence, validation, cost and proposed-diagnostic report. No generated raw results committed. |

The fixed scientific configuration is unchanged: 13 environments × 3 approved
architectures × 5 seeds; actor D1W128; UTD2; repeat2; approved raw budgets
500k/1M/2M; 5×1000-step probes, pool25600, batch256, 20 scheduled checks;
existing target, bootstrap, consecutive trigger, eligibility rule, Check1/2,
64-eps injection tolerance, precision policy, optimizer, evaluation, fork
horizon and inclusion rules. No `m` choice is made. Unset CUDA numerical
tolerances remain unset. No checkpoint component is removed.

## 3. Validation

The exact CPU commands and completed results follow. Full suite command,
from `main/`:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off \
WANDB_MODE=disabled MPLCONFIGDIR=/workspace/.cache/matplotlib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.venv/bin/python -m unittest discover -s tests -t . -p 'test_exp12_*.py' -v
```

- Full CPU suite: **201 methods: 199 passed, 2 HumanoidBench dependency skips**,
  exit0, 1527.522 s. No failures or errors.
- Final break harness (`python -m tests.exp12_break_checks`): **72/72 passed**, exit0.
- Deterministic-flag CPU subset: **17/17 passed**, 168.299 s. Runtime9,
  single-critic exact identity1, twin fork6, Angle1 parity1. This exercises the
  flag plumbing on CPU; it does not validate GPU deterministic behavior.
- Runtime subset on the new comparator: **9/9 passed**, 54.659 s.
- Post-approval Check1/fork end-to-end class: **13/13 passed**, 75.450 s.
  The full suite had already discovered tests before the single approved
  test-assertion edit; this rerun validates the final assertion. All production
  engineering fixes were present before full discovery started.
- Launcher/orchestration standalone: **25/25 passed**, 5.949 s (14+11).
- Missing-device regression before fix: 1 error; after fix: 1 pass.
- Earlier orchestration regression against old code: six failing cases and one error
  reproduced; all eleven final regressions now pass.
- Static validation: **195 parent commands, 39 approved environment/architecture configurations passed**.
- Shell syntax, diff whitespace, configuration/manifest composition,
  scientific-file invariance and manual-only launch inspection: checked
  before commit. Full tests include step accounting, probe and evaluation
  scheduling, exact identity, kill/resume and complete-state comparisons.

The earlier baseline was 187 pass + 2 dependency skips (189 total). Eleven
orchestration tests and one unavailable-device regression add twelve methods,
so final discovery contains 201 methods, with 199 passes and
the same two HumanoidBench skips. The separate launcher file is outside the
`test_exp12_*.py` pattern; never add its fourteen tests to the discovery count.

Known harmless warnings include Orbax aggregation/sharding deprecations and
Gymnasium float32 box-bound casts. No failing result is suppressed. The GPU
qualification of the aligned Check1 assertion and unset numerical tolerances
remain explicit red gates.

## 4. Remaining red/yellow gates

**Engineering:** long-run phase attribution and later-check completion;
A100 qualification of the new comparison path; canonical exact identity
completion; cache-health/recompilation attribution; multi-host claim-guard
behavior. A local CPU pass does not retire these gates.

**Methodological:** a useful CUDA diagnostics oracle and accepted thresholds;
D4W1536 hopper P/b failure; D4W1024 null firing; unselected positive-control
`m`; any subsequent probe/trigger/statistical change. Keep incomplete
confirmatory cells and n=1 treatment under existing rules.

**Infrastructure:** HumanoidBench dependencies and rendering; reviewed CUDA
environment and identical A100 model for the full grid; branch transfer to
the normal repository/Delta. No access or jobs are authorized by this report.

## 5. Runtime estimate

### Measured training component

Block B lines 172–193: probes off, 600 timed learned interaction steps,
100 extra warm-up learned steps after replay initialization, one job, four
cores. Rounded rates are usable for conditional estimates; they are not
long-horizon throughput distributions. All include environment/training
loop overhead in that window. Startup and initial random replay warm-up are
outside the timed window.

| Architecture | dog-run it/s | Myo-key-turn it/s | dog process peak GiB | dog GPU total GiB |
|---|---:|---:|---:|---:|
| D2W512 | 112.5 | 117.5 | .41 | 1.19 |
| D4W1024 | 65.8 | 66.1 | 1.34 | 2.45 |
| D4W1536 | 46.3 | 45.1 | 2.63 | 4.45 |

For raw budget B, N=B/2 interactions, replay warm-up W=5000. Learned
interactions L=N-W+1; SAC updates U=2L. Steady training hours L/(3600r):

| Architecture / proxy environment | 500k raw | 1M raw | 2M raw |
|---|---:|---:|---:|
| D2W512 / dog-run | .605 | 1.222 | 2.457 |
| D4W1024 / dog-run | 1.034 | 2.090 | 4.200 |
| D4W1536 / dog-run | 1.470 | 2.970 | 5.970 |
| D2W512 / Myo-key-turn | .579 | 1.170 | 2.352 |
| D4W1024 / Myo-key-turn | 1.030 | 2.080 | 4.181 |
| D4W1536 / Myo-key-turn | 1.509 | 3.049 | 6.128 |

500k Myo and 2M dog/Myo rows are budget extrapolations, not approved campaign
cells or additional measurements. Approved 2M cells are twin-critic
HumanoidBench; their training rate is **unmeasured**, so these rows cannot
be substituted as established HB costs.

### Separate remaining costs

Let c(a,e) be one full current/fresh paired check in seconds, F the initial
fresh-only probe, E the sum of actual episode/evaluation costs, S all required
save/restore/export costs, and C startup/compilation plus random replay
warm-up. Then nominal parent time is:

`T_parent = C + L/r + F + 20*c + E + S`.

These terms need disjoint phase attribution: nested trace timings are not
additive. Metrics/diagnostic work already included in the timed training
window must not be added again. F is not silently charged as a full paired
check; F≈c/2 is an explicit approximation only when fitting costs dominate.

| Component | Work required | Absolute A100 measurement currently available |
|---|---|---|
| Training | 245001/495001/995001 learned interactions; 490002/990002/1990002 SAC updates | Rounded dog/Myo rates above; HB unavailable |
| Probes | Per single-critic parent: 5000 fresh fits + 200000 periodic fit updates; HB doubles critic fits | Mixed null job 6042 s for 300 pairs; no per-architecture phase times |
| Evaluation | Nominal 75/100/150 episodes for 500k/1M/2M parents, including initial/final evaluations | No representative production-size episode timing in report's empty fork/eval section |
| Checkpoint/fork | 20 complete parent saves (a triggered fork save replaces its routine save) plus fresh reference; eligible fork adds exact reconstruction/restore and evaluations | No completed fork save/restore timings in report |
| Startup/compile | Backend/env/model setup, initial 4999 random interactions, compilation/cache reads | Preflight end-to-end 114/335/391 s; these combine work and cannot be used as isolated startup estimates |

6042/300 = 20.14 s is a **pooled null-pair end-to-end mean**, including setup,
compilation and reporting. It is not c(a,e). No architecture mean can exceed
6042/100=60.42 s in that particular completed mixed job, but this is not a
bound for production environments, later checks, HB or the campaign.

### Complete campaign accounting and conditional totals

195 nominal parents = 105M interactions, **208050390 SAC updates**,
**46125000 probe fit updates** (11.808B batch-example visits), 20250 nominal
evaluation episodes and 3900 complete parent saves (including the fork in
place of its routine save), plus fresh references. These are work counts,
not elapsed-hour predictions.

If all seven DMC environments share dog-run rates and all four Myo
environments share Myo-key-turn rates, non-HB steady training totals are:
**60.01 + 104.19 + 149.92 = 314.12 A100 hours**. Those assumptions have not
been validated across environments or on this corrected revision.

Ten HB parents per architecture contribute separately
`9950010/(3600*r_HB(a))` hours each. Thus baseline Exp1 GPU-hours are
`314.12 + sum(HB training) + sum(C + F + 20*c + E + S)/3600`.
There is no measured finite interval for the omitted HB/overhead terms.

Exp2 reuses the parent control trajectory; do not charge every control as a
whole new parent. For eligible scaled parents, f=kN/20, k=2..19;
H=.25N, control end Kc=max(N,f+H), injected continuation H. Additional
training is `H/r_injected + max(0,f+H-N)/r_control` per eligible parent,
plus construction/restore/Check1/2/probe/evaluation/save costs. Eligibility
rates, f distribution, injected-head `m` and injected throughput are unknown.
The development timeout supplies no estimate of any of them.

With all 130 scaled parents eligible at the latest permitted fork, extra
control ≤14M interactions and injected=17.5M interactions, adding 63M SAC
updates. Total parent+continuation work ≤271050390 SAC updates, 61875000
probe fit updates, 101850 evaluation episodes and 5200 complete
saves, including each parent fork in place of its routine save and excluding
fresh-reference/snapshot writes. This is a conditional work envelope,
not an hours bound: injected r may differ.

For cost sensitivity only, if HB has the same rate as single-critic dog-run,
injected/control rates equal parent rates, and every single-equivalent probe
pair costs the pooled 20.14 s, baseline training+probe proxy is **466.19 h**;
all eligible late forks increase this to **635.11 h**. Both omit startup,
evaluation, I/O and diagnostic/calibration runs. These are **illustrations,
not measured total estimates**. The HB-rate and pooled-probe assumptions
are especially unverified. Any packed-job rates require full-pipeline
measurement before estimating allocation-hours; training-only aggregate
packing speed is not an approved jobs-per-GPU choice.

The former ~600–700 h estimate is therefore not established or refuted.
635 h emerges under one explicit upper-work scenario *before* unmeasured
overheads, while 466 h corresponds to another. The useful new evidence is
the 314 h conditional non-HB training component and exact work accounting;
there is no honest measured complete-campaign total yet. No design is changed
to fit a preferred compute target.

### Exact missing evidence and its purpose

Paths are relative to the supplied historical
`logs/blockB_22706349/` directory. They are requests for existing evidence,
not permission to access Delta or infer absent measurements.

- `gpu2/packing/{D2W512,D4W1024,D4W1536}_x1/job_0.json`, and each existing
  packed configuration's `job_<i>.json`: unrounded rates, exact timed window,
  overlap, affinity, backend/precision and per-process peak. Corresponding
  `gpu_mem_mib.txt` gives sampled aggregate memory, not memory uncertainty.
- `gpu3/speed/myo-key-turn_<arch>/job_0.json`: unrounded per-suite rate and
  window metadata. No HB speed artifact is implied by an unavailable step.
- `gpu3/null_dog_run/null_pairs_<arch>.jsonl` and
  `null_summary_<arch>.json`: pair-level bootstrap results and noise
  distribution, not unrecorded per-probe phase durations.
- `gpu2/range_hopper_hop/range_<arch>.json`, `range_verdict.json` and any
  existing probe-curve artifacts: round-level scores/losses and exact range
  rule inputs; curves are not assumed to have been saved.
- Complete `gpu0/dev_run.log`, `gpu0/dev_run/` latest-state metadata and
  available probe/check records: last completed step/check/update and whether
  checkpoint/probe progress preceded timeout. Historical trace timestamps
  cannot be reconstructed if they were never recorded.
- Each `gpu2/identity_{cold,warm}_<arch>_<env>.log` and its parent/arm
  checkpoint metadata: which process/stage completed. The report confirms
  no identity comparison output, so no successful identity JSON is assumed.
- Existing per-architecture phase profile JSONs, **if any were generated**:
  full-check cost, save/restore, episode cost and compilation/startup timing.
  Block B's report lists none; there is no justified exact filename for
  an artifact not evidenced to exist. The proposed diagnostic produces new
  `profile.json` and traces instead.

## 6. Proposed Delta diagnostic — prepared, unsubmitted

Exact script: **`main/scripts/sbatch_exp12_runtime_diagnostic.sh`**.
One A100, four CPUs, 32 GiB; seven-minute allocation wrapper; **one fixed
300-second diagnostic plus at most 15 seconds termination grace**. Queuing
time is separate. A timeout is an incomplete diagnostic, not a scientific
failure/no-trigger and not permission to increase the timeout.

After separately authorizing branch transfer and a manual job, activate the
reviewed CUDA environment and execute this from the reviewed `main/` checkout:

```bash
OUT=/work/hdd/biqc/skaveti1/exp12/validation/runtime_diag_REVIEWED_HASH_D4W1536_profile
EXPECTED_COMMIT=$(git rev-parse HEAD)
EXPECTED_GPU_MODEL=NVIDIA\ A100-SXM4-40GB
DIAGNOSTIC_MODE=profile
DIAGNOSTIC_ARCH=D4W1536
export OUT EXPECTED_COMMIT EXPECTED_GPU_MODEL DIAGNOSTIC_MODE DIAGNOSTIC_ARCH
sbatch --export=ALL --output="${OUT}.slurm.out" --error="${OUT}.slurm.err" \
  scripts/sbatch_exp12_runtime_diagnostic.sh
```

`REVIEWED_HASH` is a label to replace with the reviewed commit. The output's
parent must already exist on persistent storage; OUT itself must be new and
outside the repository. The guard requires branch `claude/eloquent-fermat-inxqlt`,
exact EXPECTED_COMMIT, a clean tracked/untracked tree, pinned Python/JAX,
one visible GPU and its exact literal A100 model. These commands are **not
executed in Cloud**. No repository automation submits this script.

First run only D4W1536 profile. It measures training throughput, actual
environment/update counters, two full paired probe checks (one warm-up),
synthetic replay-size complete save/restore, two ten-episode evaluations
(one warm-up), process GPU peak if exposed, and compile/cache phase traces.
It has 6061 total interactions, 2124 SAC updates, 1062 `update_many` calls,
10 probe rounds and 20 critic fits. The synthetic 475000-transition replay
is an I/O benchmark; it is not a scientifically trained state. Its “fresh”
comparison is the same currently trained parameter reference used by the
existing profiler, not Exp1's saved pretraining reference. This mode does
not prove later production checks or exact full-pipeline identity.

Output: `commit.txt`, `backend.json`, `profile.log`, `profile.json`,
`trace.jsonl`, `validation.json`, `exit_status.txt`, external Slurm stdout/
stderr, and external temporary/cache material. Trace records include
monotonic timestamps, current interaction/raw step, update count, check
indices, per-stage counts/times, cache hits/misses/path and backend compile
calls/time. Timings are synchronized; serialization changes elapsed time
but not update/probe semantics. These are diagnostic timings, not an
uninstrumented production throughput promise. GPU memory availability is
reported explicitly; missing peak is an unresolved measurement, not zero.

Only after inspecting that output, separately authorized manual invocations
can select D4W1024 and D2W512 in profile mode with unique OUTs. No mode loops
over architectures or launches successors. Rates should be compared with
Block B's sizes/window definitions; there is no invented speed threshold.

For the unresolved real identity stage, the **same script** separately offers
`DIAGNOSTIC_MODE=identity_cold` or `identity_warm`, architecture D4W1024 or
D4W1536. Replace only mode/architecture and use a new OUT in the command.
It reproduces the existing dev-only dog-run identity setup: seed101,
raw budget240000, forced check2, 1000-interaction post-fork snapshot,
stop-after-snapshot, then separate identity-arm CLI and the unchanged exact
comparator. Expected parent fork12000, both snapshots13000; parent16002
SAC updates, arm2000 additional updates; parent check0 plus checks1/2,
25000 single-critic probe fits. It must progress beyond check1 and reconstruct
the actual control/identity state. Cold uses separate new parent/arm caches;
warm sequentially reuses the parent's completed cache. It does not submit
both modes automatically or select injection `m`.

Identity output: common metadata/status/log plus `parent_trace.jsonl`,
`identity_trace.jsonl`, `parent/`, `identity/`, `results/`, `identity.json`
and `validation.json` on completion. Real complete-state comparison remains
bit-exact for control/identity, including required records and panel. The
five-minute cap may expire before that full workload completes; then the
last begin/end/progress records localize it. It cannot claim a solved
runtime or a passed identity. Identity mode does not independently collect
process peak GPU memory; use the profile measurement for that limited scope.

### Criteria and interpretation

- Operational pass: correct backend/source, exit0, required finite-positive
  profile measurements and exact work counts, final trace summary without
  errors. Identity additionally requires comparator pass, no differences,
  fork12000 and both snapshots13000. No tolerance is weakened.
- Operational fail/incomplete: exception, wrong device/revision/config,
  missing/nonfinite output, wrong counts, exact identity difference, or
  timeout. Distinguish timeout from a completed identity mismatch.
- Cache health: review logs for corrupt/truncated cache reads and trace for
  recompilation after shape warm-up. Any corrupt read leaves this gate red
  even if the script's completion validator returns0. First compilation
  and intentional actor/critic shape changes are expected; module/signature
  evidence is required before calling later compiles accidental. Separate
  cold/warm traces expose startup differences only if both complete.
- Speed, peak-memory and full-campaign hour acceptance thresholds are
  **not selected**. A finite timing is measurement completion, not readiness.
  No job packing or production launch is authorized by a profile pass.
- Scientific range/null/Check2, CUDA diagnostics numerical thresholds and
  `m` remain outside this diagnostic's invented acceptance rules. Existing
  gates remain in force; modifications require the lead's approval.

## 7. Decision request

1. **Check1 test alignment — approved and implemented:** the lead explicitly
   approved replacing only the injected-Q zero-delta assertion with the
   existing scaled 64-eps bound. Production Check1, its threshold and exact
   control/identity requirements are unchanged. A100 rerun remains required.
2. **CUDA diagnostics oracle:** approve a separate calibration plan before
   choosing thresholds. The report's suggestion “10× largest deviation” is
   not adopted: long-run deviations 1.75/2.05 and a deliberate mutant passing
   show why a blind large threshold can conceal a defect. Prefer repeated
   same-program controls, precision-matched graph comparisons and mutation
   sensitivity, then present candidate bounds for approval. Exact identity
   stays exact. Current unset thresholds remain red.
3. Hopper P/b=.602 and D4W1024 null=.13 still fail their existing rules.
   Preserve both. Changing the pool, target, bootstrap, threshold, rounds or
   tolerance would change sensitivity/eligibility and requires a separate
   methodological decision supported by the raw evidence above.

No additional methodological modification is implemented. The remaining
calibration/range/null decisions stay pending; they do not prevent committing
the authorized engineering work.

## 8. Git state

Branch: `claude/eloquent-fermat-inxqlt`. New changes are committed locally
after final validation and diff review. The exact commit/working-tree and
ahead/behind state are reported with the final response; this document avoids
a self-referential commit hash. No follow-up push is authorized or performed.
Remote branch remains `origin/claude/eloquent-fermat-inxqlt` at the previously
published `c7674bfb650f6903ed443e3f1c9647ddbbfe3fb0` unless independently changed.
No merge or force-push. Raw attachment, logs, caches, profiling output and
dependency/environment artifacts stay outside the committed tree.
