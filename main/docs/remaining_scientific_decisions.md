# Remaining scientific decisions and readiness

2026-10-07. Baseline/pending diagnostic revision: `781dc6c93022a82b68a3210617db93225284b268`. Isolated implementation branch: `codex/methodology-reconstruction`. **The195-run campaign is not scientifically ready.** No queued A100 result is available here; none is inferred. No Delta access, CUDA calibration, submission or push occurred.

The engineering tree can be reviewed for a future A100 qualification after the completed CPU checks below. CPU evidence is not GPU qualification, and the existing diagnostic wrapper pins the original `claude/eloquent-fermat-inxqlt` checkout. This branch must not be substituted into the pending job or represented as the same source revision. Any future qualification of this corrected revision requires a separately reviewed launch/revision choice.

## Corrections and approval boundary

Implemented: literal twin per-network float32 subtraction then mean; finite/canonical/natural-trigger PC qualification; pre/post injection labeling; one approved range gate/exit behavior; structural/dtype/integer/nonfinite CUDA comparison guards with explicit diagnostic NaN masks; confined runtime/offline checkpoint pointers/replay header checks; duplicate/provenance-aware null resumption; non-overwriting fresh-directory manifests. No configuration, architecture, learning rate, weight decay, training budget, probe setting, trigger threshold, CI/resampling rule, evaluation schedule or primary population was changed.

**Explicit scientific approval received:** the lead answered “Approve the literal twin loss correction” to a question disclosing the1.49e-8 float32 difference and possible near-zero HumanoidBench-trigger effect. This approval covers literal longitudinal L and its validation/ancillary L reporting. Check2's mean-score paired difference remains unchanged. No other scientific change is implemented or assumed approved.

## Established Block B evidence

Authoritative attachment: `blockB_22706349.txt`, collected from `/work/hdd/biqc/skaveti1/exp12/main/logs/blockB_22706349/paste_me.txt`, supplied by the lead. SHA-256 `4f87ba9a4cada6be2eefb7a8b366200e5dce3a57edb167982136409238e258b2`. It identifies source `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`; do not relabel it as781dc6c or this branch. Raw JSON/JSONL artifacts were not attached.

### D4W1024 fresh-pair null

Report:100 pairs,13/100 per-check firings, IQM of pair-level L `7.418e-06`, SD of pair-level L `.0007396`, p95 `.001096`. Other sizes: D2W5123/100, D4W15365/100. Approved rule: excess over5% at any size requires lead consultation (D:683–715,1746); threshold remains zero. **The gate failed.** The p95 is a reported candidate, not an adopted threshold.

Source-level evidence: null mode independently initializes two same-architecture critics for each pair, shares replay/base target/minibatch draws, summarizes paired L, then applies the real five-round IQM percentile interval and strict lower-bound>0 rule. Null uses a single critic on dog-run. Single-critic fitting/trigger equations did not change between the BlockB source and781dc6c; this branch's twin-order correction therefore cannot explain that DMC failure. The stale-resume defect could contaminate evidence if output directories mixed identities/settings, but the supplied report does not prove that occurred.

Supported hypotheses, not established causes:

- Five-round percentile-IQM intervals can have unstable/undercovering tails. Each pair's fixed random initializations persist across its five rounds; conditional expected fit quality need not be equal even though neither critic is trained/degraded. This may turn initialization heterogeneity into an apparent within-pair plasticity difference.
- Finite100-pair sampling contributes variability, but is not by itself a basis for accepting13%. Under the **illustrative IID binomial assumption** p=.05, `Pr(X≥13 | n=100)=.001464348`. That assumption is not validated for the actual calibration and supplies no new hypothesis test or gate.
- A deterministic CPU toy with256 IID zero-mean Gaussian five-round loss vectors, unchanged10000-resample95% IQM interval and zero threshold, fired12/256 (4.6875%). It shows that null checks can fire and that the finite-sample interval behavior warrants examination; it does **not** reproduce13%, fit critics, prove the root cause or establish CUDA behavior. Reproducer uses `default_rng(20261007).normal(size=(256,5))` and `bootstrap_interval(x,990,k+1,10000,.95)`; no rule was changed.

Missing evidence: `gpu3/null_dog_run/null_pairs_D4W1024.jsonl` would permit exact per-pair round/CI/fired recomputation, checking nonfinite inputs, duplicates, asymmetry, seed/index identity and within-pair spread. `gpu3/null_dog_run/null_summary_D4W1024.json` would supply exact unrounded summary values/counts. The corresponding D2W512 and D4W1536 files provide architectural controls. Job source/config/runtime/launch metadata and null logs establish whether outputs were resumed/mixed. Existing legacy rows may lack full provenance; the new guard cannot retroactively certify them. Do not infer within-round SD or run-level consecutive-trigger rates from the reported across-pair SD; do not square.13 as a production false-fork estimate.

**Decision required:** disposition of the failed null gate; any adoption of `.001096`/other calibration, additional confirmation design or methodological revision. No threshold selected here.

### D4W1536 hopper-hop fresh-critic range

Report configured pool25600:

| Architecture | IQM(P) | IQM(b) | P/b | Final-loss IQM | Approved verdict |
|---|---:|---:|---:|---:|---|
| D2W512 | .4664 | .4998 | .933 | .0336 | PASS |
| D4W1024 | .4486 | .4976 | .902 | .0508 | PASS |
| D4W1536 | .3008 | .4994 | .602 | .2 | FAIL |

D4W1536 smaller pools:1600 ratio.987/final loss.00649;6400 ratio.845/final loss.0779. M(w) requires configured-pool ratio≥.9 at every size. **The gate failed.** Old standalone range utility would have accepted.602 under its superseded10–90% rule; BlockB's separate approved-rule report correctly failed it. Correcting the standalone verdict does not change any measured score or make this failure disappear.

Established computation: same fresh-network fitting path, P=b−full-poolMSE and ratioIQM(P)/IQM(b). Target network/own-mean offset and all1000 fitting steps are shared definitions with training probes. The smaller-pool result is not an approved substitute for25600. No source-level defect was established in this computation from the available evidence.

Plausible hypotheses: fixed1000-step generalization/fitting behavior over a larger replay pool; task-specific input/normalization or optimizer conditioning; TF32/sine-target sensitivity. Smaller pools also change evaluation chunk via gcd to320/1280 versus configured2560, potentially changing GPU kernel/target arithmetic, so the table cannot isolate pool size alone as the cause. It supplies no learning-curve slope or convergence evidence.

Missing evidence: `gpu2/range_hopper_hop/range_D4W1536.json` supplies exact five-round P,b,final-loss values and pool identities; corresponding D2W512/D4W1024 JSONs and `range_verdict.json` provide controls/acceptance provenance. Config/runtime/source and range logs identify the actual launch settings. **Per-step range learning curves are not saved by this utility**: they cannot be requested as if existing JSONs contain them. A future planned diagnostic would be needed to obtain them; no such CUDA run was started.

**Decision required:** response to the failed configured-pool criterion. Amendment(w) has no approved automatic next-step ladder; the old smaller-pool/more-steps fallback was withdrawn. Pool/steps/architecture/precision/acceptance remain unchanged.

## Other scientific decisions

| Priority | Question / evidence | Approval needed before change |
|---|---|---|
| 1 | Entropy conflict: M says+A/2; approved G1 and released SimBa use−A/2 with alpha(H−h*) | Explicit choice/amendment of entropy objective/target semantics. Scalar CPU gradients at H=0,A=2 are+.0099999998 for current−1 target and−.0099999998 for+1 target; this is scientific, not a harmless sign rename |
| 2 | Fresh-null13% and hopper.602 mandatory gates | Decide scientific disposition/additional evidence; no calibration/fallback adopted |
| 3 | Positive-control “success”: ranking/noise rule lacks a required amount/sign/significance of rescue | State whether ranking alone suffices or specify an approved success criterion. Example L_degraded=.8 and constant candidateL=.9/1/1.1 yields zero noise and selects last despite all recoveries negative. Preserve existing arithmetic until approval |
| 4 | Freeze injection.m from a genuine unshrunk natural-trigger PC | Lead reviews last/half/all recovery/noise evidence and freezes m. No qualifying result/m is available here; forced/shrunken diagnostic evidence cannot qualify |
| 5 | Four diagnostics on/off CUDA numerical bounds remain None | Lead approves bounds after valid GPU measurements. The existing10×largest-deviation/one-significant-digit proposal is explicitly **not approved** and is not applied |
| 6 | Exactly one eligible seed in an environment, including Check2-success-only subset | Approved uncertainty/reporting convention needed. Current confirmatory plotting raises an explicit decision-required error rather than silently emitting a degenerate interval or dropping a primary seed |
| 7 | Production GPU model, deterministic-only or warm-cache-only success, packing/runtime/storage plan | Review measured evidence and choose deployment constraints. No automatic wall-time/memory/packing threshold selected; dependencies/identity across suites still required |

No positive-control improvement criterion is added merely to obtain better outcomes, and no failed Check2 fork is excluded from the primary comparison.

## Engineering work still useful without changing methodology

- Obtain/review exact raw BlockB artifacts and source/config/runtime provenance; recompute approved equations and identify whether legacy outputs were mixed. This does not authorize cluster access in this task.
- Validate the pending781dc6c diagnostic when its report is supplied; inspect compile/restore/probe/I/O phase measurements without extrapolating an unmeasured speed or pass threshold.
- Review a source-aware scheduling preflight for DONE/FORK_READY markers and explicit large-payload checkpoint checks. Lightweight scheduling markers can suppress reruns but cannot pass the full analysis certificate. Such guards should preserve files and avoid automatically declaring invalid parents legitimate no-trigger results.
- Keep snapshots/input fingerprints and explicit full-population certification. Missing HB dependencies remain a coverage gap; synthetic twins/CPU mocks are not simulator validation.
- Review revision-specific qualification of this branch after CPU verification. The pending diagnostic and its exact original source remain preserved.

## Prioritized blockers before195-run launch

1. Resolve entropy-source conflict and decide disposition of both failed scientific gates.
2. Obtain genuine successful full-setting PC evidence, decide any extra success semantics, and freeze m through lead approval.
3. Complete CUDA correctness/identity qualification for both scaled sizes in each suite, including actual HumanoidBench twin environments; resolve GPU-only failures and approve diagnostic numerical bounds. Pending runtime evidence alone cannot satisfy these gates.
4. Approve one production GPU model and evidence-supported cache/determinism/packing/wall-time/storage plan; pass size-specific production preflights.
5. Freeze corrected approved source/config/runtime study manifest, verify195 parent commands and eligible-arm provenance, and use strict complete-result reporting. Preserve the old revision's diagnostic provenance.

## Completed CPU verification

This section is populated from completed commands before the local commit. Runtime logs, caches, temporary fixtures and retrieved reference files stay outside the committed tree. The CPU reference script is reproducible repository source and prints no production results.

| Completed check | Result |
|---|---|
| Exp12 full CPU discovery | **223 methods:221 passed,2 HB dependency skips**, no failures/errors; exit0;1605.847s |
| All other repository CPU tests | **276/276 passed**, no failures/errors/skips; exit0;777.794s |
| Final expanded break harness | **73/73 passed**, exit0; all72 original mutations retained, plus literal twin FP32-order mutation |
| Final certification + changed fixture/input regressions | **44/44 passed**, exit0;163.128s |
| Final focused reporting/PC/range/CUDA-helper/persistence/twin/manifest tests | **30/30 passed**, exit0;6.751s |
| Initial existing PC/report/manifest group |49/49 passed;121.819s (before the last added assertions) |
| Independent CPU references / configuration census |195 unique approved parent configs;39 architecture/environment combinations;130 possible scaled forks; exact paired-IQM percentile reference; scalar temperature gradient signs verified |
| Diff/syntax/preservation |Git diff whitespace and modified shell syntax pass; configs, methodology source, SAC implementation and pending diagnostic/profiling/trace scripts byte-unchanged |

Full discovery loaded code before the final offline-pointer guard, nonfinite-verdict JSON conversion and two fixture-strengthening assertions. Their final code paths were subsequently exercised by the full44-test certification/changed-fixture rerun and30-test focused rerun. These additional runs overlap discovery and are not added to its method count. No failing assertion was weakened. The full CPU coverage totals499 distinct test methods:497 passes and2 explicit dependency skips.

The two explicit HB skips are `tests.test_exp12_fork.HumanoidBenchReachEvalSeedingTest.test_reach_eval_seeding_saves_and_restores_the_training_rng` and `tests.test_exp12_pipeline.PipelinePerSuiteTest.test_humanoid_bench`, both requiring the absent HumanoidBench environment. Other available-suite tests also cannot establish missing HB execution coverage. Synthetic twin tests do not replace it.

Exact commands, from the isolated `main/`, with the existing pinned Python3.12/JAX0.4.34 venv:

```bash
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu MUJOCO_GL=disable
export EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export MPLCONFIGDIR=/workspace/scratch/mpl_methodology XDG_CACHE_HOME=/workspace/scratch/xdg_methodology
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/workspace/scaling-drl-research/.venv/bin/python -B -m unittest discover -s tests -t . -p 'test_exp12_*.py' -v
/workspace/scaling-drl-research/.venv/bin/python -B /workspace/scratch/methodology_older_runner.py
/workspace/scaling-drl-research/.venv/bin/python -B -u -m tests.exp12_break_checks
/workspace/scaling-drl-research/.venv/bin/python -B -m unittest tests.test_exp12_completeness tests.test_exp12_manifest.Exp12GridTest.test_cli_writes_the_manifests tests.test_exp12_twin_critic.TwinProbeTest tests.test_exp12_methodology_corrections -v
/workspace/scaling-drl-research/.venv/bin/python -B -m unittest tests.test_exp12_reports tests.test_exp12_manifest.Exp12GridTest.test_cli_writes_the_manifests tests.test_exp12_twin_critic.TwinProbeTest tests.test_exp12_methodology_corrections -v
/workspace/scaling-drl-research/.venv/bin/python -B scripts/methodology_cpu_reference.py
```

The outside-checkout older runner uses an `if __name__ == '__main__'` guard, adds `Path.cwd()` to sys.path, and assembles `TestLoader().discover('tests', pattern=p.name, top_level_dir='.')` for every sorted `test_*.py` except `test_exp12_*`; exit status is nonzero unless all succeed. This avoids multiprocessing-spawn import problems and does not exclude any older test module.

Completed local log fingerprints (raw logs are not committed):

| Log under `/workspace/scratch/` | SHA-256 |
|---|---|
| `methodology_exp12_full.log` | `56372c7433ed24a93242447cf6720b778ea5c278bab821f30700e9522a6668bc` |
| `methodology_older_full.log` | `2c19dba22a78539c7b328f5c35cc59e4934896f0f35a684c5f7e6b9e62e15297` |
| `methodology_break_final.log` | `83c8fbee930cc47b6b1d7226c597fab2129fb82aea22a76c20bf5abfa81ed4a0` |
| `methodology_certification_final.log` | `c451d08ef8e7d6d9e3c6d49810c1d2ac34e4911bf6973d2016f578a362371ed4` |
| `methodology_final_focused.log` | `9eef670f3cb59e2d24271ed58cad711d96f865ce1c8aa4175b0ad59cae2532fd` |
| `methodology_reference_final.log` | `6e2c35fc1702f774598ba6479ce67f8892ba6ef69029094e335926efdf755a9d` |

The initial72-check run reported71 OK and one PROBLEM from fixture masking; the final73-check rerun above supersedes it after strengthening that assertion. A focused wrong-module invocation had45 real passes plus one loader error, explained in the audit; it is not counted as a clean run. Deprecation warnings, Gymnasium bound casts and synthetic older-analysis warnings were observed without suppression.

**A100 qualification status:** CPU engineering evidence is ready for review of a separate qualification at the corrected revision; this is not CUDA-qualified or a drop-in replacement for the queued781dc6c job. **Production status: NOT READY**, pending the scientific and deployment blockers above.


No run, assertion or skip is silently converted into a scientific pass. Baseline Exp12 discovery had201 methods (199 pass+2 HB skips); the older187+2 expectation predates12 added regression methods. This branch adds22 methods; completed full discovery and final focused coverage are reported above. Full repository validation additionally includes older Angle/launcher tests, which are distinct from Exp12.
