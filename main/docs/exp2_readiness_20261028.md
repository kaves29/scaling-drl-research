# Experiment 2 implementation and readiness review — 28 October 2026

**CPU engineering review; production Exp2 NO-GO.** No job, training campaign,
new scientific probe fit, Delta preflight or GPU execution was performed. Only
the repository's tiny CPU test fixtures execute updates/fits. No main experiment,
pilot, alternative-probe branch, scientific setting or approval was changed.

## What Experiment 2 tests

At a scaled critic's first naturally eligible loss-of-plasticity trigger, retain
the complete state and compare a restored observational control with a restored
critic-injected continuation. Injection preserves the starting Q function;
subsequent actor diagnostics and raw returns may diverge. The primary comparison
is injected minus control, aligned by steps since fork, per environment and
architecture, with seed-level IQM and percentile bands. Check 2 success is never
a primary inclusion filter. Results concern eligible forks and this intervention;
they do not automatically prove a universal critic-to-actor mechanism.

## Source reconstruction and methodological boundary

Initial workspace: `/workspace/scaling-drl-research`, branch
`codex/overnight-batch1-correctness-20261023`, clean HEAD
`66d4fe368429fb71d9bebced5a4963e48548c57c`. Its older Batch1 report is not this
review's source. No user modifications were found or changed. Remote:
`https://github.com/kaves29/scaling-drl-research.git`.

Fresh fetch and `ls-remote` agree on approved integration
`integration/exp12@844e3c0cc24998b25c3d1e667bcbfe056b1b336c` (the reviewed PR1
merge; see that revision's `AGENT_HANDOFF.md:5`). The isolated review checkout is
`/workspace/exp2-readiness-review-20261028`, branch
`codex/exp2-readiness-review-20261028`. Implementation/test checkpoint:
`ee5e3d3398117197cfaab9efc657c830623ac698`. Resolve the enclosing report/PR
HEAD from Git; this report cannot contain its own enclosing SHA.

**Do not mistake integration approval for current scientific compatibility.**
The integration tree's methodology ends at amendment (z), and
`configs/base_exp12.yaml:53` still specifies five rounds × 1,000 updates,
20 checks. The later approved five-target source `main@087feab341b9be1f377697668f2d1c750ec0467f`
contains amendments (aa)/(ab), at methodology lines 565/579: five rounds ×
2,000 continuous updates, and separate Exp1/Exp2 launch qualification.
These are already recorded owner decisions, not new choices offered here.
Pinned authority: [amendments (aa)/(ab)](https://github.com/kaves29/scaling-drl-research/blob/087feab341b9be1f377697668f2d1c750ec0467f/main/.claude/methodology-exp1-exp2.md#L565).
The source/configuration/analysis alignment has not been integrated into 844.
The later main status also records Method C exploratory job22852240 at
`18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af`: hopper P/b
0.9467/0.9567/0.9552 and fresh-null 5/100. These are a dated report
at another source, not newly verified artifacts or final-source qualification.
Do not reinstate the older 1,000-step 13/100/.602 failures as current Method C
measurements, or infer current launch/job status from either report.
Tests of 844 or this narrow successor cannot certify Method C compatibility.

The separate `codex/exp2-successor@defd636e68554d60dc07ab6a30ce6ee5be60d516`
contains an Exp96 ten-target adaptation and removes historical Angle workflows.
It differs from 844 in 238 paths. PR [#2](https://github.com/kaves29/scaling-drl-research/pull/2),
head `7ef3924e0ef42583e7f885dd8fea228cb1611dbd`, targets that successor and adds
ten Delta admission/resource/funding paths. Neither is approved integration.
Their tests/staging claims are historical claims, not rerun results here.
The successor's own `docs/EXP96_EXP2_SUCCESSOR.md` explicitly says its
ten-target/50-check confirmatory result certifier is unfinished. Both analysis
files are byte-identical to 844 and still require five rounds/1,000-step curves,
20 checks and the 195-parent population (`analysis/exp12_validation.py:21,239,409,868`).
The same two analysis files are also byte-identical in published main087: its
2,000-step Method C configuration and 1,000-step curve checks disagree. This
is a confirmed source/configuration certification mismatch, not evidence that
the scientific probe is defective. A blind move to main087 does not resolve it.
A CPU-only `_check2` schema fixture declaring 2,000 updates accepts `(5,1000)`
curves and rejects `(5,2000)` with `incomplete Check 2 curve`. The validator
blob is exactly main087 (SHA256 `b54716707ea16b96d8f71cb01257af770349a9b9fd392e19cccf4bcc77a0ae18`).
See external `check_method_c_certificate.py`, `method-c-certificate-schema.json`
and its log; exit0 means the compatibility defect was reproduced. This tests
one validator subroutine with synthetic arrays, not a complete certificate or
historical outcome. Fixing the shared protocol/certifier is outside this isolated
Exp2 admission patch and requires the source integration scope to be resolved.
The minimal follow-up is a version-bound certifier that derives expected curve
lengths from the explicitly approved frozen protocol and rejects 1,000-step
curves when 2,000 are required, with valid/invalid full-package regressions.
Do not infer the protocol from observed array shapes or simply accept both lengths.
The ten-target path additionally needs its own census, schedule and IQM10 binding.
Thus neither a ten-target certificate nor an end-to-end Method C readiness claim
can be inherited from those reports. No files were imported from either branch.

PR18 source `93a9184d162b05613f5999ef1b5267f7c9ebc9b5` is a separate optional
diagnostic candidate; it does not qualify or redefine Exp2. This task did not
inspect live Slurm status or infer job state from dated reports. The user's
statement that main Exp1 has not started is context, not a scheduler observation.

## Status matrix

Unless a different revision is named, code references and fresh CPU tests below
refer to implementation checkpoint `ee5e3d3` and its unchanged 844 dependencies.
**Verified** always means the specified CPU contract, not production/scientific
qualification. Test names locate the supporting assertions in `tests/`.

| Required contract | Implementation / evidence | Status and next action |
|---|---|---|
| Natural trigger; scaled-only eligibility; first two strict firing checks by 95% | `experiments/exp12/trigger.py:42,63`; `exp1.py:120,180`; `test_exp12_phase3.TriggerTest`; positive-control natural-evidence validator | Implemented/tested for legacy 20-check schedule. Real eligible fork absent for reviewed successor. Never force one. |
| Immutable eligible fork, complete save before READY, original/control Check 1 | `exp12/fork.py:178,194`; `exp1.py:138`; `test_exp12_fork` end-to-end and kill matrix | CPU verified; READY follows successful exact control restore. Partial writes are uncommitted and safely reconstructed; published fork remains retained. CUDA/full-width durability unverified. |
| Complete model, target, optimizer, training key, replay and n-step state | `exp12/trainer.py:206,240`; `exp12/state.py:55,66,143`; `test_exp12_foundations`, `test_exp12_fork.KillMatrixTest` | CPU verified including continuation, not merely serialization of selected tensors. Full-size production recovery still requires evidence. |
| Observation normalizer; simulator/wrapper/action RNG; counters and pending metric windows | `trainer.py:215,250`; `state.py:143`; canonical 65-cell restoration; HB Reach seeding test | CPU verified on actual simulator paths. Dependency/renderer limits below; no CUDA claim. |
| Matched control/identity continuation and diagnostic RNG isolation | `fork.py:212`; `test_exp12_fork.ForkEndToEndTest.test_identity_fork_is_bit_identical` and `test_fork_is_invisible_to_the_exp1_trajectory` | CPU verified. Control and no-fork baseline match; identity snapshot has no state differences after declared non-training metadata exclusions. |
| Arm schedule/horizon and resume | `fork.py:97,247`; `exp2_arm.py:95`; `test_exp12_fork.KillMatrixTest` | CPU verified: arm +25% B, parent control max(B,fork+.25B), total ≤1.2B, evaluation every 1% B. Seven interruption locations; no changed cadence/budget. |
| Injection definition, online and lagged target Q preservation, action gradients | `injection.py:80,150,230`; `test_exp12_injection.InjectionInvariantsTest`, twin injection/end-to-end tests | CPU verified on tiny real single/twin SAC critics for last/half/all. Full-width, same-GPU-model qualification remains unverified. |
| Retained trunk moments/count; new head fresh moments/count; frozen heads receive no gradient/decay | `injection.py:118,136`; optimizer/gradient/frozen-head tests | CPU verified. No optimizer rules or parameters changed by this patch. Target Polyak follows existing approved implementation. |
| Check 1 failure stops before arm training | `fork.py:120,133`; `exp2_arm.py:144`; one-ULP and common-restore-error tests | CPU verified: exact control/identity, injected 64 float32-eps scaling in highest precision. No tolerance relaxed. |
| Check 2 paired reporting; failed rescue retained | `exp2_arm.py:61`; `analysis/exp12_validation.py:868`; `test_exp12_completeness.test_primary_contains_check2_failure` | Legacy-source CPU verified. Mean twin-score contrast remains distinct from mean per-Q plasticity loss; no formula changed. |
| Positive-control eligibility / reproducibility / m rule | `scripts/positive_control.py:58,93,179`; `m_selection.py:40`; `test_exp12_positive_control`, methodology corrections and completeness tests | CPU arithmetic/rejection verified. Forced/noncanonical fixtures cannot qualify m. Actual canonical natural full-setting development evidence and lead-frozen m absent. |
| Frozen m, no per-run retuning | `configs/base_exp12.yaml:89`; arm rejects null/unknown labels | **Blocked:** m remains null. Only the owner freezes it after reviewing real valid positive control. Tiny fixture uses test-only half; it is not a selection. |
| Source/config admission before arm startup | `exp2_arm.py:41`; new `test_exp2_source_binding.py` | **Confirmed defect fixed:** previously matching config accepted a different/missing source. Eight new tests verify exact clean full SHA, reject dirty/unknown/malformed/mismatched source, retain config refusal. |
| Paired evaluations, raw episode returns/success, probe cadence | `fork.py:212`; `exp2_ledger.py:33`; fork E2E and HB seeding tests | CPU verified for existing path. 26 post-fork eval indices including time zero; production 10 episodes per index. Tiny fixture uses one episode for speed. |
| Artifact/source/config binding; missing/duplicate/contradictory results refused | `analysis/exp12_validation.py:163,239,687,963`; `test_exp12_completeness` | Strict legacy certifier CPU verified; bad evidence fails before confirmatory publication. Progress loader is explicitly uncertified; does not count as a certificate. |
| Seed-level paired final analysis / Check2 secondary subset | `analysis/exp2_analysis.py:105,139,312`; analysis/completeness tests | Legacy analysis CPU verified; dev/identity excluded, Check2 failures included in primary. One-seed uncertainty remains an explicit owner decision. No rounds/episodes treated as independent seeds. |
| Approved Method C source + validator integration | main087 methodology (aa)/(ab); 844 config/certifier mismatch; main087 itself also retains the legacy certifier | **Missing integration:** reviewed promotion/reconciliation needed; no probe/config/Exp1 changes authorized in this isolated patch. Do not use legacy tests as a Method C certificate. |
| Ten-target Exp96 final certification (if that study is selected) | defd successor handoff; unchanged legacy validator | **Missing separate versioned certifier.** No scientific authority to replace five-target census with inferred ten-target settings. |
| Durable full-size checkpoints, resource/runtime/storage and retrieval | State layout exists; no source-matched full-size receipts inspected here | **Production-only unverified:** peak host/device memory, compile/training/probe time, atomic-write overlap, quota/inodes, recovery and verified durable transfer. No estimates or allocation changes invented. |

## Narrow correction and effect on historical results

`check_matches_parent` previously compared resolved configurations but never
compared the parent source metadata with executing source. A fixture with
identical config and commits `aaaa…`/`bbbb…` was accepted. The new rejection tests
on unmodified 844 run eight methods with ten expected assertion failures
(subtests included); the same eight methods pass after the correction.
Logs: `source-binding-before.log` and `source-binding-after.log` in external evidence.

The new guard runs before arm directory creation, metadata publication, trainer
construction, restore, injection or fitting. It requires matching full hex SHA
and explicitly clean source metadata; unavailable Git identity cannot silently
pass. Source checking reuses the repository's `code_version` mechanism, not a
claim of arbitrary runtime/library attestation. Configuration comparison is unchanged.
The separate defd successor already has a simpler source-equality check; this is
a scoped independently tested correction on approved integration, not an import
or approval of that branch.

This changes **admission only**. It cannot change a correctly admitted arm's
numerical trajectory, optimizer or random draws. Historical files remain intact.
It may reject mixed-source/dirty legacy starts that previously proceeded; those
must not be relabeled valid. Existing frozen parents must use their bound source;
this PR does not authorize rerunning or rewriting parents or carrying an old
fork onto a new code SHA. No change was established as necessary in injection,
restoration or paired analysis for the tested legacy contract.

## Validation and reproducibility

| Check (all source `ee5e3d3` unless noted) | Actual result |
|---|---|
| Full isolated repository CPU suite | 629 tests across 69 modules; zero failures/errors; 0 skips; exit0 |
| Existing break/restore mutation suite | 74/74; 70 assertion detections + four error detections; every restored case passes; exit0 |
| Canonical simulator restoration | 65/65 task/seed cells; exit0 |
| Independent methodology census/reference | 195 parents, 39 architecture/environment configs, 130 scaled candidates; exit0 (legacy 1,000-step contract) |
| New admission regression | Eight methods pass; baseline before-fix reproduction has ten expected assertion/subtest failures, exit1 |
| Retained tiny real-critic E2E fixture | Seven existing assertion methods; identity differences []; control400/injected200/fork100; exit0 |
| Method C certificate schema | Compatibility defect reproduced in two array cases; exit0, not a compatibility PASS |
| Static patch checks | Black new test, Pyflakes changed Python, AST parse, diff check: all exit0 |

Four mutation failures are intentional errors, not assertions: the old twin vmap form, unset twin TD-error variable, invalid twin actor-gradient shape, and missing twin `critic_count` metadata (the test indexes that required key). The existing harness accepts either kind of failure; its criteria were not changed. These are not four failures of restored production code.

Counts overlap: the focused regression and reused E2E assertions are included in broader coverage. Warning occurrences are retained per module in `receipt_audit.json`; logs preserve their full text (Orbax restoration/deprecations, plotting, simulator notices). No new warning suppression was introduced.

- `repository.log`: SHA256 `ba6204e1065af85fee1376601a5a51373befacdb85b7ac03d988abcc5934269f`; exit0.

- `mutations.log`: SHA256 `93bea3b7cd4c6d745d0bdf5917f7f9a1aa74030b26c931ac3102fd4565f86e69`; exit0.

- `simulators.log`: SHA256 `dcce1ae2734d65d9a220df3806dc350df0395aae3e40625e1fab16f1b946ee7f`; exit0.

- `methodology.log`: SHA256 `6e2c35fc1702f774598ba6479ce67f8892ba6ef69029094e335926efdf755a9d`; exit0.

The intentionally stopped 844 baseline exited143 before completion. Its partial receipts are preserved and excluded from final counts. The environment briefly displayed starting; the corrected-source child processes and receipts survived. The original terminal handle was lost and its outer wrapper ended after recording the repository result; its exit/cause is unobserved. Only the three unrun stages were resumed after verifying the same clean HEAD/fingerprint; see orchestration_recovery in validation.json.


Raw evidence is outside Git at
`/workspace/artifacts/exp2-readiness-20261028/`. `final/validation.json` records
exact commands, environment, exit codes, elapsed times and log SHA256s;
`final/repository/{provenance,modules,summary}.json` binds every isolated module
to exact source, interpreter, dependency versions and owned logs. Source file
hashes and comparisons are in `source_inventory.json`. Failed reproduction and
intentionally interrupted baseline logs are retained, never counted as passes.
`source_inventory.json`, `supplemental_commands.json` and `receipt_audit.json`
retain exact supplementary fixture/reproduction commands and independent receipt
checks. Final documentation-head checks are recorded separately in
`publication_receipt.json`; the full-suite claim above stays bound to `ee5e3d3`,
not transferred to another SHA. Only documentation follows that implementation.

CPU environment: Python3.12.14, JAX/JAXLIB0.4.34, Flax0.8.4, Optax0.2.3,
NumPy1.26.4, Orbax0.5.3, MuJoCo3.6.0. Actual pinned HumanoidBench source is
`cb1189039151c8aadaaa987b442da54383c87fab`, using project adapter and software EGL.
The overlay deliberately uses the project's shared simulator stack rather than
all upstream HB pins. See `exp12_cpu_validation_setup.md`; upstream dependency
lock compatibility and unrelated HB paths are not certified. No packages changed.

Reproduce from the clean tested checkout's `main/`, with fresh absolute outputs:

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu_python=/workspace/venvs/scaling-drl-hb/bin/python
"$cpu_python" -B scripts/run_exp12_cpu_validation.py --jobs 2 --out-dir /ABS/NEW/repository
"$cpu_python" -B -u -m tests.exp12_break_checks
"$cpu_python" -B scripts/check_exp12_cpu_environments.py
"$cpu_python" -B scripts/methodology_cpu_reference.py
"$cpu_python" -B -m unittest tests.test_exp2_source_binding -v
```

The retained tiny parent→fork→control/identity/injected example reuses the
existing `test_exp12_fork.ForkEndToEndTest` setup and assertions. Its fixture
is forced/test-only, not a natural trigger, positive control, current Method C
qualification, m selection or scientific result. Fixture source, logs, hashes
and metadata exclusions are retained externally. The artifact copy includes its
original temporary source paths; inspect receipts/arrays, do not reuse it as a
relocatable production certificate.

## Remaining blockers and owners

**Engineering Codex can perform after the source dependency is resolved:**
review/integrate applicable Method C source and its validator rather than
duplicate legacy paths; for Exp96, finish a separately versioned result certifier
with literal ten-target IQM, complete population/fork pairing and negative tests.
Review existing PR2 admission/resource work before any scoped adoption. Prepare
source-bound resource and artifact qualification only under a separate task.
None requires speculative redesign of the probe or scientific hypothesis.

**Real evidence required:** a source-bound natural eligible development fork,
canonical full-setting positive-control scores/noise/Check1 and reproducibility;
same-model full-width CUDA identity/injection/target/optimizer/restart checks;
actual injected resource/throughput/storage and complete paired artifacts. No
GPU schedule, source-bound spending or operational qualification is granted here.
The diagnostic pilots' success/failure cannot automatically qualify these paths.

**Owner decisions:** identify the intended five-target versus separate Exp96
campaign source/population and authorize its integration/review; later review
valid positive-control candidates and freeze m; resolve any additional positive-control
success interpretation if all candidate recoveries are negative or overshoot (the
existing smallest-within-0.10 arithmetic is unchanged); resolve documented fresh-null
and hopper-range scientific dispositions for the applicable source, any required
CUDA-bound failures, and one-eligible-seed uncertainty if it occurs. Approve actual
production source/resources/submission separately. Do not revisit already approved
(aa)/(ab), choose m now, change thresholds or force a trigger.

## Dependency-aware path to December 1 / December 5

December1 is the working completion target (34 calendar days after October28);
December5 is the hard deadline (38 days). No measured critical-path runtime or
confirmed allocation availability was supplied, so feasibility is **unestablished**.

| Sequence / parallel work | Prerequisite | Completion evidence |
|---|---|---|
| Now: review this admission patch, reconcile source/version and integration ownership | Owner source disposition / independent review | One explicit approved executable/config/validator SHA; no stale test substitution |
| Parallel with Exp1 engineering: certify applicable offline fixtures and prepare artifact/recovery checks | Same frozen protocol/source | Version-correct CPU receipts; no allocation needed |
| Exp1 may proceed independently of Exp2 positive control under approved amendment (ab) | Exp1's own scientific/CUDA/source/resource/launch gates | Actual parent evidence; no pilot or old comment used as job status |
| Natural development trajectory and approved GPU qualifications may be planned alongside parent campaign | Separate authorization and frozen source; no invented runtime | Natural eligible fork or explicit no-trigger outcome |
| Positive-control analysis and owner m freeze | Real canonical natural fork; approved fitting execution | Reproducible losses/noise/candidate recoveries, owner decision; stop if criteria fail |
| Injected arm continuation | Retained eligible parent fork + m + same-source/device and Exp2 operational approval | Check1 pass, reported Check2 outcome, full horizon, durable partial/final evidence |
| Final paired results and review before working/hard target | Complete correctly certified arms/census or explicitly unresolved missing evidence | Per-environment seed lines/bands, null eligibility and incomplete results retained |

The first natural trigger cannot be scheduled to occur, and a no-trigger or noisy
positive control is a valid blocker rather than a reason to alter the method.
Critical path is source integration → natural eligible evidence → valid positive
control/frozen m → qualified paired continuation → complete certification/review.
Exp1 and offline engineering can overlap where amendment (ab) permits; injected
arms cannot precede their own dependencies. If a dependency misses its available
window, report the incomplete scope by December1 and use December5 only within
existing authorization. No promised runtime, compute budget or success outcome.

## Next review

1. Review the narrow source-admission diff and exact-source CPU receipt.
2. Resolve the integration/source-version gap without rewriting active runs.
3. Obtain qualifying natural evidence, then review/freeze m; leave gates blocked meanwhile.
4. Authorize any later real CUDA/resource qualification as a separate operation.
5. Require the applicable final result certifier and complete retained artifacts.

Optional later literature work may inform interpretation of intervention
specificity and conditional eligibility; it is outside this implementation task
and is not authority to change the approved experiment.
