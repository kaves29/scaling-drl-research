# Experiment 2 Method C source integration candidate — 28 October 2026

Review candidate only. Production, GPU jobs and scientific qualification remain
NO-GO. PR #19 and approved integration are not modified. Exact-source CPU evidence and remaining gates are recorded below.

## Verified authority and selection

Remote refs verified before editing: integration/exp12
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`, PR #19 head
`5482b2a516de5d51dab2c1bd7a4e86a49832b768`, published main
`087feab341b9be1f377697668f2d1c750ec0467f`. Integration844 is an ancestor
of main087. A branch name alone is not approval: AGENTS.md gives chronological
`main/.claude/methodology-exp1-exp2.md` amendments and `exp12_decisions.md`
authority, and amendments (aa)/(ab) explicitly record the owner's decisions.
The older decisions text's “we use 1000” is historical and superseded by (aa).
The decisions file itself is identical between844 and087 and is not rewritten.

| Source (full SHA) | Imported paths / purpose |
|---|---|
| `c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b` | Approved Method C producer `experiments/exp12/probe.py`, sole config change `probe.steps=2000`, amendment (aa), producer tests and CPU methodology census. These exact producer/config blobs are unchanged through087. |
| `d65f9bb798f5c5c74b1857a6329669e068ddf517` | Reviewed `exp1.py` late-control schedule restore and preserved aborted fresh state; `exp12/state.py` checkpoint durability; `exp12/fork.py` atomic panel archive; fork/resume tests. Required to preserve the reviewed deferred-arm lifecycle, not a new training intervention. |
| `b225577c2ba5a18b2ad6d14c47b34a00b5443e05` | Historical range-capture tests explicitly retain their1,000-step protocol and reject relabeling Method C as historical evidence. |
| `957be5c63a48b0b90dd28ee4b361e3c6d373ab05` | Independent consolidated review documentation, retained as a dated historical source. |
| `6eb397d0c485a5db202806df8a0308e48e46b05d` | Exact amendment (ab), deferred-arm lifecycle regression and compatibility report. Complete methodology file imported byte-for-byte from087 includes (aa)/(ab). |
| PR19 `5482b2a516de5d51dab2c1bd7a4e86a49832b768` | Exact source-admission guard/test, frozen-config curve certifier/test, accurate fixture and dated reports. |
| This candidate | New producer→runner NPZ→ledger→curve certification CPU test; exact durability test class extracted without unrelated scheduler dependencies; integration map and current report/handoff. |

Machine-readable per-file source bindings and SHA256s are in
`exp2_method_c_integration_sources.json`. Runtime producer/config/restore files
are exact087 blobs; admission/certifier files are exactPR19 blobs. Imported
historical reports retain their original dates/status and are not current
qualification claims. Current status is this report.

No main087 merge is performed. Unrelated dynamic scheduler, allocation, approval
templates, launch tooling, numeric-bound tooling, pilots and newer alternative
probes are excluded. In particular, this is not promotion of main087's launch
system or its production approvals. Amendment (ab)'s permission to qualify Exp1
separately is retained; this candidate supplies and tests the deferred-arm
runtime, not a new launch system. The legacy integration launcher may still
impose stricter launch gates. Production launch-tool integration/review remains
separate; no candidate command bypasses those gates.

## Preserved semantics and CPU coverage

Five targets,2,000 continuous AdamW updates, first1,000 original draws and next
1,000 from `fold_in(idx_key,1)`; own-mean offsets, scores, bootstrap/trigger,
eligibility, injection, precision, evaluation and budgets are not retuned.
`injection.m` and force-trigger hooks remain null in canonical configuration.
Existing tiny lifecycle tests use pre-existing dev-only hooks as test fixtures;
they are not evidence of natural eligibility or positive-control sensitivity.

The added array-IO integration test uses the full canonical probe configuration
(including25,600 sample pool,256 minibatch,2,560 eval chunk,5 rounds and2,000
updates), production optimizer construction, runner saving, ledger copying and
actual arm Check2 generation. A tiny linear network and synthetic replay replace
full-width architecture/environment execution. Its unchanged loss arrays must
pass the shared curve certifier. Full-population/source certification is tested
separately with explicitly synthetic complete-study fixtures. Neither is a CUDA
qualification or a natural full-scale scientific positive control.

The runtime/config comparison against main087 (`main/experiments`,
`main/scale_rl`, base config, agent/environment configs) has exactly one
difference: PR19's stronger Exp2 source-admission guard. The selected producer/restore imports introduce no missing dependency;
new imports resolve in the approved base.

Key candidate locations: methodology amendments at567/581; producer
`probe.py:74,112,129`; raw runner archive `run_probes.py:131`; ledger
`ledger.py:55`; admission `exp2_arm.py:41`; certifier
`exp12_validation.py:387,401,888,987`; restored control schedule
`exp1.py:143`; durable state `state.py:67`; atomic panel IO `fork.py:169`.

## Combined-source validation receipt

Implementation source: `a3d7efa151ce276191851bda040e979086d0592e`.
The complete CPU suite is run on this combined branch, not a certifier-only
checkout. Fresh output: `/workspace/artifacts/exp2-method-c-integration-20261028/`.
The final documentation successor is separately tested and identified by its
external receipt; resolve its SHA from the candidate ref rather than treating
this document as self-identifying.

Exact commands from the candidate's `main/` (the final receipt records each
literal argv, cwd, SHA, exit code, test IDs and log SHA256):

```bash
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=''
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 EXP12_JAX_CACHE_DIR=off
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpu_python=/workspace/venvs/scaling-drl-hb/bin/python
out=/workspace/artifacts/exp2-method-c-integration-20261028
"$cpu_python" -B scripts/run_exp12_cpu_validation.py --jobs 2 --out-dir "$out/full-cpu"
"$cpu_python" -B scripts/check_exp12_cpu_environments.py
"$cpu_python" -B scripts/methodology_cpu_reference.py
"$cpu_python" -B -u -m tests.exp12_break_checks
```

The standalone producer-to-certifier test additionally retained its real
runner/ledger/Check2 archives via `EXP2_METHOD_C_TEST_ARTIFACTS=$out/produced-curves`.
`produced_curve_audit.json` records shapes, dtypes, finiteness and per-array/file
SHA256s. All loss curves are `(5,2000)`; raw and consolidated arrays compare
bit-exactly. Negative checks mutate copies only:1,000/1,999-entry truncations,
missing and duplicate fields reject, and top-level validation rejects the old
1,000-step frozen-study configuration. Existing source admission and complete
study tests reject wrong/unknown/dirty sources and wrong configurations.
No shape is inferred from artifact length and no tolerance is relaxed.

The checked-in full suite also tests continuous AdamW count1,000→2,000,
original first-stage draws, deterministic pairing, injected/control state,
checkpoint round trips, late-control resume, interruption before first save,
source admission and full195-parent/130-candidate certification fixtures.
The last population fixture is explicitly synthetic; it is not a reconstructed
historical campaign, natural positive control or CUDA evidence.

Environment: Python3.12.14, JAX/JAXLIB0.4.34, Flax0.8.4, Optax0.2.3,
NumPy1.26.4 and Orbax0.5.3; pinned HumanoidBench
`cb1189039151c8aadaaa987b442da54383c87fab`, project adapter and software EGL.
As documented in `exp12_cpu_validation_setup.md`, this is the project's tested
CPU/HB overlay, not upstream HB dependency-lock or CUDA qualification.
Dependency/deprecation/plotting/simulator warnings are retained in raw logs.
The readonly Mesa shader-cache warning is not an unavailable simulator or skip.

## Readiness and owner decisions

| Category | Disposition |
|---|---|
| Engineering | Combined producer/config/amendments, admission and certifier assembled from exact sources; canonical curve-IO and fail-closed checks demonstrated. Candidate is for independent review, not approved integration. |
| Scientific evidence still needed | Naturally eligible same-source fork and source-matched full-scale positive control; retained Method C null/range evidence requires any still-unrecorded owner disposition. No injection magnitude selected and no trigger forced for scientific evidence. |
| GPU/resource gates still needed | Actual final-source CUDA identity/numerics, backend/precision, checkpoint/intervention restore, live filesystem/binding, full-width memory/throughput and storage/retrieval qualification, plus explicit allocation/launch approval. CPU results do not satisfy these gates. |
| Owner decisions | Review/approve source promotion separately; resolve recorded scientific gate dispositions; freeze `m` only under the approved positive-control rule; approve resource plan and any GPU qualification/production launch separately. No new scientific parameter choice is needed to assemble this candidate. |

No methodology/source conflict was found in the selected integration: (aa)
specifies the producer's2,000-step implementation and (ab) matches the deferred
arm lifecycle. The old1,000-step certifier incompatibility is resolved by PR19's
exact patch. Old1,000-step studies remain historical and cannot be certified as
this canonical Method C. The ten-target/96-parent successor remains a separate
protocol and is not silently admitted by this five-target candidate.

The candidate excludes unreviewed launch-system promotion. A future production
launch-tool review must honor (ab)'s separate gates and current source identity;
this branch is not queue-ready. Existing artifacts tied to another commit do
not become same-source qualification evidence merely because runtime blobs
match. Any unavailable natural evidence remains unavailable.

PR19 remains draft at5482b2a, integration remains844e3c0. No merge, force-push,
production-source publication, active-pilot edit, scientific run, GPU job,
Delta preflight, submission or queue operation occurred.

### Completed CPU results

| Check | Tested source | Result |
|---|---|---|
| Complete combined-source repository suite | `a3d7efa151ce276191851bda040e979086d0592e` | 650 unique tests across 73 modules; zero failed modules; 0 skips; exit0; source fingerprint unchanged |
| Break/restore mutations | Same implementation SHA | 74/74 detected; every restored case passes; exit0 |
| Canonical simulator restoration | Same implementation SHA | 65/65 cells; exit0 |
| Independent Method C census | Same implementation SHA | 195 parents /39 architecture-environment cells, all2,000 steps; exit0 |
| Retained runner/ledger/Check2 arrays | Same implementation SHA | Three new producer-to-certifier tests pass; all loss arrays(5,2000); unmodified arrays certify and copied invalid arrays reject |

Counts overlap with regression/mutation scopes and do not represent scientific
samples. Final-head focused validation is recorded separately in
`final-validation.json`; it is rerun after this report is committed. Broad
results above remain bound to the actual implementation SHA, not silently
reassigned to a documentation successor. The only later test-file edit removes
an extra trailing blank line from the extracted durability class; its class
body remains byte-identical to the pinned source. No production runtime or
configuration bytes change after the full suite.

Warnings, per-test IDs, exact worker commands, owned log checksums and test
counts are audited in `receipt_audit.json` and `full-cpu/`. Focused production
Pyflakes, new-test Black and final diff checks are recorded in `static.json`.
The fixture retains its three pre-existing unused imports (`io`, `Path`,
`pandas`); these reproduce on the PR19 source and are reported, not hidden as
a passing lint check. Imported reviewed code is not cosmetically reformatted.

**GO for independent review of this source candidate; NO-GO for a production
Exp2 launch.** No new methodology decision was required by integration.

This candidate subsumes PR19's exact changes without modifying that PR. The
owner should choose the integration route after review; do not independently
promote both overlapping candidates as separate scientific source baselines.

Warning audit: the passing legacy Angle1 multienv rejection test emits one
`os.fork()`/multithreaded-JAX RuntimeWarning; it is retained, not a qualification
of CUDA multiprocessing. Other retained warnings concern dependency
deprecations, Gymnasium float32 bound casts and readonly plotting/shader caches.
