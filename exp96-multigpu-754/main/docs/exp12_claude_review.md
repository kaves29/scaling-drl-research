# Independent review of Claude Code's Exp1/2 findings

Review started from clean `f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6` on
`claude/eloquent-fermat-inxqlt`. All execution was local CPU validation. No Delta
checkout, running Block B files, Slurm jobs, main branch, GPU bounds, injection m,
experimental settings, numerical criteria or inclusion rules were changed.

The confirmed orchestration fix is commit
`7fbdf1421f526d8e8dfd944d0a858f9758d651b3`. Evidence hygiene and this report follow
separately. The final clean HEAD is reported with delivery; future HB qualification
must pin that actual final revision, including the spool fix and any subsequently
approved changes, rather than reusing an old implementation hash.

| Claim | Verdict | Classification and disposition |
|---|---|---|
| 1. HB derives the repository from its spool copy | CONFIRMED | Orchestration defect; reproduced and fixed, with a copied-script regression |
| 2. Unset GPU tolerances prevent HB qualification | PARTLY CONFIRMED | Intended fail-closed gate; selected HB tests need one mode's update bound, not all four bounds directly; calibration precedes qualification |
| 3. Exact twin panel comparison fails | CONFIRMED | Same known eager/JIT numerical-reference issue; suggested mixed tolerance still fails; no criterion changed |
| 4. One eligible seed stops Exp2 | CONFIRMED | Pending reporting decision; also applies to the Check-2-success secondary subset |
| 5. An incomplete run blocks all confirmatory output | PARTLY CONFIRMED | Exp1 requires all 195; Exp2 independently requires 130 scaled parents. Recovery is per affected run, not whole-grid retraining |
| 6. Generated evidence and machine paths were committed | PARTLY CONFIRMED | 64 files, 2,350,542 bytes; 46,166 text lines, only 5,891 log lines. Generated validation data, not production-grid results; hygiene corrected without history rewriting |
| 7. abb9498 is redundant | NOT CONFIRMED | Object unavailable locally; cannot inspect Claude's local-only commit. Existing 7d1bda6 demonstrably documents both tasks |
| 8. Concurrent workers share a vulnerable JAX cache | CONFIRMED | Shared default and reproducible partial-publication mechanism; exact Delta incident cause remains unproved. Cache rollout requires approval |
| 9. Old HB command and reported HEAD differ | CONFIRMED | Explained implementation/evidence ancestry; no source discrepancy then, but old pin excludes the new spool fix |

## 1. Slurm spool path

For the intended submission from `<final-checkout>/main`, without `--chdir`:

- `SLURM_SUBMIT_DIR` is `<final-checkout>/main`.
- Initial cwd is that submission directory. `--chdir` can override the initial cwd;
  it does not redefine the documented submission-directory contract.
- `$0` is the compute node's copied batch script, conventionally
  `<SlurmdSpoolDir>/job<id>/slurm_script`; the site's literal spool prefix is not
  established here. It is not the original `scripts/sbatch_exp12_hb.sh` path.

[Slurm's sbatch manual](https://slurm.schedmd.com/sbatch.html) documents cwd and
`SLURM_SUBMIT_DIR`. The
[Slurm launch source](https://github.com/SchedMD/slurm/blob/slurm-24-05-4-1/src/slurmd/slurmstepd/mgr.c)
constructs `batchdir/slurm_script` and assigns it to `argv[0]`.

The former `dirname "$0"/..` therefore entered the spool parent, not the checkout.
With successful module/conda activation, the later Git HEAD/clean-tree comparison
would fail outside the repository. It might fail earlier at cluster setup; an
actual Delta execution was neither needed nor attempted to prove the path defect.

The old CPU test copied the driver into a synthetic `scripts/` directory and ran
`bash scripts/...` with `SLURM_JOB_ID` removed. It tested exit propagation but
preserved the exact path assumption that was wrong under sbatch.

The fix uses `SLURM_SUBMIT_DIR` for batch execution and retains script-relative
resolution for local invocation. Missing/invalid submission directories fail
before cluster setup. Expected HEAD and clean-tree checks remain unchanged. No
obsolete absolute checkout is hardcoded. Submit from `main/`, as already required.

The new test executes the complete executable spool copy outside the synthetic
checkout, with a Slurm job ID and submission directory. A stand-in `module`
command records the cwd and deliberately stops before any cluster access. The
test covers default cwd, unrelated initial cwd, a checkout path with spaces, and
empty/wrong/nonexistent submission directories. On the old driver it recorded
the spool parent and failed. All nine HB bookkeeping tests pass after the fix.
This is a CPU orchestration test, not Slurm/CUDA integration qualification.

## 2. Calibration versus qualification

HB selects the real HumanoidBench pipeline test, the Reach evaluation/RNG test,
and the complete twin-critic test module. In that module,
`TwinDiagnosticsTest.test_diagnostics_never_change_the_update` calls
`check_gpu_tolerance` with `diagnostics_update/<mode>` after four updates. The
default mode is `tf32`; an inherited `--xla_gpu_deterministic_ops=true` selects
`highest_deterministic`. All four entries in `GPU_TOLERANCES` remain `None`.

The helper emits a `NUMERICS` record before `skipTest` when its bound is unset.
Unittest can return zero with skips, but `check_exp12_hb_status.py` requires a
nonzero test count and a bare `OK` summary. `OK (skipped=1)` fails qualification.
The shell records every step's code, continues collecting other evidence, and
returns the final checker's nonzero status. Existing CPU tests exercise this.

Thus the selected HB test step currently needs the relevant **update** bound to
qualify. Setting the other three entries alone cannot resolve it, and HB does not
itself measure the full four-bound matrix. The unchanged twin panel red gate is
an additional independent blocker. Missing HB packages/integration cannot be
reclassified as a qualification pass either.

There is no computational circular dependency: measurements are emitted before
the skip. The correct ordering is:

1. Gather calibration evidence on the final selected hardware/stack and required
   single/twin paths, update/training quantities and both precision modes.
2. Review helper validity, repetitions, observed worst cases and proposed bounds.
   The lead approves the acceptance criteria and values; none are set here.
3. Rerun qualification against the approved values on the final source revision.

Keep fail-closed qualification. A measurement-only HB allocation is possible only
with explicit authorization and a separately interpreted result; it must not
produce qualification PASS. No measurement-only switch or skip exception was
implemented. Existing Block B can supply preliminary calibration, subject to the
revision/helper limitations below.

## 3. Twin numerical reference

This is the same known failure, not a new implementation regression. A fresh
process reproduced the saved exact eager/panel values. Separately compiled
production Q and gradient match the production panel exactly, but share its
implementation and cannot be the independent semantic oracle.

Expected below means eager; observed means the production compiled panel.

| Element | Expected | Observed | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Q[354], maximum absolute Q difference | -1.4848878383636475 | -1.4848871231079102 | 7.152557373046875e-7 | 4.81690077072018e-7 |
| Q[183], maximum relative Q difference | -0.0026113688945770264 | -0.002610921859741211 | 4.470348358154297e-7 | 1.711879301096744e-4 |
| Gradient[82,1] | 0.0029409804847091436 | 0.0029408172704279423 | 1.632142812013626e-7 | 5.549655363235235e-5 |
| Gradient[146,0] | -0.00685656676068902 | -0.006856751162558794 | 1.8440186977386475e-7 | 2.6894198833022682e-5 |
| Gradient[164,0] | 0.04219628870487213 | 0.04219687730073929 | 5.885958671569824e-7 | 1.3948996113703742e-5 |

The exact Q assertion fails on 67/512 values. Independently applying the later
unchanged `rtol=1e-5, atol=1e-7` gradient assertion fails on 3/768 values. Giving
Q that same mixed tolerance still fails at Q[183]: its allowance is
`1.2611368894577027e-7`, versus error `4.470348358154297e-7` (3.545 times the
allowance). Copying the gradient tolerance is neither sufficient nor justified
by a numerical error model.

The independent NumPy64 forward/analytic-Jacobian characterization was rerun for
all 15 tiny panels, both eager/JIT modes and separate Q1/Q2 scales. Its output is
byte-identical to the existing characterization (SHA256
`d8c070ad1e93a1916d7936c9439d85b679ea2c8aa9abfe388325193ab15c7b8a`).

The proposed, **unadopted CPU-fixture** rule is finite arrays and, separately for
Q1, Q2 and min-Q action gradient:

`max(abs(float64(got) - reference64)) <= 64 * eps32 * max(abs(reference64))`.

Zero reference scale requires zero error. All 90 legitimate per-array/mode
comparisons would pass. Maximum legitimate error is 8.4333099944 epsilon units;
64 is the next power of two above four times that measured maximum. All 105
tested case/mutation combinations would fail: Q1/Q2/max/mean/zero gradient,
swapped Q order, and duplicated Q1. Minimum margin over the proposed bound is
42,731.215 times; the existing Q1-instead-of-min mutation margin is 45,080.131
times. This is empirical separation against these defects, not a universal
floating-point error theorem or CUDA bound. No states are dropped.

The original failing assertion first appears in `2f67915`; equivalent eager/JIT
behavior is already present in working twin commit `c293b9a`. At the pre-Exp12
baseline `6ffccba`, the original twin initializer is broken on this dependency
stack; its independently sliced single-Q path also exhibits eager/JIT rounding.
The preserved historical investigation records exact baseline values and its
limits. The fresh reproduction and unchanged characterization confirm that the
current issue is that same cross-execution reference assumption. CPU evidence
supports ordinary float32 numerical behavior, not an incorrect min derivative.
Whether the exact test fails on A100 remains unmeasured here.

Recommendation: approve/revise an independent semantic CPU-oracle budget before
changing this test. Preserve exact original/restored-control/identity checks and
the existing injected Check 1 rule. Characterize CUDA separately. The test and
its restored mutation remain red; no experiment execution changed.

## 4. Exactly one eligible seed

After completeness certification, `exp2_analysis.run_analysis` counts eligible
forks per architecture/environment, then separately counts Check-2-successful
forks. If either subset has exactly one, it raises `ReportingDecisionRequired`
before `_write_analysis`. Consequently the entire new validated Exp2 directory
is withheld: census, eligibility summary, paired/band tables, primary/secondary
figures, checks and diagnostics. A sibling `.validation_failed.json` records the
reason and census; staged output is discarded. Exp1 is not blocked by this
reporting decision. Zero eligible cells have an explicit census and are supported.
The existing one-seed regression exercises the stop.

| Option | Scientific consequence |
|---|---|
| Retain n=1 paired trajectory and point estimate, no bootstrap band | Preserves all eligible observations; explicitly cannot estimate between-seed uncertainty from one seed |
| Keep descriptive output, exclude n=1 cells from aggregate inference | Defensible if explicitly predeclared, but must not silently remove the seed from the confirmatory census or eligibility population |
| Show a labeled degenerate bootstrap band | Computable, but zero width reflects resampling one seed, not high certainty; easy to misread and not recommended |
| Collect additional seeds | Changes the approved seed population and may introduce outcome-dependent sampling; requires a separate design decision |

Exp2 currently has per-environment curves and no scalar cross-environment
aggregate. The first two options can therefore be implemented identically for
present graphs: retain the trajectory, label n=1, and make no between-seed
inference. An episode-level bootstrap would estimate a different uncertainty
quantity and cannot manufacture independent seed replication.

Recommendation for approval: retain every paired trajectory, annotate n=1,
omit its seed-bootstrap band, and label its point estimate descriptive. Apply
consistently to primary and success-only secondary subsets; do not change primary
inclusion according to Check 2 or realized sample count. No convention adopted.

## 5. Incomplete runs and recovery

Exp1 independently reconciles 3 architectures × 13 environments × seeds 1–5 =
195 parents. Exp2 independently reconciles 2 scaled architectures × the same
environments/seeds = 130 candidates, whether or not fork directories exist.
An incomplete D2W512 parent blocks Exp1, but is outside Exp2's candidate census.
An incomplete scaled parent can block both. A missing injected arm blocks Exp2;
Exp1 does not require completion of injected arms.

Completeness includes terminal DONE/state agreement, nominal endpoints, scheduled
checks/rounds/curves, required logs and provenance, and original/control fork
evidence. Eligible Exp2 forks need both arms, all 26 evaluations × 10 episodes
per arm, correct indices/times, finite returns and existing Check 1/2 evidence.
Check 2 failure remains in the primary population. B5 invalid intermediate probe
checks retain their existing treatment. Early episode termination is complete
when the required episode record exists. Pruned routine checkpoints are not
required. These protections are intentional, not reasons to discard failed runs.

Recovery workflow for review only; no commands were launched:

1. Preserve the failure log and inspect its deterministic cause. Compare the
   incomplete census entry with its source directory, DONE, run metadata,
   `state/LATEST`, required state files and, if present, `fork/FORK_READY`.
2. Confirm the owner allocation/process is no longer active before recovery.
   Never edit an active checkout/manifest or remove an active claim.
3. For an ordinary interrupted run with valid state and unchanged provenance,
   rerun the exact same approved command, seed/config, checkpoint directory,
   results root and GPU model. Exp1 restores the latest committed complete state;
   a mid-save orphan is ignored until a later successful commit removes it.
   Before the first save, the same seeded run starts again. At/after the fork,
   the immutable fork state permits the control restart. Injected arms restore
   their own state, including injection structure/optimizer state, or initialize
   from the parent's ready fork when no arm state exists.
4. Generate fresh launch manifests into a new staging directory using
   `generate_manifest.py --grid exp12` with the same roots and approved frozen m.
   `classify_exp12` emits fresh/resume runs and omits DONE runs. Arms are emitted
   only for ready scaled forks and grouped by recorded device model. Regenerate
   after new forks appear; the launcher does not create arm manifests itself.
   Use only the files emitted by this generation. Old device-arm manifests can
   remain in a reused directory when that device has no pending arms; staging
   avoids accidentally selecting such stale queue files.
5. Review only the affected commands. Run the existing overlap check against
   every manifest that could coexist; retain distinct checkpoint/log paths and
   invocation logs. Pass the launcher explicit absolute `--phase-files`, since
   its defaults are Angle 1 phases. Its child commands execute from `main/`.
   Preflight and authorization are still required before any allocation/run.
6. The claim launcher rechecks DONE and uses atomic mkdir for a fresh claim;
   live owners are skipped and nonzero commands release their claim for retry.
   Its stale reclamation is a multi-operation unlink/rmdir/mkdir sequence, not
   a transactional ownership transfer. An ownerless claim is held indefinitely,
   and a child that exits zero without DONE leaves an unresolved claim. For
   recovery, serialize stale-claim handling after verifying no live owner;
   do not assume arbitrary concurrent stale takeovers are proven safe. The
   launcher can repeatedly retry deterministic errors: diagnose them before
   authorizing retries, rather than consuming allocation time indefinitely.
7. Successful entry points write complete state/ledgers before atomic DONE.
   Resume rewrites exported scientific records from checkpointed lists, removing
   any uncommitted tail; kill/resume tests cover pre-check, mid-interval,
   mid-save, fork-write and post-fork control/injected interruptions.
8. Revalidate the complete study and publish into a new empty analysis directory.
   Input hashes/directory snapshots are checked again before atomic publication;
   stale output is not overwritten. Exploratory output is separately labeled
   and cannot supply a confirmatory certificate.

A DONE marker is a scheduling hint, not proof that all analysis evidence is
valid. If DONE exists but required data are corrupt/missing, the generator will
skip that run and the validator will still refuse it. Do not remove DONE to
silently reselect observations. Investigate whether a missing export can be
reconstructed from intact saved state; otherwise propose a preserved-artifact,
same-seed affected-run rerun for lead review. A revision/runtime/config change
requires review too: the frozen study contract rejects incompatible launches.
There is no requirement to rerun healthy completed runs merely because one
ordinary interruption occurred.

## 6. Evidence hygiene

The exact pre-cleanup inventory is below. Totals: 19 logs (842,199 bytes, 5,891
lines), 21 JSON files (1,171,898 bytes, 39,212 lines), 12 NPZ files (188,644
bytes), 2 CSV files (124,015 bytes, 730 lines), and 10 Python reproduction scripts
(23,786 bytes, 333 lines). Twenty-one text files contain `/Users/` or macOS
temporary-directory paths. These paths identify local machines/usernames, not
portable execution locations. No committed production checkpoint or full study
dataset was found in these two evidence directories.

The logs, arrays and detailed JSON/CSV dumps are generated validation results,
not small unit-test fixtures. Committing them in bulk was inconsistent with
keeping generated results outside the tracked source tree. No maintained code in `experiments/`, `analysis/`, `scripts/` or `tests/`
loads these archived dumps. The ten historical reproduction scripts under
`docs/` are optional diagnostics; reconstruct their archived inputs when needed. The maintained
NumPy oracle and synthetic complete-study fixture live in `tests/` and generate
their inputs rather than loading this evidence. Replacing removed dumps with
synthetic fixtures is unnecessary.

Cleanup preserves all summaries/provenance and the sole historical exact record:

- The complete 64-file bundle is already remotely accessible at
  [f9615de](https://github.com/kaves29/scaling-drl-research/tree/f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6/main/docs).
  No history is rewritten. A full copy can be reconstructed locally with
  `git archive f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6 main/docs/exp12_followup_evidence main/docs/exp12_approved_evidence`.
- Keep 15 small tracked files (39,922 bytes): ten reproduction scripts, both
  metadata files, both SHA256 inventories, and the per-commit file inventory.
  Fingerprints describe the archived bundle, not files promised to exist in
  every current working tree.
- Untrack 49 generated files (2,310,620 bytes) while preserving existing local
  copies. Ignore new generated log/NPZ/CSV/JSON files in these evidence folders.
- Retain the historical investigation and implementation reports with exact
  scientific/numerical summaries. Their evidence links point to the pinned
  historical snapshot. New raw review outputs remain outside the tracked tree.

This cleans future checkouts; historical Git objects, including machine paths,
remain accessible. It does not erase prior publication or reduce existing Git
history size. This is repository hygiene only, with no test/execution behavior
change and no evidence destruction.

## 7. Claude's abb9498

`git cat-file -t abb9498` reports that the object is unavailable. No merge or
cherry-pick is justified on an unseen local-only patch. Existing
`7d1bda6d6aeeea22fef1692be46bf56409ade38e` changes only methodology amendment (z),
recording both h1-run-v0 and h1-reach-v0 input dimensions, actor/per-Q/twin counts
and task-specific ratios. The current tree contains those changes. This supports
the sufficiency of our documentation, not a claim of patch equivalence with an
unavailable commit.

## 8. Persistent compilation cache

Inspection is of installed JAX 0.4.34, the pinned cluster version, not an
assumption about the latest release. `configure_compilation_cache` defaults to
`main/jax_cache/<device_kind>` and sets the compile-time/entry-size thresholds to
zero. Independent processes on the same model use the same directory. The
function's exact-path override is `EXP12_JAX_CACHE_DIR`; `off` returns without
clearing an independently inherited JAX cache setting.

| Caller | Current ownership |
|---|---|
| Block B ordinary lanes/packing workers | Inherit one shared model directory unless explicitly overridden; multiple independent writers can overlap |
| Block B identity cells | Explicit per-cell parent/arm cold directories; warm parent and arm sequentially reuse a cell's shared directory |
| Block HB ordinary tests/profile | Per-job/per-step override directories; a whole unittest step can still spawn multiple child processes inheriting that same directory |
| Block HB identity cells | Inherit caller/default cache; neither explicit per-cell cold/warm separation nor isolation from other jobs is established |
| Production manifests/claim launcher | No process-specific cache override added; Exp1 and injected-arm entry points configure the shared model directory |
| Single process | No concurrent self-writer across processes, but sequential restarts reuse the directory and can inherit an interrupted write |

At the default unlimited cache size, installed `LRUCache.get/put` use no lock.
`put` checks existence and directly writes the final cache file, without atomic
temporary-file publication; an existing file is not replaced. Each independent
training program is its own JAX process zero, so coordinated-distributed
rank-zero documentation does not serialize the independent workers.
[Pinned LRU source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/lru_cache.py).

A fresh local two-process probe paused the actual LRU writer after publishing
all but five bytes of a compressed placeholder, then read via another LRU
instance. Decompression failed with exactly `Error -5 while decompressing data:
incomplete or truncated stream`. Completing the write made the same entry
readable. A separate interrupted entry was not repaired by a later `put`.
This deliberately forced interleaving proves the mechanism, not its incidence
or the cause of the historical Delta warning. Storage failures or killed writers
can produce the same symptom.

[Pinned compiler source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/compiler.py)
catches read/decompression errors by default and recompiles; enabling
`jax_raise_persistent_cache_errors` instead propagates the exception. Corrupted
compressed bytes are not partially deserialized and executed. That fallback
does not prove CUDA autotuning will produce a bit-identical executable.
[Pinned serialization source](https://github.com/jax-ml/jax/blob/jax-v0.4.34/jax/_src/compilation_cache.py).

Minimal proposed design, **not implemented without approval**:

- Ordinary processes get a fresh private writable directory per process and
  restart, using a unique random name rather than PID alone across nodes. Repeated
  configuration within a process reuses its own directory. Record the actual
  directory/provenance. Avoid shared active writers and stale interrupted files.
- Preserve explicit identity cold/warm paths and ordering; no automatic rewrite
  of those cases. Qualify HB's explicit identity cache plan separately.
- If warm-start reuse is needed, copy a completed immutable template into each
  private writable cache only after its sole producer exits. Stock JAX writes
  atime sidecars on reads, so pointing workers at a truly read-only shared cache
  is not a complete solution. Template use is additional scope to review.
- Define genuine `off` behavior against inherited JAX settings and verify it in
  fresh processes; do not silently change the current off semantics here.
- Test simultaneous-process directory separation, repeated calls, restarts,
  explicit overrides/off, and preservation of cold/warm scenarios; then require
  CUDA original/control and 1,000-step identities plus cost/packing remeasurement.

Even without changing flags, dtype, precision or model code, replacing a warm
hit with a cold compile can change executable reuse/autotuning and wall time.
The user's instruction requires approval when there is any such possibility.
Therefore `precision.py`, Block B, the launcher and identity cache scenarios
remain unchanged. Claude's uncommitted file is unavailable and was not copied.

## 9. Commit/HB pin relationship

The exact chain at review start was:

`2dc8734fc79c04d44d5072644e9eb45f8389f90f`
→ `8d4cf46d09ddc8e03049156158c9a4b32fa0edf7`
→ `f9615de5a3beaa380ef2e11bccaa1d5d1a5161a6`.

`2dc8734` completed the implementation/test tree. `8d4cf46` preserved historical
audit evidence; `f9615de` added the final report and CPU evidence. Their complete
diff is confined to `main/docs/`; experiment/analysis/script/test/config source
was identical. The former HB command intentionally targeted a proposed separate
clean checkout at the implementation revision. That checkout was not created
and the command was not executed. The distinction was documented, but the two
hashes should not be treated interchangeably for exact qualification provenance.

The new spool fix is a code-changing successor. Future HB must use the final
clean source revision containing it, plus all approved numerical/helper/cache
work, and record that same full hash in the submission, Git check and evidence.
Neither `2dc8734` nor `f9615de` qualifies the new driver. No HB command was run.

## 10. Integrated readiness and verification

Current CPU validation results are recorded below after completion. These runs
cannot establish a CUDA result, freeze m, approve a new bound or change the
seed-inclusion/reporting convention.

Change inventory:

- `7fbdf14`: `main/scripts/sbatch_exp12_hb.sh` and
  `main/tests/test_exp12_hb_qualification.py` only; submission-directory fix and
  one additional regression, with no existing criterion weakened.
- The delivery report/hygiene commit: `main/.gitignore`, this report,
  `main/docs/exp12_approved_implementation_report.md`,
  `main/docs/exp12_followup_investigation.md`, and the 49 exact `Archive` entries
  in the inventory below. It changes no executable/test/config source. The full
  delivery hash and final Git status are reported in the accompanying response;
  a committed document cannot embed its own content-derived Git commit hash.
- No new changes were pushed. Origin remains at the already-published f9615de
  snapshot; the branch's new commits are local. No merge/history rewrite occurs.

### What older Block B can establish

The documented submitted revision is `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`.
No current job state, Delta log or result was fetched in this review. That revision
predates the final twin support, HumanoidBench episodic correction, original-to-
restored-control Check 1 guard, independent oracle/reporting and completeness
protection. Its optional HB cases cannot qualify the final twin implementation.

Useful provisional evidence can include single-Q DMC/MyoSuite calibration and
resource measurements, natural-trigger positive-control observations, range/null
checks, and cold/warm identity results under exactly their recorded source,
config, stack, model and cache conditions. Review raw statuses and quantities;
do not infer success from a driver exit or a submission record. These can inform
planning and proposed values, without constituting final-HEAD qualification or
freezing m. Inspect relevant single-Q code differences before transferring any
measurement, and repeat acceptance tests under the final execution configuration.

The numerical helper is unchanged and remains problematic: it converts leaves
to float64 before checking floating dtype, and NaN arithmetic can leave a
reported maximum at zero. It neither validates matching tree structure nor
treats counters/RNG as discrete. Single/twin update tests compare actor/critic/
target/temperature train-state leaves, agent RNG and common info arrays. Training
deviation measures only the agent checkpoint tree; its metadata-difference count
is context, not an acceptance assertion on replay, physics or normalization.
Consequently older scalar `NUMERICS` records are calibration candidates to inspect,
not authoritative bounds on complete state. No bad value should be approved
merely because it came from Block B.

The four unresolved bounds are:

- `diagnostics_update/tf32`
- `diagnostics_update/highest_deterministic`
- `diagnostics_training/tf32`
- `diagnostics_training/highest_deterministic`

Approve the helper's finite/mask/structure/discrete-state contract before adopting
them. A matching NaN should be allowed only for explicitly approved unavailable-
diagnostic fields; unexpected nonfinite values must not mask errors. The present
TF32 quantity compares diagnostics-on TF32 against diagnostics-off highest, so
it combines precision and instrumentation differences. Changing to a same-
precision isolation quantity also requires approval. The historical proposed
10-times/one-significant-digit rule is not an automatically adopted criterion.

The exact identity comparator covers agent leaves, observation normalization,
filled replay arrays, environment replay/RNG metadata, global RNG, counters,
meters/logs and selected extra state, plus panel equality. Its known omissions
include `buffer_meta.pkl` (cursor/count/n-step queue), symmetric agent path/key
accounting and unrecognized extra-state keys. Do not describe it as exhaustive
state identity. Correcting these acceptance checks requires the lead's decision.

### What HB currently schedules and what remains

The path fix does not add workloads. HB schedules:

- Actual h1-run pipeline integration; h1-reach evaluation/global-RNG restoration;
  the 23 tiny twin-critic tests, including the unchanged red panel test and the
  currently skipped GPU update-deviation test.
- Six exact identity cells: D4W1024 and D4W1536 × dog-run, myo-key-turn and
  h1-run-v0; seed 101, reduced 240,000 raw-step budget, forced trigger at check 2,
  snapshots exactly 1,000 further interaction steps.
- h1-run-v0 twin profiling for D2W512, D4W1024 and D4W1536: 600 timed steps,
  100 warmup steps and one probe repetition, recording speed/overhead/peak bytes.
- Main/HB environment and GPU-model checks, per-step statuses/timeouts and final
  fail-closed output validation.

HB does not schedule full-size injected Check 1 (its full-size arms are identity
arms), full-size h1-reach identities, explicit cold/warm pairs, both precision
modes/repetitions, the single-Q diagnostic tests, training-bound measurement,
twin training isolation, or complete final-tree CUDA suite/mutations. It also
does not establish full-grid packing, long-run performance/storage, production-
size resume, positive-control/range/null acceptance, or frozen m. No extra
expensive cases or acceptance exceptions were silently added.

| Gate | Current status / evidence required |
|---|---|
| HB submission-directory resolution and failure accounting | CPU verified; copied-script and negative bookkeeping cases pass; actual Slurm integration unrun |
| Confirmatory completeness and separate twin reporting | CPU verified with synthetic full populations/break cases; real frozen study/population not collected |
| Baoding exact environment replay | CPU verified; no assertion/tolerance changed |
| Twin independent oracle | Characterization reproduced; unchanged test/restored mutation RED; CPU criterion requires lead decision |
| One-seed reporting | Explicit stop remains; lead decision required for primary and secondary subsets |
| Cache ownership | Risk and local reproduction confirmed; design pending approval; CUDA executable/identity/cost effects unqualified |
| Deviation helper, state comparator coverage and four bounds | RED/unresolved; contract and measured criteria require lead approval |
| Final-tree CUDA identity/injected checks | Unverified for prescribed sizes/tasks/modes/cache plan on selected model |
| HB environment/integration and twin resource fit | CPU bookkeeping only; cloned stack, real GPU behavior/time/memory still need verification |
| Positive control, frozen m, range/null, packing/preflight | Final accepted evidence/lead-frozen m still required; Block B outcomes not assumed |
| Submission | Neither HB nor the full grid authorized/submitted in this review |

Before recommending **qualification HB submission**, require the conjunction of:

1. Reviewed independent numerical criterion and passing relevant restored CPU
   tests/mutations, plus approved helper/comparator coverage and required measured
   update bound; alternatively an explicit measurement-only authorization with
   no claim of qualification PASS.
2. A separate clean checkout at the exact final full hash; correct main/HB cloned
   stacks, pinned HB dependency, selected literal GPU model, EGL and captured
   provenance. Submission from that checkout's `main/`, no running B files reused.
3. Lead-approved disposition of missing injected/task/mode/repetition cases and
   explicit cold/warm/cache-isolation plan; honest skip/failure accounting.
4. Reviewed time/memory/storage feasibility and distinct output paths, followed
   by explicit job-submission authorization. No job is submitted by this report.

Before **full Exp1/2 grid launch**, additionally require:

1. Relevant final-tree CPU and CUDA suites/mutations pass with no unexplained
   failures/skips; approved helper/state accounting and four measured CUDA bounds
   pass reruns on the chosen model/stack, with declared single/twin coverage.
2. Exact original/control and 1,000-step identity checks at both scaled sizes
   across required suites/tasks and approved cache/mode cases; full-size injected
   Check 1 at the eventual frozen m. No CPU bound is silently reused on CUDA.
3. Accepted natural-trigger D4W1536 dog-run development positive control outside
   seeds 1–5, prescribed recovery/noise/shared-offset checks and lead-frozen m;
   accepted current-size range/null checks under unchanged stop rules.
4. Production-size save/resume preflight and accepted measured cost, memory,
   packing, concurrency, cache and storage behavior; distinct paths/claims and
   reviewed retry/recovery handling.
5. Independent 195-run study manifest, 130-parent census/provenance, unchanged
   complete observation grids, resolved n=1 reporting, reviewed B/HB failures
   and explicit full-grid authorization. Partial data remain exploratory only.

### CPU results

| Current validation | Result |
|---|---|
| Full Exp12 CPU discovery | 181 run: 178 passed, 1 failure, 0 errors, 2 HB skips; 680.715 s |
| New tests included in that run | 32 pass: completeness 19, HB plumbing 9, oracle arithmetic 2, twin reporting 2 |
| Focused older environment-state + determinism suites | 31 run, 31 pass, no failure/error/skip; 1.174 s |
| Standalone HB bookkeeping after formatting | 9 run, 9 pass, no failure/error/skip |
| Existing break/restore harness | 72 checks: 71 OK, 1 PROBLEM; expected process exit 1 retained |
| Fresh twin probe | Same 67 Q mismatches and 3 later gradient violations; separately compiled production functions match panel exactly |
| Independent numerical characterization | 15 panels, both modes, seven semantic mutations each; byte-identical to preserved characterization |
| Cache publication probe | Forced two-process read during write reproduces truncated-stream Error -5; interrupted existing entry is not repaired |
| Shell/format/diff/archive checks | Bash syntax, Black, Git whitespace, original-file fingerprints and archived-link Git-object checks pass |

The one failure is exactly
`tests.test_exp12_twin_critic.TwinCheck1Test.test_panel_values`.
The mutation PROBLEM is the Q1-instead-of-min gradient mutation: both mutated
and restored variants fail the existing test, so its restored gate is not green.
The skips are the real HB pipeline and Reach evaluation/RNG test because the
local HB package/environment is unavailable. These are not counted as passes.
The prior full older-suite 262/262 result is historical; this review reran the
31 relevant environment-state tests, not all unrelated older tests.

Validation used Python 3.12.13, JAX/JAXlib 0.4.34, CPU backend, the existing
research venv and the already-present rliable 1.2.0/arch 7.2.0 dependency overlay.
No dependencies were installed/upgraded. The executable/test tree is that of
7fbdf14; subsequent changes are documentation, ignore rules and untracking only.
Raw current outputs remain in `/tmp/exp12-claude-review/` on this machine;
summary/fingerprints are retained here without committing fresh raw dumps.

Commands, from `main/`, with environment
`JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
PYTHONPATH=/tmp/exp12-validation-deps MPLCONFIGDIR=/tmp/exp12-mpl`:

```text
../.venv/bin/python -m unittest discover -s tests -t . -p 'test_exp12_*.py' -v
../.venv/bin/python -m tests.exp12_break_checks
../.venv/bin/python -m unittest -v tests.test_exp12_hb_qualification
../.venv/bin/python -m unittest -v tests.test_angle2a_env_state tests.test_angle2a_env_state_determinism_smoke
../.venv/bin/python scripts/characterize_exp12_twin_numerics.py --out /tmp/exp12-claude-review/twin_characterization.json
```

For the optional archived twin probe, add `.:tests` to PYTHONPATH and supply an
output directory. The standalone cache probe used installed LRUCache in fresh
spawned CPU processes without compiling a model.

Setup errors are separate from final suite results: an initial backend probe
without `JAX_PLATFORMS=cpu` selected experimental Metal and aborted; it was
corrected to CPU. The first discovery invocation omitted the documented analysis
overlay (170 run, one known failure, 14 missing-rliable errors, two HB skips).
The corrected complete discovery result above supersedes that environment-only
failure. The first optional probe also needed the documented main/tests import
paths. No assertion or criterion was changed to resolve any setup error.

| Local current artifact | SHA256 |
|---|---|
| `cpu_suite_qualified_env.log` | `257dd6fe29a608944fa0419797abbdc3d40cda2d505cda7b14ffc98d798b0537` |
| `mutations.log` | `aa2637bc9bbf57df0947bd2489839e0332ec1e963a082ca58c04f3f899ad53cc` |
| `baoding.log` | `a685bcb2eb9b8938979fc811f2be634914d47051452467584fc9bd695ebe118b` |
| `hb_plumbing.log` | `a2130c822fa817577b6c8de703ebbed221fb48e74993d135d721050cb847d8ba` |
| `cache_race_probe.json` | `524abdaab5b2e3b4fb24402c2cb715d3f56dabac7b1239fad75896497daaa47a` |


## Exact evidence inventory at f9615de

Paths below are relative to `main/docs/`. `Archive` means untracked from the
current source tree with the exact version preserved at f9615de and in local
ignored copies; `Keep` means small tracked provenance or reproduction code.
No item is required as an input fixture by the maintained unittest suite.

| Path | Bytes | Text lines | macOS path | Disposition |
|---|---:|---:|---|---|
| `exp12_approved_evidence/commit_files.json` | 6,568 | 131 | No | Keep |
| `exp12_approved_evidence/exp12_cpu.log` | 337,991 | 2,175 | Yes | Archive |
| `exp12_approved_evidence/hb_plumbing_cpu.log` | 1,326 | 13 | No | Archive |
| `exp12_approved_evidence/metadata.json` | 1,878 | 57 | No | Keep |
| `exp12_approved_evidence/mutations_cpu.log` | 153,672 | 1,118 | Yes | Archive |
| `exp12_approved_evidence/older_cpu.log` | 312,825 | 1,589 | Yes | Archive |
| `exp12_approved_evidence/sha256.json` | 644 | 8 | No | Keep |
| `exp12_approved_evidence/twin_numpy_characterization.json` | 127,368 | 3,835 | No | Archive |
| `exp12_followup_evidence/analysis_acceptance.json` | 23,966 | 985 | No | Archive |
| `exp12_followup_evidence/analysis_acceptance_probe.py` | 5,386 | 65 | No | Keep |
| `exp12_followup_evidence/baoding_6ffccba.log` | 3,602 | 137 | Yes | Archive |
| `exp12_followup_evidence/baoding_6ffccba/baoding.json` | 18,690 | 721 | No | Archive |
| `exp12_followup_evidence/baoding_6ffccba/baoding_arrays.npz` | 5,588 | 0 | No | Archive |
| `exp12_followup_evidence/baoding_b554d45.log` | 3,602 | 137 | Yes | Archive |
| `exp12_followup_evidence/baoding_b554d45/baoding.json` | 18,690 | 721 | No | Archive |
| `exp12_followup_evidence/baoding_b554d45/baoding_arrays.npz` | 5,588 | 0 | No | Archive |
| `exp12_followup_evidence/baoding_current_1.log` | 3,632 | 137 | Yes | Archive |
| `exp12_followup_evidence/baoding_current_1/baoding.json` | 18,720 | 721 | Yes | Archive |
| `exp12_followup_evidence/baoding_current_1/baoding_arrays.npz` | 5,588 | 0 | No | Archive |
| `exp12_followup_evidence/baoding_current_2.log` | 3,632 | 137 | Yes | Archive |
| `exp12_followup_evidence/baoding_current_2/baoding.json` | 18,720 | 721 | Yes | Archive |
| `exp12_followup_evidence/baoding_current_2/baoding_arrays.npz` | 5,588 | 0 | No | Archive |
| `exp12_followup_evidence/baoding_differences.csv` | 10,081 | 61 | No | Archive |
| `exp12_followup_evidence/baoding_exact_summary.json` | 32,332 | 991 | Yes | Archive |
| `exp12_followup_evidence/baoding_original_6ffccba.log` | 2,864 | 43 | Yes | Archive |
| `exp12_followup_evidence/baoding_original_b554d45.log` | 2,864 | 43 | Yes | Archive |
| `exp12_followup_evidence/baoding_original_current_1.log` | 2,908 | 43 | Yes | Archive |
| `exp12_followup_evidence/baoding_original_current_2.log` | 2,908 | 43 | Yes | Archive |
| `exp12_followup_evidence/baoding_probe.py` | 2,843 | 38 | No | Keep |
| `exp12_followup_evidence/cuda_metric_probe.py` | 591 | 12 | No | Keep |
| `exp12_followup_evidence/cuda_metric_scope.json` | 95 | 5 | No | Archive |
| `exp12_followup_evidence/exp12_restore.json` | 698 | 31 | No | Archive |
| `exp12_followup_evidence/exp12_restore_probe.py` | 1,093 | 23 | No | Keep |
| `exp12_followup_evidence/exp1_ledger_acceptance.json` | 642 | 28 | No | Archive |
| `exp12_followup_evidence/exp1_ledger_probe.py` | 859 | 16 | No | Keep |
| `exp12_followup_evidence/metadata.json` | 1,592 | 65 | No | Keep |
| `exp12_followup_evidence/numpy_reference.py` | 3,044 | 46 | No | Keep |
| `exp12_followup_evidence/original_test_results.json` | 1,386 | 37 | Yes | Archive |
| `exp12_followup_evidence/pre_exp12_single_pair.npz` | 11,238 | 0 | No | Archive |
| `exp12_followup_evidence/pre_exp12_twin_probe.json` | 17,026 | 16 | No | Archive |
| `exp12_followup_evidence/pre_exp12_twin_probe.py` | 1,927 | 31 | No | Keep |
| `exp12_followup_evidence/run_originals.py` | 1,401 | 20 | Yes | Keep |
| `exp12_followup_evidence/sha256.json` | 5,454 | 57 | No | Keep |
| `exp12_followup_evidence/summarize_evidence.py` | 2,663 | 33 | No | Keep |
| `exp12_followup_evidence/twin_c293b9a.log` | 1,384 | 56 | No | Archive |
| `exp12_followup_evidence/twin_c293b9a/twin.json` | 292,081 | 10,015 | No | Archive |
| `exp12_followup_evidence/twin_c293b9a/twin_arrays.npz` | 32,216 | 0 | No | Archive |
| `exp12_followup_evidence/twin_c293b9a/twin_params.npz` | 15,892 | 0 | No | Archive |
| `exp12_followup_evidence/twin_current_1.log` | 1,384 | 56 | No | Archive |
| `exp12_followup_evidence/twin_current_1/numpy64_reference.json` | 1,126 | 37 | No | Archive |
| `exp12_followup_evidence/twin_current_1/numpy64_reference.npz` | 10,730 | 0 | No | Archive |
| `exp12_followup_evidence/twin_current_1/twin.json` | 292,111 | 10,015 | Yes | Archive |
| `exp12_followup_evidence/twin_current_1/twin_arrays.npz` | 32,216 | 0 | No | Archive |
| `exp12_followup_evidence/twin_current_1/twin_params.npz` | 15,892 | 0 | No | Archive |
| `exp12_followup_evidence/twin_current_2.log` | 1,384 | 56 | No | Archive |
| `exp12_followup_evidence/twin_current_2/twin.json` | 292,111 | 10,015 | Yes | Archive |
| `exp12_followup_evidence/twin_current_2/twin_arrays.npz` | 32,216 | 0 | No | Archive |
| `exp12_followup_evidence/twin_current_2/twin_params.npz` | 15,892 | 0 | No | Archive |
| `exp12_followup_evidence/twin_differences.csv` | 113,934 | 669 | No | Archive |
| `exp12_followup_evidence/twin_later_gradient_assertion.log` | 501 | 12 | No | Archive |
| `exp12_followup_evidence/twin_original_2f67915.log` | 1,892 | 32 | Yes | Archive |
| `exp12_followup_evidence/twin_original_current_1.log` | 1,914 | 32 | Yes | Archive |
| `exp12_followup_evidence/twin_original_current_2.log` | 1,914 | 32 | Yes | Archive |
| `exp12_followup_evidence/twin_probe.py` | 3,979 | 49 | No | Keep |
