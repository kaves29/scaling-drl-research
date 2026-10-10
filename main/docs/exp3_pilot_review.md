# Independent review of the Exp3 pilot implementation (2026-10-10)

Reviewed: `codex/exp3-pilot-implementation` @ `142a7fc92ec8d19ff58e6022c6c60af84bc9ed4d`, based on
`integration/exp12` @ `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. Review branch: `claude/exp3-pilot-review`.
No methodology is chosen or changed here, nothing is merged and no GPU job is submitted.

> **Round 2 update (same branch).** Two corrections to round 1, marked inline: C2 (exp2_arm *does* write an
> injected state at the fork step, then deletes it) and S1 (post-fork evaluations do not advance the training key).
> A new critical engineering defect (C3: the source recorder could not resume after a crash between checkpoints)
> is fixed and tested. The prospective capture plan is in [exp3_capture_plan.md](exp3_capture_plan.md), and the
> S1–S9 decision table is in [exp3_owner_decisions.md](exp3_owner_decisions.md).

**Specification caveat.** The owner's approved Pilot 1–3 specification text is not in the repository or in this
review session. The only written statements are Codex's `docs/exp3_pilots.md` and `exp3_implementation_plan.md`.
This review checks the code against those documents and against the owner's stated focus areas: causal validity,
matching, SAC gradients, per-state measurement, isolation, passive fidelity, ordering, keys, twin aggregation,
target assignment and interpretation. Points that depend on the exact approved wording are marked **(spec)**.

## Verdict
- **The core mechanics are sound.**
  - **Pilot 1:** per-state SAC gradients, matched base noise across actors, shared actor/Adam state across
    conditions, the retained entropy term and min-of-twins aggregation are all correct.
  - **Pilot 2:** passive delivery is faithful. A new positive control here shows that once the action-sampling key
    split is emulated, the passive U learner reproduces the *active* U learner bit for bit (CPU).
  - **Pilot 3:** target-policy assignment, shared base noise, the frozen evaluator and the twin fit are correct.
- **Two issues are critical:**
  - one causal-validity defect: Stage B accepted mismatched injection provenance. It is fixed on the review branch;
  - one feasibility problem: the existing Exp1/Exp2 entry points cannot produce the matched artifacts Pilots 1 and 3
    need, and no Exp2 forks exist yet. The documents present artifact reuse as conditional on compatibility; in
    practice almost nothing is reusable.
- **Before any execution,** several scientific decisions and a GPU qualification adapted to the production TF32
  precision are needed.

## Findings by priority

### CRITICAL
**C1. Stage B could pair an I stream with a passive I learner injected differently (fixed here).**
- *Defect:* `make_passive` (`runner.py`) checked the stream's arm label but not its injection. The recorder
  (`recording.py`) did not store the source arm's `(m, seed)`. A spec with a different `injection_m` or
  `injection_seed` was therefore accepted without error.
- *Consequence:* Stage B's injected-critic learner on the I stream would not be the critic that generated that
  stream, which breaks the causal pairing.
- *Fix:* the recorder now stores `trainer.extra_state["injection"]`, and the resume check includes it.
  `make_passive` requires that injection to equal the spec for `source="i"`, and `None` for a U stream. Codex's
  fixture provenance was extended accordingly.
- *Test:* `tests/test_exp3_review.py::test_stage_b_passive_injection_must_match_the_injected_source_arm`. With the
  guard removed (mutation), it fails.

**C2. Historical and as-designed Exp1/Exp2 runs cannot supply Pilot 1/3 matched U/I artifacts, or Pilot 2 streams.**
- *Only the latest state survives.* `commit_state_dir(..., keep_previous=False)` (`experiments/exp12/state.py:46`)
  deletes every routine state except LATEST. A finished cell therefore retains only:
  - the fork state (at the fork step);
  - the final control state, at `control_end_step = max(N, fork + 0.25N)` (`fork.fork_plan`);
  - the final injected state, at `arm_end_step = fork + 0.25N`.
- *Matched post-fork pairs are rare.* `require_matched` demands equal steps, so a pair exists only when
  `fork ≥ 0.75N`, and then only at the endpoint.
- *Fork-time Pilot 1 is unreachable.* That branch needs an injected checkpoint carrying injection provenance at the
  fork step. **Correction (round 2):** `exp2_arm` does write one, its first save after injection and Check 1
  (`if latest is None: save(trainer)`), but the next routine save deletes it. `require_matched(fork, fork, fork)`
  refuses (`test_fork_time_pilot1_needs_an_injected_fork_step_artifact_that_exp2_does_not_retain`). Retention
  (round 2) keeps it, and Pilot 1 then runs at fork time on real entry-point artifacts
  (`tests/test_exp3_capture.py`).
- *No streams exist.* Ordered U/I streams exist only if recording was enabled during the source run.
- *No Exp2 forks exist.* The Block B dev run never triggered, and no grid has run.
- *Discrepancy:* `exp3_pilots.md` says existing checkpoints "can be reused if compatibility passes". Structurally,
  only the fork state, and endpoints of late forks, can be.
- *Decisions needed (owner):*
  - Record streams and retain matched checkpoints during the future Exp2 runs, or authorize separate continuations
    later. The first is much cheaper, but retention changes what Exp2 runs write.
  - For fork-time Pilot 1, whether "I at fork" may be constructed as fork + existing injection (as Pilot 2 already
    does), justified by Check 1.

### SCIENTIFIC (owner decisions; nothing chosen here)
**S1. Passive vs active learners differ only in the actor-sampling key stream (demonstrated).**
- The active loop advances `agent._rng` once per training `sample_actions` call and per routine-evaluation action.
  **Correction (round 2):** post-fork evaluations do *not* advance it: `fork.post_fork_eval` evaluates under an
  isolated key and restores the training key (`experiments/exp12/fork.py:224-234`). Passive delivery advances it
  only through updates.
- With that one split emulated before each arrival, passive U equals active U bit for bit: actor, critic, target and
  temperature (`test_passive_fidelity_when_action_key_consumption_is_emulated`). This also proves that arrival
  order, recorded normalization, replay contents, matched sampling and the update kernel are faithful.
- Without the emulation, the passive and active learners diverge. Both passive arms still share keys, so the
  passive U vs passive I contrast stays matched.
- *Decision:* whether the passive protocol should reproduce the source key stream. Round 2 makes this possible
  without any counting model: the recorder now captures the source agent key at each arrival, and on the real Exp1
  entry point a passive U learner given those keys reproduces the retained active U snapshot bit for bit
  (`tests/test_exp3_capture.py`). A call-count capture was tried and rejected, because post-fork evaluations call
  `sample_actions` under an isolated key.

**S2. Measurement precision.**
- `run()` calls `set_matmul_precision()`, which is TF32 on GPU. Every Pilot 1 per-state gradient, cosine, norm and
  Q value, and every Pilot 3 fit, is therefore computed in TF32 on the A100.
- Only `updated_policy_arrays` and Check 1 use FP32 ("highest").
- Small or nearly orthogonal per-state signals (cosines, magnitude ratios) are the quantities most sensitive to
  TF32's 10-bit mantissa, and CPU tests (FP32) cannot reveal the effect.
- *Decision:* the precision for Exp3 *measurements*, kept separate from the training precision.

**S3. Which actor is measured, (spec).**
- Pilot 1 also measures the *fork* actor against *post-fork* U/I critics (actor label `f`).
- Pilot 3's final panel guidance is hard-coded to the fork actor with fork normalization.
- Confirm both against the approved specification.

**S4. Pilot 1 outcomes come from a single AdamW step** (lr 1e-4) followed by deterministic evaluation. They are
matched (same evaluation seed and key) but almost certainly uninformative. Codex labels them descriptive; any
longer horizon is a separate decision.

**S5. Pilot 1's fork-time Check 1 uses a new panel.** Its actions are sampled from the fork policy, not the approved
256-pair `fork/panel.npz`, contrary to the matrix row "fork Check1 needs original panel". Pilot 2 does use the
approved panel. Its `check1(pre, after, control=pre)` makes the control part trivially pass. That is reasonable
for a single process, but it should be stated.

**S6. Injection inputs for Pilot 2.**
- m is not frozen (EXPERIMENT_STATUS: "m is not frozen and cannot be chosen by an agent").
- `injection_seed` is a free integer, while `exp2_arm` injects with `seed = cfg.seed`.
- For Stage A to mirror Exp2, the owner should bind m to the frozen value and the seed to the cell seed. Stage B is
  now enforced by C1.

**S7. Further protocol inputs that need decisions.**
- `chunk_size` defines Pilot 1's noise draw: `fold_in(key, chunk_start)`. Results therefore depend on it; it must
  be fixed per protocol, as Codex notes.
- Pilot 3 trains post-fork critics on fork-replay batches.
- Manifest cells carry no architecture, though each (env, seed) can have two scaled parents.
- The MyoSuite pair is unselected (Codex flags this).

**S8. Pilot 3 target log density.** It is computed from the retained latent; it matches production's
`dist.log_prob(dist.sample())` on unsaturated draws (test), and deliberately differs at saturation. This is
acceptable and documented, but it is not literally the production critic target.

**S9. Put the approved Exp3 specification in the repository** (methodology authority) before execution, so that
reviews and qualification can cite it.

### ENGINEERING
**E1. The tiny GPU harness (`scripts/exp3_gpu_validation.py`) will very likely fail for numerical reasons, not
defects, and it omits the GPU-critical tests.**
- It enables TF32 and then runs `GuidanceTest` and `TargetTest`. Those compare separately compiled programs with
  CPU-oracle tolerances (`rtol=2e-5`) and bitwise `tree_equal`.
- It excludes `FixtureTest`: passive isolation and restart, recording on/off parity, and the end-to-end pilots.
- A FAIL would be uninterpretable, and a PASS would not cover the passive and recording paths.

**E2. Memory.** `make_passive` restores two full fork replays per source. Pilot 1 stores
`panel × actor_params × 4 B × 4` gradient arrays per actor (three actors). Measure both at real width before
choosing panel and chunk sizes.

**E3. Stream provenance** lacked the injection record (fixed with C1). Streams recorded before this fix would be
refused by the new guard, which is the intended behaviour.

**C3 (round 2; CRITICAL for production use, fixed). The source recorder could not resume after a crash between
checkpoints.**
- *Defect:* the recorder published a chunk whenever `chunk_size` arrivals were pending, which can happen between
  routine saves. After a timeout or preemption, the job restores the last routine save, but the committed stream
  is already ahead of it, so the recorder refuses to continue ("source state and committed stream cursor differ").
  Production Exp1 parents run across several Slurm segments, so every recorded designated parent would have died
  on its first mid-interval interruption.
- *Fix:* during source recording, arrivals are published only together with a routine save under
  `<run_dir>/state`. Validation snapshots elsewhere do not publish. The end-of-train flush is removed; the final
  save, or sealing at scope exit, publishes the tail. A parent that crashed before its fork starts its stream
  fresh at the fork, where the existing fork-step check still refuses any missing prefix.
- *Cost:* pending arrivals are bounded by the save interval (25,000 arrivals; about 180 MB for dog-run), not by
  `chunk_size`.
- *Test:* `test_recorder_resumes_after_a_crash_between_checkpoints`. It fails on the unfixed recorder with exactly
  that error, and passes after the fix, with the resumed stream and final state equal to the uninterrupted ones.
- *Codex test adjusted:* `test_process_local_source_recorder_restart_is_exact_and_restores_original_loop` saved
  its emulated checkpoint outside `<run_dir>/state`. It now saves to the routine root, where Exp1 and Exp2 resume
  from. Its assertions are unchanged.

### OPTIONAL
- **O1.** Require `sanity` in every Pilot 1 intervention list. It is the in-run check that the substitution path
  reproduces ordinary SAC. In a new test, direction and magnitude reduce to the ordinary gradient in both spaces
  when U equals I.
- **O2.** Pilot 1 receipts could record the per-chunk keys explicitly. They are derivable today.

## What was verified (CPU, JAX 0.4.34, `MUJOCO_GL=disable`, no EGL renderer in this container)
| Check | Result |
|---|---|
| Codex's suite at `142a7fc` (`tests.test_exp3_pilots`, `tests.test_exp3_streams`) | 39/39 OK, 469 s |
| No existing file modified by `142a7fc` | yes: additions only (`git diff --diff-filter=MDR` empty) |
| New `tests/test_exp3_review.py`: 6 guidance/target tests | OK |
| New: 6 passive tests on Codex's fixture | OK |
| Full suite on the review branch (Codex's 39, with the C1 fixture update, plus 12 review tests) | OK (the run executed 72 test cases, because the fixture class was also collected through an import, since corrected); review module alone: 12/12 OK |
| Mutation: C1 guard removed | the Stage B review test fails, as intended |

The new tests cover:
- base noise identical across different actors;
- per-state gradients independent of other states;
- the panel mean of per-state gradients equal to the ordinary gradient;
- direction and magnitude reducing to ordinary SAC when the signals coincide, in both spaces;
- interventions changing only the critic term;
- Pilot 3 log density and target equal to production on unsaturated draws;
- passive isolation: with no injection, the two arms stay bit-identical through every update;
- injection propagating to the actor;
- passive replay and normalization equal to the active source;
- the passive-fidelity positive control (S1);
- the Stage B injection guard;
- the unreachable fork-time Pilot 1 path.

## Round 2 verification (CPU)
| Check | Result |
|---|---|
| Real-entry-point capture (`tests/test_exp3_capture.py`, 6 tests): real `exp1.run` with a forced dev trigger plus a real injected `exp2_arm.run`, both recorded and retained | 6/6 OK. C1 provenance from the real arms; retained steps exactly as predicted; routine roots unchanged; lean snapshots pair; Pilot 1 runs at fork time and post-fork; **recorded parent bit-identical to an unrecorded parent**; passive U with captured keys equals the retained active U, and without them does not |
| C3 crash-between-checkpoints test | fails on the unfixed recorder ("source state and committed stream cursor differ"); OK after the fix |
| C1 mutation (guard removed), real-entry-point test | the test fails, as intended |
| Harness self-checks (`tests/test_exp3_gpu_harness.py`) | 6/6 OK |
| Full Exp3 suite (Codex's 39 + review 13 + capture 6 + harness 6) | **64/64 OK**, 814 s (log SHA-256 `f557c7f2…e8c77e`) |

## Minimum GPU validation before any execution (not submitted)
1. **Harness repair (done in round 2, `scripts/exp3_gpu_validation.py`):**
   - bitwise exactness suites under each requested precision;
   - oracle comparisons and TF32 sensitivity, measured and reported only;
   - a repeat-determinism probe that separates GPU nondeterminism from defects.
2. **GPU determinism of the pairing premise:** two passive runs from one fork must be bit-identical, and U vs
   uninjected I must be bit-identical, on the A100. If they are not, the passive U−I contrast contains GPU noise,
   and its size must be measured before any interpretation.
3. **One real-width cell** (e.g. dog-run D4W1536, from an actual or authorized fork): restore and Check 1; one Pilot
   1 chunk at the proposed chunk size; a short Pilot 2 continuation, recorded with recording on/off parity; a few
   Pilot 3 updates. Report peak device and host memory, timings and storage per arrival.
4. Separately, MyoSuite restore and evaluation, and HumanoidBench twins if Exp3 uses them.

None of these qualify scientific conclusions. They establish that the instruments measure what they claim on the
production hardware.
