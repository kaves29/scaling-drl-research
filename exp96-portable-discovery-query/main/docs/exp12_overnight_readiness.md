# Exp1/Exp2 overnight runtime and readiness audit

Original investigation date: 2026-10-06 (America/New_York). That investigation
executed on the Cloud CPU machine. No Delta access, Slurm submission, production training,
GPU execution, methodological change, tolerance change, merge, or push occurred.

**The 195-run grid is not computationally qualified. The six-hour timeout is
not solved.** Check 00 proves fresh probing completed, not meaningful SAC
progress or absence of natural degradation. `SKIPPED_NO_TRIGGER` after an
incomplete parent is an orchestration outcome, not a scientific null result.

## 1. Source provenance and authoritative inputs

The initial Cloud checkout was on `work` at
`6ffccbaf11bd8f60ce663b59ac546e9ff7cffd9c`. Before investigation changes, native
HTTPS Git access verified the requested remote branch, fetched its latest tip,
and checked out **`claude/eloquent-fermat-inxqlt`** at
**`ca95b1e4ea9262eb9abde406848769349eeb8e17`**. The checkout was clean.
This is the investigation's exact starting HEAD. Fetching the named branch
populated FETCH_HEAD rather than a remote-tracking ref in this narrow checkout;
the local named branch was created from that verified FETCH_HEAD.

Consulted inputs:

- `CLAUDE.md` and `.claude/methodology-exp1-exp2.md`, including amendments
  (a)–(z). This Exp12 document supersedes the older Angle architecture/UTD rules.
- `.claude/{compute-and-data-safety,testing-and-verification,jax-rl-safety,
  experimental-isolation,reproducibility}.md`.
- `docs/exp12_runtime_investigation.md` and `docs/exp12_claude_review.md`.
- Relevant decision/qualification and historical performance sections of
  `docs/exp12_decisions.md`, `docs/exp12_cuda_commands.md`,
  `docs/exp12_coverage_audit.md`, `docs/exp12_master_summary.md`, and
  `docs/exp12_approved_implementation_report.md`.
- Current configs, manifests, Block B/HB scripts, entry points, trainer,
  SAC scan, probes, checkpoint/state, fork, instrumentation, and test helpers.

Later amendments and the approved implementation report supersede historical
claims of single-Q HB, D6W1536, incomplete reporting, and older test counts.
Nothing in the historical reports supplies missing current A100 evidence.

## 2. Identity workload: exact decomposition

Call chain: `scripts/exp12_blockB.sh:step_identity` → `run.py` →
`experiments/exp1.py:run` → `Exp12Trainer.start/train` →
`RunProbes.capture_fresh/maybe_check` → `run_probe_networks` →
`probe.run_probe/probe_round/_fit`; fork creation goes through
`fork.write_fork`, `trainer.save`, control reconstruction/restore and Check1.
The second process is `experiments/exp2_arm.py:run`, with `fork.arm=identity`.
`scripts/compare_identity_fork.py:compare` compares the resulting snapshots.
Block HB uses the same identity budget/forced-fork path.

The configured raw budget is 240,000, hence N=120,000 interactions, with
5,000 replay warm-up, UTD=2, checks every 6,000 interactions, forced check 2,
and snapshots 1,000 interactions after fork. Default CLI checkpoint settings
do not cause additional routine saves before the validation stops.

| Component | SAC updates | Probe optimizer updates, single Q | Purpose and identity relevance |
|---|---:|---:|---|
| Start, five eval episodes, train reset, replay warm-up to interaction 5,000 | 0 before fresh callback | 0 | Creates a meaningful simulator/replay/RNG/normalization state. Startup evaluation is inherited from the pipeline and consumes state; its exact cost is not intrinsic to the identity assertion. |
| Fresh parameter save and check 0, before the first SAC update | 0 | 5×1,000 = 5,000 | Establishes the scientific fresh reference. Required for the complete Exp1 path, not for parameter/optimizer/replay continuation alone. |
| Learning from interaction 5,000 through 6,000 inclusive | 2,002 | 0 | Nonzero optimizer/target/temperature state is essential to a useful restore test; this exact prefix length is inherited setup. |
| Paired check 1 at 6,000 | 0 | 2×5×1,000 = 10,000 | Current/fresh probe records and first forced firing check exercise pipeline state. Not intrinsically required to prove exact SAC continuation. |
| Learning 6,001 through 12,000 | 12,000 | 0 | Advances to the second scheduled check; the length follows the validation budget/check schedule. |
| Paired check 2 at 12,000 | 0 | 10,000 | Supplies the second firing check, completing the approved consecutive-check fork gate. Not a natural-trigger measurement because firing is forced in dev mode. |
| Fork state, fixed panel, original Check1; rebuild/restore control; control Check1; eval index 0, ten episodes | 0 | 0 | Complete save/restore and original-to-control equality are essential. Trigger/probe scheduling and evaluation integration additionally qualify the full pipeline. |
| Restored control from 12,001 to 13,000, then identity_control save | 2,000 | 0 | One required 1,000-interaction exact continuation. No check 3 or next post-fork evaluation is due. |
| Identity process: restore same fork; Check1; eval index 0; initial arm save | 0 | 0 | Independent restoration and state isolation are essential. Initial arm save and eval are pipeline/resume integration, beyond a bare numeric continuation test. |
| Identity arm 12,001 to 13,000, then identity_identity save and compare | 2,000 | 0 | Second 1,000-interaction continuation and complete-state comparison are essential. No injection or Check2 occurs. |
| **Total** | **18,002** | **25,000** | Parent 16,002 SAC; identity arm 2,000 SAC. |

The next post-fork eval is at 13,200; check 3 is at 18,000. Both snapshots
are at 13,000. Startup/base eval plus two fork evaluations total 25 episodes.
There are four full state writes, two full agent restores, and one fresh
reference write. Structural metrics run at the 2,000-interaction boundaries;
bootstrap resampling is statistical work, not SAC/probe optimizer updates.
Current HB has two Q networks, so the same 18,002 SAC *iterations* operate on
twins and the probe fits total **50,000**, rather than 25,000, updates.

This is confirmed inherited full-pipeline setup, not an accidental second UTD.
Removing the prefix/probes changes which integration properties the gate tests,
even though exact restoration can be tested with less setup.

**Proposed minimal focused validation, not implemented as a replacement:** use
the production architecture and a nontrivial trained, mid-episode state with
nonempty optimizer moments, target state, replay cursor/queue, diagnostic window,
normalization and all RNGs; save the original panel and complete fork state;
restore both independent continuations through the production restore path;
retain the approved 1,000-interaction horizon and exact comparisons of every
state component and Check1. A separate focused gate need not reproduce a forced
trigger or three expensive probe checks to prove that assertion. Keep existing
pipeline/probe-isolation coverage. Replacing the current qualification gate,
changing the validation starting trajectory or its probe/eval schedule, or
dropping its integration claims requires the lead's approval. No work was removed.

## 3. Full step accounting

`configs/base_exp12.yaml` resolves
`N = B_raw / (num_train_envs × action_repeat)`.
`experiments/exp12/envs.py:create_envs` rejects train/eval counts other than one.
`RepeatAction.step` advances up to two underlying simulator steps per interaction,
stopping early on termination/truncation. Recorded raw steps are the *nominal*
counter `2 × interaction_step`; actual physics steps can therefore be smaller.
This is existing approved step accounting, not literal measurement of all physics
calls. No counter convention was changed.

`trainer.train` adds one transition per interaction. With n_step=1,
`buffer.can_sample` becomes true after adding transition 5,000, on that same
iteration. Fresh probing precedes its first update. The update accumulator adds
2 once; exactly two replay batches of 256 are sampled and stacked; one
`agent.update_many` call delegates to `_update_sac_networks_scan`, whose leading
axis/scan length is 2. **The scan does not apply UTD again.** Each iteration
updates actor, critic, target, and temperature; these are constituents of one
SAC update, not four independent applications of UTD.

For any learned prefix K≥5,000:

`U(K) = 2(K − 5,000 + 1)`.

| Suite/budget class | Raw B | Interactions N | Nominal SAC updates U(N) | Check interval N/20 |
|---|---:|---:|---:|---:|
| Medium DMC, two tasks | 500,000 | 250,000 | 490,002 | 12,500 |
| Hard DMC + MyoSuite, nine tasks | 1,000,000 | 500,000 | 990,002 | 25,000 |
| HB, two tasks | 2,000,000 | 1,000,000 | 1,990,002 | 50,000 |

Three architectures × thirteen environments × five seeds = 195 parents.
Per architecture: 35M interactions across 65 parents, 69,350,130 SAC updates.
Total: **210M nominal raw steps, 105M interactions, 208,050,390 SAC updates**.
Twin HB adds network computation per iteration, not environment/update counts.

Twenty scheduled checks plus one fresh check are distinct from training.
For q Q networks, a scheduled check fits `2q×5×1,000 = 10,000q`
optimizer updates; fresh check fits 5,000q. Nominal total is 205,000q per parent.
Per architecture: 55 single-Q parents and 10 twin parents, hence 15,375,000
probe fit updates; all three architectures total **46,125,000**.
Each probe fit has batch 256, so these are 11,808,000,000 fit-example visits,
plus target/full-pool passes. These counts are not additional SAC learning.

Actor-gradient cosine executes on SAC update numbers divisible by 30. It adds
diagnostic gradient computation, not optimizer updates. Metrics sample replay
at every 2,000 interactions; evaluation uses separate environments. Their extra
replay/RNG consumption is part of the existing trajectory and cannot be silently
removed. Probe streams are separately keyed by seed/check/round.

Nominal base evaluation is five episodes at startup and each 50k boundary,
with fifty episodes at the final boundary: 75 / 100 / 150 episodes for the
250k / 500k / 1M interaction budgets. Across the nominal grid: **20,250 episodes**.
Late control extension also receives the existing final-multiplier evaluations
on each subsequent 50k boundary; it must be counted, not assumed free.

Production manifests explicitly set checkpoint_interval=N/20 and start_frac=0.
No-fork parents write 19 routine states plus the final state, retaining only the
latest routine generation. A fork substitutes an immutable fork save for the
routine save at its check, restores control, and sets
`Kc=max(N,f+N/4)`. Parent checks continue to Kc on the original N/20 grid.
Injected arm trains N/4 interactions, i.e. N/2 SAC updates, with five later
paired checks. Its initial state + four intermediate + final state gives six
complete saves. Check2 fits injected/control/fresh: 15,000q additional probe
updates. Both arms evaluate at indices 0…25, ten episodes each: 520 episodes
per eligible fork pair. Base control evals remain additional.

Inspected paths rule out a hidden multiplicative UTD/scan/vectorization factor.
No budget/UTD/architecture/precision discrepancy with the latest amendments was
identified in this accounting. This does not establish CUDA execution speed.

## 4. Plasticity probe performance

`RunProbes` creates optimizer transformations once per instance. `_base_targets`
and `_fit` are module-level JITs, with architecture/optimizer/chunk static and
params/pool/keys/indices dynamic. `_fit` runs one `lax.scan` over 1,000 batches;
optimizer state is correctly fresh per fit. It does not reconstruct a Python
model or optimizer transformation inside that scan. Target parameters use the
same original architecture even in injected arms.

Each round samples its pool from filled replay with its own NumPy generator,
normalizes it using current training statistics, transfers observations/actions
to device, generates one shared random-function target and one minibatch order,
and fits every current/fresh Q separately. Individual offsets, two full-pool
passes per fit, the same target/minibatches across paired fits, five distinct
round keys, and twin averaging are required by the approved procedure.
`run_probe` transfers returned curves/scalars together after all rounds; fitted
weights are not serialized. Targets are not independently regenerated for Q1/Q2.

New production-shape **lowering only**, on CPU; no production fitting executed:

| Critic, dog 223/38 | Parameters/Q | StableHLO bytes, 10→1,000 steps | Dots | Whiles |
|---|---:|---:|---:|---:|
| D2W512 | 4,337,153 | 141,855→141,883 | 29→29 | 3→3 |
| D4W1024 | 33,854,465 | 229,071→229,099 | 49→49 | 3→3 |
| D4W1536 | 75,947,521 | 229,071→229,099 | 49→49 | 3→3 |

The 1,000-step fit is not unrolled into 1,000 Python copies. Changing check/round
numbers does not itself change a static JIT argument. New optimizer function
identities after reconstruction, committed restored arrays/sharding, injection
structure, or changed abstract input types can retrace/compile. This is different
from recompiling on every optimizer iteration.

Twin expansion slices both required Q networks and fits them separately on shared
inputs. Vectorizing all fits, fusing reductions/normalization, changing PRNG
generation, donation, or skipping the fresh reference might change floating-point
execution or isolation. No such optimization was implemented. Model/optimizer
state copies and host transfers remain hypotheses to time on CUDA, not claimed
hundreds-of-hours bugs. The fit cost is substantial by design: 5,000 probe
updates per fitted Q/check and many dense forward/backward evaluations.

## 5. Checkpoint restore, exactness, and size

`git show ca95b1e^:main/scale_rl/agents/sac/sac_agent.py` confirms the old typed
restore loaded actor/critic/target/temperature/key state, then an unconstrained
second *complete* restore recovered `churn_ref_batch`. The current implementation
reads that reference's metadata, supplies its real shape/dtype/sharding in the
typed target, and makes one payload restore. Metadata inspection is not a second
payload restore. Avoided API traffic is not proof of doubled physical disk reads
or an A100 speedup: TensorStore/OS caches and filesystem latency matter.

The restore audit covers:

| State | Persistence/restore path |
|---|---|
| Actor, all online Qs, all target Qs, Adam moments/counts, temperature and optimizer, model counters, JAX key, churn batch | SACAgent save/load via Orbax typed TrainState tree; twin Q axes preserved |
| Observation mean/variance/count | ObservationNormalizer's `obs_rms.pkl`, required on restore |
| Filled replay arrays | `state.save_buffer/load_buffer`, `buffer.npz` |
| Replay cursor, filled count, n-step queue | `buffer_meta.pkl`, restored without resampling |
| Interaction/update/accumulator counters, current observations/timestep | trainer `meta.pkl` |
| Both environments' reset RNG + action history, vector action-space RNG | ExactRestore reset/action replay; global RNGs restored afterward |
| NumPy/Python RNG, meters/media/log rows, probe/fork/eval records, diagnostic reference/window, agent window buffers | trainer metadata and diagnostics load_state |
| Injected structure and twin metadata | Injection seed/m in extra_state rebuilds matching structure before payload restore; critic layout/config/runtime records preserve count |

New adversarial twin restore coverage uses a trained twin agent and a saved
three-row churn batch, rather than assuming the normal reference shape. One
payload restore, every model/optimizer/counter/key leaf, and the next paired
updates remain exactly equal. Existing original/control, single/twin identity,
and interruption/resume tests are rerun separately from the known red numerical
oracle. No required state was removed.

**New confirmed comparison defect, fixed:** `state_differences` did not read
`buffer_meta.pkl`; it also zipped agent leaves without matching paths, and
checked metadata/normalization/replay keys only from one side. The new comparator
checks the tree structure and matching paths, both key sets, and replay metadata.
Existing numeric assertions/tolerances are unchanged. Valid identical states
still compare equal; cursor/count/queue and same-leaf-count key substitutions
now fail. This repairs coverage of the existing exactness requirement, not the
training procedure or scientific criteria.

For P float32 parameters per Q, online weights + two Adam moments + target
weights require approximately **16P bytes per Q**. Dog replay costs
`4×(2×223+38+3)=1,948` bytes/transition. Only filled entries are saved, not the
unfilled million-transition allocation.

| Dog critic | Critic state bytes | Replay at 500k | Approximate combined bytes |
|---|---:|---:|---:|
| D2W512 | 69,394,448 | 974,000,000 | 1,043,394,448 |
| D4W1024 | 541,671,440 | 974,000,000 | 1,515,671,440 |
| D4W1536 | 1,215,160,336 | 974,000,000 | 2,189,160,336 |

These exclude small actor/metadata overhead, checkpoint-format overhead and
injected frozen/new-head state. Even tiny D4W1536 checkpoints legitimately exceed
1GB because the critic itself dominates. Fresh dog D4W1536 parameters add
303,790,084 bytes once. HB D4W1536 critic state is approximately 2.42GB before
replay. At the identity fork dog replay is 23,376,000 bytes (12k transitions).

Temporary coexistence of old/new routine state protects atomic LATEST recovery;
immutable fork and fresh reference remain. Repeated CSV/ledger/curve exports and
large metadata/optimizer writes amplify storage work. Their actual Delta cost is
unmeasured. Asynchronous checkpointing or omitting state needs a recovery design;
neither was implemented. Routine checkpointing cannot explain the fresh dev stall
before check1, since the first periodic save follows that check.

## 6. JAX 0.4.34 compilation and persistent caches

Inspected the installed, pinned `jax/_src/{lru_cache,compiler,compilation_cache}.py`.
Default cache max_size is -1. With eviction disabled, LRU get/put do not lock;
put writes the final path directly, and does not replace an existing entry.
Each independent job is JAX process zero; rank-zero-only publication does not
serialize unrelated workers. Cache reads also write access-time sidecars.

New controlled two-process experiment paused the real LRU writer after a flushed
partial publication. The reader reproduced **`Error -5 while decompressing data:
incomplete or truncated stream`**. Completing the write allowed a correct read;
an independently interrupted existing entry was not repaired by another put.
This proves the mechanism, not occurrence/rate in Block B. Compiler `_cache_read`
catches such failures and falls back to compilation unless the existing
raise-persistent-cache-errors setting is enabled. Failed compressed artifacts are
not partly deserialized and executed; fallback compilation/autotuning is still
not an exact-CUDA-result guarantee.

| Execution | Writers/reuse topology |
|---|---|
| Block B dev/preflight/null/tests/packing/speed | Default main/jax_cache/device_kind unless caller overrides; lanes and packing workers can overlap |
| Block B cold identity | Private cell cache_parent/cache_arm; sequential processes |
| Block B warm identity | New cell cache_shared; arm reuses that parent only after it completes |
| Block HB ordinary tests/profile | Per-job/per-step path; spawned subprocesses within a test step inherit/share it |
| Block HB identity | Caller/default path, without explicit per-cell cold/warm override |
| Production manifests + claim launcher | Child inherits caller cache settings; no process-specific cache assigned |
| Ordinary single process | No concurrent independent self-writers, but restarted processes can encounter interrupted entries |

Cold identities already use separate directories, so concurrent shared-cache
writing is not a necessary explanation of all cold identity timeouts. A warm
identity *parent* starts with a new directory; its timeout does not measure warm
arm reuse. Private directories can still suffer filesystem latency or expensive
cold compilation.

Per-process private writable caches eliminate the overlapping-writer mechanism.
They also replace executable reuse with independent compilation/autotuning;
CPU equality cannot establish CUDA numerical equivalence. Therefore **no cache
layout rollout** was made. Proposed qualification: retain explicit identity
scenarios, record unique ordinary-process cache paths, test same-process reuse,
restart separation, sequential completed-template reuse and independent cold
compilation, then require the original exact CUDA identity criterion. Do not
silently use a read-only shared directory: reads write atime sidecars. The current
`off` sentinel returns early and does not clear an inherited JAX cache config;
that issue is recorded, not silently changed.

The new tiny CPU trace exercised a single-writer cache, then a sequential warm
injected process: parent 425 reads/30 hits/395 backend compiles; arm 290 reads/
282 hits/8 backend compiles, zero read errors. This demonstrates cache event
coverage and reuse on this CPU setup, not speed or exact qualification on A100.

## 7. Preflight versus production scaling

The reported 391s preflight and >21,600s incomplete dev attempt differ by >55.24×
wall time. Their workloads differ substantially:

| Quantity | D4 scaled preflight, parent+arm | Nominal dog dev | Ratio |
|---|---:|---:|---:|
| Raw parent budget | 4,000 | 1,000,000 | 250× |
| Replay warm-up | 50 | 5,000 | 100× |
| SAC updates | 4,902 | 990,002 before extension | 201.959× |
| Probe steps/round | 10 | 1,000 | 100× |
| Probe batch | 32 | 256 | 8× |
| Pool | 256 | 25,600 | 100× |
| Probe chunk / chunks | 256 / 1 | 2,560 / 10 | 10× / 10× |
| Probe optimizer updates | 2,700 | 205,000 | 75.926× |
| Probe fit-example visits | 86,400 | 52,480,000 | 607.407× |
| Complete saves | 14 | 20 nominal, including applicable fork substitution | Size/cadence dependent |
| Eval episodes | 57 | 100 nominal + conditional fork/extension evals | Episode/suite dependent |

Preflight's 2,700 probe updates = fresh50 + parent2,000 + arm500 + Check2 150.
SAC = parent3,902 + arm1,000. Through *dev check1* only: 40,002 SAC updates
(8.160× the entire preflight), and 15,000 probe updates ×256 = 3,840,000 fit
examples (44.444× the entire preflight). Through check0 alone: 1,280,000 fit
examples (14.815×), but zero completed SAC updates before that callback returns.
Dev startup has five eval episodes and no routine checkpoint before check1;
preflight's much greater save/eval count cannot be attributed to that dev interval.

Runtime cannot be obtained by multiplying 391s by any single count ratio. A
weighted model is

`R = (Σ n_production,j × measured_cost_production,j + compile_production)
     / (Σ n_preflight,j × measured_cost_preflight,j + compile_preflight)`.

Probe batch/pool shapes change device efficiency; saves include the same large
model but different replay; startup and eval do not scale with training budget.
No per-stage preflight/Delta measurements are available, so neither a numerical
expected wall ratio nor an honest residual number can be computed. The residual
is `T_observed − T_predicted(stage measurements)` and remains unmeasured.

If the reported null really completed 100 pairs at each of three sizes in 6,042s,
the mixed-size mean is 20.14s/pair and each size's average is at most 60.42s/pair
under nonnegative work. These are aggregate-average bounds, not a maximum for
one check, cold compile bounds, or production rates. They discourage assuming
hours of ordinary steady-state probe work, but do not identify the dev stall.
The latest established stage remains fresh check0. Up to 40,002 updates could
have occurred before/inside check1, or the process could have stalled on the first
SAC compile. Missing Delta artifacts prevent a sharper conclusion.

## 8. Runtime/progress instrumentation

The trace CLI remains strictly opt-in; ordinary training does not import/install
its monkeypatch wrappers. No PRNG key is generated/advanced by instrumentation,
and wrapper arguments/results and the original computation are forwarded.

New engineering changes:

- Flush `trace_start` and `backend_init` begin **before** `jax.devices()`.
  Backend initialization previously happened before the first record; a startup
  failure left an empty trace and an unclosed file. Failures now record their
  type/time and close the trace.
- Add monotonic timestamps, PID, stage call indices and probe check indices;
  retain wall timestamps, durations and flushed JSONL progress.
- Time probe-object/fresh-reference initialization and injection setup explicitly.
- Record cache-directory returns, cache hits/misses, and post-fork episode counts.
- Preserve an original training exception when final progress/synchronization
  also fails; log `progress_error` instead of replacing the original failure.

Coverage: backend compilation/module names; trainer/start/eval; environment steps;
replay sample; SAC updates; targets/fit/each round/whole check; structural metrics;
complete/agent/replay/meta save/restore; fork; original/control/injected Check1;
Check2; injection; post-fork evaluations; persistent-cache reads/writes/init.
Progress includes interaction/env/update counters, completed check index, eval
episodes, and accumulated stage calls/time. Frequent fine stages emit their first
three calls plus aggregate progress; coarse stages emit every begin/end.

`--synchronize` blocks on returned JAX work and the trainer's actor/critic/target/
temperature/key state at boundaries. This makes diagnostic device timing meaningful
but removes asynchronous overlap and adds barriers; it is not uninstrumented
throughput. Without it, inner device calls can measure dispatch. Whole probes
already return host data. Nested totals overlap; do not sum backend_compile,
update_many, probe_round, _fit and synchronization into an independent wall total.
Use a later paired unsynchronized run to quantify overhead. Events are flushed
to the OS, not fsynced durable storage against machine/storage loss. Python/import
and hardware-detection work before entering RuntimeTrace remains outside its
backend_init timer; capture external wall time and stderr too.

New equivalence test compares real tiny probed, trained and injected persisted
state with/without synchronized tracing, including replay metadata bytes. It
asserts 42 interactions, 66 SAC updates, two scheduled checks, 15 rounds and one
injection. The CLI smoke used separate tiny dev parent/injected processes with
400/100 interactions, full functional fork/Check1/Check2, 5×20 test probes,
64-pool/16-batch, and a 20-step test horizon. Both reached DONE. Totals:

| Stage | Parent calls | Injected arm calls |
|---|---:|---:|
| SAC update_many (two updates each) | 391 | 100 |
| Probe rounds / individual fits | 105 / 205 | 30 / 65 |
| Complete saves / restores | 2 / 1 | 2 / 1 |
| Injection / Check2 | 0 / 0 | 1 / 1 |
| Check1 | 1 | 2 |
| Post-fork evaluations | 26 | 26 |

All instrumentation smoke timings are local CPU plumbing evidence. These small
fixture settings were not substituted for approved production checks.

## 9. Confirmed bugs, fixes and broader bottlenecks

| Finding | Classification and scope | Action |
|---|---|---|
| Identity workload is full pipeline plus 1k continuation | **CONFIRMED ROOT CAUSE** of the misleadingly small workload description; not the one-hour timeout cause | Exact decomposition; gate unchanged |
| Duplicate full agent restore | **CONFIRMED CONTRIBUTING FACTOR**, inherited I/O bug; already fixed at starting HEAD | Re-audited old/current source; new twin/reference-shape adversarial test |
| Identity comparison omits replay metadata and can miss structure/extra keys | **CONFIRMED ROOT CAUSE** of validation blind spots; unrelated to dev runtime | Fixed comparator; adversarial regressions |
| Backend-init trace opacity, missing injection/probe-init/cache outcome/progress fields, exception masking | **CONFIRMED ROOT CAUSE** of instrumentation gaps; not the six-hour timeout | Fixed and exercised |
| Unlocked partial cache publication and unrepaired interrupted entries | **CONFIRMED CONTRIBUTING FACTOR** as a reproducible recompilation mechanism; **UNRESOLVED** incidence/cost in Delta | Controlled two-process repro; no default-policy rollout |
| Stale claim takeover race | **CONFIRMED ROOT CAUSE** of duplicate ownership under a forced stale-worker interleaving; **UNRESOLVED** campaign incidence | Local repro; no launcher rewrite |
| SAC scan length/UTD/vectorization accidentally multiply work | **RULED OUT** in inspected production path | One env, one accumulator addition, scan2 |
| Probe fit is Python-unrolled 1,000 times | **RULED OUT** | Production-shape lowering |
| Periodic saves, fork restores, Check2 or post-fork eval cause the fresh dev pre-check1 stall | **RULED OUT** under the reported fresh-run/check0-only assumptions | These stages have not yet been reached |
| SAC compile/steady throughput, environment/replay, metric-window transfers/SVD, cache FS latency or memory pressure cause the dev stall | **UNRESOLVED** | Bounded stage trace / missing artifacts needed |

Broader review found these material risks:

- During random warm-up, policy sampling still runs after the first interaction
  before the action is replaced by a random action. It updates normalization and
  consumes the existing agent RNG. Removing that computation alone changes the
  trajectory; it is not an authorized equivalent shortcut. It precedes check0.
- Policy action selection depends on the latest update and reads back to the
  CPU, intentionally serializing environment/action/training. Acting with stale
  parameters or changing vectorization would change the algorithm.
- Replay advanced indexing followed by `np.array` copies, repeated KL reference
  uploads, array stacking/host normalization, pending metric transfers and eager
  structural metrics/SVD deserve CUDA measurement. Source establishes the work,
  not its contribution to the six-hour stall. No speculative rewrite was made.
- Control rebuilds environments/models/optimizers after fork; injected heads add
  static structures/programs and frozen weights. These must be priced separately.
  No unchanged online/target/optimizer state can simply be dropped.
- Post-fork evaluation intentionally reconstructs/seeds an environment at each
  index and restores training RNGs. Parallelization changes episode/batch streams.
- Atomic new-generation saves and repeated ledger/CSV/curve exports can stress a
  shared HDD/metadata filesystem. Node-local staging requires a persistence/crash
  recovery plan; timing /tmp alone does not validate persistent Delta storage.
- Resume uses committed LATEST and rewrites records from persisted lists; unfinished
  work after the last checkpoint can repeat by design. DONE avoids ordinary
  reexecution but is only a scheduling hint, not a completeness certificate.
- Launcher nonzero children can be retried indefinitely; ownerless claims can
  remain held; zero exit without DONE is unresolved. Stale takeovers are unsafe
  if concurrent. In a local forced interleaving, B reads the old owner and pauses;
  A reclaims/writes its new owner; B then deletes A's owner/claim and succeeds:
  **A=true, B=true, final owner=B**. No scheduler command was used. Serialize stale
  recovery after verifying no active owner. A durable multi-host lock/fencing
  repair needs the intended filesystem's semantics and interruption tests;
  no speculative flock/rename implementation is claimed qualified here.

## 10. Tests and evidence

Runtime: Python 3.12.14, CPU JAX/JAXlib 0.4.34, Flax 0.8.4, Optax 0.2.3,
Orbax 0.5.3, dm_control 1.0.38, MyoSuite 2.12.2, rliable 1.2.0. Repository
requirements installed unchanged; `uv pip check` passed for all 117 installed
packages. A repeated requirements install and the saved complete install script
both passed without changing manifests/lockfiles. Only an initial uv cache write
to the read-only home failed; UV_CACHE_DIR=/workspace/.cache/uv corrected setup.

From main/, all focused/full/break CPU validation uses:

```bash
export JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off
export WANDB_MODE=disabled MPLCONFIGDIR=/workspace/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

| Check | Current-instance outcome |
|---|---|
| Starting-HEAD focused runtime/resume/single/twin fork selection | 12 tests, all passed; 244.723s |
| Updated focused selection | 15 tests, all passed; 247.271s |
| New trained twin restore with nondefault churn shape | 1 test passed; 14.794s |
| New exception-preservation test | 1 test passed; 0.003s |
| Regression against pre-fix state/startup trace | 2 tests: expected four failing subtests/structure assertions and one startup error; 12.264s; old behavior failed |
| Old train wrapper under exception-preservation regression | Expected error from masking the original exception, 0.002s; fixed wrapper passes |
| Old cache/progress wrappers under new outcome regression | Expected missing-field error, 0.001s; fixed wrappers preserve the returned values and report both outcomes |
| Previous duplicate-restore method under the original call-count test | Both trained/untrained subcases fail `2 != 1`, 5.892s; current method passes |
| Traced tiny CLI parent and injected arm | Both exit 0, both DONE, Check1/Check2/injection/cache events present |
| Production probe StableHLO lowering | Three sizes; 10/1,000 step dot/while counts unchanged |
| Two-process cache mechanism | Truncated read reproduced; complete read succeeds; existing partial entry remains unchanged |
| Stale-claim mechanism | Both stale contenders return ownership; local-only forced schedule |
| Full Exp12 CPU discovery | 189 tests in 1,500.194s; 187 passed, 2 dependency skips, zero failures/errors; exit 0 |
| Existing break/restore suite | 72/72 OK, exit 0; every mutation detected and every restored test passed; in-process patches only |

The historical `TwinCheck1Test.test_panel_values` numerical-oracle gate passed
on this CPU host, including the restored break check. Its source and
`rtol=1e-5, atol=1e-7` assertion are unchanged. This is a host-specific result,
not a fix or a resolution of the historical red gate, and it establishes no
CUDA qualification. Exact original/control/identity comparisons remain exact.

Added five tests: complete replay/structure/key comparison, backend startup flush,
cache outcomes/post-fork progress, exception preservation, trained twin reference
shape. Extended the existing trace equivalence test to include injection and
probe initialization. No test was disabled, tolerance loosened, or result altered.

Commands:

```bash
../.venv/bin/python -m unittest -v tests.test_exp12_runtime \
  tests.test_exp12_foundations.ResumeExactnessTest \
  tests.test_exp12_fork.ForkEndToEndTest.test_identity_fork_is_bit_identical \
  tests.test_exp12_twin_critic.TwinForkEndToEndTest
../.venv/bin/python -m unittest discover -s tests -t . -p 'test_exp12_*.py' -v
../.venv/bin/python -m tests.exp12_break_checks
```

Raw local evidence is outside Git: `/tmp/exp12-*-focused.log`,
`/tmp/exp12-{full-cpu,break-checks,twin-restore,trace-exception,trace-old-exception,trace-old-cache,old-restore-regression,new-tests-before}.log`,
`/tmp/exp12-{probe-lowering,cache-mechanism,claim-race}.json`, and
`/tmp/exp12-stage-smoke/{parent,arm}.{log,jsonl}` plus tiny generated states.
It is current-instance evidence, not portable published research data.

## 11. Minimum proposed A100 measurement — NOT submitted

First recover existing Block B artifacts locally if available; no additional GPU
time is needed for that. Required files: dev and identity logs/status TSVs, speed/
packing `job_0.json`, null summaries, probe/checkpoint metadata/timestamps and cache
errors. None were present in this Cloud checkout.

If new measurement is needed, begin with **one process on one A100 of the final
literal model, four CPUs, 32G host RAM, seven-minute allocation**, one diagnostic
attempt capped at 300s +15s kill grace (**≤0.0875 workload GPU-h**; allocation
maximum 0.1167 GPU-h). Use a reviewed checkout of the requested branch after
the diff is committed/reviewed; pin its exact final full hash. Do not execute on
an active Block B checkout. This is a proposal, not authorization to submit.

Suggested SBATCH resource lines, matching the repository's existing account and
partition declarations (site availability remains unverified here):

```bash
#SBATCH --account=biqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
```

The reproducible, manually submitted implementation is
[`scripts/sbatch_exp12_runtime_diagnostic.sh`](../scripts/sbatch_exp12_runtime_diagnostic.sh).
It contains the resource directives above and does not call `sbatch` or `srun`.
No existing launcher, CI workflow, startup instruction, or production manifest
invokes it. Submit only after the lead separately authorizes Delta execution.

After activating the existing pinned CUDA environment, from the reviewed clean
`main/` checkout, the eventual command is:

```bash
conda activate scaling-drl-py31213
export EXPECTED_COMMIT='<published packaging commit hash>'
export EXPECTED_GPU_MODEL='<reviewed literal A100 device_kind for the grid>'
export OUT='/new/absolute/persistent/diagnostic-directory'
sbatch --export=ALL \
  --output="${OUT}.slurm-%j.out" --error="${OUT}.slurm-%j.err" \
  scripts/sbatch_exp12_runtime_diagnostic.sh
```

The output parent must already exist. OUT must be new and outside the checkout.
Slurm stdout/stderr are sibling files so their creation cannot dirty the source
checkout or pre-create OUT. All diagnostic output, temporary replay/checkpoints,
and the isolated cache reside beneath OUT. Site account/partition availability
and the intended persistent filesystem remain unverified here.

The script resolves the checkout through SLURM_SUBMIT_DIR, not Slurm's copied
script path. It refuses a missing allocation, wrong branch/hash, dirty source,
reused/relative/in-repository output, non-A100 expected model, wrong literal
GPU model/backend, multiple devices, or a different Python/JAX version.
Device initialization, workload, and output validation all share the same
300-second timeout with 15-second kill grace. It records the command's final
status, including timeout, without retrying or launching anything else.

No scientific knob is overridden: architecture, batch, UTD, precision, action
repeat, 5×1,000 probe, pool25,600/batch256 remain production settings. The short
measurement window is a diagnostic prefix, not a reduced scientific budget.
Warm-up ends at interaction6,001, crossing the structural-metric boundary6,000;
the subsequent sixty-interaction timed window contains no logging/eval/save
boundary. It tests steady SAC separately from that window cost. There are
2,124 SAC updates in total through6,061. The script performs a full paired probe
warm-up plus one full paired timed check, synthetic95%-replay save/restore, and
two ten-episode fork evals. `TMPDIR` places the existing helper's otherwise
temporary checkpoint/replay writes on the requested filesystem. Report the
synthetic data as I/O benchmarking, never exact scientific continuation evidence.

`profile_exp12` uses identical trained parameters under its current/fresh labels
for timing, not an actual stored initialization reference; it emits no scientific
plasticity claim/ledger. It also does not call the Exp1 fresh callback before
the first SAC update. These limitations matter if the initial production lifecycle
is responsible for the stall. Its projected probe count treats initial fresh
work as a full paired check; use the separate accounting model below instead.

Expected files: `commit.txt`, `backend.json`, `profile.log`, `trace.jsonl`,
`profile.json`, `validation.json`, and `exit_status.txt`; sibling Slurm stdout/
stderr files. Partial attempts can lack backend/profile/validation output.
Expected evidence: complete profile JSON only on successful completion; flushed
trace even on timeout; compile/module/cache read/write durations and hit outcomes;
per-fit/round/check timings; progress/update counters; structural metric boundary;
environment/replay/sample-action/update timings; full save/restore bytes/times;
evaluation cost; peak device memory; log and exit status. Nested timing totals
are not additive. Source/runtime/model and output ownership must match.

Stop/fail conditions: wrong branch/hash/model/backend/JAX version, dirty or reused output,
nonzero exit/timeout, OOM/nonfinite output, cache errors, missing output/end stage,
or unexpected counts. The embedded validator requires a finite positive measured
rate/check/save/restore/evaluation duration and state byte count; one dog-run
D4W1536 result with nominal N=500,000; synthetic replay count475,000; ten episodes
per post-fork eval; one timed probe repeat; an error-free final trace summary;
last training progress6,061 interactions/2,124 SAC updates; and stage totals
1,062 update_many calls,10 probe rounds,20 individual fits. Each round fits
both current and fresh critics; the two full paired checks each have five rounds. These are engineering
completeness/count checks derived from the existing implementation, not new
scientific thresholds. `validation.json` says explicitly that successful
completion is **not scientific qualification**. If peak device telemetry is
unavailable it records that fact; memory qualification remains incomplete.
**No automatic retry, timeout increase, second cell,
positive control, HB job, or grid follows.** Preserve the last flushed stage.

| Measurement outcome | What it resolves / next hypothesis |
|---|---|
| Slow/hung initial SAC backend_compile | Compilation/autotuning/startup becomes localized; inspect module/cache/model/error evidence |
| Compile ends, updates/interaction are slow | Separate synchronized update, action, env/replay and memory costs; compare counters with 2 updates/interaction |
| Metric boundary dominates while pure window is fast | Prior speed windows omitted important work; price metric windows explicitly |
| A full check is costly | Per-round target/fit/pool timing measures intentional probe cost or an identified stage problem; no probe definition changed |
| Save/restore dominates | Use byte counts/component timings on that filesystem; a later reviewed local-storage comparison can distinguish storage overhead |
| All isolated stages are fast | This does not solve dev; recover logs or run a capped actual Exp1 fresh-callback lifecycle trace |
| Cold read errors in a new private directory | Concurrency is insufficient as explanation; investigate interrupted writer/storage/cache initialization evidence |

Only after this attempt completes, propose separate bounded follow-ups:
an unsynchronized same-command run to quantify barrier overhead, a *sequential*
reuse of the completed cache with new output/TMPDIR to measure warm compilation,
and a new-private-cache repetition to compare independent cold compilation. Each
has its own 300s cap and its own reviewed interpretation; none automatically runs.
Comparing these cases shows compile reuse costs, not production policy equivalence.
Never induce corrupt executable caches on Delta.

If needed, the actual production-lifecycle diagnostic is the same trace wrapper
around `run.py --experiment exp1 --config_name base_exp12`, D4W1536 dog-run,
run_role=dev, seed990, a new absolute results/checkpoint root, checkpoint_interval
25000/start_frac0, *no* scientific overrides, with the same300s cap. It retains
capture_fresh before first SAC, unlike profile. Timeout can leave only check0;
flushed update counters then distinguish completed training from initial probing.
It is an incomplete diagnostic and must never be analyzed as a no-trigger run.

## 12. Transparent compute model

Use measurements per architecture **and environment/suite**, with separate twin
and injected-m cases. Seconds are exclusive stage costs, or rates measured for
a composite interval with the included work explicitly identified. Do not add
inclusive trace totals to composite training wall time again.

For cell a,e,s define N=B/2, W=5,000, G=N/20, log cadence L=2,000,
base eval cadence E=50,000, fork f if eligible, H=N/4,
Kc=max(N,f+H), and q=1 for DMC/Myo or2 forHB.
Measure these unknowns on the selected A100/stack/filesystem:

- `i_a,e`: startup/model/environment/first compilation cost, exclusive of eval
  and warm-up; `w_a,e`: actions/environment/replay work through interaction W−1,
  excluding the fresh probe and already-counted compilation.
- `t_a,e`: learned interaction seconds for ordinary SAC (includes action/env,
  two replay batches/updates and normal host work, excludes metrics/probes/evals).
- `p0_a,e`: actual fresh check seconds; `p_a,e`: ordinary full paired check
  including pool/target/fit/transfer/bootstrap/NPZ overhead as chosen consistently.
- `m_a,e`: logging/structural-window cost beyond the pure interaction rate.
- `v_a,e(j,n)`: evaluation cost at boundary j for n episodes, including any
  environment construction; `z_a,e(r)`: save cost with replay size r.
- Eligible-only: control rebuild/restore/panel/Check1/fresh-reference load `Fc`,
  injection/rebuild/restore/Check1/Check2/setup `Fi`, injected learned interaction
  `tI_a,e,m`, injected paired check `pI`, post-fork eval `vF`, injected save `zI`.

No-fork nominal parent:

```text
T0(a,e,s) = i + w + (N-W+1)*t
          + p0 + 20*p
          + floor(N/L)*m
          + v(start,5) + sum_{j=E,2E,...,N} v(j, 5*(10 if j+E>N else 1))
          + sum_{k=1..20} z(min(k*G, replay_capacity))
          + export/finalization_cost
```

Warm-up w includes actions/environment interactions1…W−1; define its measurement
boundary consistently so interaction W is counted exactly once with the learned
rate. Startup and first-update compilation can instead be put into one explicit
initialization term if they cannot be separated reliably.

For an eligible parent, substitute Kc for the training end and metric/eval end,
retain N in all schedule fractions/final-eval tests, use Kc/G scheduled paired
checks, add Fc and 26 post-fork evaluations. Count the fork save once at f instead
of its routine save. Since approved f is kG and H=5G, parent complete saves
including the immutable fork number Kc/G (20…24), plus the fresh reference save
cost if it is not already in p0/i. Extra control training is
`max(0,f+H−N)*t`, at most0.20N*t, not another whole run.

Injected arm:

```text
TI(a,e,s,f,m) = Fi + H*tI + 5*pI + sum_{j=0..25} vF(j,10)
              + sum_{k=0..5} zI(min(f+k*G, replay_capacity))
              + export/finalization_cost
```

Fi includes the immediate Check2 (three fitted critics/Q, 15,000q updates),
not five paired Check2s; explicitly separate it if measured. H interactions
have 2H SAC iterations, not another replay warm-up. Injected speed, memory,
checkpoint sizes and compile cost depend on the frozen head m and cannot be
substituted from the original critic without measurement.

```text
Exp1_nominal_GPU_h = sum_{a,e,s} T0 / 3600
Exp1_actual_GPU_h  = (sum T0 + sum_eligible (Tcontrol - T0)) / 3600
Exp2_injected_GPU_h = sum_eligible TI / 3600
Campaign_GPU_h = Exp1_actual_GPU_h + Exp2_injected_GPU_h
```

Divide the indicated sums by3600 consistently. For expected eligibility r_a,e
and conditional fork distribution pi_a,e(k), use

`E[campaign] = Σ T0 + 5 Σ_scaled_a,e r_a,e Σ_k pi_a,e(k)
               × (Tcontrol(k)−T0 + TI(k,m))`, then /3600.

There are130 possible scaled forks. No eligibility rate is inferred from the
dev timeout. Report zero/full-eligibility scenarios and observed conditional
fork rates separately. Uncertainty uses measured per-stage low/high values,
startup/cache/storage scenarios and r/pi ranges; missing terms stay unknown,
not zero. A single dog/A100 check cannot qualify all39 architecture/environment
cells or twin/injected cases. Packing requires measured *aggregate* throughput;
allocated GPU-hours and per-run single-GPU accounting must be distinguished.

Calibration, positive control, null checks, identity qualification, preflight,
failed attempts and resume/retry overhead are additional budget lines. They are
not included in the 195 nominal parents, and cannot be treated as zero when
planning the complete allocation request.

Cost per1%B interval (N/100 learned interactions) away from probes/checkpoints:
`(N/100)*t + metric windows + due evaluations`. Over five intervals add one
paired probe and save; amortized probe/save contribution per1%B is `(p+z)/5`.
Post-fork add one ten-episode eval each interval and price injected tI separately.
For dog-run this is5,000 interactions and10,000 SAC updates per1%B.

The previous **600–700 A100-hour estimate remains unvalidated**. No CPU timings,
parameter-count scaling, null aggregate total, or speed-test overall wall time
is used here as a measured A100 training rate.

## 13. Outstanding lead decisions and external evidence

Required decisions before changing the corresponding behavior:

1. Whether to replace/supplement the inherited full-pipeline identity gate with
   the focused equivalent continuation design and which integration claims to keep.
2. Cache isolation/warm-template policy and its CUDA qualification cases; no
   numerical-equivalence assertion from CPU alone.
3. The existing independent twin numerical-oracle criterion and measured CUDA
   diagnostic bounds; exact original/control/identity rules remain unchanged.
4. The n=1 reporting convention, frozen injection m, and any scientific
   interpretation/inclusion changes. None were selected here.
5. Explicit authorization for any future Delta allocation/submission. This report
   is not a submission request and has submitted nothing.

Required outside evidence: actual Block B artifacts; bounded A100 stage timings;
production-size exact continuation and injected Check1; HB renderer/simulator/
twin integration; intended filesystem and concurrency/recovery qualification.
No scientific speed/memory/1%B acceptance threshold is set by the diagnostic.
Any such threshold, treating this short/synthetic profile as campaign or
scientific acceptance, a different grid GPU model if not yet approved, changes
to identity/Check1/Check2 criteria or tolerances, or adopting private caches as
production policy require the lead's decision. A diagnostic timeout is never
scientific evidence of no degradation. The diagnostic does not perform injection,
Check1, Check2, production-size exact identity, or the Exp1 fresh callback.

No result requiring these is claimed resolved. CPU development capabilities and
saved setup are distinct from full-grid scientific/computational qualification.

## 14. Reusable Cloud configuration and original audit source state

Saved draft fields: complete **install_script**, **start_skill**, and the
inspected repository membership at the exact starting remote commit, mount
`scaling-drl-research`. The script creates/reuses the Python3.12 venv outside
tracked source, installs unchanged requirements plus pinned Black, checks
dependencies/backend, and uses writable workspace caches. It was executed and
passed in this instance. Startup records branch/methodology checks, activation/
CPU/headless/offline settings, meaningful test commands, no-worktree guidance,
known red gates and no-Delta/no-production constraints. No secret or network
policy change was needed. DMC/Myo CPU development and analysis dependencies are
installed; HB and CUDA qualification remain separate.

Draft save is confirmed, not publication or validation of a restored fresh task.
Review/save the changes in environment settings, then publish the environment
to retain the prepared snapshot. Local raw /tmp evidence is not guaranteed to
survive restoration; no fresh-task restoration claim is made.

The original audit ended at HEAD **`ca95b1e4ea9262eb9abde406848769349eeb8e17`** on
`claude/eloquent-fermat-inxqlt`. **No commits created; no pushes.** Its reviewable
working-tree changes were:

```text
 M main/experiments/exp12/runtime_trace.py
 M main/experiments/exp12/state.py
 M main/tests/test_exp12_runtime.py
?? main/docs/exp12_overnight_readiness.md
```

The changed source repairs comparison/instrumentation coverage, not the research
algorithm. The report/model/benchmark proposal is new documentation. Dependencies,
caches, test outputs and helper setup script are outside tracked source; source
manifests/lockfiles/configs remain unchanged. That checkout was not a clean
qualified CUDA revision. Final test outcomes and Git/format checks follow.

## 15. Final validation record

Completed full discovery output:

```text
Ran 189 tests in 1500.194s
OK (skipped=2)
```

Command exit status was 0: 187 test methods passed, two were skipped, with zero
failures/errors. The two explicit skips are
`HumanoidBenchReachEvalSeedingTest.test_reach_eval_seeding_saves_and_restores_the_training_rng`
and `PipelinePerSuiteTest.test_humanoid_bench`, both because `humanoid_bench` is
absent. Foundation `ALL_ENVS`/`SUITES` also conditionally omit HB simulator
subcases when the package is absent; those subcases were **not run**, rather
than silently counted as passes. The 11 enabled DMC/Myo environment restore
cases ran. HB configuration/unit coverage is not HB simulator qualification.

`python -m tests.exp12_break_checks` completed with **72 OK, zero PROBLEM,
exit 0**. Every deliberately broken behavior failed its designated test and
every restored behavior passed. The harness applies only process-local mock
patches. Its elapsed wall duration was not recorded; no timing estimate is
invented. The historical twin numerical-oracle gate passed here unchanged,
not through a tolerance adjustment or an algorithmic fix.

Final validation also passed: Black checks on runtime_trace.py and
test_exp12_runtime.py, Black on only the edited state.py line range,
`git diff --check`, and `bash -n` on the entire proposed diagnostic payload.
`profile_exp12.py --help` parsed successfully on CPU; the proposed CUDA payload
was **not executed**. No production-size training/fit, CUDA throughput or
Delta filesystem measurement was performed.

At the original audit completion, branch/HEAD/status matched section 14.
No commits or pushes had occurred and no generated results/caches were in
the tracked diff. The saved environment draft
was reread successfully (revision 2): install/start/repository fields are
present, no secrets/runtime requirements were added, and publication remains
pending user review. These successful CPU gates establish development and
engineering regression readiness, not readiness of the 195-run A100 grid.


## 16. Repository packaging for normal checkout / Delta review (2026-10-07)

The user explicitly authorized committing and pushing the legitimate work to
`claude/eloquent-fermat-inxqlt`, without merge, force-push, Delta access, or job
submission. Sections14–15 retain the original audit's pre-commit provenance;
this packaging step adds the manual diagnostic script and its portable CPU
launcher checks. The exact delivery hash is the commit containing this report
update and is reported after push; the script requires that reviewed full hash
through EXPECTED_COMMIT.

Workspace inspection found only the three documented Python changes and this
report as initial nonignored changes. Ignored checkout files were the local
venv/dependencies and Python bytecode; Cloud setup scripts, dependency caches,
raw logs/results, helper reproductions, and profiling/state artifacts outside
Git remain uncommitted. Staging uses explicit file paths only.

The Python source/regressions have not changed since the completed 189-test
CPU suite and 72-check break run: their modification times precede both logs,
and the full diff matches the reviewed audit changes. The logs were rechecked:
187 passed +2 HB dependency skips, no failures/errors;72/72 break checks OK.
Pinned dependencies remain compatible (117 packages). Repeating the entire
25-minute suite is unnecessary for this packaging-only change. Fresh focused
runtime validation passed **8/8 in54.067s**, exit0. The final launcher checks
passed **12/12 in3.557s**, exit0, including agreement with the real probe
round/fit scheduling function. The original broad Exp12 discovery still counts
189 methods; the 12 standalone launcher tests are additional coverage, not
replacements for any existing gate. Shell syntax, embedded Python/Black,
changed-file Black checks, and Git whitespace checks passed.

The launcher checks use a temporary local Git fixture and mocked Python/timeout
executables; the real output validator runs against synthetic JSON, and one
check exercises real CPU probe scheduling with expensive fitting mocked. They never
invoke Slurm, CUDA, or Delta. They cover allocation/revision/branch/cleanliness,
output reuse/location, model class, bounded command arguments, nonzero exit,
timeout, nonfinite/missing output, cache errors, and counter mismatches. They are
engineering checks, not GPU evidence. Run them from `main/` with:

```bash
python -m unittest -v tests.test_runtime_diagnostic_launcher
```

No scientific config, methodology file, training/probe implementation, numerical
tolerance, statistical rule, or production launch policy was changed. The only
precision-related launch setting added is enforcing the already-approved
x64-off setting; the existing entry point still selects the approved TF32 policy.
The private cache is restricted to this proposed measurement, never installed
as a production default. Existing numerical gates remain untouched.

An initial launcher-test invocation from the repository root could not import
`experiments`; using the documented `main/` working directory resolved it.
No Slurm command was executed. Reference scans found only the report and test
referencing the diagnostic, so no automatic submission path was added.

Files in this delivery:

- `main/experiments/exp12/runtime_trace.py`
- `main/experiments/exp12/state.py`
- `main/tests/test_exp12_runtime.py`
- `main/tests/test_runtime_diagnostic_launcher.py`
- `main/scripts/sbatch_exp12_runtime_diagnostic.sh`
- `main/docs/exp12_overnight_readiness.md`
