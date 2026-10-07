# Block B runtime investigation

Investigated locally on 2026-10-06 (America/New_York), starting at
`d3123c3c6db1d69e7a982dd1c7970d413139c23b`, branch
`claude/eloquent-fermat-inxqlt`. Reported Delta evidence concerns job 22706349
at the older `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309` revision.
No Delta connection, checkout access, job submission/cancellation, or GPU execution
was used in this investigation. Scientific settings and timeout values are unchanged.

## 1. Diagnosis and confidence

**The six-hour dev timeout's root cause is not established by the evidence available
on this Mac.** There are no copied job logs, checkpoint metadata, status TSVs,
packing JSONs, or cache timestamps here. The supplied observation of check 00 is
enough to establish completed warm-up and initial probing, not training throughput
or natural failure to trigger. A precise claim that compilation, probes, or storage
consumed six hours would be invented.

Two concrete findings:

* **CONFIRMED ROOT CAUSE of the identity workload being much larger than “1,000
  steps”:** 1,000 is continuation length after a forced fork. The parent first
  performs 12,000 interactions, production probes at checks 0, 1, 2, and fork setup.
  Parent plus identity arm perform approximately 18,002 SAC updates and 25,000
  probe optimizer updates. This establishes workload scope, not why it took an hour.
* **CONTRIBUTING FACTOR, confirmed engineering bug and fixed:** every agent restore
  read the complete agent payload twice to recover one small churn reference batch.
  The fix uses checkpoint metadata to recover that batch's shape and performs one
  payload restore. Exact CPU continuation passes. This cannot explain a timeout
  before the first scheduled probe/fork, because no restore has occurred there.

| Suspected cause | Classification for the extreme timeout | Evidence / limit |
|---|---|---|
| SAC throughput, environment stepping, replay/host work | UNRESOLVED | No timed training window or exact counters from the job |
| Initial training compilation / CUDA autotuning | UNRESOLVED | Fresh-only null probes do not compile SAC; tiny CPU compilation is insufficient evidence about CUDA |
| Compilation on every changing update counter | RULED OUT in the local short production call path | Update JIT cache stabilizes at 3 startup signatures; later counters reuse it |
| 1,000-fold Python unrolling of probe fit | RULED OUT | Production-shape lowered graph has the same dot/while count at 10 and 1,000 steps |
| Intentional probe work | CONTRIBUTING FACTOR | Real procedure is substantial; preflight reduces probe fit-example work by hundreds of times per check |
| Probe execution alone taking hours/check | UNRESOLVED, inconsistent with the reported null throughput under comparable conditions | Null completed 300 production paired checks in 6,042 s; no per-size breakdown available |
| Shared-cache corruption | UNRESOLVED for dev; RULED OUT as a necessary explanation of all cold identity failures | Cold identity parents each have a private directory and run sequentially |
| Persistent-cache filesystem latency | UNRESOLVED | Private directories still use the same filesystem |
| Duplicate complete agent restore | CONTRIBUTING FACTOR, fixed | Two payload restores reduced to one; occurs after fork/resume |
| Periodic checkpoint serialization before check 1 | RULED OUT for a fresh dev run | First periodic save follows completion of check 1 at interaction 25,000 |
| Fresh-reference serialization | CONTRIBUTING FACTOR | One parameter-only save precedes check 00; reportedly completed |
| Evaluation / Check1 / Check2 / fork restoration | RULED OUT as post-start causes of a fresh dev stall before check 1 | Only startup evaluation has occurred; fork earliest at check 2 |
| Hidden 2×/4×/20× SAC update factor | RULED OUT by code accounting | UTD 2; one environment; scan length 2; probe updates separate |
| Repeated initialization and compilation after fork | CONTRIBUTING FACTOR | Control rebuilds a trainer intentionally; new optimizer function identities can retrace; backend persistent-cache reuse is separate |
| Structural metrics and metric-window flush | UNRESOLVED contribution | Run every 2,000 interactions, including before replay warm-up; speed timed window excludes them |
| Timeout simply too short | UNRESOLVED and not a fix | Increasing it would conceal the missing stage evidence |

## 2. Exact progress that can and cannot be established

These bounds assume a fresh invocation with the script defaults, and that the
reported artifact search found all completed probe files. They are not an inspection
of the live job.

| Workload | Latest confirmed progress | What remains unknown |
|---|---|---|
| Dev D4W1536 dog-run, 21,600 s timeout | At check-00 completion: interaction 5,000; nominal raw env step 10,000; SAC update step 0; five startup eval episodes completed | Later counters, timestamps, in-memory metrics; whether stalled in first SAC compile, training, structural metrics, or check 1 |
| Each of four cold identities, 3,600 s timeout | Process timed out; no stage artifacts supplied | Even whether parent reached check 00, fork, continuation, or identity-arm launch |
| First warm D4W1024 dog identity, 3,600 s timeout | Same: timeout only | No parent-completion message or arm artifacts supplied |
| Next warm D4W1536 dog identity | Reported started | No completion/progress evidence supplied |

For dev, absence of check 01 bounds the fresh process at **at most interaction
25,000 / nominal env step 50,000 / SAC updates 40,002** if it is inside that check.
It can also still be at interaction 5,000 with zero completed SAC updates. Check 00
is written after all its rounds finish, before the first SAC update. It is not the
first 5%-of-budget check. The first periodic checkpoint is after check 01; no such
checkpoint is established by the supplied evidence. Startup evaluation is the only
evaluation established; nominal training evaluation starts at interaction 50,000.

No measured dev env/s, updates/s, seconds per 1%B, or full-run projection can be
calculated without timestamps and counters. The upper counter bounds divided by
21,600 give **at most 2.315 nominal env/s and 1.852 SAC updates/s** over the whole
attempt, including startup. These are bounds, not steady-state measurements.
Failure to complete the first 5%B interval within 6 h corresponds to more than
4,320 s per 1%B when amortizing that failed interval. Linear extrapolation would
exceed 120 h per nominal run, but this is a pathological diagnostic extrapolation,
**not a campaign forecast**.

Recovering exact progress needs local copies of `gpu0/dev_run.log`, the run's
`logs/*.csv`, `probes/check_00.npz` and any later files, `state/LATEST` and its
`meta.pkl`, `fork/*`, each identity log/tree, `status/*.tsv`, packing/speed
`job_0.json`, and cache-read warnings/timestamps. Checkpoint metadata gives exact
persisted counters; probe NPZ alone has curves, not an independently timestamped
training counter. CSV is written on saves, not continuously. Cache mtimes show
cache activity, not completed environment or optimizer steps. None of these
unavailable observations is replaced with an assumption.

## 3. Execution paths and runtime model

Sources: `scripts/exp12_blockB.sh`, `run.py`, `scripts/preflight_checkpoint_check.py`,
`scripts/probe_fresh_checks.py`, `scripts/profile_exp12.py`, `experiments/exp1.py`,
`experiments/exp2_arm.py`, and `experiments/exp12/{trainer,run_probes,probe,fork}.py`.
The relevant old/current paths are unchanged apart from current twin support and
the restore optimization made here; current Block B also filters GPU UUID output.

All workloads use one train/eval environment, repeat 2 and UTD 2 unless explicitly
noted below. A SAC update means one actor, one critic, one temperature optimizer
step and one target update; these are not four additional UTD passes.

| Workload | Budget and SAC update count | Probe/evaluation/save/fork schedule | Architecture / limit / representativeness |
|---|---|---|---|
| Dev | 1M raw, 500k interactions; 990,002 SAC updates through nominal end | Fresh at 5k; paired checks every 25k through 500k; metrics every 2k; eval at 0 and every 50k; checkpoints every 25k after check; natural eligible fork only, control may extend to 600k | D4W1536 dog-run seed102; 21,600 s; production procedure, dev role |
| Identity cold/warm | Configured 240k raw /120k interactions; actually parent stops at 13k and arm after 1k more | Fresh 5k; checks 6k and12k, forced fork12k; parent snapshot13k; identity snapshot13k; base eval only at0 (5 episodes); post-fork eval0 with10 episodes in each arm; no later eval before snapshots; four complete saves across parent+arm plus fresh reference | D4W1024/1536 ×dog/Myo/HB; 3,600 s includes both processes; full-sized probe/architecture, reduced validation budget |
| Null | No SAC training or optimizer updates; warm-up to5k separately for each size | 100 pairs/size, each two independently initialized fresh critics, 5×1000 fits/critic, bootstrap10k; no complete training checkpoints/fork; startup5 eval episodes per size | Three sizes dog-run;14,400 s; representative probe kernels, no SAC throughput |
| Speed / packing | Warm-up5,000+100, then600 timed interactions; 1,402 SAC updates total,1,200 timed | No probe callback; startup1 eval episode; structural metrics at2k/4k before learning; no timed-window metric boundary/eval/save/fork | Three sizes Myo/HB;900 s; training only. Packing dog-run uses1/2/3/4 processes,4 CPU cores each |
| Preflight default | 4k raw /2k interactions; min replay50;3,902 SAC updates |20 checks every100 interactions;5×10 probe steps,pool256,batch32; startup5 eval episodes; save every200,final2k; no default-architecture fork | D2W512; outer1,200 s, inner900 s/process; plumbing only |
| Preflight scaled | Same parent3,902 updates plus1,000 arm updates | Forced fork200; parent ends2k, injected arm700; Check2; post-fork eval every20 interactions,1 episode,26 perarm;10 complete parent states incl fork/final and4 arm states; no base periodic eval before2k | D4W1024/1536; same limits; injected m=last by preflight default; plumbing, not production probe cost |

If dev forks at interaction f, the parent ends at `max(N,f+0.25N)` and an injected
arm trains `0.25N` extra interactions. Base evaluations remain on the50k grid and
post-fork evaluations are additional: indices0..25,10 episodes each, every1%B.
For nominal dog-run without a fork the base evaluation count is100 episodes:
5 at startup,9×5 intermediate,50 at final. Late control extension can add more
base evaluations under the existing final-evaluation logic; it must be costed
from the actual fork plan, not assumed to be part of the nominal100.

Expected logical large-program families per fresh single-Q process are action
sampling, SAC update scan, target generation, and probe fit. Current/fresh fits
reuse the same fit function when static definitions/optimizer and abstract inputs
match. There is no compilation per scan iteration. Local tiny training observed
3 startup update signatures and2 action signatures, then reuse. Exact total XLA
compilations is backend/cache/shape dependent; hundreds of small eager primitive
compiles are also possible. Rebuilding the trainer/optimizer after fork can retrace
families again; injected structure adds a distinct fit/update family. An exact
CUDA compile count cannot be asserted without logs.

Use the stage model, with nested stages distinguished:

`T = init + initial_eval + replay_warmup + fresh_save + fresh_probe + initial_SAC_compile
     + learned_interactions*t_steady + metrics_windows*t_metrics
     + scheduled_probes + base_evals + periodic_saves
     + fork_save/restore/Check1 + post_fork_evals + Check2(injected only)`.

## 4. Preflight versus production

D4W1536's 391 s preflight is not a production throughput measurement. The dev
attempt spent at least55.24× that wall time; the completed workload fraction is
unknown. Important multipliers are:

| Quantity | Scaled preflight, parent plus injected arm | Production dev | Work multiplier |
|---|---|---|---|
| Nominal raw parent budget |4,000|1,000,000|250×|
| Replay warm-up interactions |50|5,000|100×|
| Nominal parent SAC updates |3,902|990,002|253.7×; versus preflight+arm4,902:202.0×|
| SAC batch / UTD / architecture |256 /2 /D4W1536|same|1×|
| Steps per probe fit |10|1,000|100×|
| Probe minibatch |32|256|8×; fit-example work800× per corresponding fit|
| Pool / evaluation chunk |256 /256|25,600 /2,560|pool100×; chunk10×; chunk count1→10|
| Rounds / nominal checks |5 /20|5 /20|1×|
| Total probe optimizer updates |2,700 including parent,arm,Check2|205,000 nominal parent|75.93×; fit examples607.4×|
| Probe work through dev check1 |all preflight:86,400 fit examples|3,840,000|44.44×|
| Probe work through dev check0 |all preflight:86,400 fit examples|1,280,000|14.81×|
| SAC work through dev check1 |all preflight:4,902 updates|40,002|8.16×|
| Complete checkpoint writes |14 small-replay full-model saves|none before completion of check1|No periodic-checkpoint multiplier explains this interval|
| Eval episodes |57 (5 startup+52 post-fork)|5 established before check1|Dev has less eval work before check1|

These ratios are operation counts, **not runtime ratios**. In particular one cannot
multiply391 s by800: the preflight time includes14 full-model saves, compilation,
training, and evaluation, while800 applies to corresponding probe minibatch work.
Batch-dependent CUDA efficiency and pool passes further prevent direct scaling.
The reported null measurement independently argues against assuming all six hours
are the intended probe workload. Remaining unexplained overhead is localized only
to the unobserved interval after fresh probing and before check1 completes;
stage traces are required to identify it.

## 5. Step/update audit

`N = num_env_steps / (num_train_envs*action_repeat)`; both environment counts must
be1 (`create_envs` rejects others). RepeatAction executes up to2 underlying steps,
with early termination allowed; recorded raw counters use the configured repeat.
With n_step1 and min_length5,000, updates begin on interaction5,000 itself:

`SAC_updates(N) = 2*(N-5,000+1)`.

Trainer samples exactly2 replay batches, stacks leading dimension2 and invokes
`update_many` once/learned interaction. The SAC `lax.scan` length is that leading
dimension; it does not multiply UTD again. `first_update_step` is dynamic. Actor
gradient cosine is computed on update numbers divisible by30, not30 extra updates.
Metrics may draw an additional replay sample but do not train. Eval consumes
separate environment steps; post-fork eval restores training JAX/NumPy/Python RNGs.
Probe RNGs and copies are separate from training. A paired single-Q check performs
10,000 probe optimizer steps; a twin check20,000, fitting each network separately.
Neither increments the SAC update counter. The bootstrap10,000 resamples is NumPy
statistical computation over5 losses, not10,000 critic-training passes.

For the nominal195-parent population there are105M interactions /210M nominal raw
steps, and208,050,390 SAC update iterations after warm-ups. Twin HB updates operate
on two networks per iteration; that is extra device work, not extra environment steps.

## 6. Probe cost and compilation audit

Each round creates one random same-architecture target network, forwards the pool,
and fits each critic with freshly initialized AdamW state for1,000 steps. Each fit
also forwards the pool before and after optimization. Target initialization uses
the original architecture even in injected arms. Pools are sampled with replacement
from filled replay and normalized using training observation statistics. Five
rounds use distinct deterministic probe streams. Returned losses/scalars transfer
to host once after all rounds; pools/index arrays transfer to device per round.
Only curves/scalars are compressed into NPZ, not fitted network weights.

Production-shape **lowering only, no heavy CPU/GPU fit execution**, dog223/38:

| Critic | Parameters | Fit StableHLO bytes:10→1000 steps | Dot operations | While operations |
|---|---:|---:|---:|---:|
|D2W512|4,337,153|141,855→141,883|29→29|3→3|
|D4W1024|33,854,465|229,071→229,099|49→49|3→3|
|D4W1536|75,947,521|229,071→229,099|49→49|3→3|

The loop body is traced once. This rules out Python graph unrolling, not CUDA
compiler/autotuning problems. `_fit` and `_base_targets` are module-level JITs;
round/check indices and parameters are dynamic inputs. There is no per-round
optimizer factory in `RunProbes`. Required fresh optimizer *state* is constructed
inside each fit. New `RunProbes` objects after restore construct new optimizer
transformations; static function identity may cause a retrace, with backend cache
reuse still possible. No numerical/optimizer restructuring was made speculatively.

An approximate forward-equivalent check cost is two critics×five rounds×
`(3*1000*256 + 3*25,600)` examples. Dominant dense FLOP cost scales roughly with
parameters: ratios1:7.81:17.51 for the three sizes. This is an arithmetic cost model,
not a measured A100 time. Twin checks roughly double the fitted-network work.

**Actual A100 per-size check times remain unavailable.** The reported null pass
covers300 paired checks in6,042 s:20.14 s/pair averaged across sizes, including
initialization, warm-up, bootstrap and logging. If all100 pairs/size actually
completed with defaults, each size's average is bounded above by60.42 s/pair
(nonnegative work in the other sizes). This is not a cold-compile bound nor a
per-size measurement. Obtain the three `null_summary_*.json` and pair timestamps
before using it as a production projection. The prior3/16/33.4 s forecasts remain
estimates; this investigation has not measured them anew.

## 7. Checkpoints and I/O

Complete state contains online actor/critic, their Adam moments/counts, target
critic, temperature and optimizer, model update counters, JAX key, churn reference
batch, observation RMS, filled replay arrays/cursor/n-step queue, NumPy/Python RNGs,
both environment replay/RNG states, vector action-space RNG, current observations
and timestep, meter/media windows, historical metric/eval rows, probe records,
fork/evaluation state, actor diagnostic reference/window state, and pending agent
window buffers. None was dropped.

For a single critic with P parameters the online weights+Adam2 moments+target
weights require approximately16P bytes. The >1GB D4W1536 preflight states are
therefore expected even with a tiny replay buffer; not evidence of serializing
the full unfilled million-transition replay allocation. `save_buffer` writes only
`[:_num_in_buffer]`, uncompressed NPZ.

| Dog-run state | Critic component bytes | Replay at500k transitions | Approximate total before small actor/metadata overhead |
|---|---:|---:|---:|
|D2W512|69,394,448|974,000,000|1.043GB|
|D4W1024|541,671,440|974,000,000|1.516GB|
|D4W1536|1,215,160,336|974,000,000|2.189GB|

Replay is1,948 B/transition for dog-run (two223-vectors,38 actions,three scalar
float32 arrays). At5k it is9.74MB; at the12k validation fork23.376MB.
The fresh D4W1536 reference is another303.79MB once. HB twin D4W1536's critic
component alone is approximately2.42GB, with HB replay on top. Injected critic
size depends on m and includes frozen parts, new heads, optimizer state and target;
do not assume its checkpoint or training cost equals the original critic's.

Dev/grid saves happen after scheduled checks, plus the immutable fork and final
state. New state directories protect crash recovery; LATEST switches only after
a complete save, then older periodic states are pruned. The fork and fresh
reference remain independently available. Two complete periodic generations can
temporarily coexist, plus the fork. Re-saving unchanged large model arrays is
expected between snapshots; removing them or changing checkpoint cadence requires
a recovery design and approval. No evidence establishes OCDBT, Orbax, or filesystem
serialization as the dev bottleneck; only the duplicate restore is confirmed here.

Restore previously loaded the large agent tree with typed targets and then loaded
the entire tree again without targets to recover the churn batch. It now reads
metadata for that batch's real shape/dtype/sharding, includes it in the typed
target, and restores the full payload once. At D4W1536 that avoids requesting a
second ~1.215GB agent payload per restore (about2.42GB for twin). Actual physical
disk traffic/speedup depends on OS/TensorStore caches and must be measured; two API
calls do not prove twice the physical disk reads.
`git log -S` and blame place the second restore's introduction at `b5f7a89`,
before Exp1/2; it is also present in the actual Block B revision. This is an
inherited I/O inefficiency, not evidence that Exp1/2 changed restored values.

## 8. JAX persistent cache

Pinned local JAX0.4.34 source inspected: `jax/_src/lru_cache.py`,
`compilation_cache.py`, `compiler.py`. With the default unlimited LRU size, writes
go directly to the final file, without the eviction-enabled file lock. Readers
can observe a partial file; an existing malformed file is not overwritten on a
later put. The prior forced two-process reproduction on this installed version
reported `Error -5 ... incomplete or truncated stream`, successful reading after
write completion, and an existing partial entry not repaired. This is a forced
interleaving reproduction, not proof that job22706349 encountered it.

| Execution | Current cache topology |
|---|---|
| Block B ordinary dev/preflight/packing/tests/null/speed processes | Default `<main>/jax_cache/<device_kind>`; shared across lanes and packing workers |
| Block B cold identity | Unique cell/mode `cache_parent` and `cache_arm`; parent then arm sequential |
| Block B warm identity | New cell `cache_shared`; parent first populates it, arm later reuses it |
| Current HB ordinary tests/profile | `$OUT/cache/<step>`; separate step directory, but subprocesses in a test step inherit/share it |
| Current HB identity | No explicit HB identity override; inherits caller cache or falls back to per-device default |
| Grid manifests | No explicit cache assignment; default per-device directory shared by concurrent workers unless caller overrides |

The first warm parent is not warmed by the cold cases: its directory is different.
“Warm” describes arm reuse after parent completion. A parent timeout can prevent
the warmed arm from ever running. Thus it neither proves nor disproves an arm-side
warm-cache speedup. Private cold parents also mean cross-process directory sharing
is not a necessary cause of all the identity timeouts. Common filesystem metadata
load, genuinely expensive compilation, or slow execution remain possible.

**Proposal, not implemented:** assign each ordinary Python process a unique writable
directory under a chosen cache root; choose it once per process and reuse on later
`configure_compilation_cache()` calls; record it in launch metadata; preserve explicit
caller settings for reviewed cold/warm identity scenarios. Use an atomic publication
or reviewed read-only completed snapshot if cross-process reuse is desired. A mere
run directory shared by subprocesses is insufficient. This changes caching policy,
not precision/RNG/config or JAX's executable cache key. However independent cold CUDA
compilation/autotuning can select different executables under nondeterministic
backend behavior. CPU equality cannot certify the exact CUDA identity contract.
Accordingly no cache-default rollout is made here; qualification must compare
same-cache execution, independent-cache execution, and sequential warm reuse under
the approved identity criterion. The existing `off` sentinel also returns early
without clearing an inherited JAX cache configuration; no behavior change to it was
made in this task. The local CPU processes started without an inherited cache setting.

## 9. Identity timeout diagnosis

For the default240k raw validation configuration:

* Parent random/data warm-up through interaction5,000, initial five-episode eval,
  fresh parameter save, fresh probe5,000 fit updates.
* SAC begins at5,000; check1 at6,000 (2,002 SAC updates), check2 at12,000
  (14,002 SAC updates),10,000 probe fit updates per check.
* Forced fork12,000, complete state save, panel, control trainer reconstruction
  and restore, exact Check1 and10-episode control eval0.
* Control runs1,000 additional interactions (2,000 SAC updates), snapshots13,000,
  stops. No scheduled check3 or next post-fork eval is due during that continuation.
* Identity process reconstructs/restores, checks exact identity, evaluates10 episodes,
  initially saves state, runs1,000 interactions/2,000 updates, saves snapshot, compares.
  No injection and no Check2 are performed in the identity arm.

Parent total16,002 SAC updates; identity2,000. The parent has not performed a
near-full120k-interaction experiment, but its setup is far larger than the2,000
SAC updates of the continuation alone. Four complete checkpoint writes and two
complete restores are required by the present test implementation. This is
consistent with a more expensive validation than its shorthand implies. **The
exact stage causing each3,600 s timeout remains unknown.** Avoid shortening probes,
skipping the fork path, or changing the starting checkpoint without lead approval.

## 10. Engineering changes

* `scale_rl/agents/sac/sac_agent.py`: one agent payload restore, including the
  churn reference recovered from metadata. Format, required contents, counters,
  optimizer and continuation remain unchanged.
* `experiments/exp12/runtime_trace.py` and `scripts/trace_exp12_runtime.py`:
  opt-in wrappers around the existing entry point. They report begin/end stages,
  aggregate calls/times, progress counters/check/evaluation counts, backend compiles,
  cache read/write/init, environment/replay/update/structural-metric work, probe
  targets/fits/rounds/checks, saves/restores, fork, Check1 and Check2. JSONL is flushed
  as work progresses and refuses to overwrite an existing trace. Normal experiment
  execution does not import/install these wrappers.

`--synchronize` blocks on returned JAX work and complete agent state at instrumented
boundaries. This deliberately removes asynchronous overlap and adds barrier overhead;
it measures synchronized diagnostic execution, not uninstrumented throughput. Barrier
call/time totals are recorded separately, and nested stage totals overlap. Without
that flag, device-only call times may measure dispatch; naturally blocking actions,
host transfers and whole probes still include their waits. Use paired short runs to
quantify instrumentation overhead. Do not sum nested compile/update/probe or barrier
totals as independent wall time.

Local real tiny DMC parent+injected pipeline, D1W16 actor8,400 parent interactions,
100 arm interactions, probe5×20/pool64/batch16, episode limit20, cache off:

| Synchronized CPU stage | Calls | Inclusive total seconds |
|---|---:|---:|
| Overall trace |1|13.4445|
| Backend compilation |468|5.8201|
| Trainer initialization |3|1.1499|
| SAC update_many |491|5.5813|
| Probe rounds / individual fits |135 /270|1.5837 /1.2707|
| Structural metrics |25|1.6392|
| Complete saves / agent saves |4 /4|0.1307 /0.1171|
| Complete restores / agent restores |2 /2|0.09578 /0.08992|
| Replay save / load |4 /2|0.00151 /0.00131|
| Check1 comparisons / panel Q+gradient |3 /4|0.000203 /0.5197|
| Check2 |1|0.5597|
| Post-fork evaluation |52|1.7249|
| Train environment stepping |500|0.07384|
| Cache initialization / disabled cache lookup |2 /468|0.0000294 /0.00770|
| Synchronization |10,181|0.4110|

Cache writes are absent because cache was off. This tests the trace and actual
execution paths; it does **not** measure production architecture cost or A100
read/write latency. Earlier local short training without probe callbacks showed
update-cache counts1,2,3,3,3 as interactions10,11,12,20,21 completed; warm interaction21
took about1.34ms on this tiny CPU configuration. Neither number is a production rate.

## 11. Regression and equivalence tests

`tests/test_exp12_runtime.py` adds:

1. One payload restore for both untrained(None churn reference) and trained agents.
   On the old implementation both subcases fail `2 != 1`. The fixed implementation
   checks exact saved values and exact subsequent optimizer/state/output equality.
2. Real tiny pipeline with and without synchronized tracing: complete persisted
   state has zero differences, including an explicit byte comparison of replay
   cursor/count/n-step metadata. Expected42 interactions,66 SAC updates,two scheduled
   checks,15 probe rounds are asserted.
3. Exception preservation and refusal to overwrite traces.

Final focused tests:3/3 passed in14.349 s, including the additional model-counter
coverage. The full CPU suite ran184 tests in657.175 s:181 passed,1 existing failure,
2 HumanoidBench dependency skips. The existing failure is
`TwinCheck1Test.test_panel_values`:67/512 Q entries differ, maximum absolute
difference7.1526e-7. The later gradient assertion is unchanged and remains subject
to the earlier numerical-criteria decision. Mutation/break checks:71 OK,1 existing
PROBLEM because that same restored twin test remains red. Both aggregate commands
exit1; this is not reported as a green qualification suite. Existing exact fork,
crash-resume and twin injection checks pass. Formatting and `git diff --check` pass.

Validation used Python3.12.13 /JAX0.4.34 CPU, `JAX_PLATFORMS=cpu`,
`MUJOCO_GL=disable`, `EXP12_JAX_CACHE_DIR=off`, `WANDB_MODE=disabled`,
`PYTHONPATH=/tmp/exp12-validation-deps`, and `MPLCONFIGDIR=/tmp/exp12-mpl`.
From `main/`, the commands were `../.venv/bin/python -m unittest discover -s tests
-t . -p 'test_exp12_*.py' -v` and `../.venv/bin/python -m tests.exp12_break_checks`.

Raw evidence remains local under `/tmp/exp12-runtime-*`, including focused old/new
logs, production-shape lowering summary, synchronized traces, full-suite and break
logs. Large raw outputs are not added to Git. The report records summarized evidence.

## 12. Remaining GPU uncertainties and release gates

Required before claiming a production runtime diagnosis: exact partial job counters
or a bounded production-path stage trace; CUDA initial compile and warm training
rates; structural-metric/window costs; each production-sized probe round/check;
checkpoint write/restore cost on the intended filesystem; cache hit/miss/read errors
and compile durations. HB additionally needs current twin kernels, both environments'
restore/evaluation behavior and memory, and independent identity qualification.

The restore optimization has exact CPU evidence and still needs the production
CUDA identity gate. The prior twin compiled/eager test remains red, GPU diagnostic
bounds remain unapproved/unmeasured, and methodology decisions from the earlier
audit remain pending. The runtime investigation does not clear Block HB or the grid.
No scientific criterion or inclusion decision was inferred from a performance issue.

## 13. Compute estimate

There is no honest new measured full-grid A100 forecast from the supplied totals.
The speed PASS wall times85/72/80 s include startup/warm-up, not just the600-step
window. They only bound the timed training rates below by7.06/8.33/7.50 interactions/s
respectively; the actual `it_per_s` values are in missing `job_0.json` files. They
cannot be substituted as measured steady-state speeds. The null total constrains
probe-average cost, not SAC throughput. The timed-out attempts are not extrapolated
as normal runs.

A reproducible **conditional forecast using the prior unverified rate estimates**
illustrates the population and eligibility effect. From the manifest, each
architecture has35M nominal interactions across65 parents:25M non-HB,10M twin HB.
Using non-HB76/61/43.6 it/s and HB88/56/32 it/s, with single-Q probe estimates
3/16/33.4 s and double for HB:

| Architecture | Nominal parent training GPU-h | Parent probes GPU-h (20.5 paired equivalents) |
|---|---:|---:|
|D2W512|122.94|1.28|
|D4W1024|163.45|6.83|
|D4W1536|246.08|14.26|
|Total195 parents|532.47|22.38|

This gives554.85 GPU-h before base evaluation, initialization/compilation, metrics
overhead beyond the rate benchmark, checkpoints, eligible arms and control extension.
It is a one-process-per-GPU accounting model; packing gains require measured aggregate
throughput. A crude dog-length evaluation proxy adds about24.5 h for20,250 nominal
base eval episodes; real Myo/HB episode lengths and environment costs differ.

If all130 scaled parents are eligible, original-architecture rates would add
**102.38 h of arm training**, plus up to**81.91 h of extra control training** for
latest eligible forks. Actual injected training adds
`40.86*k_1024 + 61.52*k_1536` hours, where each k is its measured slowdown relative
to the original critic on the same environment mix; these multipliers are unknown
and depend on frozen m. Both arms' post-fork evaluations add67,600 episodes at full
eligibility, about81.7 h under the same crude dog-length proxy, before extra probes,
Check2, injection compilation and saves. No eligibility rate is inferred from the
dev timeout.

Thus the illustrative zero-eligible scenario is roughly580–620 h; a full-eligibility
scenario at k=1 is roughly780–900+ h, with further injected slowdown and filesystem
cost unbounded by current evidence. These are low-confidence planning scenarios,
not measured revised predictions. **600–700 h is not a credible validated estimate
for the complete possible Exp1/2 campaign.** It could describe a limited eligibility
scenario under the old throughput assumptions. If the extreme behavior is intentional
and sustained, no trustworthy range can be given without its stage rates. After the
confirmed restore fix, the forecast changes only by avoided restore I/O; its A100
time saving is unmeasured and cannot repair a pre-check1 slowdown.

## 14. Smallest proposed follow-up, NOT executed

First obtain the already-generated job artifacts locally: zero additional GPU cost.
If they do not identify the stall, use a reviewed isolated checkout (never the
running Block B tree), one A100 matching the eventual grid, one process, D4W1536
dog-run. No HB install, identity grid, or new experiment campaign is required to
identify the failing stage. Example command to prepare for later approval/execution:

```bash
# Inside an independently approved allocation and reviewed checkout/main only.
# OUT must be a new absolute diagnostic directory on the intended filesystem.
EXP12_JAX_CACHE_DIR="$OUT/cache_single_writer" WANDB_MODE=disabled \
  timeout -k 15 300 python scripts/trace_exp12_runtime.py \
  --out "$OUT/trace.jsonl" --synchronize --progress-every 100 -- \
  scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D4W1536 --warmup_steps 30 --train_steps 60 --probe_repeats 1 \
  --fork_timing --eval_cost --out "$OUT/profile.json"
```

This retains actual architecture, batch, precision, UTD, repeat and5×1000/pool25,600
probe settings. Training length is an explicitly short diagnostic, not a reduced
confirmatory run. It performs two whole paired probes (compile warm-up+one timed
check), a synthetic replay-filled95%-budget save/restore to measure bytes/I/O, and
two ten-episode post-fork evaluations. It does not make a scientific degradation
claim or validate exact identity. The synthetic replay benchmark must be labeled as
such; its sampled reward/termination values are not a research trajectory.

Hard cap300 s plus15 s kill grace, at most0.0875 A100-h for this attempt. Stop and
inspect the last flushed begin/end/progress stage if incomplete; no automatic retry,
timeout increase or subsequent cell. Expected output: initial compile durations,
warm60-interaction training rate, per-round/fit/check times, save bytes/durations,
restore duration, evaluation time, private single-writer cache timings and summary.
Exact restore continuation still needs the separately reviewed identity gate.

Only if the first trace completes, propose a second short unsynchronized run to
measure instrumentation overhead and a sequential warm-cache run to measure cache
reuse, each separately bounded. Compare intended storage with node-local storage
only if the first trace identifies I/O as dominant. Reproduce the original Exp1
first-update path with its fresh callback if the profile succeeds but job artifacts
still indicate a stall there: standalone profile is not identical to that lifecycle.
No recommendation to submit HB follows merely from a successful microbenchmark.

## 15. Source state and validation status

At the close of the investigation, before preparation for Cloud continuation,
HEAD was `d3123c3c6db1d69e7a982dd1c7970d413139c23b`; no commits or pushes had been
performed for the investigation. The working tree contained the restore fix and new
trace CLI/module, runtime tests and this report. The two earlier local commits were
ahead of origin in that historical snapshot. The dirty tree must not be
presented as a clean qualified CUDA revision.

Local status at the close of the investigation (historical, before publication):

```text
## claude/eloquent-fermat-inxqlt...origin/claude/eloquent-fermat-inxqlt [ahead 2]
 M main/scale_rl/agents/sac/sac_agent.py
?? main/docs/exp12_runtime_investigation.md
?? main/experiments/exp12/runtime_trace.py
?? main/scripts/trace_exp12_runtime.py
?? main/tests/test_exp12_runtime.py
```

No timeout, assertion, tolerance, measured scientific quantity, training/probe/eval
schedule, cache-default policy or methodological inclusion rule was changed.

### Continuation in Cloud

The five files listed above were reviewed for publication on
`claude/eloquent-fermat-inxqlt`. The report and source/tests are portable; the raw
`/tmp/exp12-runtime-*` evidence and local dependency overlay are not included in
Git and will not be present in Cloud. Use the repository's dependency setup rather
than assuming the local `/tmp/exp12-validation-deps` path exists there.

The publication review reran 12 focused CPU tests, all passing: 10 runtime/single-Q
identity/twin-fork tests in 42.975 s, and 2 complete resume/mutation tests in
55.598 s. These retain exact value, optimizer/counter, continuation and complete
persisted-state comparisons. Formatting and staged whitespace checks passed. The
earlier full-suite and break-check results in section 11 remain historical results;
the existing twin-panel failure was not changed or hidden.

From `main/` with repository dependencies installed, rerun the focused selection:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=disable EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled \
  python -m unittest -v tests.test_exp12_runtime \
  tests.test_exp12_foundations.ResumeExactnessTest \
  tests.test_exp12_fork.ForkEndToEndTest.test_identity_fork_is_bit_identical \
  tests.test_exp12_twin_critic.TwinForkEndToEndTest
```

Continue from this report and the existing decision/qualification documents. The
six-hour timeout remains unexplained without copied Block B artifacts or an
independently approved bounded CUDA diagnostic. Exact production CUDA continuation
is still a release gate. Publishing this engineering work does not authorize Delta
access, jobs, the proposed benchmark, HB submission, the grid, or methodological
changes. Obtain the published source revision with `git rev-parse HEAD`; the
historical dirty-tree snapshot above is provenance, not the publication status.
