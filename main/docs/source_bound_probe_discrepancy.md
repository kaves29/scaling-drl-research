# Final source-bound paired-probe discrepancy investigation

## Diagnosis

**The cause remains unidentified.** Confidence is high in the recorded paired
sign contrast and limited in a mechanistic explanation. At check35/350,000
interactions, all32 medium/large DMC parents fit better than their contemporary
fresh controls, whereas all32 medium/large Myo parents fit worse. This is not
explained by target-variance subtraction or an initial-reference plotting choice.
No explanatory normalization, routing, target, optimizer or precision defect
has been demonstrated, so no production “fix” is implemented.

The strongest measured starting-state evidence is one original pen-small case:
large trained initial minibatch residuals, substantial prediction deviations
from its stored own offset, and a large remaining final pool error. It supports
a starting-residual/finite-budget mechanism **in that case**, but cannot explain
the medium/large DMC–Myo sign split without the missing original predictions,
pools and curves. Mean Q, rank loss or the falling fresh score cannot supply
those missing measurements. The proposed pilot discriminates these mechanisms;
it does not promise causal identification from additional observational logs.

This continues PR13–15, based on PR15 `822c2e5aae1aa99b970fdce8e0b4bff94e7d4724`.
Original campaign source is `f597ee9898e4d401912092c336c3e5330cf2fda1`.
The previous [component accounting](probe_difficulty_accounting.md),
[forensic report](forensic_probe_audit_20261024.md),
[metric follow-up](diagnostic_probe_disagreement_audit_20261024.md) and
[dataflow map](forensic_probe_dataflow.md) remain valid within their stated scope.
No previous control is relabeled as a new original-runtime test.

## Original-data paired contrast

Primary quantity is source IQM over within-round float32 fresh-score minus
trained-score. In real arithmetic this is trained final MSE minus fresh final
MSE; negative means trained better. The original NPZ's maximum difference
between source FP32 score subtraction and real-valued MSE subtraction is
7.8976154e-7. This roundoff is retained, not “corrected.” Shared b cancels before
IQM; subtraction of marginal IQMs is not substituted.

All96 canonical parents are included, with seven overlapping pilot records
excluded from independent-seed counts. New tables cover4,738 paired checks;
previous accounting retains48,340 round records including initial anchors.
Check35 has eight seeds in every cell; all96 also reach39, after which crashes
reduce coverage. Plots connect only observed checks. No failed run is imputed.

|Environment|Critic|Trained better at35|Mean paired IQM|95% seed-bootstrap interval|
|---|---|---:|---:|---|
|dog-run|D2W512|0/8|0.120598|[0.093608, 0.148318]|
|dog-run|D4W1024|8/8|-0.012484|[-0.016309, -0.009044]|
|dog-run|D4W1536|8/8|-0.029829|[-0.032722, -0.026872]|
|humanoid-run|D2W512|8/8|-0.022373|[-0.026679, -0.017998]|
|humanoid-run|D4W1024|8/8|-0.051968|[-0.060712, -0.042820]|
|humanoid-run|D4W1536|8/8|-0.151214|[-0.161654, -0.141548]|
|myo-key-turn|D2W512|0/8|0.592313|[0.541971, 0.645118]|
|myo-key-turn|D4W1024|0/8|0.458922|[0.445074, 0.474061]|
|myo-key-turn|D4W1536|0/8|0.464284|[0.412632, 0.516190]|
|myo-pen-twirl|D2W512|0/8|7.433970|[4.213995, 10.304484]|
|myo-pen-twirl|D4W1024|0/8|0.697953|[0.297236, 1.449907]|
|myo-pen-twirl|D4W1536|0/8|2.846334|[0.824150, 4.888074]|

Intervals resample eight training seed labels10,000 times with the fixed PR15
bootstrap algorithm. Rounds are not independent seeds. These exploratory
condition summaries do not replace the approved main-experiment estimator or
establish a population effect from two tasks/domain.

At check5/50,000 interactions, all32 medium/large DMC parents already beat
fresh, and all32 corresponding Myo parents already lose. Exceptions to a simple
“DMC always wins” description include small dog:7/8 trained better at5 but0/8
at35; small humanoid:2/8 at5 but8/8 at35. Some medium dog trajectories change
sign near the final check. Pen has large between-seed magnitude differences
while retaining the same late sign. See `paired_common_summaries.csv` and
`paired_seed_checks.csv`; each row binds run ID, check, raw JSONL row index
(zero-based), source and seed. Initial fresh decline occurs in all96 at35,
not just DMC, and none of those trajectories is monotonically declining.

The dedicated source-bound figure `paired_all12.pdf/png` shows the actual
same-task contrast. PR15's all12 nine-panel component trajectories and
DMC/Myo common-check seed uncertainty plots are retained in the evidence bundle,
with their original hashes and explicit blank unavailable component panels.
The paired Myo–DMC difference at35 remains positive at each architecture;
PR15 gives means3.96403/.610663/1.74583 for small/medium/large. These are
conditional descriptive contrasts after matching architecture/check/seed labels,
not fully adjusted environmental causal effects.

## Recovery: exact scope and failure evidence

A fresh scan of artifact, attachment, shared and library roots enumerated207
candidate files and10 ZIP/tar containers. Only one standalone original NPZ is
provenance-bound; archive copies do not add independent checkpoints. ZIP members
were listed and small NPZ bytes hashed, with no supplied pickle deserialized or
untrusted script executed. No original parameter/pool/RMS/teacher bundle was
recovered. Prior repository/scratch scans and PR15's inventory remain available.

`recovery_inventory.csv` lists48 early/late NPZ retrieval targets: checks5 and35
for a typical and an exceptional parent in **each of12 conditions**. Typical is
closest to the check35 paired-loss median, exceptional is farthest from it,
with deterministic seed tie-breaking. Selection is outcome-dependent and is
not an unbiased sample. Each requested row has run ID, seed, source, exact
expected path and an explicit unavailable hash/status. A49th row is the sole
available original case. Paths follow source config and earlier access records;
**they are retrieval targets, not live existence claims**.

Original root:
`/projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z/campaign/`.
Target hierarchy:
`checkpoints/exp1/<architecture>/<environment>/seed_<seed>/probes/check_XX.npz`.
Also request `probes/probe_checks.csv`, `fresh_critic/`, `run_metadata.json`,
`state/LATEST`, and `fork/state/LATEST`. Resolve actual pointers and hash the
complete referenced Orbax/replay/RMS/simulator state; never substitute the latest
state for check35. Routine states may have been pruned by the original retention
policy. A matching saved initial/late state does not by itself supply the exact
historical sampled pool, teacher or runtime; record reconstruction qualification.

Current access evidence (`access_attempts.json`):

- Delta project path and Mac context path are not mounted. No configured VPN,
  TCP destination grants, outbound identity or secret bindings exist here.
- CONNECT to each Delta login through the configured TCP proxy fails with
  connection refused **at the proxy**; this is not a Delta authentication result.
- HEAD requests through the inherited HTTPS proxy to both the original W&B
  host `api.forge.coreweave.com` and `api.wandb.ai` fail with proxy403 before
  reaching an authenticated API. No raw socket bypass or interactive login used.
- No local-agent execution connector is available. Previous Mac SSH access is
  not access from this Cloud process. No Slurm or remote storage query succeeded.

The supplied W&B file lists have19 unresolved `artifact/.../wandb_manifest.json`
references, size0/MD5null; manifest bodies are absent. The user's reported
read-only63b389qb query finding only a history artifact is useful external
context, not a new API verification here. We do not infer that other runs lack
remote checkpoints. Follow these references and original storage once a supported
read-only route is available; no placeholder is interpreted as an artifact.

The available archive is63b389qb, D2W512/pen/seed5/check50,
`raw_archived/D2W512_myo-pen-twirl_seed5_check50.npz`, SHA256
`66a223890220be7aa9d022e9cdfbcd90909d6580132e20315492a0ff307839ca`.
It retains10 paired rounds of b/final_loss/score/offset and2×10×2000 changing
minibatch losses. Scores bind exactly to raw W&B records. Median final MSE is
.1422703 fresh and14.8014 trained; MSE/b medians .285551 and29.4504. No original
full-pool initial/intermediate predictions are retained. The previous actual-scale
roundoff and starting-deviation bounds are in the bundled PR15 CSVs.

## Component accounting and what cannot be inferred

PR15 extracted every available per-round component. Only10 of48,340 round rows
have b/final errors; no other b/MSE was recovered in this task. Unavailable
components remain blank. MSE/b is undefined when b≤1e-12, without adding epsilon.
No b=.5 imputation is made. No source IQM or acceptance tolerance is changed.

Thus fresh drift cannot be quantitatively attributed to Δb versus ΔMSE_f over
time or by domain. The one original endpoint proves two rounds have lower b
than their own initial fresh score (and hence than initial b), but does not
identify the average drift or fresh-error trend. PR15's algebraic alternative
histories are explicitly **not estimates of original components**.

The observed paired contrast needs neither missing b nor that decomposition.
Its absolute magnitude and sign are identified from round operands. Sine-range
bounds additionally show substantial pen fitting error beyond anything target
variance could absorb, but do not explain the DMC advantage or establish every
Myo cell's absolute-MSE ordering. No trained-versus-fresh starting residual or
DMC endpoint catch-up can be reconstructed from mean training Q or minibatch
losses alone.

## Factor-by-factor disposition and executable checks

The [evidence ledger](../audits/probe_discrepancy/factor_evidence_ledger.csv)
records expected/actual behavior, source lines, measured magnitudes, affected
conditions, required direction, confounding and the smallest discriminator for
10 factors. Source citations bind the complete original SHA through the prior
[dataflow map](forensic_probe_dataflow.md), rather than mutable PR line numbers.

New CPU checks execute **selected original Git-source functions**, with imports,
decorators and annotations excluded, not copied production modules:

1. Original `sample_pool` runs on tiny controlled arrays. It normalizes once,
   preserves RMS/training RNG, and routes the selected observation/action rows.
2. Original `probe_round` runs with stand-in teacher/fitter and NumPy index
   generation. It routes distinct parameter objects, one shared teacher/base,
   identical pool/index objects and2,000 index rows. This performs **zero fits**
   and does not certify GPU numerics, real optimizer updates or original states.
3. Original `RunProbes.current_critic` selects the live trained object rather
   than the fresh sentinel. Original `paired_loss` reproduces the sole NPZ's
   score subtraction without modifying operands.
4. Nine AST guards verify own offset, continuous optimizer-state carry, final
   fitted full-pool evaluation, teacher construction, paired arguments/keys,
   one normalization and configured AdamW. Six controlled source mutations are
   rejected. These guards inspect source structure, not a numerical optimizer
   or original installed package. Earlier PR13 numerical controls are cited only.

Confirmed reporting defects remain distinct: stale W&B fresh IQM summary is
already bypassed by raw reconstruction; it cannot account for the paired split.
The inherited rank/dormancy naming limitations are not pair-arithmetic errors.
The source Myo factory does not forward its seed to `make`, and trainer resets
are unseeded at this boundary; exact same-seed simulator replay remains a gate,
not a newly demonstrated explanation of the paired sign or a silent seeding fix.
No original simulator execution is attempted in this task.

There is no new isolated production patch because no explanatory defect meets
all required tests. Changing target scale, offset, fitting length, normalization,
LR/weight decay, clipping or statistical thresholds would be a methodological
change, not a justified correction on this evidence.

## Revised ten-run simultaneous pilot — blocked, not executable

This supersedes the14-run serial draft. **Exactly ten pilot executions**, five
concurrent processes on allocated GPU0 and five on allocated GPU1; one barrier
release, no waves, replacements, retries or completion-driven admission.
Both GPUs carry the same two-large/two-medium/one-small condition mixture.
The [v2 manifest](../audits/probe_discrepancy/pilot_manifest.DRAFT.json) keeps
`launch_approved=false`, `executable=false`, `packing_qualified=false`, with
no diagnostic implementation SHA or output root assigned.

### Allocation and interpretation

|Pair|Task|Critic|Seed on both GPUs|Historical parent|GPU0 / GPU1 execution|
|---|---|---|---:|---|---|
|C01|dog-run|D4W1536|3|htg273zh|C01G0 / C01G1|
|C02|humanoid-run|D4W1024|5|chhgyiuq|C02G0 / C02G1|
|C03|myo-key-turn|D4W1536|2|deaf606s|C03G0 / C03G1|
|C04|myo-pen-twirl|D4W1024|6|w5gyiy2y|C04G0 / C04G1|
|C05|myo-pen-twirl|D2W512|5|63b389qb|C05G0 / C05G1|

Each row executes **once on each GPU with identical scientific configuration
and seed**. All ten output/checkpoint directories, process logs, completion
receipts and W&B IDs are distinct. Pairing balances every selected environment,
architecture, seed and condition exactly across devices. It deliberately
purchases device/process replication rather than ten independent seed draws:
there are five distinct scientific configurations, not ten independent seeds.
Same nominal seed alone does not guarantee identical realised Myo trajectories;
original simulator/RNG capture and qualification remain necessary. Differences
between device replicas are not automatically hardware causal effects.

The large dog/key pair contrasts domains at D4W1536; medium humanoid/pen does
so at D4W1024. Pen medium/small tests a size contrast in the task with highly
variable deficits. The small pen/seed5 counterpart anchors the one original
NPZ whose arrays bind to historical W&B. All four tasks and three sizes are
represented. The other four seeds retain the earlier median-based selections;
this is outcome-selected diagnostic replication. There is one unique seed per
cell, incomplete task×architecture coverage, and no within-cell independent-seed
uncertainty estimate. Task and size differences cannot be fully disentangled.

These runs can locate when the paired error sign emerges, inspect actual
normalization/teacher/residual/optimizer-routing evidence and detect mismatched
identities or numerical artifacts. They can refute simple starting-state or
endpoint-catch-up accounts in these cells. They **cannot** prove a general
DMC-versus-Myo effect, prove a plasticity hypothesis, assign a causal fraction
from observational correlations, or guarantee a fix. GPU replication is not an
additional scientific baseline. No corrected arm exists because no explanatory
defect has been established; no main experiment follows automatically.

### Scientific invariants and exact retained evidence

Every execution preserves its hash-bound original resolved scientific config:
500,000 nominal interactions/1M raw steps, original natural-fork untreated
extension, all50 checks plus initial anchor, ten rounds, pool25,600, batch256,
continuous2,000-step AdamW1e-4/.01 fitting, original teacher architecture/keys,
`sin(100000*f)`, TF32/non-matmul-float32 rules, rewards/discounts, replay/RMS,
statistics and thresholds. The five original scientific config hashes are
pinned in `source_config_bindings.json`. No concurrency-dependent science setting
or shorter endpoint is used to make packing fit.

**Identical target clarification:** original fresh/current fits share exact
inputs, teacher, sine base and minibatch sequence; each retains its own mean
prediction offset. Their literal targets therefore differ by the intended
constant. Preserve and hash **both** arrays, shared base and both offsets;
verify the shared base/input hashes. Calling those two literal arrays identical
would be incorrect; replacing own offsets with a shared offset would change the
method. The pilot makes neither change. Any newly requested genuinely identical
literal targets would require separate scientific review.

For every run, retain without smoothing/overwriting:

- **Every scheduled check/round:** original NPZ/CSV b, initial/final scores as
  actually measured, fresh/current final pool MSE, own offsets, paired float32
  operands/IQM and2,000 pre-update minibatch losses. Keep fresh-only check0
  distinct. Missing/invalid values are explicit, never zero-filled.
- **Checks0/5/35/50 and actual extension terminal check, all rounds:** exact raw
  replay indices, replay size/capacity/counters and raw observation/action pool;
  normalized pool, RMS count/mean/variance/epsilon and scales; teacher parameters,
  seed/fold-in keys, pre-scale outputs, phase, sine base and literal targets;
  dtype/shape/byte-order-aware hashes and source/run/check/round lineage. Preserve
  raw arrays and value ranges/nonfinite/duplicate counts, not only summaries.
- **Matched state/learning dynamics:** fresh/current initial trees/hashes,
  own-offset full-pool initial Q/variance/residual/covariance, full-pool Q/MSE at
  0/100/500/1000/2000; existing minibatch curves remain separately labeled.
  Keep complete training checkpoint/optimizer state, replay, simulator state,
  all RNG states/counters, checkpoint pointer/hash and full config/executable/
  baseline source/runtime hashes. Never regenerate targets/indices to record them.
- **Telemetry:** allocation node/physical GPU UUID/Slurm/cgroup/CPU affinity;
  per-process RSS and CPU/I/O counters; GPU VRAM/utilization/compute-process
  mapping; free host/cgroup/storage headroom; node-wide disk counters labeled
  as shared; cold compilation, steady SAC throughput, pool/teacher/fit/full-pool
  diagnostic/checkpoint I/O timing and byte counts; ready/release/exec timestamps.
  Source-bound adapter events must distinguish compile time from steady work.
- **Identity/failure:** unique canonical per-run directory plus deployment
  namespace/W&B ID/project, source-scientific and actual-deployment config hashes.
  Preserve all failed/partial outputs, compiler/process logs and receipts.
  No deletion, replacement slot, automatic restart or extra pilot attempt.

The retention adapter is **not implemented/reviewed/qualified** by this revision.
It must record existing values without training RNG consumption or state changes.
Extra forwards/capture can alter compilation/runtime; existing diagnostics-on/off
and continuous/segmented criteria remain unchanged. No numerical threshold is
relaxed. The outstanding A100 equivalence issue remains a launch gate.

### Mechanism discrimination, without altering the hypothesis

Starting-state support requires the trained Myo disadvantage to be present in
same-pool initial residuals and to track final gaps; comparable initial errors or
DMC starting worse then winning weaken that account. Measure centered variance
and prediction–base covariance separately; mean Q cannot replace them.

Endpoint catch-up requires actually measured early same-pool trained losses
above fresh, then convergence/overtaking by2,000. If DMC starts better, that
specific catch-up account is refuted. Do not add fitting steps or substitute
minibatch points. Decompose fresh-score drift into observed Δb and ΔMSE_f;
relate these descriptively to measured replay/RMS/teacher changes. Runtime-defect
claims still require a reproducible source/runtime mismatch, correct direction/
magnitude and an independent check. Mixed or null results are valid outcomes.

### Actual Delta status and resource feasibility

**Five-process-per-GPU qualification is unavailable.** This Cloud task has no
Delta mount, VPN, TCP destination grant or configured remote credential. The
read-only inspection captures absent `sinfo`, `scontrol` and `nvidia-smi` binaries
here; these are Cloud observations, not Delta measurements. Earlier TCP proxy
connection refusal/W&B proxy403 remain access evidence, not evidence that Delta
storage or GPUs are unavailable. No live node, scheduler, CPU/RAM, VRAM, disk
load or exact-source compilation/throughput profile has been obtained.

The existing `exp96_campaign.py`/`sbatch_exp96_allocation.sh` route is a one-GPU,
96-parent, completion-driven worker. Its packing estimates/old profiles do not
qualify this ten-run simultaneous layout. It is not reused or silently modified.

The **exact candidate request**, conditional on measured feasibility, is one
`gpuA100x4` node, account `biqc-delta-gpu`, two A100-SXM4-40GB GPUs, one Slurm
controller task,32 CPUs,128GiB host RAM and48h. Within that reservation, ten
children receive three disjoint CPU IDs each and the controller two; each GPU
hosts the same five-condition mix. Actual CPU topology/affinity, memory cgroup,
GPU UUIDs and scheduler limits must match the qualification receipt. If that
reservation is not available or sufficient, this request remains blocked.

Historical selected W&B runtime values, counted once for each of ten executions,
sum to134.313h. They are process-runtime proxies, not allocation/GPU usage.
Neither division by five nor simple multiplication of a single-run profile
establishes packed runtime. A48h two-GPU cap reserves96 physical GPU-hours;
completion within it is **unproven**. Storage200–500GiB remains a planning range.
Two large≈75.7M-weight critics plus two medium and a small per GPU can have
substantial simultaneous teacher/optimizer/compilation peaks; five fits must be
measured, not assumed from parameter bytes or disabled JAX preallocation.

Representative qualification must use the final clean executable source,
adapter, full unchanged configs, exact five-process mixture **on both GPUs at
once**, same runtime/driver/device model, binding/allocator/CPU policy and
telemetry. Retain traces covering all ten live processes, cold compilation,
steady SAC updates, a full scheduled ten-round paired2,000-step probe and
checkpoint/storage overlap, plus projection of all unchanged endpoints/extensions
with reviewed headroom. Per-process serial runs, a one-process smoke, a five-run
wave followed by another wave, synthetic receipts or a different source do not
qualify it. CPU mocks test orchestration only. No qualification job is submitted
here; any future qualification workload/budget needs separate review.

There is **no currently qualified executable alternative** for the unimplemented
adapter. The closest exact-count alternative if packing fails would be two GPUs
with lower concurrency and waves after separate qualification; it violates this
request and is **not** an allowed fallback. If simultaneous ten-run launch is
non-negotiable, propose a separately reviewed larger-memory/more-GPU allocation
supported by actual Delta inventory and requalify it. That changes the requested
layout and needs approval. The controller never makes either substitution.

### Reviewed launch path, command and fail-closed preflight

The new `scripts/sbatch_echocritic_diagnostic_pilot.sh` starts one Slurm step
exposing **both** allocated devices to `audits.probe_discrepancy.launch`.
The controller preserves the two inherited `CUDA_VISIBLE_DEVICES` tokens;
each child receives the appropriate physical token **before JAX import**.
Each child sees local GPU0, while the manifest's GPU0/GPU1 refer to the two
allocation slots. Never assume they are physical node indices0/1. UUID, model,
CPU affinity, host-memory allocation/cgroup and runtime checks fail closed.

All ten children are spawned once and report ready before a common GO file.
They must all be alive at release. Release-to-exec skew is recorded and bounded
by a reviewed operational limit; this is not a requirement that heterogeneous
runs execute training steps in lockstep. Each PID then execs only the reviewed
adapter. No dispatcher replenishes a free slot. The child bootstrap initializes
and verifies JAX, then execs the adapter; that extra startup/context lifecycle
is part of the source/configuration requiring packing qualification.

The controller monitors GPU/host/cgroup/storage headroom and allocation time.
On failed readiness, child failure, resource/gate mismatch or timeout it stops
remaining children, retains all partial evidence and writes failed/partial
receipts. Process exit0 alone is insufficient: distinct source/plan/config-bound
adapter completion receipts must also attest full endpoints and diagnostics.
Even engineering completion is not scientific certification. Fast transient
memory peaks can occur between telemetry samples; representative peak profiling
is still required. Adapter signal handling/checkpoint durability need review.

Preflight requires source/manifest/plan hashes, clean matching checkout, reviewed
implemented adapter, pinned Python/JAX/jaxlib/Optax/CUDA/driver, exact allocation,
new root and globally unused W&B identities. Ten independent gate receipts with
hash-checked underlying evidence are mandatory: adapter review, scientific
review, CPU restoration/equivalence, CUDA smoke, simultaneous five/five profile,
storage profile, fresh live Delta layout, W&B identities, allocation review and
owner launch. The profile must fit measured VRAM/host RAM and projected full
endpoints within reviewed caps. Expired layout evidence, wrong mixture/source,
missing raw receipts or still-closed manifest flags block command generation.

The [review checklist](diagnostic_pilot_10_review_checklist.md) lists the exact
remaining gates. `launch_approval.template.json` contains no approvals. Only
readiness flags/status may be sealed after qualification; `plan_hash` binds every
other plan field, including configurations, layout, executable SHA, root and
runtime policy. Approval additionally binds the complete sealed manifest bytes.
Changing any substantive field requires new matching receipts. Do not edit flags
by hand to bypass gates or reuse old source smoke/allocation approvals.

From the reviewed checkout's `main/`, the supported interface is:

```bash
python -m audits.probe_discrepancy.launch preflight \
  --manifest "$SEALED_EXTERNAL_MANIFEST" --approval "$REVIEWED_EXTERNAL_APPROVAL" \
  --checkout "$FROZEN_CHECKOUT"
python -m audits.probe_discrepancy.launch command \
  --manifest "$SEALED_EXTERNAL_MANIFEST" --approval "$REVIEWED_EXTERNAL_APPROVAL" \
  --checkout "$FROZEN_CHECKOUT"
```

These document future agent-operated execution, not commands for the owner to
run. There is no submit subcommand and this task executes no `sbatch`. The
`command` action prints a fully quoted exact command **only after preflight**.
Its conditional shape (placeholders are not an authorized runnable command) is:

```bash
sbatch --parsable --account=biqc-delta-gpu --partition=gpuA100x4 \
  --nodelist=<source-qualified-node> --nodes=1 --ntasks=1 --gres=gpu:2 \
  --cpus-per-task=32 --mem=128G --time=2880 \
  --job-name=echocritic-diagnostic-10 --chdir=<frozen-checkout>/main \
  --output=<new-root>/slurm-%j.out --error=<new-root>/slurm-%j.err \
  <frozen-checkout>/main/scripts/sbatch_echocritic_diagnostic_pilot.sh \
  <qualified-python> <sealed-manifest> <reviewed-approval> <manifest-sha256> <approval-sha256>
```

Actual node/UUID/Python/root values cannot be filled honestly before live
qualification and approvals. Current preflight returns BLOCKED/nonzero and
prints no sbatch command. A direct manual `sbatch` bypass could consume an
allocation before startup rejection; use only the gated command route. At
startup the wrapper/controller repeat all gates and verify submitted file
hashes before any child can start, preventing a changed-file launch. No old
2/4h engineering budget or96-parent approval transfers to this pilot. No Slurm
job, merge or main experiment launch is authorized by this revision.

## Validation and reproducibility

Scripts live in `main/audits/probe_discrepancy`; the
[README](../audits/probe_discrepancy/README.md) gives reproduction commands,
not a request for owner execution. Evidence lives outside Git at
`/workspace/artifacts/source-bound-discrepancy-pilot/`. The tracked receipt binds
source/code/inputs, manifest, access log and checksums. Only audit/launch-control scripts, synthetic CPU tests, a blocked design and
documentation change. The adapter and actual Delta qualification remain absent.

Validation:72 CPU tests pass, including six source guard mutations,
ten prior accounting tests, gate rejection checks and a mocked all-ten barrier/
five-five dispatch. Fourteen Matplotlib/Pyparsing API
deprecation warnings are reported; no test fails/skips. Nine source guards,
controlled source routing and actual NPZ pairing pass. Existing raw reader
rechecks697 hashes and all4,834 IQMs. Black/Pyflakes/compileall/diff checks pass.
No broad fit/training test suite, original NN forward, new fit, RL training,
GPU/Slurm job, production change, historical rewrite or merge was performed.
