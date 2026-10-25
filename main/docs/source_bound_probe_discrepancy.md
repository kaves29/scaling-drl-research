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

## Fourteen-run pilot design — unapproved, not executable

The reviewable [manifest](../audits/probe_discrepancy/pilot_manifest.DRAFT.json)
preserves original source/configuration settings, hashes every full resolved
historical config and identifies exact historical parents. Eight large runs
(two seeds/task), four medium (one/task) and two small contrasts total14.
“Exceptional” means magnitude at35, not an invented opposite-sign case.

|Pilot|Environment|Critic|Seed|Selection|Original run ID|
|---|---|---|---:|---|---|
|P01|dog-run|D2W512|5|typical|7ic002ar|
|P02|dog-run|D4W1024|2|typical|2j9z776f|
|P03|dog-run|D4W1536|3|typical|htg273zh|
|P04|dog-run|D4W1536|6|exceptional|q0859o4f|
|P05|humanoid-run|D4W1024|5|typical|chhgyiuq|
|P06|humanoid-run|D4W1536|1|typical|jiotx3f8|
|P07|humanoid-run|D4W1536|4|exceptional|vi8gd0nw|
|P08|myo-key-turn|D4W1024|5|typical|zlwpdeln|
|P09|myo-key-turn|D4W1536|2|typical|deaf606s|
|P10|myo-key-turn|D4W1536|3|exceptional|gxta0ldp|
|P11|myo-pen-twirl|D2W512|7|typical|515e9nqz|
|P12|myo-pen-twirl|D4W1024|6|typical|w5gyiy2y|
|P13|myo-pen-twirl|D4W1536|5|typical|14fmtkz6|
|P14|myo-pen-twirl|D4W1536|7|exceptional|6apfwv2e|

All are source-bound **replications of selected existing seeds**, not14 new
independent confirmatory observations. Medium/small cells have one seed each;
they are diagnostic contrasts, not powered architecture estimates. Historical
run_role/config flags do not confer confirmatory status on this new pilot.
Original versus “corrected” arms are not invented because no correction exists.

The training source is f597ee9; the diagnostic adapter source is deliberately
null until implemented, reviewed and frozen. This task prepares the specification,
not an unqualified runner. A recording build is a different executable revision
and cannot borrow old-SHA GPU qualification. No commands/queue/submission are
embedded. Original result/W&B roots are excluded; the new root is unassigned.

### Endpoints and required retained evidence

Preserve500,000 nominal interactions, action_repeat2/1M raw steps, original
rewards/discounts/normalization, all50 ten-round scheduled checks, initial
warmup anchor, continuous2,000 AdamW updates, target formula and original
natural-fork untreated continuation. Do not shorten training/probes for cost or
force a trigger. Capture selected full evidence at checks0/5/35/50 and the
actual terminal check if an extension follows; do not interrupt the schedule.
Check0 has fresh only.

- Every check: original NPZ/CSV round b, final MSE, scores, offsets and pre-update
  minibatch curves; run/check/round IDs, source/runtime/config hashes and typed
  counters. Preserve original values and pair arithmetic independently.
- Selected checks: original raw replay indices/pool/actions, normalized pool,
  RMS count/mean/variance, training/fresh parameter hashes and trees, complete
  training checkpoint including optimizer/RNG/counters/simulator state.
- Per selected round: actual teacher parameters/key, teacher output, scaled
  phase, sine base, literal per-critic targets and minibatch-index sequence;
  loss/target dtype, nonfinite counts and value/distribution summaries.
- Pool forward predictions and MSE at0/100/500/1000/2000 for both critics;
  centered initial variance, covariance with base and residual distribution.
  Save observed Q/target vectors; do not infer them from minibatches or add
  updates. Keep raw fitting curves separate. No intermediate parameter-tree
  dump is required if exact observed vectors and their step/identity binding
  are retained.

Additional forwards/capture can change compilation/runtime even with no intended
parameter change. A future adapter must use existing values where possible,
never resample/reinitialize to obtain them, and demonstrate diagnostics-on/off
parameter, optimizer, outputs and RNG equivalence under **unchanged existing
criteria**. It must not count an extra pre/post diagnostic as an optimizer step.
The original GPU segmented-equivalence failure is unresolved and must not be
waived or hidden by this instrumentation. No such fit or GPU gate runs here.

### Discrimination rules decided before these new runs

- **Starting-residual explanation:** supports the simple version if Myo's
  trained disadvantage is already present in initial same-pool errors/centered
  residuals and tracks its final disadvantage within selected seeds; weakens it
  if initial errors are comparable or DMC starts worse but finishes better.
  Covariance versus variance is measured separately. Association alone is not
  a causal fraction of final error.
- **Endpoint catch-up:** supports it where trained DMC starts worse, loses at
  early same-pool endpoints, then reaches/eclipses fresh at2,000. Reject this
  trajectory description where DMC starts better or never catches up. A
  longer-fit experiment is not part of this plan.
- **Changing task:** separate Δb and Δfresh MSE within each recorded seed/check.
  Describe input diversity/RMS/teacher distribution changes against those
  components. This distinguishes variance-driven score drift from fitting-error
  drift without claiming that observational co-movement identifies its cause.
- **Runtime defect:** require an actual same-round identity, normalization,
  key, dtype, arithmetic or state mismatch against the source/retained arrays,
  then reproduce independently. A common offset or fixed-pool alternative is
  not silently introduced. Actual target-addition bounds versus measured gaps
  test direct roundoff; accumulated-path/sine effects require separate evidence.
- **Mixed/null result:** preserve it. If the14 runs do not discriminate, report
  the remaining ambiguity. No automatic larger campaign, retuned optimizer or
  “corrected DMC” acceptance requirement follows. Causal interventions would
  need a separately approved design.

### Resource estimate, stopping rules and exact launch gates

Selected W&B `_runtime` values sum to206.882 hours. These are process-runtime
metadata, sometimes from incomplete/crashed parents, not allocated physical
GPU-hours or a same-source isolated profile. As a conservative serial one-GPU
planning range propose207–414 GPU-hours, **unmeasured and not authorized**.
One40GB A100 process at a time,16 CPUs/64GiB host RAM, no packing; propose
200–500GiB durable storage pending actual state dimensions/retention profiling.
Large critics have≈75.7M weights (~.28GiB per float32 tree); teacher trees,
full training/optimizer/replay states and multiple checks dominate storage.
These estimates include contingency, not a qualified per-run walltime or charge.
The count is small but the unchanged recipe is not a cheap pilot. Prioritize
original artifact recovery before spending this budget; no historical2/4-hour
engineering budget transfers to it.

Remain blocked until all the following are reviewed for the **same final SHA**:

1. Supported read-only Delta/W&B route; actual source/artifact/runtime inventory.
2. Owner approval of this14-run scientific manifest, exploratory selection,
   observational endpoints, full source settings and remaining uncertainty.
3. Implement/review/freeze the diagnostic adapter and a fresh, clean source
   checkout. Bind training baseline plus adapter commit, full configs, new
   output/project identities and immutable queue hash. Never relabel f597 as
   the new executable SHA or reuse an old campaign root.
4. Focused CPU tests plus required simulator/checkpoint/optimizer/RNG restoration
   and diagnostics-on/off checks; reconcile PR12/PR14 changes explicitly rather
   than silently integrating unrelated fixes. CPU passes do not qualify GPU.
5. Same-source/runtime/model CUDA startup, original-vs-recording forward/update
   and continuous/segmented equivalence with existing tolerances; resolve the
   outstanding A100 failure. Preserve TF32/x64 rules and pinned software identity.
6. Source-bound unchanged campaign smoke/resource gates, live Slurm availability,
   account/quota, I/O/storage capacity, one-GPU resource profile, absolute budget
   and allocation caps. The full96-parent campaign gate cannot be repurposed as
   approval for a14-parent plan; retain or explicitly review the separate route.
7. Explicit owner launch authorization and genuine source-bound scientific/
   allocation approvals. None is supplied or synthesized here. Exp2 injection,
   main experiment launch and PR merges remain unauthorized.

On invalid/nonfinite data, identity/parameter/optimizer/RNG mismatch, corrupted
checkpoint, duplicate run, exceeded resource cap or any failed gate: preserve
artifacts and mark the attempt failed/incomplete, with no automatic retry or
new allocation. A scientific null or opposite sign is a result, not a failure.
Stop after the14 approved attempts/endpoints and review; never autoqueue main.

## Validation and reproducibility

Scripts live in `main/audits/probe_discrepancy`; the
[README](../audits/probe_discrepancy/README.md) gives reproduction commands,
not a request for owner execution. Evidence lives outside Git at
`/workspace/artifacts/source-bound-discrepancy-pilot/`. The tracked receipt binds
source/code/inputs, manifest, access log and checksums. Only audit scripts,
synthetic CPU tests, a non-executable design and documentation change.

Validation:22 CPU tests pass (12 new source/design checks including six guard
mutations, plus10 PR15 accounting tests). Fourteen Matplotlib/Pyparsing API
deprecation warnings are reported; no test fails/skips. Nine source guards,
controlled source routing and actual NPZ pairing pass. Existing raw reader
rechecks697 hashes and all4,834 IQMs. Black/Pyflakes/compileall/diff checks pass.
No broad fit/training test suite, original NN forward, new fit, RL training,
GPU/Slurm job, production change, historical rewrite or merge was performed.
