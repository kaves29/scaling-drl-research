# Exp1/Exp2 scientific and validation status

## Amendment (ab): independent launch gates (13 October 2026)

The owner approved conditional separation: Exp1 may launch after its own
scientific, final-source GPU, allocation and approval gates; Exp2 positive
control is no longer a prerequisite for Exp1 parents. Exp2 still requires a
natural eligible fork, source-matched positive control, frozen injection m and
its separate approval. The trigger checkpoint is retained for deferred arms;
the control continuation starts immediately from that same state. This is a
launch-order amendment only. Current status for **both** experiments is NO-GO
until the remaining individual gates pass. See
[`exp12_ab_deferred_fork.md`](exp12_ab_deferred_fork.md).

## Consolidated Method C engineering review (13 October 2026, Eastern)

The owner-authorized Method C settings remain unchanged. Claude's engineering
fixes at `b225577c2ba5a18b2ad6d14c47b34a00b5443e05` were incorporated
on `codex/exp12-method-c-consolidated`; resolve its final SHA from Git.
Review findings and qualification limits are in
[`method_c_consolidated_review.md`](method_c_consolidated_review.md). Exp1 and
Exp2 remain NO-GO pending the recorded scientific disposition, final-source
CUDA qualifications, Delta allocation preflight and explicit launch approvals.

## Method C owner decision (12 October 2026, Eastern)

The owner authorized Method C implementation: 2,000 continuous fit steps
with the same five target functions, original first-1,000 minibatches and
continued AdamW state. Zero trigger threshold, 95% bootstrap and two
consecutive scheduled checks remain. The approved-settings table below now
records Method C (amendment (aa)); the earlier 1,000-step probe describes
historical sources only. A/B/C job 22852240 on
exploratory source `18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af` observed
D4W1536 Hopper-Hop P/b 0.9467/0.9567/0.9552 and D4W1024 5/100 single-check
fresh-null firings. Neither establishes trained-critic sensitivity or a
two-consecutive-check false-trigger rate. See
[`method_c_exp12_readiness.md`](method_c_exp12_readiness.md) for the final
source's qualification and launch blockers. No GPU job or production run is
authorized by this status update.

## D4W1536 hopper capture update (2026-10-11)

Attached `hopper_capture_22817406.tar.gz`, SHA-256
`e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868`,
records an A100/CUDA run at clean source
`3d75a05789b9815cefd08cf3d8981a4be34c2352`. All eight capture files
match the embedded SHA-256 manifest. The five configured-pool fresh-critic
1,000-step fitting curves remain downward-trending at the final step; final
full-pool losses are .21933/.19825/.17626/.20636/.19532. Saved replay-pool
samples match the approved seeded with-replacement draws from 5,000 warm-up
rows exactly; normalized observations, teacher target variance and P=b-loss
arithmetic agree. P/b=.6023976348 again fails the approved >=.9 rule. This
establishes underfitting within the approved budget, with no confirmed
implementation defect and no proof of the cause of slow fitting. It does not
establish trained-critic plasticity degradation. The legacy sections below
describe the evidence available at their historical writing time. Current
launch commands and owner decisions are in
[`exp12_oct12_execution_handoff.md`](exp12_oct12_execution_handoff.md).

## Raw-null update, 2026-10-11

Both uploaded D4W1024 datasets now have verified unique rows, paired losses,
IQMs, CIs and firing flags: seed990=13/100 and seed991=6/100. Both fail the
approved ≤5/100 rule. Seed991 contains clean A100 source/runtime provenance;
the original source is documented by the separate Block B archive. Neither
dataset establishes the production consecutive-check false-trigger rate.
Hopper D4W1536 configured-pool P/b=.6024 still fails and original stepwise
curves/targets are unavailable. Details and remaining actions are in
`exp12_remaining_qualification.md`; no methodology or threshold changed.

## Dynamic scheduler preparation, 2026-10-11

The isolated `codex/exp12-production-readiness` branch prepares two
independently queued A100 allocations with one shared persistent Exp1/Exp2
ledger; see [exp12_dynamic_launch.md](exp12_dynamic_launch.md). This is
engineering preparation, not GPU validation or launch authorization. The
195-parent population, natural trigger, injection, scientific thresholds and
statistics remain unchanged. D4W1024 null, D4W1536 hopper range, six
final-source CUDA identity cells, positive-control/m, entropy and numerical
decisions still block the confirmatory campaign. Pilot 22800749 is unaffected.

## Production-readiness update, 2026-10-11

The W&B-online D4W1536 dog-run seed102 development pilot is queued as Slurm
job 22800749; its outcome is unknown here. The separate review at
`codex/exp12-production-readiness` is based on
`7d0883cfa99d27fed062d77a68cbbfa27e560061`. It does not touch the pilot.
See `exp12_production_decision_2026_10_11.md` for current NO-GO blockers and
the same-source Exp2 arm preflight. This update does not supersede original
evidence SHAs or certify a GPU gate.

## Isolated readiness update, 2026-10-10

No scientific status is promoted by the new `codex/exp12-launch-readiness` work.
Approved integration remains844e3c0. The owner reports small A100 cross-process
resume PASS at844e3c0 (job22783981) and independent D4W1024 null6/100 at9d6a82d
(job22779974). New raw artifacts are not available here for independent audit.
Both null samples13/100 and6/100 fail the existing≤5/100 criterion; threshold0
and every scientific setting remain unchanged. Full-width qualification and
guarded parent/pilot preparation are implemented, not GPU-qualified. Read
[exp12_full_grid_readiness.md](exp12_full_grid_readiness.md) for separate pilot,
Exp1-grid and Exp2 NO-GO lists and
[exp12_null_threshold_decision.md](exp12_null_threshold_decision.md) for the owner
decision. All earlier evidence tables below retain their historical scope.

The [readiness CPU receipt](exp12_launch_validation_report.md) records620 broad
passes/one optional skip,51 focused passes,74 mutation checks and65 restoration
cells. Production settings and all195 parent fingerprints remain unchanged.

This is a status index, not a methodology amendment. Read the canonical
[methodology and amendments](../.claude/methodology-exp1-exp2.md) and the
[approved decisions](exp12_decisions.md). Historical reports retain their original
revision and publication snapshots; [AGENT_HANDOFF.md](AGENT_HANDOFF.md) supplies
current integration provenance. Investigation scripts are opt-in tools, not
production entry points or authorization to adopt exploratory criteria.

## Approved settings retained unchanged

| Contract | Approved implementation |
|---|---|
| Grid and population | Actor D1W128; critics D2W512, D4W1024, D4W1536; 13 tasks x seeds 1–5 = 195 Exp1 parents; 130 potentially eligible scaled parents. |
| DMC tasks/budgets | dog-run, dog-trot, humanoid-run, humanoid-walk, humanoid-stand: 1M raw steps each; swimmer-swimmer15 and hopper-hop: 500k each. |
| MyoSuite tasks/budgets | myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach: 1M raw steps each. |
| HumanoidBench tasks/budgets | h1-reach-v0, h1-run-v0: 2M raw steps each. |
| Training | UTD=2, action repeat=2, one training env, batch256, replay1M, warmup5000; AdamW lr1e-4, actor/critic decay .01, temperature decay0; initial alpha .01; target tau .005; gamma .99 DMC/HB and .95 Myo. |
| Critic/entropy | Single Q for DMC/Myo; independent clipped twins for HB. Retain approved G1 target H*=−0.5|A|; unsigned original wording remains a documented interpretation question, not permission to change the sign. |
| Probe | Method C (amendment (aa)): five paired rounds, each with a fresh optimizer and its own teacher, fitted for 2000 continuous steps (original first-1000 minibatch stream, then its separate follow-up stream; parameters and AdamW state carried through), batch256, pool25600, sin(1e5 teacher) targets, fixed per-fit offset, fresh-critic comparison; 20 scheduled checks. Preserve literal per-network FP32 loss subtraction then twin mean. |
| Trigger | IQM, 10000 percentile-bootstrap resamples, 95% CI, strict lower bound >0; first two consecutive eligible checks, completed by check19 (95% budget). |
| Exp2 | Scaled-only natural eligible forks; complete-state restoration; matched control/injected arms; 25%-budget continuation capped at total1.2B. Post-fork evaluation every1% B, 10 episodes. |
| Injection | Approved Nikishin implementation and identity preservation; m is not frozen and cannot be chosen by an agent. |
| Checks/precision | Control/identity Check1 representation-exact Q and dQ/da; injected arm existing64 float32-eps scaled rule; Check2 paired reporting with failures retained. Preserve existing production precision and diagnostic FP32 policies, without selecting pending CUDA bounds. |
| Completeness | Immutable 195-parent population, provenance-aware arms and strict incomplete/missing/duplicate result refusal. Thin scheduling markers are not scientific certificates. |

The authoritative equations, evaluation details and amendments remain in the
linked specifications and production configs; this summary does not replace them.
All scientific configurations, SAC/training/probe/statistical implementation and
authority files remain byte-unchanged from recovered dcb7491. The reviewed
precision-module exception changes persistent-cache I/O publication only; it does
not change any precision setting or compiled payload.

## Evidence and validation

| Evidence | Exact source | Interpretation |
|---|---|---|
| Approved integrated candidate | `9d6a82d4aef81299901177865e383640439cde2a` | Scientific/engineering baseline; methodology reconstruction a3f44aa and candidate9938aef are already ancestors. |
| A100 job22765261 | `9d6a82d4aef81299901177865e383640439cde2a` | Independent audit in a100_job22765261_independent_audit.md; profile reached step6061/2124 SAC updates and completed step6000 transfer. Partial profile is not full-budget or identity qualification. |
| Exact comparator/reporting fixes | `dcb7491b61c094a961155c0c5e0a9d77d5d79f13` | Prior363 CPU tests,73 mutation checks,65 restoration checks; recovered/published. CUDA qualification of these fixes remains pending. |
| Isolated transfer experiment job22764211 | `18243cbc877b7dac9c1a6bb4b0fe71b2aa012fd7` | User-reported bulk timeout versus grouped success; historic diagnostic files now preserved. Not a substitute for full combined qualification. |
| Block B job22706349 | `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309` | Original null/range evidence must not be relabeled as final-source evidence. |
| Claude scientific follow-up | `8a888ef998dd2055d43faea26b78e8f555d00bc6` | Reports/scripts preserved with original history; findings and exploratory alternatives remain distinct from approved qualification. |
| Current integrated CPU checks | `53cc476b16bb7707484c7c049d2e77c0600568e2` | 613/613 repository tests (67 modules),5/5 additional analysis tests,73/73 mutations,65/65 restorations,195/195 exact config matches. Final protocol edits are documentation only; receipts and hashes are in AGENT_HANDOFF.md. |

## Unresolved scientific and GPU decisions

- **Fresh-null:** D4W1024 13/100 exceeds approved ≤5% gate. Original-data
  integrity is reported verified in null_followup_original_results.md. The
  pre-specified pair-level W test was not significant; post-hoc pair-structure
  findings are exploratory. Threshold remains zero; no p95 alternative is adopted.
  Lead disposition and any seed991 replication require approval.
- **Hopper range:** D4W1536 P/b=.602 is below approved .9. CPU investigations
  support a fitting-transient/budget limitation; no fitting budget, optimizer,
  architecture or acceptance rule is changed. Final-source GPU evidence and
  owner disposition remain pending.
- **Positive control:** recover a canonical natural full-setting dev fork,
  qualify noise/sensitivity and m comparisons, and obtain lead-frozen m. Current
  selection-rule behavior with negative recoveries/overshoot is reported, not
  silently repaired by introducing a new scientific criterion.
- **Identity GPU staging:** D4W1536 identity_warm has an observed-cost subtotal
  about329s before omitted work, exceeding the unchanged300s internal budget.
  Seven-minute Slurm allocation does not remove the internal cap. Hold this test;
  different staging or resources require explicit approval.
- **Remaining qualification:** final-source CUDA identity/injection/restore
  coverage for both scaled sizes and all suites, actual HB twins, pending CUDA
  numerical bounds, runtime/cache/production allocation and storage/recovery
  evidence. No CPU test grants these passes.
- **Other decisions:** unsigned entropy wording clarification, one-eligible-seed
  uncertainty convention, invalid-null-pair interpretation and any additional
  positive-control success semantics remain owner matters.

Status: **engineering integration CPU-validated and READY for shared development;
production launch BLOCKED; final-source GPU/scientific qualification UNVERIFIED.** No Delta access,
job submission, scientific change, resource increase, or main merge is authorized
by this document.

## Current cache/report integration and next development gate

The pinned scientific handoff cb33733, pilot report3734215 and atomic cache patch
d23191b are now reviewed and incorporated with original history. Cache follow-ups
are engineering-only. That cache-review executable/test checkpoint is
`813de566c4b169961fd4ec6a13063ac0ffa809d5`; resolve its enclosing published
integration HEAD as described in AGENT_HANDOFF.md. Final edits are documentation.
PR #1 at4afc772d is now separately authorized/reviewed, with original history
preserved and diagnostic-only follow-ups at
`78d04d167a5f65c35f6905cb261876bfe709d85a`. All production/scientific files
remain byte-unchanged from8bb0c0f. Exact final integration HEAD is delivered
separately; use a pinned detached checkout for qualification.

Fresh focused CPU coverage:74 passes/1 optional eviction skip out of75 tests;
25/25 fork/kill/resume/identity methods;56/56 methodology/config/analysis methods;
74 mutations (final deterministic cache oracle also checked independently);
65/65 canonical simulator restoration;195/195 exact resolved config matches.
Strict checkpoint-state restore checks passed3/3; intentional diagnostic metric
NaNs remain governed by the unchanged metric behavior. The final cache module
and CPU-adapted command harness passed7/8 with the same optional filelock skip.
The earlier613-test full suite is historical baseline evidence, not a newly run
full suite for this cache revision. See AGENT_HANDOFF.md for scope and receipts.

[exp1_a100_minimal_gate.md](exp1_a100_minimal_gate.md) prepares cache, restore,
in-process exact Check1/identity and pilot-size dog-run/D4W1536/seed102 smoke
checks, keeping one A100/four CPUs/32GB/seven minutes and the300-second workload
cap. None has run on CUDA at this revision. The historical kill tests force CPU
training subprocesses; an A100 rerun must not be called GPU resume qualification.
Full-width cold/warm identity and actual GPU cross-process restart remain gaps.

After those gates pass, the proposed full-length **development** pilot can be
considered for an explicitly approved allocation/execution. Its existing full
scientific settings and natural trigger remain unchanged. The eight-hour pilot
report allocation is a proposal, not an authorization. A seven-minute diagnostic
cannot run the full parent. Require genuine GPU resume evidence first if restart
qualification is a launch prerequisite. This is not clearance for the195-parent
campaign or Exp2, and no m, numerical bound, statistical alternative or failure
disposition is selected. No jobs were submitted or resources changed.

## First targeted A100 qualification after PR #1 review

[pr1_gpu_resume_review.md](pr1_gpu_resume_review.md) records23 final probe unit
passes,46 focused regression passes/one optional dependency skip, complete
eight-child CPU kill/resume PASS, five full-state equivalence comparisons
against the original probe, and195 unchanged parent fingerprints. Final
completed CPU endpoints are300 interactions/582 updates; actual step60/update102
restores, completion and state placement are observed. None is GPU evidence.

The first prepared job is the explicit crash95 GPU shard with two references
and a real checkpoint restore/replay, using the tracked
`scripts/sbatch_exp12_gpu_resume_gate.sh`. It requires exact source verification,
pinned JAX/CPU+CUDA, GPU-backed trained/saved/restored parameters, complete states
and approved exact comparison. Logs, specs, receipts and original checkpoint
are preserved even if the300-second workload cap interrupts execution. Exit2
reference variability is an observation with an unproven cause, not authorization
to relax identity or choose deterministic flags. Any nonzero/partial gate blocks
qualification. Timing on A100 is unmeasured.

The full-width identity_warm hold and all scientific decisions remain unchanged.
Current-source GPU Check1/fork and D4W1536 checkpoint smoke are subsequent relevant
checks; a tiny resume pass cannot be called full-width/Exp2 qualification. Beyond
GPU gates and trigger disposition, the owner must approve the full-length
development/positive-control role and appropriate allocation/execution. No
eight-hour resource proposal is adopted here. No demonstrated unresolved
production engineering defect was introduced by this review.
