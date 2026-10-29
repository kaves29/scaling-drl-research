# Exp1/Exp2 scientific and validation status

## Owner-directed CPU review (28–29 October 2026)

Separate successor `codex/exp20-owner-decisions-cpu-20261028` preserves PR20 c931 and integration844. Owner selected PR20’s combined route; do not apply PR19 twice. The only runtime addition is signed per-m raw plasticity-loss reporting; methodology/config/probe/trigger/training/pilots remain unchanged relative to c931.

See the [single consolidated review](exp20_owner_cpu_review_20261028.md) and [recorded answers](exp2_owner_review_decisions_20261028.md). Implementation `03aa0b0b6179db01fb1bda8a7315d64f38006207`:654 unique CPU tests/74 modules,74 mutation checks,65 legacy simulator restoration cells; exact final-head source/receipts are external under `/workspace/artifacts/exp20-owner-cpu-20261028/`. Legacy13-task/195-config coverage is engineering compatibility, not restoration of the official scope.

The four-task/96-parent ten-sine historical source is distinct from PR20’s canonical five-sine protocol. All96 retained identities/configs verify;32 medium/large Myo histories contain eligible detector sequences, none observed in32 DMC histories (four DMC histories partial). These are not certified disk forks or null false-positive estimates. Missing5/100 raw calibration rows remain unavailable. No m was frozen, no scientific/GPU qualification claimed and no campaign or job execution performed. Official Exp3 method/capture decisions remain unspecified; the deadline covers all three official experiments. GO for review; NO-GO for production. The sections below retain historical snapshots and do not grant current authorization.


## Method C source integration candidate (28 October 2026)

Separate review branch: `codex/exp2-method-c-integration-candidate-20261028`,
from approved integration844. Exact Method C producer/config and (aa)/(ab)
amendments are combined with PR19's admission/certifier fixes; see the
[current integration report](exp2_method_c_integration_20261028.md) and
[per-file source map](exp2_method_c_integration_sources.json).

Implementation `a3d7efa151ce276191851bda040e979086d0592e` passes 650 unique CPU tests /73 modules,
74 mutation checks and65 simulator restoration cells. Final tested successor
SHA and focused results are in the external final validation receipt.

Separate draft review PR: https://github.com/kaves29/scaling-drl-research/pull/20

Current candidate defaults: five paired rounds,2,000 continuous AdamW updates
per target, first/next1,000 streams unchanged; no selected m or scientific
forced trigger. Approved Method C runtime matches main087 except PR19's
stricter source admission. This is GO for review, NO-GO for production.
Natural same-source evidence, owner-frozen m, CUDA/resource qualification and
launch-tool review/approval remain separate. No GPU jobs, training campaign,
production source promotion or active-pilot changes occurred. PR19 remains
draft at5482b2a; integration remains844e3c0.

The sections below are retained historical source/status snapshots. Their
1,000-step integration settings describe844, not this candidate's defaults.

## PR #19 Method C certifier follow-up (28 October 2026)

The stale `(5,1000)` curve checks are corrected at implementation checkpoint
`dbf9eb85ab25fedf67f2e560b5f118831fa696c6`: expected length is bound to the
canonical frozen configuration, so Method C requires exactly2,000 entries per
round. Producer, streams, continuous AdamW state, scoring and triggers are unchanged.
[Follow-up report](exp2_method_c_certifier_20261028.md) traces the active offline
Exp2 publication path and records80 PR CPU tests,75 on an isolated approved
Method C source with the same patch,74 mutation checks and65 simulator cells.
The independent Method C CPU checkout is `aff6b72282582c3ebee4e957d4557613003d5d09`;
it is not an integration merge or GPU qualification. PR19 remains draft:
https://github.com/kaves29/scaling-drl-research/pull/19.

Approved integration still lacks the Method C producer/config/amendment promotion.
Natural same-source evidence, owner-frozen m and GPU/resource/launch gates remain
blocked. Older readiness sections below retain their original source snapshots.
No training campaign, job, pilot, m or scientific setting was changed.

## Exp2 implementation/readiness review (28 October 2026)

Codex's isolated `codex/exp2-readiness-review-20261028` starts from approved
integration `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. Implementation checkpoint
`ee5e3d3398117197cfaab9efc657c830623ac698` adds only an Exp2 arm startup
source-admission guard and eight regressions; scientific/Exp1/pilot files are unchanged.
See [source-bound readiness review](exp2_readiness_20261028.md) for full CPU
receipts, version comparison, blockers and the December1/December5 dependency plan.

Production Exp2 remains **NO-GO**. Approved integration retains the legacy
five-round/1,000-step protocol; published main087 records approved Method C
(aa)/(ab), but its shared certifier still requires 1,000-step curves. A CPU array
fixture reproduced that compatibility defect. Neither the unintegrated Exp96
successor/PR2 nor alternative PR18 supplies a current Exp2 certificate. No m was
chosen, no source branch promoted, and no GPU/job/preflight/submission was run.
Natural source-bound evidence, owner-frozen m, correct versioned integration and
actual CUDA/resource qualification remain separate gates. Historical status below
retains its original scope; do not infer live jobs from it.

This is a status index, not a methodology amendment. Read the canonical
[methodology and amendments](../.claude/methodology-exp1-exp2.md) and the
[approved decisions](exp12_decisions.md). Historical reports retain their original
revision and publication snapshots; [AGENT_HANDOFF.md](AGENT_HANDOFF.md) supplies
current integration provenance. Investigation scripts are opt-in tools, not
production entry points or authorization to adopt exploratory criteria.

## Historical integration844 settings (not this candidate)

| Contract | Approved implementation |
|---|---|
| Grid and population | Actor D1W128; critics D2W512, D4W1024, D4W1536; 13 tasks x seeds 1–5 = 195 Exp1 parents; 130 potentially eligible scaled parents. |
| DMC tasks/budgets | dog-run, dog-trot, humanoid-run, humanoid-walk, humanoid-stand: 1M raw steps each; swimmer-swimmer15 and hopper-hop: 500k each. |
| MyoSuite tasks/budgets | myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach: 1M raw steps each. |
| HumanoidBench tasks/budgets | h1-reach-v0, h1-run-v0: 2M raw steps each. |
| Training | UTD=2, action repeat=2, one training env, batch256, replay1M, warmup5000; AdamW lr1e-4, actor/critic decay .01, temperature decay0; initial alpha .01; target tau .005; gamma .99 DMC/HB and .95 Myo. |
| Critic/entropy | Single Q for DMC/Myo; independent clipped twins for HB. Retain approved G1 target H*=−0.5|A|; unsigned original wording remains a documented interpretation question, not permission to change the sign. |
| Probe | Five paired rounds, fresh optimizer, 1000 fitting steps, batch256, pool25600, sin(1e5 teacher) targets, fixed per-fit offset, fresh-critic comparison; 20 scheduled checks. Preserve literal per-network FP32 loss subtraction then twin mean. |
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
