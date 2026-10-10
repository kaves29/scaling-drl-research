# Exp1/Exp2 scientific and validation status

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
| Probe | Five paired rounds, fresh optimizer, 1000 fitting steps, batch256, pool25600, sin(1e5 teacher) targets, fixed per-fit offset, fresh-critic comparison; 20 scheduled checks. Preserve literal per-network FP32 loss subtraction then twin mean. |
| Trigger | IQM, 10000 percentile-bootstrap resamples, 95% CI, strict lower bound >0; first two consecutive eligible checks, completed by check19 (95% budget). |
| Exp2 | Scaled-only natural eligible forks; complete-state restoration; matched control/injected arms; 25%-budget continuation capped at total1.2B. Post-fork evaluation every1% B, 10 episodes. |
| Injection | Approved Nikishin implementation and identity preservation; m is not frozen and cannot be chosen by an agent. |
| Checks/precision | Control/identity Check1 representation-exact Q and dQ/da; injected arm existing64 float32-eps scaled rule; Check2 paired reporting with failures retained. Preserve existing production precision and diagnostic FP32 policies, without selecting pending CUDA bounds. |
| Completeness | Immutable 195-parent population, provenance-aware arms and strict incomplete/missing/duplicate result refusal. Thin scheduling markers are not scientific certificates. |

The authoritative equations, evaluation details and amendments remain in the
linked specifications and production configs; this summary does not replace them.
All configurations, production source and scientific authority files are byte-
unchanged from recovered dcb7491 in this integration.

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

## Later remote work outside this validated baseline

The final refresh observed scientific handoff `cb33733aee96e882b10417316227a64bf732f38d`,
pilot-readiness branch `3734215e7c546e53871a7da87d6871b627bafbbe`, and atomic-cache
correction branch `d23191befd1d6a3254c61ac1edbad363b9b744e5`. These are not integrated
or covered by this baseline's CPU results. See AGENT_HANDOFF.md for ownership and
review actions. No pilot command, new resource estimate or cache change is adopted
by this publication.
