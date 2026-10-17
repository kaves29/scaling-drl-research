# Branch consolidation review

Snapshot: 17 October 2026, after `git fetch --prune` from GitHub. Default branch
is `main` at `087feab341b9be1f377697668f2d1c750ec0467f`. Counts are unique
commits versus that main (`+main`) and versus the *proposed local cleanup tree*
(`+candidate`); they do not measure scientific value. This file is a deletion
plan, not deletion authorization. GitHub PR API returned Forbidden, so open PR
ownership could not be verified from this environment. Local worktrees were
listed, but remote collaborators may have other active checkouts.
Read-only Git ref inspection found `refs/pull/1/head` at `4afc772d5f2beab625e3b513e6f53cc72ea8e1c6`;
the ref alone does not say whether PR 1 is open or closed.

The cleanup candidate merges the current Exp96 lineage, the two unique
seven-parent Exp1 pilot commits, and the two numerical-isolation harness commits,
preserving ancestry. The Exp96 lineage already contains the updated Exp1/Exp2
recording-isolation work and ten-target candidate infrastructure. Its tip is not
itself approval to launch either study. The Exp3 branches are not folded into
Exp1/Exp2 main merely because they are recent.
The candidate merge adds no diff relative to `be1ffae` in `scale_rl/`,
`experiments/exp1.py`, `experiments/exp2_arm.py`, `experiments/exp12/`,
`configs/base_exp12.yaml`, `configs/base_exp12_ten.yaml`, or `requirements.txt`.
The seven-parent pilot adds an optional scheduler checkout argument plus
isolated runner, W&B helper, draft resources and tests; the numerical harness
adds isolated scripts/tests. The frozen `087feab` five-target config differs
from the newer lineage only in W&B `project_name` and a comment; do not call
that difference a scientific probe/training change or a source-equivalent job.

## CPU validation of the candidate

Using Python 3.12.14/JAX 0.4.34 on CPU, the isolated repository runner recorded
87 successful modules and 773 tests run. One optional compilation-cache test
skipped because `filelock` is absent; no module failed. Its source fingerprint
was `53997b282fd1280c3a9d5d6dd7ba9a41515c55660af6b61a2e9b084595f0a227`.
The 74 declared mutation checks passed. Environment restoration passed for all
65 task-seed cells (13 tasks × 5 seeds). The independent reference reported
195 parent commands, 39 architecture-environment configurations, and 130
scaled eligibility candidates. Focused pilot, harness, Exp96, and scheduler
tests passed 45/45. These CPU checks do not certify A100 concurrency,
statistical calibration, or source-bound launch approvals.

| Remote branch | Tip SHA | Latest commit purpose | +main | +candidate | Proposed disposition | Evidence/condition |
| --- | --- | --- | ---: | ---: | --- | --- |
| `main` | `087feab341b9be1f377697668f2d1c750ec0467f` | Parse reservation metadata strictly; block malformed unrecorded-failure records | 0 | 0 | KEEP | Frozen five-target scientific ancestor and current default. |
| `claude/eloquent-fermat-inxqlt` | `312bb01a7a6aa2e3a34a74dd557f05543081d0f5` | Keep CPU backend available in CUDA runtime diagnostic | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `claude/exp1-pilot-readiness` | `2879b64499490e93bc6374465a6b7b2bcbd2724e` | A100 Exp1 test failures: root causes (harness device-pinned comparison, over-strict Check 1 test, Hydra isolation; all fixed by 9d6a82d), GPU resume gap, minimal A100 gate and resume probe | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `claude/exp12-gate-decision-memo` | `ec74dee69b6b08c6ec07cbd386c2234f8a9c0092` | Exp12 gate decision memo: null and hopper range dispositions, launch path | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `claude/exp12-launch-blocker-audit` | `bc61fd6b555be39efac38b37609090f86dec0c8e` | Exp1/Exp2 launch-blocker audit: hard blockers, parallel work, deferrals, GO/NO-GO | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `claude/exp12-method-c-fixes` | `b225577c2ba5a18b2ad6d14c47b34a00b5443e05` | Pin historical range-capture test to its 1,000-step setting under Method C | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `claude/exp12-min-qualification` | `087feab341b9be1f377697668f2d1c750ec0467f` | Parse reservation metadata strictly; block malformed unrecorded-failure records | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `claude/exp12-production-launch` | `a092101952a29b0cf745f5564a8f3f5be299a63b` | Add read-only pilot trigger-trajectory analysis (numpy only) | 4 | 4 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `claude/exp3-capture-fixes-review` | `91e370a0c983452f3a5c1ce82b166ebd51d78ef6` | Review Codex Exp3 capture fixes; fix false retention conflicts on logically identical re-saves; reconcile methodology with recovered request | 9 | 9 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `claude/exp3-pilot-review` | `0b70602bbb5313a25879a526642cf9d5cb933e31` | Proposed Exp3 methodology and execution readiness (documentation only) | 8 | 8 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `claude/exp3-retention-integrity-fixes` | `459e700f61a88f55f03c9d6bc765122313bede84` | Exp3 retention integrity: validate existing snapshots and incoming checkpoints; explicit capture verdicts | 11 | 11 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `claude/gpu-resume-probe` | `4afc772d5f2beab625e3b513e6f53cc72ea8e1c6` | GPU cross-process resume probe and Block B A100 Exp1 test-failure analysis | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `claude/jolly-keller-lwr6xi` | `8709696d2a9272857a9569590b6a00fa51555e40` | Add Modal smoke-test harness for Angle 2A/2B/2C | 1 | 1 | INVESTIGATE | Historical independent implementation; verify reproducibility before archive. |
| `claude/null-threshold-decision` | `52fa7bfb6c33309b3b5afe42a5ff07368f036d81` | Fresh-null trigger-threshold decision framework for the owner: options vs seed-991 readings (analysis only, no setting changed) | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `claude/precision-cache-fix` | `d23191befd1d6a3254c61ac1edbad363b9b744e5` | Persistent compilation cache: write entries atomically (fixes "Error -5 ... truncated stream" reads) | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `claude/scientific-validation-investigation` | `cb33733aee96e882b10417316227a64bf732f38d` | Scientific handoff for integration/exp12: established results with source SHAs, seed-991 status unverified, blockers and decisions | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/a100-exact-identity-audit` | `dcb7491b61c094a961155c0c5e0a9d77d5d79f13` | Enforce exact identity comparisons and classify diagnostic control flow | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/a100-runtime-debug` | `f6b7f73b018eac365819ce64c90d22942b211f42` | Investigate A100 timeout and trace logging boundaries and synchronization failures | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp1-a100-gate-plan` | `8bb0c0fb80bdd5eb128c4d6c848c56c125420a3f` | Document reviewed cache integration and bounded Exp1 A100 development gates | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp1-seven-parent-pilot` | `cad7088dc2c32cecec74ed857d007a374b2fac75` | Tighten pilot checkpoint telemetry and format helpers | 2 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-d-option-followup` | `891dd7aa9df74446cb642ebbde624a992d547295` | Prepare separate ten-target exploratory follow-up | 3 | 3 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/exp12-d1-d2-consolidated` | `18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af` | Consolidate bounded D1 D2 exploratory A100 diagnostic | 2 | 2 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/exp12-d1-d2-exploration` | `777c579c8e02f51313c000d81ad142bec74ede51` | Add isolated D1 D2 probe-budget comparison and evidence replay | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/exp12-deferred-fork-ab` | `6eb397d0c485a5db202806df8a0308e48e46b05d` | Separate Exp1 launch gate from deferred Exp2 qualification | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-full-length-pilot-plan` | `ff8a10022c72d109836a859eae4ba30a7a730adb` | Audit uploaded gate receipts and ground pilot resource estimates | 3 | 3 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/exp12-gate-attempt-accounting` | `07721bf27055c2c407f635149bd9bbf54e97e989` | Validate retry-aware full-width gate attempt provenance | 1 | 1 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/exp12-launch-readiness` | `7d21eaa106a336168464cd5418a59136b6936a64` | Record Exp12 launch gates, null decisions and verified CPU readiness | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-method-c-consolidated` | `957be5c63a48b0b90dd28ee4b361e3c6d373ab05` | Consolidate Method C engineering fixes and independent review | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-method-c-production` | `c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b` | Implement continuous 2000-step Method C plasticity probe | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-metric-integration` | `9d6a82d4aef81299901177865e383640439cde2a` | Integrate bounded metric transfer into approved Exp12 candidate | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-numerical-isolation-harness` | `60f2787df10619f8f4ea678f6ebe73fc55f3d35a` | Checksum numerical isolation evidence on exit | 2 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-pilot-wandb-online` | `7d0883cfa99d27fed062d77a68cbbfa27e560061` | Enable online W&B for prepared Exp1 pilot launcher | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-production-readiness` | `1ea37e49a9279bc75e8390f2e702b746fff2eed8` | Prepare source-bound Exp12 positive control and launch handoff | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-qualification-debug` | `ae1e455c0de57ddd8304601ca35f62b334849766` | Complete CPU debugging of recording-isolation qualification | 3 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp12-recording-isolation` | `12839f656e53b00a3327384f2cb4fc1be47e73fe` | Keep Exp1/Exp2 actor diagnostics compiled across recording modes | 1 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/exp3-capture-fixes` | `853b6a523799132cf96a2e889150c10e3ea1c2ac` | Publish independent Exp3 capture review, source scope and validation receipts | 8 | 8 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `codex/exp3-pilot-implementation` | `142a7fc92ec8d19ff58e6022c6c60af84bc9ed4d` | Document Exp3 reuse, validation evidence and explicit execution interfaces | 3 | 3 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `codex/exp3-retention-final-review` | `0b287e479834de0fa095b2161749fc8eca353751` | Review Exp3 retention integrity with adversarial regressions | 10 | 10 | INVESTIGATE | Exp3 scope; preserve unique history outside Exp1/Exp2 main. |
| `codex/exp96-kaggle-t4-smoke` | `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` | Add Kaggle T4 Exp96 engineering smoke notebook | 10 | 0 | KEEP until jobs finish | Active frozen Delta source. |
| `codex/exp96-pending-gate` | `bd568d482c261f067add0b1d44bb57a6aceb52b1` | Gate pending Exp96 allocations before parent admission | 9 | 0 | KEEP until jobs finish | Active frozen Delta source. |
| `codex/method-c-96-infrastructure` | `f467bd92ab6b8dcb848641c4c508c7213a3f7761` | Add versioned 96-parent Method C campaign infrastructure | 8 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/method-c-ten-cpu-qualification` | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | Qualify versioned ten-target Method C on CPU | 7 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/method-c-ten-function-design` | `2137ccd78ecb7569903b7daf0d06601ac0bf7bf0` | Prepare isolated ten-target Method C validation design | 6 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/method-c-trained-pilot-prep` | `3c04a874d19be96b4dd0b4dd9df7ae9571081229` | Capture pilot resource telemetry and preserve workload exit | 5 | 5 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/method-c-validity-controls` | `539ad4d1362e16fef48d739075dc5ffb88668793` | Add exploratory Method C construct validation controls | 5 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/metric-transfer-fix` | `18243cbc877b7dac9c1a6bb4b0fe71b2aa012fd7` | Drain metric host copies per update group | 2 | 2 | INVESTIGATE | Unique commits not integrated; review evidence/use before archiving. |
| `codex/review-a100-resume-pr1` | `1999d35ebe6f56a6c278e42eedca5fc923fe0013` | Distinguish cache review checkpoint from latest diagnostic source | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/review-exp12-atomic-cache` | `813de566c4b169961fd4ec6a13063ac0ffa809d5` | Make atomic-publication mutation oracle deterministic | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/review-exp12-pilot-report` | `dc8fc37b1463b976dd24e45e4fe20da099ba2dcf` | Review and preserve pinned Exp1 development readiness report | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `codex/review-exp12-scientific-handoff` | `5365e43077631b09d62933eb130762790f283588` | Review and integrate pinned scientific handoff without adopting proposals | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |
| `delta-fixes` | `d52b0c121723b1ba9ba33850296b8e741ad25c25` | Fix checkpoint-path bug and add throughput optimizations | 1 | 1 | INVESTIGATE | Historical independent implementation; verify reproducibility before archive. |
| `integration/exp12` | `844e3c0cc24998b25c3d1e667bcbfe056b1b336c` | Integrate independently reviewed PR 1 and bounded A100 restore qualification | 0 | 0 | ARCHIVE after review | Tip is in candidate history; tag before remote deletion. |

## Safe sequence, pending owner review

### Local-only refs and worktrees

Local branches with no same-name remote ref are `codex/scaling-drl-consolidation`
(this candidate), `codex/numeric-isolation-probe-review` at
`122979546c1351ca9f2ed2a3403692cb4c224dcf` (unique local review),
`codex/exp3-capture-adversarial-review` at
`5764bd09d13c8c22fd6c0a81ead237f2f2ed2fdf` (Exp3 work), and historical
`work` at `6ffccbaf11bd8f60ce663b59ac546e9ff7cffd9c`. Preserve their commits
before pruning local refs. Ten local Git worktrees were inspected. Eight were
clean. `/workspace/method-c-validity-controls`, on
`codex/method-c-96-infrastructure`, has two modified files and one untracked
test; it was left untouched. The cleanup checkout had expected staged and
unstaged changes at inventory time. Remote collaborator checkouts and open PRs
remain unverified because this workspace's GitHub PR API returned Forbidden.

### Future branch responsibilities

- `main`: owner-reviewed shared source after consolidation; changes require a
  reviewed task branch and must retain source-specific approval boundaries.
- `codex/scaling-drl-consolidation`: review-only integration candidate until
  approved and merged; no Delta job should point at its moving tip.
- `codex/exp96-pending-gate` and `codex/exp96-kaggle-t4-smoke`: frozen source
  references for jobs `22945337` and `22952754` while those jobs remain active.
  Keep them even if their commits also become ancestors of main.
- Future `codex/<task>` and `claude/<task>` branches: one scoped owner and
  deliverable each; archive after review and after active job/PR references end.

### Execution order

1. Finish CPU validation and inspect the exact cleanup diff. Compare scientific
   config, SAC/probe, training budgets, thresholds, and numerical behavior with
   the intended parent commits. A future main move must be a reviewed merge or
   fast-forward that retains `087feab`, `bd568d`, and `be1ffae` as ancestors.
2. Keep both active Delta source branches while jobs `22945337` and `22952754`
   are pending or running. Their independent checkouts and source-bound approval
   files must not be retargeted to the candidate. A PASS at `be1ffae` does not
   satisfy the unchanged `bd568d` gate.
3. Before deleting any remote branch, resolve open PRs and active collaborators,
   create and verify an immutable archival tag at its exact full SHA, and record
   the tag mapping. Historical scientific branches and all Exp3 unique commits
   remain discoverable by tags and Git history. No tags or deletions are done
   in this task without a reviewed final plan.
   Proposed tag namespace: `archive/2026-10-17/<original-branch-name>`, whose
   target must equal the full tip SHA in the table above. Protect tags from
   movement and verify them on GitHub before considering a branch deletion.
4. After main is approved and merged, use `main` for new shared work and at most
   two active task branches: one for the active Exp96 deployment until it ends,
   and one for the next isolated engineering/scientific task. Do not rename or
   update the existing Delta jobs to achieve this count.

## File disposition

The only tracked generated artifact confirmed in the current tree is root
`__pycache__/sparseppo.cpython-313.pyc`; it is removed from the candidate.
The existing `.gitignore` already covers Python bytecode, local checkpoints,
logs and cache. Historical Angle experiment files are still imported by
`experiments/__init__.py`, Exp1/Exp2, or tests, so deleting them would break
current imports and reproducibility. Historical reports and evidence manifests
remain tracked because they explain scientific and engineering provenance.
No dependency is removed without an import/runtime proof. No checkpoints,
results, or Delta evidence are moved or deleted.

Examples of **historical-document archive candidates** are
`main/docs/exp12_oct12_execution_handoff.md`,
`main/docs/exp12_overnight_readiness.md`,
`main/docs/exp12_production_decision_2026_10_11.md`, and
`main/docs/exp12_launch_validation_report.md`. They describe earlier readiness
snapshots, not the current launch command, but each is referenced from two to
five other tracked files. Moving them requires a separate link migration and
review of source-specific evidence citations; they remain in place here.
