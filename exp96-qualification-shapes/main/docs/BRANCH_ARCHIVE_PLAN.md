# Remote branch archival plan

Snapshot: 17 October 2026, compared with the proposed consolidation tree.
This is a review list, **not** deletion authorization. Re-fetch and verify every
live tip, open PR, collaborator worktree, and active source-bound deployment
immediately before any future deletion. Do not move a frozen checkout or rewrite
a commit. `main` is never a deletion candidate.

The review branch will merge or fast-forward into `main` only after owner review.
After that, the 31 ancestor branches below have their exact tips in main history.
The 20 divergent branches have annotated archival tags at their exact tips,
published and dereferenced against the live remote tips on 17 October 2026.
Their unique commits are not silently merged into the current
Exp1/Exp2 source. In particular, Exp3 and unapproved scientific alternatives
stay historical until separately reviewed. The two Exp96 frozen source branches
remain temporarily while they are referenced by deployment workflows.

## Candidate branches with tips already in consolidation history

| Branch | Exact tip | Condition before deletion |
| --- | --- | --- |
| `claude/eloquent-fermat-inxqlt` | `312bb01a7a6aa2e3a34a74dd557f05543081d0f5` | After main integration and open PR/worktree check. |
| `claude/exp12-method-c-fixes` | `b225577c2ba5a18b2ad6d14c47b34a00b5443e05` | After main integration and open PR/worktree check. |
| `claude/exp12-min-qualification` | `087feab341b9be1f377697668f2d1c750ec0467f` | After main integration and open PR/worktree check. |
| `claude/gpu-resume-probe` | `4afc772d5f2beab625e3b513e6f53cc72ea8e1c6` | After main integration and open PR/worktree check. |
| `claude/precision-cache-fix` | `d23191befd1d6a3254c61ac1edbad363b9b744e5` | After main integration and open PR/worktree check. |
| `claude/scientific-validation-investigation` | `cb33733aee96e882b10417316227a64bf732f38d` | After main integration and open PR/worktree check. |
| `codex/a100-exact-identity-audit` | `dcb7491b61c094a961155c0c5e0a9d77d5d79f13` | After main integration and open PR/worktree check. |
| `codex/a100-runtime-debug` | `f6b7f73b018eac365819ce64c90d22942b211f42` | After main integration and open PR/worktree check. |
| `codex/exp1-a100-gate-plan` | `8bb0c0fb80bdd5eb128c4d6c848c56c125420a3f` | After main integration and open PR/worktree check. |
| `codex/exp1-seven-parent-pilot` | `cad7088dc2c32cecec74ed857d007a374b2fac75` | After main integration and open PR/worktree check. |
| `codex/exp12-deferred-fork-ab` | `6eb397d0c485a5db202806df8a0308e48e46b05d` | After main integration and open PR/worktree check. |
| `codex/exp12-launch-readiness` | `7d21eaa106a336168464cd5418a59136b6936a64` | After main integration and open PR/worktree check. |
| `codex/exp12-method-c-consolidated` | `957be5c63a48b0b90dd28ee4b361e3c6d373ab05` | After main integration and open PR/worktree check. |
| `codex/exp12-method-c-production` | `c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b` | After main integration and open PR/worktree check. |
| `codex/exp12-metric-integration` | `9d6a82d4aef81299901177865e383640439cde2a` | After main integration and open PR/worktree check. |
| `codex/exp12-numerical-isolation-harness` | `60f2787df10619f8f4ea678f6ebe73fc55f3d35a` | After main integration and open PR/worktree check. |
| `codex/exp12-pilot-wandb-online` | `7d0883cfa99d27fed062d77a68cbbfa27e560061` | After main integration and open PR/worktree check. |
| `codex/exp12-production-readiness` | `1ea37e49a9279bc75e8390f2e702b746fff2eed8` | After main integration and open PR/worktree check. |
| `codex/exp12-qualification-debug` | `ae1e455c0de57ddd8304601ca35f62b334849766` | After main integration and open PR/worktree check. |
| `codex/exp12-recording-isolation` | `12839f656e53b00a3327384f2cb4fc1be47e73fe` | After main integration and open PR/worktree check. |
| `codex/exp96-kaggle-t4-smoke` | `be1ffae7a8dd9d43bc35a6c8c4b0be9e5582b574` | Retain until source-bound deployment is complete; then confirm no open PR/worktree. |
| `codex/exp96-pending-gate` | `bd568d482c261f067add0b1d44bb57a6aceb52b1` | Retain until source-bound deployment is complete; then confirm no open PR/worktree. |
| `codex/method-c-96-infrastructure` | `f467bd92ab6b8dcb848641c4c508c7213a3f7761` | After main integration and open PR/worktree check. |
| `codex/method-c-ten-cpu-qualification` | `5fe2ff65ae96cf900b23bf9810f79464730def9d` | After main integration and open PR/worktree check. |
| `codex/method-c-ten-function-design` | `2137ccd78ecb7569903b7daf0d06601ac0bf7bf0` | After main integration and open PR/worktree check. |
| `codex/method-c-validity-controls` | `539ad4d1362e16fef48d739075dc5ffb88668793` | After main integration and open PR/worktree check. |
| `codex/review-a100-resume-pr1` | `1999d35ebe6f56a6c278e42eedca5fc923fe0013` | After main integration and open PR/worktree check. |
| `codex/review-exp12-atomic-cache` | `813de566c4b169961fd4ec6a13063ac0ffa809d5` | After main integration and open PR/worktree check. |
| `codex/review-exp12-pilot-report` | `dc8fc37b1463b976dd24e45e4fe20da099ba2dcf` | After main integration and open PR/worktree check. |
| `codex/review-exp12-scientific-handoff` | `5365e43077631b09d62933eb130762790f283588` | After main integration and open PR/worktree check. |
| `integration/exp12` | `844e3c0cc24998b25c3d1e667bcbfe056b1b336c` | After main integration and open PR/worktree check. |

## Divergent branches requiring archival refs

For each row, an annotated tag was published and verified:
`archive/2026-10-17/<branch>` pointing at the exact tip below. This preserves
all reachable historical objects without putting a superseded implementation
into the current source tree. Only then consider deletion after checking that no
active PR, collaborator or deployment still relies on the branch.

| Branch | Exact tip | Unique commits | Proposed archive tag |
| --- | --- | ---: | --- |
| `claude/exp1-pilot-readiness` | `2879b64499490e93bc6374465a6b7b2bcbd2724e` | 1 | `archive/2026-10-17/claude/exp1-pilot-readiness` |
| `claude/exp12-gate-decision-memo` | `ec74dee69b6b08c6ec07cbd386c2234f8a9c0092` | 1 | `archive/2026-10-17/claude/exp12-gate-decision-memo` |
| `claude/exp12-launch-blocker-audit` | `bc61fd6b555be39efac38b37609090f86dec0c8e` | 1 | `archive/2026-10-17/claude/exp12-launch-blocker-audit` |
| `claude/exp12-production-launch` | `a092101952a29b0cf745f5564a8f3f5be299a63b` | 4 | `archive/2026-10-17/claude/exp12-production-launch` |
| `claude/exp3-capture-fixes-review` | `91e370a0c983452f3a5c1ce82b166ebd51d78ef6` | 9 | `archive/2026-10-17/claude/exp3-capture-fixes-review` |
| `claude/exp3-pilot-review` | `0b70602bbb5313a25879a526642cf9d5cb933e31` | 8 | `archive/2026-10-17/claude/exp3-pilot-review` |
| `claude/exp3-retention-integrity-fixes` | `459e700f61a88f55f03c9d6bc765122313bede84` | 11 | `archive/2026-10-17/claude/exp3-retention-integrity-fixes` |
| `claude/jolly-keller-lwr6xi` | `8709696d2a9272857a9569590b6a00fa51555e40` | 1 | `archive/2026-10-17/claude/jolly-keller-lwr6xi` |
| `claude/null-threshold-decision` | `52fa7bfb6c33309b3b5afe42a5ff07368f036d81` | 1 | `archive/2026-10-17/claude/null-threshold-decision` |
| `codex/exp12-d-option-followup` | `891dd7aa9df74446cb642ebbde624a992d547295` | 3 | `archive/2026-10-17/codex/exp12-d-option-followup` |
| `codex/exp12-d1-d2-consolidated` | `18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af` | 2 | `archive/2026-10-17/codex/exp12-d1-d2-consolidated` |
| `codex/exp12-d1-d2-exploration` | `777c579c8e02f51313c000d81ad142bec74ede51` | 1 | `archive/2026-10-17/codex/exp12-d1-d2-exploration` |
| `codex/exp12-full-length-pilot-plan` | `ff8a10022c72d109836a859eae4ba30a7a730adb` | 3 | `archive/2026-10-17/codex/exp12-full-length-pilot-plan` |
| `codex/exp12-gate-attempt-accounting` | `07721bf27055c2c407f635149bd9bbf54e97e989` | 1 | `archive/2026-10-17/codex/exp12-gate-attempt-accounting` |
| `codex/exp3-capture-fixes` | `853b6a523799132cf96a2e889150c10e3ea1c2ac` | 8 | `archive/2026-10-17/codex/exp3-capture-fixes` |
| `codex/exp3-pilot-implementation` | `142a7fc92ec8d19ff58e6022c6c60af84bc9ed4d` | 3 | `archive/2026-10-17/codex/exp3-pilot-implementation` |
| `codex/exp3-retention-final-review` | `0b287e479834de0fa095b2161749fc8eca353751` | 10 | `archive/2026-10-17/codex/exp3-retention-final-review` |
| `codex/method-c-trained-pilot-prep` | `3c04a874d19be96b4dd0b4dd9df7ae9571081229` | 5 | `archive/2026-10-17/codex/method-c-trained-pilot-prep` |
| `codex/metric-transfer-fix` | `18243cbc877b7dac9c1a6bb4b0fe71b2aa012fd7` | 2 | `archive/2026-10-17/codex/metric-transfer-fix` |
| `delta-fixes` | `d52b0c121723b1ba9ba33850296b8e741ad25c25` | 1 | `archive/2026-10-17/delta-fixes` |

## Proposed small active set

1. `main`: sole authoritative integration and development baseline after the
   reviewed PR is merged. Both 195-parent and 96-parent protocols remain versioned
   and separately gated.
2. One scoped task branch only while it has active review work.
3. If a frozen Exp96 source branch is still operationally required, keep that
   branch temporarily as the second task branch; the other frozen SHA remains
   reachable in main history and an archival tag. Do not delete either until
   the deployment owner confirms that references have ended.

No remote branch is deleted by this cleanup PR. The branch count cannot reach
the target of two or three active branches until owner-approved archival and
deletion occur.
