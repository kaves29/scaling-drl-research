# Shared agent handoff

## Recording-isolation qualification harness correction (14 October 2026)

`codex/exp12-qualification-debug` builds on recording-isolation candidate
`12839f656e53b00a3327384f2cb4fc1be47e73fe`. Resolve the branch tip from
Git. Its harness now checks the selected precision in each triple and compares
saved meters after excluding only the three declared recording-only metrics.
The continuation recovered `723da0d1c0c6f95910468a4aae9c94f1c40f60ed`, inspected
job 22900932's original Delta artifacts and verified their checksum manifest.
The TF32 exception is confirmed as the harness expecting `highest`; its triple
stopped before writing a fingerprint. The historical highest triple passed.
Both 400-interaction CPU triples now pass with probes, alongside regression and
mutation/restoration coverage. The final stable CPU suite passes 760 tests with
five HumanoidBench skips across 79 modules. Final-checkpoint step checks, failed-arm receipts
and accurate backend scope are added. See
[the debugging report](exp12_qualification_job22900932_debug.md) for evidence,
validation scope and the Delta CPU-account limitation. The updated source still
requires GPU qualification. No production source or scientific setting changed;
no GPU job was submitted.

## Actor-diagnostic recording isolation candidate (14 October 2026)

Codex's isolated `codex/exp12-recording-isolation` branch starts at frozen
`087feab341b9be1f377697668f2d1c750ec0467f`. The Exp1/Exp2 recording
switch now leaves compiled KL/saturation computation in both modes and filters
OFF metrics only after host transfer. Resolve the branch tip from Git. The
historical true-no-diagnostics A100 result from job 22897269 remains BLOCKED;
recording isolation does not establish computational non-invasiveness. No GPU
qualification or production submission is authorized by this candidate.

## Conditional Exp1/Exp2 launch approval (13 October 2026)

Owner-selected amendment (ab) is implemented on the isolated
`codex/exp12-deferred-fork-ab` branch based on consolidated
`957be5c63a48b0b90dd28ee4b361e3c6d373ab05`; resolve the final branch
tip from Git. Parent approval no longer requires Exp2 positive control.
Exp2 allocation approval still requires its source-bound natural positive
control, frozen m and separate decision. The retained fork-state path and
deferred-arm CPU evidence, distinct GO/NO-GO lists and minimal final-source
GPU plan are in [exp12_ab_deferred_fork.md](exp12_ab_deferred_fork.md).
No GPU qualification, allocation approval or production submission is implied.

## Method C consolidated engineering candidate (13 October 2026, Eastern)

Codex independently reviewed Claude's `claude/exp12-method-c-fixes` through
`b225577c2ba5a18b2ad6d14c47b34a00b5443e05` against Method C base
`c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b`. The candidate branch is
`codex/exp12-method-c-consolidated`; resolve its final SHA from Git. The
finding-by-finding review, remaining owner/GPU gates and source links are in
[`method_c_consolidated_review.md`](method_c_consolidated_review.md). No
production or Delta job is approved by this handoff.

## Method C production implementation (12 October 2026, Eastern)

Owner authorized Method C with 2,000 continuous fitting steps, preserved five
targets/first-1,000 minibatches and AdamW continuity, zero threshold, 95%
bootstrap and two consecutive checks. Codex's isolated implementation branch
is `codex/exp12-method-c-production`, based on the published
`codex/exp12-production-readiness` tip
`1ea37e49a9279bc75e8390f2e702b746fff2eed8`. Resolve its final published
SHA from Git. The current readiness verdict and guarded commands are in
[`method_c_exp12_readiness.md`](method_c_exp12_readiness.md). This does not
approve GPU jobs, allocation A/B, the parent grid, or Exp2 injected arms.

## Production execution preparation (2026-10-11)

Codex's latest CPU-only handoff is
[`exp12_oct12_execution_handoff.md`](exp12_oct12_execution_handoff.md).
The published `codex/exp12-production-readiness` tip was
`3d75a05789b9815cefd08cf3d8981a4be34c2352` at the start of this task;
resolve the final tip from Git after publication. The new A100 hopper capture
job 22817406 confirms the unchanged D4W1536 hopper-hop failure and preserves
all five fitting curves and teacher/replay pools. Both D4W1024 null datasets
still fail. No GPU job or methodology change was made in this task, and pilot
22800749 remains untouched. Six final-source CUDA identities, one shared-mount
preflight, a final-source natural positive control, resource evidence and owner
scientific decisions remain outstanding.

## Remaining Exp1/Exp2 qualification (2026-10-11)

The seed-990 and seed-991 D4W1024 null JSONL files were independently
validated (13/100 and 6/100; both fail). `exp12_remaining_qualification.md`
records the conditional pair/replay interpretation, the absent original
hopper fitting curves, a source-bound two-node filesystem/binding preflight,
a one-A100 packing profile and the six identity commands. No GPU job was
submitted and pilot 22800749 remains untouched. Scientific dispositions,
final-source positive control/m and actual allocation approvals remain open.

## Dynamic allocation task (2026-10-11)

Codex's current isolated task branch is `codex/exp12-production-readiness`,
based on `dafd7d3f3fa9f070ecb2047519f68e43094b4b62`. The new persistent
scheduler, A/B Slurm wrapper, approval template and focused CPU checks are
described in [exp12_dynamic_launch.md](exp12_dynamic_launch.md). Resolve its
final source SHA from Git; this handoff does not approve either allocation,
Exp2 injection, scientific-gate disposition or production. Pilot 22800749
remains untouched. The historical fixed-wave commands are not the intended
campaign path.

## Production-readiness review (2026-10-11)

Codex task branch `codex/exp12-production-readiness` starts at the explicitly
requested W&B-online source `7d0883cfa99d27fed062d77a68cbbfa27e560061`.
See `exp12_production_decision_2026_10_11.md` for the blocker ledger, proposed
allocation and prepared commands. The branch adds an Exp2 arm/parent source
preflight and focused regression; it does not modify queued pilot job 22800749,
scientific settings, GPU evidence or integration. Resolve the branch's final
commit from Git after publication. Claude retains scientific-gate decision
ownership; the research owner must disposition null/range and freeze m.

## Isolated launch-readiness task (2026-10-10)

Owner: Codex. Approved task base is844e3c0cc24998b25c3d1e667bcbfe056b1b336c;
branch `codex/exp12-launch-readiness`. Integration remains untouched. New full-width
qualification and guarded launch preparation are described in
[exp12_full_grid_readiness.md](exp12_full_grid_readiness.md); null count memo in
[exp12_null_threshold_decision.md](exp12_null_threshold_decision.md).
Resolve this task's enclosing final SHA from Git/the delivery. No source/scientific
integration or job submission is implied. Claude's ongoing Exp3 review and the
published Exp3 implementation branch are untouched. No current PR number is
assumed from an unavailable GitHub API.

Owner-reported GPU resume22783981 PASS and null replication22779974 count6/100
are not independently verified without their new artifacts. The original
13/100 and new6/100 both fail the unchanged empirical gate. Existing small
resume evidence is not full-width qualification. Pending owner actions: provide
raw new evidence; decide natural-zero-trigger dev disposition; authorize a staged
full-width gate if desired; approve actual pilot allocation/role. Production
requires the separate grid/scientific/Exp2 lists in the new report. Tests and
publication verification are recorded in the delivery and validation report.

[CPU validation receipt](exp12_launch_validation_report.md): fresh base suite
620 passed/one optional skip (621 run), final focused51/51, mutation74/74,
environment restoration65/65, all195 parent fingerprints unchanged. New GPU
paths are prepared, not executed; no scientific gate is promoted.

Branch: `integration/exp12`. This file establishes the shared protocol; it does
not authorize scientific changes, main merges or Delta submissions.

## Exact source checkpoint and live integration HEAD

The current integrated executable/test tree is pinned at
`78d04d167a5f65c35f6905cb261876bfe709d85a` (reviewed PR #1 plus
engineering-only GPU restore/evidence guards and the bounded gate wrapper).
The previous executable checkpoint was
`813de566c4b169961fd4ec6a13063ac0ffa809d5`. Final handoff/status edits and
integration merge do not change that tested executable tree. Resolve the exact published integration HEAD with:

```bash
git fetch origin '+refs/heads/integration/exp12:refs/remotes/origin/integration/exp12'
git rev-parse refs/remotes/origin/integration/exp12
git ls-remote origin refs/heads/integration/exp12
```

The two results must agree. The final full SHA is also delivered in the task
completion report. This document intentionally distinguishes its literal source
checkpoint from its enclosing documentation commit to avoid a stale/self-referential
HEAD claim. Update the approved base when a subsequent integration is authorized.

## Provenance

- Approved scientific/A100 baseline: `9d6a82d4aef81299901177865e383640439cde2a`.
- Recovered exact-comparison and fork/stop reporting fixes:
  `dcb7491b61c094a961155c0c5e0a9d77d5d79f13`.
- Non-squashed scientific-history merge:
  `c6c2aec623540de1c4b2b74ae4abdab5c5deb519`, parents dcb7491 and
  `8a888ef998dd2055d43faea26b78e8f555d00bc6`.
- Claude's original commits: `f12e00ff71d4c266ed1afeca3e5693f260d48c89`,
  `5487f6270aab2faceb7b35762080cac3511f7d54`,
  `8a888ef998dd2055d43faea26b78e8f555d00bc6`.
- Historical diagnostic preservation: `53cc476b16bb7707484c7c049d2e77c0600568e2`;
  three files copied byte-for-byte from `18243cbc877b7dac9c1a6bb4b0fe71b2aa012fd7`.
  Its production mitigation was already integrated; it was not merged again.

## Ownership, active tasks and PRs

| Owner | Task | State / next action |
|---|---|---|
| Codex | Establish integration, preserve diagnostics, publish shared protocol | Complete integration and CPU verification; publication is authorized after these checks. Exact live remote verification is supplied in the delivery report. |
| Codex | Engineering/runtime/infrastructure/integration | Pinned handoff/cache/pilot review branches are listed below; start future work from the exact published integration HEAD in the delivery report. |
| Claude | Scientific investigation and validation | Original analysis now included. Start a new isolated task branch from integration; investigate remaining gates without adopting alternatives. |
| Research owner | Scientific and deployment decisions | Review blockers in EXPERIMENT_STATUS.md; authorize any follow-up experiments or changed staging. |

At the previous2d baseline publication, the public PR page showed zero open
and zero closed PRs; this remains a historical snapshot, not a current API result.
No PR was created by that original branch-establishment task; current PR #1 is reviewed below. Recheck GitHub before claiming
that another agent has no active work; inventory is a snapshot, not a lock service.
Repository protection/CI settings and collaborator permissions are unverified.

## Pinned remote work now reviewed

The owner authorized these exact pins after publication of baseline2d007663.
Separate review branches preserve the original non-squashed history; their
sequential integration includes only the reviewed changes. GitHub CLI/API
credentials are unavailable (`gh auth status`: invalid injected GH_TOKEN); native
Git fetch/push works. The user explicitly allowed review branches instead of PRs.
No PR numbers/links are invented. These are the reviewed branch checkpoints:

| Review branch | Exact reviewed tip | Scope |
|---|---|---|
| codex/review-exp12-scientific-handoff | `5365e43077631b09d62933eb130762790f283588` | Merge of cb33733; one historical handoff document. |
| codex/review-exp12-pilot-report | `dc8fc37b1463b976dd24e45e4fe20da099ba2dcf` | Merge of pinned3734215 atop the accepted handoff; one pilot report. |
| codex/review-exp12-atomic-cache | `813de566c4b169961fd4ec6a13063ac0ffa809d5` | Merge of d23191b; two review follow-ups preserve empty-key errors, clean failed publications, strengthen bytes/concurrency coverage and make the mutation oracle deterministic. |

No scientific proposal or resource request in either historical report is adopted.
The subsequently authorized PR [#1](https://github.com/kaves29/scaling-drl-research/pull/1),
head `4afc772d5f2beab625e3b513e6f53cc72ea8e1c6`, was reviewed on
`codex/review-a100-resume-pr1`; original history is preserved by merge
`09a0af582e4e49e3ec8dff3384292bf5385b89a5`, followed by tested engineering
corrections at `78d04d167a5f65c35f6905cb261876bfe709d85a`. It adds
investigation/diagnostic files only. Review/receipt details are in
[pr1_gpu_resume_review.md](pr1_gpu_resume_review.md). GitHub CLI credentials
remain unavailable, and the public API CONNECT proxy returned403; native Git
merge/push and ancestry verification are separate from an unverified PR UI badge.
`claude/jolly-keller-lwr6xi` and `delta-fixes` remain INVESTIGATE and untouched.
No broad cleanup is active.

Latest PR-specific checks:

- Final23/23 probe unit tests pass, including fallback, failed/no-op restore,
  missing/corrupt state, undeclared omissions, premature endpoint and retained logs.
- Focused47-test regression passes46 with one optional cache-eviction/filelock skip.
- Final complete eight-child CPU probe passes all5/60/95 scenarios; both saved
  step60/update102 restores are observed completed. All completed states are300/582.
- Its five complete states exactly match the original PR CPU probe's counterparts;
  diagnostic observations introduce no observed CPU state/trajectory difference.
- 195/195 config fingerprints still match the original2d baseline.
- Gate Bash/embedded Python checks pass; no automatic submission; actual wrong-SHA
  preflight exits2 and retains status/checksums before backend startup.

These CPU results do not qualify GPU behavior. Raw evidence is under
`/workspace/scratch/exp12-pr1-review/`; report includes the final receipt checksum.
First GPU priority: tracked `scripts/sbatch_exp12_gpu_resume_gate.sh`, explicit
crash95 shard, exact delivered integration SHA. It retains the approved
one-A100/four-CPU/32GB/seven-minute allocation and300-second total workload cap.
Later relevant GPU shards: in-process exact Check1/fork and full-width dog-run
checkpoint smoke. No resource increase, trigger alternative or GPU pass is adopted.

Current engineering receipts (source/cache production at27da3eb, with the final
813de56 test-only oracle additionally checked; final shared documents do not alter
execution):

| Check | Result |
|---|---|
| Focused cache/runtime/launcher/comparator/checkpoint/orchestration suite | 75 tests:74 pass,1 optional filelock/eviction skip; exit0 |
| Full relevant fork/kill/resume/identity classes | 25/25 pass; exit0, including all kill subtests |
| Methodology/manifest/hardware/isolated null-follow-up suite | 56/56 pass; exit0 |
| Mutations | 74/74 pass; final deterministic cache oracle also independently rejects mutant and passes restored implementation |
| Final cache module | 8 tests:7 pass,1 optional eviction skip; exit0 |
| Strict checkpoint-state restore command helper | 3/3 pass; existing intentional diagnostic-NaN metric behavior preserved |
| Canonical simulator restoration | 65/65 pass, including actual H1 reach/run |
| Resolved configs against untouched2d baseline | 195/195 exact fingerprint matches |
| Independent methodology census | 195 parents,39 cells,130 scaled eligibility candidates; exit0 |
| Mock-CUDA controls | Both reported startup cases pass on untouched baseline and candidate in this pinned environment; reported failures not reproduced here |
| Prepared A100 command harness | Bash/Python syntax checked; CPU-only adapted cache execution exits0 and retains valid JSON; CUDA assertions were not executed |

An initial test command named a nonexistent `test_exp12_production_config` module;
the corrected56-test invocation passed. Draft command checks also caught stdin
multiprocessing and skipped-TestCase JSON serialization defects; the delivered
file-based guarded harness corrects both. An overbroad draft exact comparator
rejected intentional unscheduled `actor_grad_cosine` NaNs in metrics; the scoped
state comparator passes while retaining the existing metric oracle. These were
validation-command defects, not production regressions or relaxed scientific
rules. Raw attempts/receipts remain outside Git at
`/workspace/scratch/exp12-pilot-integration/review/`.

Next A100 stages, exact checkout/setup/log-preservation commands, coverage limits
and acceptance criteria are in [exp1_a100_minimal_gate.md](exp1_a100_minimal_gate.md).
No Delta access, GPU run or submission occurred. Codex owns next engineering
review; the owner must authorize execution and the full-length pilot allocation.
Claude owns scientific gate disposition and any new investigation branch.

## Claude synchronization

From a clean normal checkout (do not switch another active agent's checkout):

```bash
set -euo pipefail
test -z "$(git status --porcelain --untracked-files=all)"
git fetch origin '+refs/heads/integration/exp12:refs/remotes/origin/integration/exp12'
agent_base=$(git rev-parse refs/remotes/origin/integration/exp12)
test "$(git ls-remote origin refs/heads/integration/exp12 | cut -f1)" = "$agent_base"
# Compare agent_base to the exact owner-approved SHA in the delivery report.
# Use a new, non-existent path and a unique task branch; do not reuse active work.
git worktree add -b claude/SCIENTIFIC_TASK ../scaling-drl-claude-SCIENTIFIC_TASK "$agent_base"
cd ../scaling-drl-claude-SCIENTIFIC_TASK
cat AGENTS.md
cat main/docs/EXPERIMENT_STATUS.md
git rev-parse HEAD
```

After authorized task work and verification, commit and push only that task
branch, create a PR targeting `integration/exp12`, and report its full SHAs and
PR URL. Do not merge it or submit jobs without authorization. No submission
command is implied by synchronization.

## Validation receipt

Completed on source checkpoint `53cc476b16bb7707484c7c049d2e77c0600568e2` using the existing
pinned CPU/HumanoidBench environment. Final workflow/status edits are documentation
only; executable/configuration bytes and the fingerprint remain unchanged.

| Check | Result |
|---|---|
| Full available repository suite | **613/613 tests across 67 modules**, zero failures/errors/skips; exit0 |
| Claude's isolated null-follow-up regressions | **5/5 tests**; exit0 (additional to the repository suite) |
| Break-and-restore mutations | **73/73**; exit0 |
| Canonical environment/seed restoration | **65/65**, including actual CPU H1 reach/run wrappers; exit0 |
| Resolved configs vs approved9d6 | **195/195 exact matches**; exit0 |
| Independent methodology census | 195 parents,39 architecture/environment cells,130 potential scaled forks; exit0 |
| Preserved transfer diagnostic | CPU bulk/groups,1001 metric groups: both exit0, values bit-exact |
| Preserved CPU transfer reference | group counts1/1001; exit0 |
| Preservation and syntax | Exact Codex fix blobs,17 Claude files and3 historical files retained;14 analysis scripts parse; shared-document links resolve |

Every repository module receipt, process exit, unique test ID count and owned log
SHA-256 was independently verified. CPU coverage is not CUDA/scientific qualification.
Raw receipts, logs and generated evidence remain outside Git at
`/workspace/scratch/exp12-shared-baseline/` in this Cloud workspace; copy them to
durable evidence storage separately if needed. These paths are not production
configuration or a guarantee of availability in another environment.

- Executable/configuration fingerprint:
  `411197db644f43400172b97248090d9f6f63251a0691b620ac2e9e820b92ecbd`.
- Completed full-suite summary SHA-256:
  `c49b52875e140dbb9852f5e986fb0f90b05689565220629bc24f7bdd98f68524`.
- Completed module ledger SHA-256:
  `46fbdbbb04a088f61c92bd6efec0ce4c871f8584d69286b7942b2f14582d726f`.

The existing reproducible commands, from `main/`, are:

```bash
cpu_python=/absolute/path/to/pinned-HumanoidBench-CPU-environment/bin/python
cpu_output=/absolute/new/output/path
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$cpu_python" -B scripts/run_exp12_cpu_validation.py --jobs 2 --out-dir "$cpu_output"
"$cpu_python" -B -m unittest scripts.sci_investigation.test_null_followup -v
"$cpu_python" -B -u -m tests.exp12_break_checks
"$cpu_python" -B scripts/check_exp12_cpu_environments.py
"$cpu_python" -B scripts/methodology_cpu_reference.py
```

Use a working renderer and the pinned dependency environment documented in
exp12_cpu_validation_setup.md; do not count missing dependencies as qualification.
The preserved metric diagnostic can be run with `--platform cpu --groups 1001`
and a fresh external `--out-dir`; this does not request CUDA or submit a job.

## Remaining handoff blockers

The pinned cache/report work is now incorporated; A100 qualification of the final
source remains pending. Production scientific qualification remains blocked by fresh-null, hopper range,
positive-control/m and final-source CUDA coverage. D4W1536 identity_warm is held
under the existing 300-second internal timeout/seven-minute Slurm allocation;
changing staging or budgets requires owner approval. No current CPU result grants
production-launch authorization. Historical Angle/Modal and delta-fixes branches
remain untouched pending individual investigation.
