# Shared agent handoff

Branch: `integration/exp12`. This file establishes the shared protocol; it does
not authorize scientific changes, main merges or Delta submissions.

## Exact source checkpoint and live integration HEAD

The current integrated executable/test tree is pinned at
`813de566c4b169961fd4ec6a13063ac0ffa809d5` (reviewed pinned Claude reports plus atomic cache fix and
engineering review follow-ups). The previous executable checkpoint was
`53cc476b16bb7707484c7c049d2e77c0600568e2`. Final handoff/gate/status edits are
documentation only. Resolve the exact published integration HEAD with:

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
No PR is created by this branch-establishment task. Recheck GitHub before claiming
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
The pilot branch advanced to `2879b64499490e93bc6374465a6b7b2bcbd2724e` after the
pinned373 revision; its failure-analysis document and GPU-resume probe are not
included. They require a separate review. `claude/jolly-keller-lwr6xi` and
`delta-fixes` remain INVESTIGATE and untouched. No further broad cleanup is active.

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
