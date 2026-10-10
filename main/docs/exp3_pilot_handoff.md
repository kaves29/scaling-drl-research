# Exp3 implementation handoff

Owner: Codex (engineering); scientific choices and execution: research owner.
Task branch `codex/exp3-pilot-implementation`; approved base
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. Resolve the delivered final task SHA
from Git; this document cannot include its own enclosing commit's final SHA.
Do not merge into integration/main or modify the queued Delta jobs.

Implementation, interfaces, exact future execution commands, artifact-reuse
matrix, scientific alternatives, missing data and GPU plan:
[exp3_pilots.md](exp3_pilots.md) and
[exp3_implementation_plan.md](exp3_implementation_plan.md).

CPU engineering evidence in this workspace:

- `exp3-reviewed-tests.log`:37/37 pass. All three pilots and StageB execute;
  passive/target/source recording restarts compare exact state; intentional
  replay, optimizer, RNG, provenance, stream-order/checksum and dtype/signed-zero
  mismatches are refused. Saturated frozen-actor actions retain finite density.
- `exp3-existing-regression.log`:38/38 existing tests pass, covering fork-unit,
  runtime/restore, representation-exact comparison, production manifest, twin
  defects and SAC checkpoint behavior.
- Final parameter-space sanity/full controls preserve ordinary SAC gradient
  computation; their focused rerun and all-three-pilot smoke are recorded in
  `exp3-final-update-regression.log`:9/9 pass. Completion marker and diagnostic
  array publication are atomic.
- `exp3-final-compatibility-tests.log`:8/8 pass after final artifact-counter and
  atomic-output guards, including all-three-pilot smoke and both added regressions.
  Together with the37-test full run, all39 current Exp3 tests were exercised.
- `exp3-config-isolation.json`:337 baseline tracked files byte-unchanged;
  195/195 complete resolved parent configuration fingerprints match base844e3c0.
- `exp3-methodology-reference.log`:independent census195 parents/39 cells/130
  eligible scaled candidates, unchanged entropy and paired-statistical references.
- Additional existing full fork/control/injected/identity and exact-resume
  regressions are recorded in `exp3-fork-restoration-regression.log`:15/15 pass.

Detailed scope and evidence checksums: [exp3_validation_report.md](exp3_validation_report.md).

Evidence files are under `/workspace/scratch/`, never committed. Installed CPU
environment: `/workspace/scratch/a100-audit-venv/bin/python`, Python3.12.14,
JAX/JAXLIB0.4.34. It is not Delta's Python3.12.13/A100 environment. Tiny fixture
numbers and CPU-oracle tolerances are test parameters, not approved pilot settings.
No CUDA execution, real Exp3 pilot, source continuation or Slurm submission occurred.
No integration/main branch was edited. Historical sources are retained.

Pending owner decisions: select the TWO MyoSuite tasks out of four; approve all
null diagnostic settings/normalization/alpha/intervention choices and budgets;
identify actual compatible fork and matched U/I artifacts; authorize minimum
missing source recording and GPU qualification if desired. The20-cell manifest
can be generated only after environment selection; an example test pair is not
adopted. GPU timing/memory/passive-arm cost is unmeasured.

GitHub CLI reports the injected token invalid, although native Git transport
works. Publication and PR creation must be checked independently. Do not invent
a PR URL or claim a merge if API creation fails; the prepared PR body and compare
link can be used from a normally authenticated checkout. Next agent should inspect
the exact pushed SHA and scope before proposing integration.
