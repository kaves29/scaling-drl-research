# Shared Codex and Claude Code instructions

GitHub is the shared source of truth. Read `main/docs/AGENT_HANDOFF.md` and
`main/docs/EXPERIMENT_STATUS.md` before starting work. The application, research
specifications, tests, and existing docs live under `main/`.

## Authority and scientific safety

Explicit user instructions take precedence. Exp1/Exp2 authority is
`main/.claude/methodology-exp1-exp2.md`, including chronological amendments, and
approved answers in `main/docs/exp12_decisions.md`. The older
`main/.claude/research-methodology.md` governs its historical Angle scope; it does
not override Exp1/Exp2. Investigation reports and exploratory scripts are
analysis, not authority to adopt new settings.

Never silently change methodology, metrics, RNG, optimizer/initialization
semantics, precision, tolerances, thresholds, statistical rules, resource limits,
or training/probe budgets. Request explicit owner approval for such changes or
unresolved scientific choices. Keep confirmed observations, hypotheses, and
scientific qualification separate. Passing CPU tests does not establish GPU or
scientific qualification. Do not submit Delta jobs without authorization. Do not
merge into main, delete branches, force-push, or rewrite shared history.

## Ownership and Git workflow

- Codex owns engineering, runtime, infrastructure, and integration.
- Claude owns scientific investigation and validation.
- Both start new tasks from the latest approved `origin/integration/exp12` HEAD,
  recording its full SHA. If the remote tip is not approved, stop and resolve it.
- Use a separate task branch (`codex/<task>` or `claude/<task>`) and isolated
  checkout/worktree for each concurrent task. One writer per active branch.
- Never directly edit another agent's active branch. Handoff via GitHub PRs;
  integration changes require independent review and task authorization.
- Commit and push completed shared changes to the authorized task branch. Report
  exact base/final SHAs, PR URL, changed paths, checks and remaining blockers.
  If push or PR access is unavailable, report the precise blocker; preserve an
  exact bundle and checksums rather than inventing publication or PR links.
- Preserve experimental provenance and historical work. Keep generated results,
  logs, caches, environments and large evidence outside Git. Retain artifact
  checksums, exact source revisions, commands and validation receipts.

## Verification and handoff

Inspect status/diff before committing; keep changes minimal and scoped. Run
relevant regression tests and required CPU mutation/configuration/restoration
checks using `main/docs/exp12_cpu_validation_setup.md`. Use fresh output paths
outside Git. Preserve runner exit statuses and distinguish passed, failed,
skipped and unrun coverage. Report integration conflicts/regressions; never
change scientific settings to make checks pass.

Update the handoff and experiment-status documents with source/evidence SHAs,
ownership, active tasks, PRs, validation and unresolved decisions. Resolve the
current integration tip from Git; a committed file cannot contain its own final
commit hash. Historical reports retain their original source/status snapshots.

## Code style

Follow existing JAX/Flax conventions, PEP 8 and Google Python style; use Black.
Prefer small changes and terse comments explaining only non-obvious behavior.
Do not hardcode experimental parameters that belong in configuration.
