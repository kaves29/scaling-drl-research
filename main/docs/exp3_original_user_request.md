I want you to implement the COMPLETE infrastructure for my Experiment 3 pilot studies in the existing scaling-drl-research repository.

This is a substantial research-engineering task. Work as a senior reinforcement learning research engineer, prioritizing scientific correctness, reproducibility, efficiency, and integration with the existing codebase.

IMPORTANT CONTEXT

Our current validated integration branch is `integration/exp12` at:

844e3c0cc24998b25c3d1e667bcbfe056b1b336c

Exp1/Exp2 engineering validation is ongoing, and two Delta jobs are already queued.

Do not modify, cancel, interfere with, or duplicate those jobs.

Do not directly edit the integration branch. Create a separate `codex/exp3-pilot-implementation` branch/worktree based on the verified integration commit.

Read AGENTS.md, CLAUDE.md, the scientific methodology, decisions log, experiment status, handoff documentation, and existing Exp1/Exp2 implementation before designing anything.

OBJECTIVE

Implement the infrastructure for three Experiment 3 pilots, targeting:

- Dog-run
- Humanoid-walk
- Both existing MyoSuite environments
- Five seeds per environment

Resolve exact environment identifiers and seed conventions from the repository. Do not invent them.

These are DEVELOPMENT PILOTS, not the final Exp3 scientific experiments.

Preserve the approved Exp1/Exp2 methodology exactly. Exp3 must not change existing training behavior.

ARCHITECTURAL REQUIREMENTS

1. Reuse existing SAC agents, actor/critic implementations, checkpointing, fork infrastructure, replay systems, optimizer logic, evaluation and metrics wherever possible.

2. Avoid duplicating the SAC training pipeline.

3. Keep Exp3 isolated in clearly named modules, configurations, tests, scripts and output directories.

4. Make all pilot configurations explicit, versioned and reproducible.

5. Implement checkpoint/artifact compatibility validation. Fail clearly if an Exp2 artifact lacks required actor, critic, optimizer, replay, RNG, normalization or provenance information.

6. Distinguish reusable historical artifacts from information that cannot be reconstructed. Never fabricate transition histories or missing optimizer states.

7. Support single-Q and twin-Q critics correctly, including the existing SAC critic aggregation semantics.

8. Preserve current scientific precision, normalization, target-network, entropy and optimizer behavior unless a pilot explicitly requires a documented diagnostic intervention.

PILOT 1 — ACTOR GUIDANCE ACROSS THREE ACTORS

At matched post-fork checkpoints, evaluate untreated critic Q_U and injected critic Q_I under:

- Untreated actor pi_U
- Injected actor pi_I
- Original fork-time actor pi_F

For each within-actor comparison, match states, reparameterization noise/actions, alpha and actor optimizer state.

Implement one common prespecified state panel, with optional source-specific panels.

Measure the chain:

action-value gradients
→ critic-derived actor parameter gradients
→ full SAC actor gradients
→ actual optimizer updates
→ policy-function changes.

Preserve per-state/per-action measurements before aggregate statistics.

Measure gradient direction, magnitude, and optionally local variation where scientifically justified.

Include fork-time equivalence assertions.

Implement configurable diagnostic interventions:
- Direction-only substitution
- Magnitude-only substitution
- Full critic-signal substitution
- Sanity/control condition

Keep these interventions clearly separated from ordinary SAC updates.

Ensure any optimizer-update comparison uses matched actor parameters and optimizer state.

Do not interpret gradient differences alone as evidence of harmful guidance. Include outcome evaluation interfaces for intervention consequences.

PILOT 2 — PASSIVE LEARNING AND EXPERIENCE SOURCE

Implement Stage A first:

P_U<-U: untreated passive learner using untreated-source experience.
P_I<-U: injected passive learner using exactly the same untreated-source experience.

Both initialize from the appropriate common fork state.

Both remain passive and never contribute to their own training replay.

Match:
- Initial replay
- Ordered transition arrivals
- Replay sampling keys between passive learners
- Training update budget and timing
- Actor/critic initialization
- Normalization
- Evaluation protocol

Each passive learner maintains its own actor, critics, target networks, entropy state and optimizers according to ordinary SAC.

Implement deterministic transition-stream recording and replay, with checksums, metadata and explicit arrival order.

Do not assume existing replay snapshots contain chronological source trajectories.

Also implement optional Stage B, an all-passive 2×2:

P_U<-U
P_I<-U
P_U<-I
P_I<-I

Keep active U/U and I/I trajectories as references, not substitutes for passive learners.

Make Stage B opt-in and independently configurable.

PILOT 3 — OPTIONAL TARGET-POLICY ASSIGNMENT

Implement as an opt-in diagnostic, not an automatically launched experiment.

Clone one selected critic/optimizer checkpoint into two identical diagnostic learners.

Feed identical transition batches and use one common frozen target-Q evaluator.

Fix alpha at a documented protocol-derived value.

Construct SAC targets using frozen pi_U versus frozen pi_I, with shared base action-noise draws.

Train for a short, configurable, prespecified update budget.

Measure:
- Target differences
- Critic adaptation
- Q and action-gradient changes on fixed panels
- Optional guidance to a common actor

Implement a configurable alternative-evaluator sensitivity check.

Do not introduce online actor changes into this frozen-actor diagnostic.

DATA REUSE AND EFFICIENCY

Before implementation, audit exactly which existing Exp1/Exp2 artifacts can be reused.

Design an artifact-dependency matrix for every pilot, environment and seed.

Reuse existing checkpoints, evaluation states, replay snapshots, actor/critic states and recorded transitions when valid.

Avoid unnecessary environment interaction and duplicated training.

Use streaming or chunked transition storage rather than keeping entire histories in RAM.

Avoid redundant GPU copies, unnecessary JAX recompilation, excessive checkpointing and unbounded metric storage.

Implement opt-in source-stream recording for future Exp2 runs without altering their learning trajectories. Test that enabling recording leaves training states and outputs unchanged.

If historical streams are unavailable, identify the minimal new source runs needed; do not silently substitute different data.

BENCHMARKING

Provide instrumentation to measure:
- Environment stepping
- SAC update time
- Transition delivery
- Evaluation overhead
- Compilation overhead
- Peak memory
- Checkpoint and stream storage

Do not assume the previously estimated 0.4–0.7 passive-arm cost is accurate.

TESTING

Implement extensive CPU tests covering:

- Artifact compatibility and missing data
- Exact checkpoint restoration
- Fork-time equivalence
- Common-state and common-action comparisons
- Correct SAC actor gradients
- Shared replay sampling
- Passive learners not generating training experience
- Transition ordering and replay fidelity
- Single-Q and twin-Q behavior
- Correct target construction
- Optimizer-state matching
- RNG reproducibility
- Isolation of interventions
- Resume/restart correctness
- No changes to existing Exp1/Exp2 behavior

Include small end-to-end CPU smoke tests for all three pilots.

Run existing relevant regression suites and configuration-fingerprint comparisons.

Create explicit negative tests that fail when a supposedly matched condition is intentionally mismatched.

DELIVERABLES

1. Complete implementation in a separate task branch.
2. Clear directory structure and documented interfaces.
3. Exp3 pilot configurations for the four environments and five seeds.
4. Artifact-reuse and missing-data report.
5. CPU test suite and results.
6. Minimal GPU validation plan.
7. Estimated incremental GPU workload, clearly separating reused data, required new source runs, and diagnostic training.
8. Exact commands for future authorized execution.
9. A list of any scientific choices that require my approval before experiments run.

Do not invent unspecified numerical settings such as panel size, update duration, checkpoint cadence or intervention normalization. Implement configurable choices and flag them for approval.

WORKFLOW

First audit the existing architecture and produce a concise implementation plan.

Then implement in logical, reviewable commits.

Run tests continuously and fix engineering defects.

Do not stop after scaffolding or documentation. Complete all feasible code, tests, configuration interfaces and CPU smoke tests.

If scientific ambiguity prevents implementing a specific behavior, isolate that decision behind a configuration interface, document the alternatives and continue with independent components.

Do not change scientific methodology or existing Exp1/Exp2 settings.

Do not merge into integration/main, submit GPU jobs, or run the actual Exp3 pilot experiments.

When complete, push your branch and open a PR against `integration/exp12`.

Report the exact branch, commit SHA, tests, remaining decisions, and GPU readiness.