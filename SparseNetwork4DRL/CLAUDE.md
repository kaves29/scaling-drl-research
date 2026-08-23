# EchoCritic Research Codebase

## Project
- This is a causal/mechanistic DRL research project.
- Main question: how critic scaling in SimBa-based SAC produces critic pathology and whether it propagates to actor-side behavior.
- Optimize for scientific validity, reproducibility, auditability, and controlled experimentation.

## How Claude Should Work
- Inspect the actual repository before making assumptions.
- Make the smallest correct change.
- Do not perform unrelated refactors.
- Separate confirmed facts, inferences, and hypotheses.
- If a decision can change experimental validity and is not explicitly settled, stop and ask.
- Never silently choose a methodological default.
- Preserve existing behavior unless the requested change requires it.

## Decision Priority
1. Explicit user instruction in the current task
2. Current locked research methodology in `.claude/rules/research-methodology.md`
3. Other project rules
4. Existing code behavior
5. Claude's engineering judgment

If these conflict, do not silently reconcile them. Surface the conflict.

## Codebase
- JAX/Flax SAC
- SimBa critic architecture
- DMC + MyoSuite
- Experiment routing through `run.py`
- Angle 1 and Angle 2A are currently implemented research pipelines.

## Core Research Config
- Actor: depth 1 / width 128
- Default critic: depth 2 / width 512
- Scaled critics: depth 5 / width 768 and depth 7 / width 1024
- UTD = 5
- Angle 1 = 5 seeds
- Angle 2A = 5 seeds
- Current experiment environments = [your exact current environment list]

## Required Working Behavior
- Do not hardcode experimental parameters that belong in configuration.
- Do not change an algorithm, metric, optimizer, normalization, sampling procedure, or evaluation protocol because another approach seems better.
- Treat research outputs as immutable evidence unless explicitly asked to recompute them.
- Before expensive runs, perform the smallest practical validation.
- Before declaring success, verify the actual output/result, not just the exit code.

## Required Final Report
For non-trivial tasks report:
- files changed
- what changed
- why
- tests/validation run
- errors encountered
- assumptions
- what I should double-check