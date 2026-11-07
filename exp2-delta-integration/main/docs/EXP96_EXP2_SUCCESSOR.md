# Exp96 Experiment 2 successor — engineering handoff

This branch descends from `a5c376d740b072d5211e55d94860a61dbd054fec`.
It does not change that frozen Exp1 source or any Delta checkout. The older
Angle 1/2A/2B/2C methodology is retired by owner direction. Its active entry
points, angle-specific configs, scripts and tests were removed on this branch;
historical Git commits preserve reproducibility. The current source of truth
is `.claude/methodology-exp1-exp2.md`, `docs/exp12_decisions.md`, and the
versioned ten-target Exp96 decision in `docs/exp96_launch_decision_2026-10-18.md`.

## Current implementation inventory

| Requirement | Current path | Status |
|---|---|---|
| Natural operational trigger and checkpoint | `experiments/exp12/trigger.py`, `experiments/exp1.py`, `experiments/exp12/fork.py` | Implemented; no forced trigger |
| Complete replay, normalization, RNG, optimizer and simulator restoration | `experiments/exp12/state.py`, `experiments/exp2_arm.py` | Implemented; final-source GPU proof still required |
| Paired control/injection and Check 1/2 | `experiments/exp1.py`, `experiments/exp2_arm.py`, `experiments/exp12/injection.py` | Implemented; positive control and selected m pending |
| Policy KL/churn, actor gradient diagnostics, critic plasticity | `experiments/exp12/diagnostics.py`, `analysis/exp2_analysis.py` | Implemented; no new MC Q-error or frozen-actor analysis is authorized |
| Source-bound 96-parent arm discovery | `scripts/exp12_dynamic_scheduler.py`, `scripts/exp96_exp2_arms.py` | Implemented; only naturally eligible D4W1024/D4W1536 parents |
| Durable claims, validation, retry and per-arm outputs | `scripts/exp12_dynamic_scheduler.py`, `scripts/exp96_exp2_arms.py` | Implemented; A100 qualification pending |
| Eight-A100 execution | `scripts/sbatch_exp96_exp2_arms.sh` | Two nodes × four A100s, one arm process per GPU until measured resource approval |
| Confirmatory Exp96 paired analysis certification | `analysis/exp2_analysis.py`, `analysis/exp12_validation.py` | **Unfinished:** older certifier assumes 20 checks/five targets; use only exploratory outputs until a ten-target certifier is approved and tested |

The previous integrated candidate `9938aefae0cee48ebaf570c2e627daa01b60b4cc`
is a historical five-target integration ancestor; current fork and injection
modules remain the reusable implementation. The eight-GPU Exp1 successor's
7/5/4 packing is not applied to injected arms.

## Source identity and admission

The arm trainer runs in a **clean checkout of the exact parent commit** named
in the 96-parent queue and approval. The newer scheduler checkout is separately
clean and SHA-pinned. The ten-target positive-control report binds the parent
source SHA and the qualifier/scheduler SHA separately; the admission gate
rechecks its paired score/loss series and chosen m. Arm identity includes environment, architecture, seed,
resolved arm config hash, parent source ID, exact fork-file SHA-256, chosen m,
and parent commit. Each arm stores `source_config_hash` for revalidating an
interrupted start against the immutable parent fork. The shared durable ledger
prevents duplicate claims across allocations; DONE is accepted only after
checkpoint, Check 1, Check 2, schedule and result validation. Failed attempts
are retained and require the existing source-bound review-retry process.

The source-bound approval template is
`configs/exp96_exp2_approval.template.json`. It is intentionally unapproved
and has placeholder measurements. The owner must approve the selected m from
a **same-source natural-trigger** positive control. The actual CPU, host RAM,
VRAM, per-group arm runtime and allocation values require measured Delta
evidence. Four A100s per node is the Delta `gpuA100x4` topology; one process
per A100 is a conservative admission policy, not an optimized packing result.

After approval and at least one natural fork, from the successor's `main/`
directory on Delta, obtain the command without submitting:

```bash
python -m scripts.exp96_exp2_arms submission-command \
  --queue /ABS/EXP96_PARENT_QUEUE --approval /ABS/EXP2_APPROVAL.json
```

It prints a source-bound `sbatch` command for one or two four-A100 nodes.
`discover` reports the current eligible arm count without launching GPUs.
Do not reserve GPUs when that count is zero. A failed attempt is never silently
accepted: use `review-retry --id ID --decision /ABS/SOURCE_BOUND_DECISION.json`
only after inspecting preserved attempt evidence.

## Remaining before Exp2 launch

1. Exp1 parents must produce complete, naturally eligible source-bound fork
   states. Non-triggering parents are null eligibility results.
2. Run and review the same-source positive control; owner freezes m and signs
   the separate Exp2 decision. The owner approved the ten-target adaptation:
   `positive_control.py` accepts only the canonical `base_exp12_ten` development
   configuration at D4W1536 dog-run with a seed outside 1–8. It uses the same
   natural trigger, 2,000-step fit, matched pools/targets, recovery definition,
   pooled within-series noise cutoff 0.10, and smallest-m-within-0.10 rule.
   The report records the development seed, source and analysis SHAs, probe
   configuration, criteria, per-round losses and chosen m. Five-target evidence
   cannot qualify the Exp96 intervention. No ten-target positive-control GPU
   evidence has yet been produced, and no m is approved for Exp96.
3. Measure one representative arm's restore, injection, checkpoint, VRAM,
   CPU/RAM and runtime on A100, including Check 1/2 and interruption recovery.
   Eight-GPU arm scheduling requires the approved per-group profiles; no
   7/5/4 parent packing inference is valid for injected arms.
4. Complete a versioned ten-target/50-check Exp2 result certifier before any
   confirmatory paired plots. The existing paired raw-return/IQM analysis can
   be generated as exploratory progress but its historical certificate is
   not a valid Exp96 certificate.

For the positive-control development run, use a clean checkout of the **exact
Exp96 parent source**, a dedicated result/checkpoint directory and a positive
`DEV_SEED` outside 1–8. The existing Exp1 entry point supplies the natural
trigger and immutable fork. Run the development parent from its frozen
`main/` checkout, then run the qualifier from the clean Exp2 successor's
`main/` checkout. The report records both different source SHAs:

```bash
python -u run.py --experiment exp1 --config_name base_exp12_ten \
  --overrides env_name=dog-run --overrides env=dmc_hard \
  --overrides critic_num_blocks=4 --overrides critic_hidden_dim=1536 \
  --overrides run_role=dev --overrides seed="$DEV_SEED" \
  --overrides env.num_env_steps=1000000 \
  --checkpoint_dir "$DEV_RUN" --checkpoint_interval 10000 \
  --checkpoint_start_frac 0.0
cd "$EXP2_SCHEDULER_MAIN"
python -u scripts/positive_control.py \
  --run_dir "$DEV_RUN" --out_dir "$NEW_POSITIVE_CONTROL_DIR"
```

The development run requires separate GPU authorization. The second command
does no SAC training or new environment interactions, but full-width ten-round
probe fitting still needs GPU time. Preserve its JSON and NPZ and record their
hashes; a stop report or no natural trigger is not an m qualification. The
five-target code path and its source-bound reports remain separately usable.

No Delta job or scientific amendment is authorized by this document.
