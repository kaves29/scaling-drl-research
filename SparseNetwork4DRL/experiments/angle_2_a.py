"""Angle 2A: scaled/degraded critic vs. independently-trained healthy
reference critic, evaluated at the critic's own seed-specific Angle 1
degradation onset.

See experiments/angle_2a/ for the implementation, split by concern:
  config.py        - required-architecture validation, no hidden defaults
  onset_lookup.py  - deterministic per-(architecture,environment,seed) Angle 1
                     onset lookup (never averaged/borrowed)
  env_state.py     - exact dm_control state capture/restore for MC rollouts
  agent_runner.py  - trains one fully independent agent to an exact step
  probes.py        - probe sampling, cross-critic Q eval, MC rollouts, errors
  storage.py       - persistent probe-level dataset (canonical, independent of WandB)
  matchup.py       - orchestrates one D-vs-R matchup end to end

Two real matchups (scaled_a vs reference, scaled_b vs reference), each using
its OWN scaled architecture's OWN onset step - never averaged, never
substituted. `D_5x768` and `D_7x1024` are each a fresh, independent agent.

The reference side (`R_2x512`) is likewise trained fully independently per
matchup - a separately-initialized agent (own actor, own critic, own replay
buffer, own environment, own RNG stream), trained online from scratch for
exactly that matchup's own onset_step, with no state shared across matchups.
This matches research-methodology.md's Angle 2A section verbatim ("Each
scaled critic gets its own independently-trained reference critic, matched
to its own t* - never shared across two scaled architectures") and its
"Must Never Change Silently" list. run_matchup() already does exactly this
by default (reference_handle=None) - it is the same code path the
null-baseline matchups below already use, so no separate reference-training
primitive is needed here; the two real matchups just call run_matchup()
twice, independently, exactly like the null-baseline loop does.

(An earlier version of this module trained ONE shared reference trajectory
and snapshotted it twice - one snapshot per matchup's onset step - which
.claude/rules/angle2.md described as intentional but which
research-methodology.md's Angle 2A section and "Must Never Change Silently"
list explicitly forbid. That shared-trajectory design has been removed; see
the 2026-08-28 audit notes in the End-of-Task Summary for the full
before/after and why. experiments/angle_2a/agent_runner.py's
train_reference_agent_with_snapshots()/ReferenceTrajectory/
ProbeCapture.snapshot() supported only that removed design and were deleted
as dead code alongside it.)

An optional healthy-vs-healthy null baseline reuses each matchup's
already-looked-up onset step (a real Angle-1-derived number, not invented)
and, as always, requires fresh, independent training for BOTH sides, since
Angle 1 does not persist checkpoints/replay buffers in a form Angle 2A could
probe directly - see the "null baseline" note in the Angle 2A deliverables
for why.
"""

import os

import omegaconf
from dotmap import DotMap

from experiments.angle_2a.config import architecture_label, validate_angle2a_config
from experiments.angle_2a.errors import Angle2AOnsetLookupError
from experiments.angle_2a.matchup import run_matchup
from experiments.angle_2a.onset_lookup import lookup_critic_degradation_onset
from experiments.registry import register_experiment

import hydra


@register_experiment("angle_2_a")
def run(args: dict) -> None:
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # See the matching comment in experiments/angle_1.py: hydra.initialize()
    # resolves a relative config_path relative to *this file's* directory,
    # not the process CWD, so it must be made absolute first.
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)

    # Deliberately NOT a whole-tree OmegaConf.resolve(cfg): the top-level
    # critic_num_blocks/critic_hidden_dim in base_angle2a.yaml are mandatory
    # (`???`) precisely so nothing can silently fall back to them; resolving
    # the whole tree would touch (and fail on) those unused placeholders even
    # for a fully-valid Angle 2A config. Only the fields Angle 2A actually
    # needs are accessed (and thus lazily resolved) below.
    architectures = validate_angle2a_config(cfg)

    seed = int(cfg.seed)
    environment = str(cfg.env_name)
    onset_source_experiment = str(cfg.angle_2_a.onset_source_experiment)
    onset_ledger_root = str(cfg.angle_2_a.onset_ledger_root)
    num_probes_per_source = int(cfg.angle_2_a.num_probes_per_source)
    num_mc_rollouts = int(cfg.angle_2_a.num_mc_rollouts)
    run_null_baseline = bool(cfg.angle_2_a.run_null_baseline)

    reference = architectures["reference"]
    reference_label = architecture_label(reference)

    matchup_specs = [
        ("matchup_1", architectures["scaled_a"]),
        ("matchup_2", architectures["scaled_b"]),
    ]

    # Phase 1: look up BOTH onsets first (each independently, from its own
    # scaled architecture's ledger entry - never averaged, never one
    # substituted for the other).
    onsets = {}
    for matchup_name, scaled_architecture in matchup_specs:
        scaled_label = architecture_label(scaled_architecture)
        try:
            onsets[matchup_name] = lookup_critic_degradation_onset(
                architecture=scaled_label,
                environment=environment,
                seed=seed,
                source_experiment=onset_source_experiment,
                ledger_root=onset_ledger_root,
            )
        except Angle2AOnsetLookupError:
            print(
                f"[angle_2_a] {matchup_name} ({scaled_label} vs {reference_label}, "
                f"env={environment}, seed={seed}) requires manual review: could not "
                f"retrieve a usable Angle 1 critic-degradation onset. See error below."
            )
            raise

    # Phase 2: run each real matchup with its own independently-trained
    # reference agent. run_matchup() trains R fresh (own actor/critic/
    # buffer/env/RNG) whenever no reference_handle is supplied - the same
    # code path the null-baseline loop below already relies on - so simply
    # not passing reference_handle here gives each matchup a genuinely
    # separate reference agent, trained to its own onset_step, with nothing
    # shared between matchup_1 and matchup_2.
    for matchup_name, scaled_architecture in matchup_specs:
        scaled_label = architecture_label(scaled_architecture)
        onset = onsets[matchup_name]

        print(
            f"[angle_2_a] {matchup_name}: D={scaled_label} (fresh) vs "
            f"R={reference_label} (fresh, independent) env={environment} "
            f"seed={seed} -> stopping at onset_step={onset.onset_step} "
            f"(from run_key={onset.run_key})"
        )

        run_matchup(
            matchup_name=matchup_name,
            scaled_architecture=scaled_architecture,
            scaled_architecture_label=scaled_label,
            reference_architecture=reference,
            reference_architecture_label=reference_label,
            onset_step=onset.onset_step,
            onset_source_run_key=onset.run_key,
            base_cfg=cfg,
            seed=seed,
            environment=environment,
            experiment_name="angle_2_a",
            num_probes_per_source=num_probes_per_source,
            num_mc_rollouts=num_mc_rollouts,
            output_root="results/angle_2a",
            wandb_project=str(cfg.project_name),
        )

    # Phase 3: null baseline - unchanged: healthy-vs-healthy, both sides
    # freshly trained per matchup, independent of the real matchups above.
    if run_null_baseline:
        for matchup_name, _scaled_architecture in matchup_specs:
            onset = onsets[matchup_name]
            null_matchup_name = f"null_{matchup_name}"
            print(
                f"[angle_2_a] {null_matchup_name}: healthy-vs-healthy null "
                f"baseline at the same onset_step={onset.onset_step} "
                f"(reference architecture {reference_label} on both sides, "
                f"two freshly trained agents - see module docstring for why "
                f"this can't reuse the original Angle 1 checkpoints)."
            )
            run_matchup(
                matchup_name=null_matchup_name,
                scaled_architecture=reference,
                scaled_architecture_label=f"null_{reference_label}",
                reference_architecture=reference,
                reference_architecture_label=reference_label,
                onset_step=onset.onset_step,
                onset_source_run_key=onset.run_key,
                base_cfg=cfg,
                seed=seed,
                environment=environment,
                experiment_name="angle_2_a",
                num_probes_per_source=num_probes_per_source,
                num_mc_rollouts=num_mc_rollouts,
                output_root="results/angle_2a",
                wandb_project=str(cfg.project_name),
            )

    print(f"[angle_2_a] done. seed={seed} environment={environment}")
