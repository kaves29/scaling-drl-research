"""Angle 2C: Distortion Decomposition.

See experiments/angle_2c/ for the implementation, split by concern:
  errors.py        - shared exception types
  config.py         - required-field validation, no hidden defaults
  loader.py         - loads Angle 2B's already-persisted (s,a)/nabla_a Q/
                      Q(s,a)/actor-gradient arrays with zero recomputation
  properties.py     - the three candidate property formulas (direction,
                      magnitude/bias, instability) + null-comparison
                      transforms
  perturbation.py   - local-instability perturbed-action nabla_a Q variance
  reconstruction.py - co-occurrence reconstruction test (direction vs
                      magnitude), via the chain rule, against g_{D|R}
  onset_lookup.py   - single-frozen-point proxy comparison against Angle
                      1's onset ledger (see its module docstring for what
                      this can and cannot actually establish)
  storage.py        - persistent results (canonical, independent of WandB)
  matchup_2c.py     - orchestrates one full analysis per matchup_name

For every matchup_name in angle_2_c.matchup_names, runs the full Angle 2C
analysis and mirrors a run-level summary to WandB. Zero new training, zero
new environment interaction, zero new state-action sampling anywhere in
this module or anything it calls - see research-methodology.md's Angle 2C
section and CLAUDE.md's "do not change an algorithm... because another
approach seems better" (the three property formulas and the reconstruction
test are implemented exactly as specified, not reinterpreted).
"""

import os

import hydra
import omegaconf
import wandb
from dotmap import DotMap

from experiments.angle_2c.config import validate_angle2c_config
from experiments.angle_2c.matchup_2c import run_angle_2c_analysis
from experiments.registry import register_experiment


@register_experiment("angle_2_c")
def run(args: dict) -> None:
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # See the matching comment in experiments/angle_1.py, angle_2_a.py, and
    # angle_2_b.py: hydra.initialize() resolves a relative config_path
    # relative to *this file's* directory, not the process CWD, so it must
    # be made absolute first.
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)

    run_cfg = validate_angle2c_config(cfg)

    for matchup_name in run_cfg.matchup_names:
        print(
            f"[angle_2_c] environment={run_cfg.environment} seed={run_cfg.seed} "
            f"matchup={matchup_name}: loading Angle 2B outputs from "
            f"'{run_cfg.angle_2b_results_root}' and Angle 2A checkpoints "
            f"from the default root (zero training/env interaction/new "
            f"sampling)..."
        )
        result = run_angle_2c_analysis(
            environment=run_cfg.environment,
            seed=run_cfg.seed,
            matchup_name=matchup_name,
            num_perturbations=run_cfg.num_perturbations,
            perturbation_sigma=run_cfg.perturbation_sigma,
            analysis_seed=run_cfg.analysis_seed,
            onset_source_experiment=run_cfg.onset_source_experiment,
            onset_ledger_root=run_cfg.onset_ledger_root,
            angle_2b_root=run_cfg.angle_2b_results_root,
            output_root=run_cfg.output_root,
        )

        if result.non_result:
            print(
                f"[angle_2_c] {matchup_name}: NON-RESULT - none of the three "
                f"candidate properties cleanly separated from its null. "
                f"Reported as a legitimate finding, not papered over."
            )
        else:
            print(
                f"[angle_2_c] {matchup_name}: diverging_properties="
                f"{result.diverging_properties} dominant_property="
                f"{result.dominant_property} "
                f"onset_timing_consistent={result.onset_timing.consistent if result.onset_timing else None}"
            )

        _log_to_wandb(result, wandb_project=str(cfg.project_name))

    print(f"[angle_2_c] done. seed={run_cfg.seed} environment={run_cfg.environment}")


def _log_to_wandb(result, wandb_project: str) -> None:
    run = wandb.init(
        project=wandb_project,
        group=f"angle_2c_{result.environment}_seed{result.seed}",
        job_type=result.matchup_name,
        name=f"angle2c-{result.matchup_name}-{result.environment}-seed{result.seed}",
        config=result.run_metadata,
        reinit=True,
    )
    try:
        run.summary.update(
            {
                "metadata_path": str(result.output_paths["metadata"]),
                "null_distribution_csv_path": str(result.output_paths["null_distribution_csv"]),
                "properties_path": str(result.output_paths["properties"]),
            }
        )
        run.log(result.run_metadata)
    finally:
        run.finish()
