"""Angle 2B: Actor-Facing Signal Falsification.

See experiments/angle_2b/ for the implementation, split by concern:
  errors.py         - shared exception types
  config.py          - required-field validation, no hidden defaults
  checkpoint_io.py    - loads a frozen Angle 2A agent snapshot with zero
                        training/environment interaction
  sampling.py         - deterministic 60-state batch construction
  gradients.py        - frozen counterfactual actor-gradient primitive +
                        distortion metrics (D_dir, D_mag, D_grad)
  null_baseline.py    - empirical healthy-critic null distribution, reused
                        from Angle 2A's own null-baseline snapshots
  statistics.py       - (mean + 2*std) null-exceedance comparison, the same
                        criterion used throughout this study
  matchup_2b.py        - orchestrates one full analysis (primary + secondary
                        + null comparison) for one (environment, seed, matchup)
  storage.py           - persistent results (canonical, independent of WandB)

For every matchup_name in angle_2_b.matchup_names, runs the full Angle 2B
analysis and mirrors a run-level summary to WandB. Zero training, zero
environment interaction anywhere in this module or anything it calls - see
research-methodology.md's Angle 2B scope boundary and the "Things Claude
Must Never Change Silently" list (never let Angle 2B turn into a
continued-training experiment - that boundary belongs to Angle 3 only).
"""

import os

import hydra
import omegaconf
import wandb
from dotmap import DotMap

from experiments.angle_2b.config import validate_angle2b_config
from experiments.angle_2b.matchup_2b import run_angle_2b_analysis
from experiments.registry import register_experiment


@register_experiment("angle_2_b")
def run(args: dict) -> None:
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # See the matching comment in experiments/angle_1.py and angle_2_a.py:
    # hydra.initialize() resolves a relative config_path relative to *this
    # file's* directory, not the process CWD, so it must be made absolute first.
    hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)

    run_cfg = validate_angle2b_config(cfg)

    for matchup_name in run_cfg.matchup_names:
        print(
            f"[angle_2_b] environment={run_cfg.environment} seed={run_cfg.seed} "
            f"matchup={matchup_name}: loading frozen Angle 2A snapshots from "
            f"'{run_cfg.angle_2a_results_root}' (zero training/env interaction)..."
        )
        result = run_angle_2b_analysis(
            environment=run_cfg.environment,
            seed=run_cfg.seed,
            matchup_name=matchup_name,
            null_seeds=run_cfg.null_seeds,
            analysis_seed=run_cfg.analysis_seed,
            num_states_per_source=run_cfg.num_states_per_source,
            angle_2a_root=run_cfg.angle_2a_results_root,
            output_root=run_cfg.output_root,
        )

        print(
            f"[angle_2_b] {matchup_name}: "
            f"primary D_dir={result.primary['d_dir']:.4f} "
            f"(exceeds_null={result.null_comparison['d_dir'].exceeds_null}), "
            f"D_mag={result.primary['d_mag']:.4f} "
            f"(exceeds_null={result.null_comparison['d_mag'].exceeds_null}), "
            f"D_grad={result.primary['d_grad']:.4f} "
            f"(exceeds_null={result.null_comparison['d_grad'].exceeds_null}); "
            f"secondary D_dir={result.secondary['d_dir']:.4f} "
            f"D_mag={result.secondary['d_mag']:.4f} "
            f"D_grad={result.secondary['d_grad']:.4f}; "
            f"null_n={len(result.null_pairs)}"
        )

        _log_to_wandb(result, wandb_project=str(cfg.project_name))

    print(f"[angle_2_b] done. seed={run_cfg.seed} environment={run_cfg.environment}")


def _log_to_wandb(result, wandb_project: str) -> None:
    run = wandb.init(
        project=wandb_project,
        group=f"angle_2b_{result.environment}_seed{result.seed}",
        job_type=result.matchup_name,
        name=f"angle2b-{result.matchup_name}-{result.environment}-seed{result.seed}",
        config=result.run_metadata,
        reinit=True,
    )
    try:
        run.summary.update(
            {
                "metadata_path": str(result.output_paths["metadata"]),
                "null_distribution_csv_path": str(result.output_paths["null_distribution_csv"]),
                "gradients_path": str(result.output_paths["gradients"]),
            }
        )
        run.log(result.run_metadata)
    finally:
        run.finish()
