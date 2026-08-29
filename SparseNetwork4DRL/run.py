import argparse

# Must run before anything below - `import experiments` (next block) has the
# side effect of importing experiments.angle_1, which does `import jax`, and
# JAX_PLATFORMS/PYOPENGL_PLATFORM/HIP_VISIBLE_DEVICES must already be correct
# before JAX initializes at all (see utils/hardware.py). This runs
# unconditionally here, before argparse even parses --experiment, so it's
# not contingent on which experiment is requested or on any particular
# module's import order.
from utils.hardware import configure_hardware_env

configure_hardware_env()

from experiments.registry import UnknownExperimentError, get_experiment, list_experiments
# Importing experiments registers every experiment module (angle_1, angle_2_a, ...)
# as a side effect. See experiments/__init__.py.
import experiments  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--experiment",
        type=str,
        default="angle_1",
        help=f"Which experiment pipeline to run. Available: {', '.join(list_experiments())}",
    )
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="base_sac")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=100_000)
    parser.add_argument("--checkpoint_start_frac", type=float, default=0.4)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        experiment_fn = get_experiment(args.experiment)
    except UnknownExperimentError as e:
        parser.error(str(e))

    print("RUNNING SUCCESSFULLY, STANDBY!")
    experiment_fn(vars(args))
