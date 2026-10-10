"""Exp3 engineering validation on an ALREADY allocated A100; never submits a job.

Three separately reported sections, none of which selects a numerical tolerance:

1. exactness: unittest suites whose criterion is bitwise equality (pairing/isolation,
   passive replay fidelity, passive and target restart, recording on/off parity,
   real-entry-point capture). Run under each requested matmul precision.
2. oracle: alternative computation graphs for the same quantity (per-state Jacobian
   row vs that row's own gradient, panel-gradient update vs update_actor, frozen-target
   formula vs production) at 'highest' and 'tensorfloat32'. REPORTED as max abs/rel
   differences; acceptability is an owner decision.
3. tf32_sensitivity: every Pilot 1 measurement field at 'highest' vs 'tensorfloat32'
   on synthetic inputs at an approved production architecture. REPORTED only.

Verdicts: EXACTNESS_PASS (oracle/TF32 MEASURED, awaiting owner tolerances),
EXACTNESS_FAIL, NONDETERMINISTIC (a repeated identical computation differed, so
exactness failures are not attributable), INCOMPLETE (skips, nothing run).
--cpu-dry-run checks the harness itself and can never qualify a GPU.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

EXACTNESS = (
    "tests.test_exp3_capture_boundaries",
    "tests.test_exp3_streams",
    "tests.test_exp3_review.PassiveReviewTest",
    "tests.test_exp3_capture",
    "tests.test_exp3_pilots.FixtureTest.test_passive_shared_sampling_no_training_experience_and_resume_exact",
    "tests.test_exp3_pilots.FixtureTest.test_passive_end_to_end_restart_matches_uninterrupted_checkpoint",
    "tests.test_exp3_pilots.FixtureTest.test_target_end_to_end_restart_matches_uninterrupted_and_alternative_evaluator",
    "tests.test_exp3_pilots.FixtureTest.test_process_local_source_recorder_restart_is_exact_and_restores_original_loop",
    "tests.test_exp3_pilots.FixtureTest.test_source_instrumentation_benchmark_has_no_trajectory_changes",
    "tests.test_exp3_pilots.FixtureTest.test_stream_recording_preserves_complete_source_state",
)
EXIT = {"EXACTNESS_PASS": 0, "EXACTNESS_FAIL": 1, "NONDETERMINISTIC": 2}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exactness-precisions", required=True, nargs="+", choices=("highest", "tensorfloat32"))
    p.add_argument("--sensitivity-env", required=True, help="an Exp12 task name, for its observation/action sizes")
    p.add_argument("--sensitivity-arch", required=True, help="an approved Exp12 architecture, e.g. D4W1536")
    p.add_argument("--sensitivity-states", required=True, type=int, help="synthetic states (engineering size)")
    p.add_argument("--cpu-dry-run", action="store_true", help="harness self-check on CPU; never a GPU pass")
    p.add_argument("--skip-exactness", action="store_true", help="harness self-check only; verdict INCOMPLETE")
    return p.parse_args(argv)


def check_source(root, expected):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != expected or len(head) != 40:
        raise ValueError("validation source revision mismatch")
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                                     text=True).strip()
    if status:
        raise ValueError("validation requires a clean checkout")
    return head


def run_suite(names, precision, log_path):
    import jax

    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    start = time.perf_counter()
    with open(log_path, "x") as log, jax.default_matmul_precision(precision):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skips": len(result.skipped), "seconds": round(time.perf_counter() - start, 1),
            "passed": result.wasSuccessful() and not result.skipped and result.testsRun > 0,
            "log": Path(log_path).name}


def measurements(args):
    import jax

    from experiments.exp12.envs import create_envs
    from experiments.exp3 import gpu_checks
    from generate_manifest import EXP12_ARCHS, EXP12_ENVS
    from tests.exp12_helpers import compose

    envs, archs = dict(EXP12_ENVS), {a[0]: a for a in EXP12_ARCHS}
    if args.sensitivity_env not in envs or args.sensitivity_arch not in archs:
        raise ValueError("sensitivity env/arch must be an approved Exp12 task and architecture")
    if args.sensitivity_states <= 0:
        raise ValueError("explicit positive --sensitivity-states required")
    group = envs[args.sensitivity_env]
    env_type = {"dmc_hard": "dmc", "dmc_medium": "dmc", "myosuite_simba": "myosuite"}.get(group, group)
    env = create_envs(env_type=env_type, seed=1, env_name=args.sensitivity_env, num_train_envs=1, num_eval_envs=1,
                      rescale_action=True, no_termination=False, action_repeat=2, reward_scale=1.0,
                      max_episode_steps=1000)[0]
    obs_dim, act_dim = env.single_observation_space.shape[0], env.single_action_space.shape[0]
    env.close()
    arch = archs[args.sensitivity_arch]
    cfg = compose([f"env_name={args.sensitivity_env}", f"env={group}",
                   f"critic_num_blocks={arch[1]}", f"critic_hidden_dim={arch[2]}"])
    twin = bool(cfg.agent.critic_use_cdq)
    u = gpu_checks.make_agent(cfg.agent, obs_dim, act_dim)
    i = gpu_checks.make_agent(cfg.agent, obs_dim, act_dim)
    # A distinct second critic; synthetic, untrained, engineering only.
    i._critic = i._critic.replace(params=jax.tree_util.tree_map(lambda x: x * 1.01, i._critic.params))
    batch = gpu_checks.synthetic_inputs(0, args.sensitivity_states, obs_dim, act_dim)
    key = jax.random.PRNGKey(0)
    out = {"inputs": {"env": args.sensitivity_env, "arch": args.sensitivity_arch, "observation_dim": obs_dim,
                      "action_dim": act_dim, "states": args.sensitivity_states, "twin": twin,
                      "note": "untrained networks and synthetic states: engineering sensitivity, not pilot data"},
           "repeat_bitwise": {}, "oracle": {}, "status": "MEASURED; no tolerance approved"}
    for precision in gpu_checks.PRECISIONS:
        with jax.default_matmul_precision(precision):
            out["repeat_bitwise"][precision] = gpu_checks.repeat_bitwise(u, i, batch["observation"], key, twin)
            out["oracle"][precision] = gpu_checks.oracle_measurements(u, i, batch, key, twin)
    out["tf32_sensitivity"] = gpu_checks.tf32_sensitivity(u, i, batch["observation"], key, twin)
    return out


def verdict(report, dry_run, skipped_exactness):
    if skipped_exactness or not report["exactness"]:
        v = "INCOMPLETE"
    elif not all(report["measurements"]["repeat_bitwise"].values()):
        v = "NONDETERMINISTIC"
    elif any(s["skips"] or s["tests_run"] == 0 for s in report["exactness"].values()):
        v = "INCOMPLETE"
    elif all(s["passed"] for s in report["exactness"].values()):
        v = "EXACTNESS_PASS"
    else:
        v = "EXACTNESS_FAIL"
    return f"CPU_DRY_RUN_{v}" if dry_run else v


def main(argv=None):
    args = parse(argv)
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    if not out.is_absolute() or root == out or root in out.resolve().parents:
        raise ValueError("absolute evidence path outside the checkout required")
    head = check_source(root, args.expected_commit)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "main"))
    import jax
    import jaxlib

    from experiments.exp12.precision import configure_compilation_cache, runtime_info, set_matmul_precision

    if jax.__version__ != "0.4.34" or jaxlib.__version__ != "0.4.34":
        raise ValueError("pinned JAX/JAXLIB 0.4.34 required")
    if args.cpu_dry_run:
        if jax.default_backend() != "cpu":
            raise ValueError("--cpu-dry-run is for the CPU backend only")
    elif (jax.default_backend() != "gpu" or len(jax.devices()) != 1
          or jax.devices()[0].device_kind != "NVIDIA A100-SXM4-40GB"):
        raise ValueError("one actual A100-SXM4-40GB required; CPU fallback refused")
    if not jax.local_devices(backend="cpu"):
        raise ValueError("SAC explicit CPU initialization requires the CPU backend too")
    set_matmul_precision()
    configure_compilation_cache()
    (out / "backend.json").write_text(json.dumps({
        "commit": head, "runtime": runtime_info(), "device_kind": jax.devices()[0].device_kind,
        "python": sys.version, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpu_dry_run": args.cpu_dry_run,
    }, indent=2))
    report = {"commit": head, "exactness": {}, "measurements": None}
    os.chdir(root / "main")
    t0 = time.perf_counter()
    report["measurements"] = measurements(args)
    report["measurement_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "measurements.json").write_text(json.dumps(report["measurements"], indent=1))
    if not args.skip_exactness:
        for precision in args.exactness_precisions:
            report["exactness"][precision] = run_suite(EXACTNESS, precision, out / f"exactness_{precision}.log")
            (out / "validation.partial.json").write_text(json.dumps(report, indent=1))
    report["verdict"] = verdict(report, args.cpu_dry_run, args.skip_exactness)
    report["scope"] = ("tiny-fixture exactness plus production-width synthetic sensitivity; not real-width pilot, "
                       "source-run, MyoSuite/HumanoidBench or scientific qualification")
    (out / "validation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"verdict": report["verdict"]}))
    return EXIT.get(report["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
