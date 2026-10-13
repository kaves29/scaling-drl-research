"""HumanoidBench restoration on CPU across a process boundary, with the real H1 environments.

Existing twin-critic fork tests drive hopper with a forced twin critic; existing HumanoidBench
tests restore the simulator, not the trainer. Here each H1 task runs the real Exp 1 entry point
with the twin critic its suite configures, then:
  - a job killed before and after the fork resumes in a new process to a state bit-identical to
    an uninterrupted run (Exp 1 resume fidelity);
  - the identity arm, restored from the parent's fork in a new process, matches the parent's
    in-process control after the snapshot steps (the CUDA identity case's procedure, on CPU).
Tiny networks and probe budgets keep it CPU-sized; it does not qualify CUDA numerics.
Needs humanoid_bench and MUJOCO_GL=osmesa (CPU hosts) or egl (GPU nodes).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import compose, load_agent_tree, state_differences, tiny_overrides  # noqa: E402
from experiments.exp12 import fork  # noqa: E402
from experiments.exp12.state import latest_state_dir  # noqa: E402

HAVE_HB = importlib.util.find_spec("humanoid_bench") is not None
RUNNER = str(Path(__file__).resolve().parent / "exp12_subprocess_runner.py")
STEPS, FORCED_CHECK, INTERVAL = 400, 5, 20  # checks every 20 steps -> fork at 100
TASKS = ("h1-reach-v0", "h1-run-v0")


def hb_overrides(env_name, results_root, extra=()):
    return tiny_overrides(env_name, "humanoid_bench", steps=STEPS, extra=[
        "run_role=dev", f"testing.force_trigger_check={FORCED_CHECK}", "fork.architectures=[D1W8]",
        "env.max_episode_steps=40", "fork.eval_episodes=1", "fork.identity_snapshot_steps=50",
        f"results_root={results_root}", *extra])


@unittest.skipUnless(HAVE_HB, "needs humanoid_bench (MUJOCO_GL=osmesa on CPU hosts)")
class HumanoidBenchCpuRestoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, run_dir, overrides, experiment="exp1", **crash):
        spec = {"experiment": experiment, "overrides": overrides, "checkpoint_dir": run_dir,
                "checkpoint_interval": INTERVAL, **crash}
        path = os.path.join(self.tmp, f"spec_{os.path.basename(run_dir)}.json")
        with open(path, "w") as f:
            json.dump(spec, f)
        gl = os.environ.get("MUJOCO_GL", "osmesa")
        env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl)
        return subprocess.run([sys.executable, RUNNER, "exp1", path], env=env, capture_output=True, text=True)

    def _ok(self, result):
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])

    def test_the_suite_configures_the_twin_critic(self):
        for task in TASKS:
            with self.subTest(task=task):
                cfg = compose(hb_overrides(task, os.path.join(self.tmp, "r")))
                self.assertTrue(bool(cfg.agent.critic_use_cdq), "HumanoidBench must use clipped double Q")

    def test_killed_jobs_resume_bit_identically_in_a_new_process(self):
        for task in TASKS:
            results = os.path.join(self.tmp, task + "_results")
            overrides = hb_overrides(task, results)
            reference = os.path.join(self.tmp, task + "_reference")
            self._ok(self._run(reference, overrides))
            final = latest_state_dir(Path(reference) / "state")
            self.assertTrue(fork.is_ready(reference), "the forced fork did not happen")
            for name, step in (("before the fork", 67), ("after the fork, in the control", 130)):
                with self.subTest(task=task, kill=name):
                    run_dir = os.path.join(self.tmp, f"{task}_kill_{step}")
                    killed = self._run(run_dir, hb_overrides(task, results + f"_{step}"), crash_step=step)
                    self.assertNotEqual(killed.returncode, 0)
                    self.assertFalse((Path(run_dir) / "DONE").exists())
                    self._ok(self._run(run_dir, hb_overrides(task, results + f"_{step}")))
                    self.assertEqual(state_differences(final, latest_state_dir(Path(run_dir) / "state"),
                                                       ignore_meta=("wandb_run_id",)), [])

    def test_identity_arm_restored_in_a_new_process_matches_the_control(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import compare_identity_fork

        for task in TASKS:
            with self.subTest(task=task):
                results = os.path.join(self.tmp, task + "_identity_results")
                overrides = hb_overrides(task, results, extra=["testing.stop_after_identity_snapshot=true"])
                parent = os.path.join(self.tmp, task + "_parent")
                arm = os.path.join(self.tmp, task + "_identity")
                self._ok(self._run(parent, overrides))
                self._ok(self._run(arm, overrides + [f"fork.source={parent}", "fork.arm=identity"],
                                   experiment="exp2_arm"))
                control_check = fork.validate_control_restore(parent, results)
                self.assertTrue(control_check["pass"])
                self.assertEqual(control_check["max_eps_units"], 0.0)
                result = compare_identity_fork.compare(Path(parent), Path(arm))
                self.assertTrue(result["pass"], result)
                self.assertEqual(result["fork_step"], FORCED_CHECK * INTERVAL)
                self.assertEqual(result["interaction_step"], FORCED_CHECK * INTERVAL + 50)
                params = load_agent_tree(latest_state_dir(fork.fork_dir(parent) / "state"))["critic"]["params"]
                self.assertIn("VmapSACCritic_0", params)  # the twin critic is what was restored


if __name__ == "__main__":
    unittest.main()
