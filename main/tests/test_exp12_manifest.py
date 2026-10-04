"""Phase 6: Exp 1/2 manifests, DONE/resume classification, arm jobs per device, launcher and preflight."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import omegaconf

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import generate_manifest as gm  # noqa: E402
from exp12_helpers import CONFIG_PATH  # noqa: E402
from experiments.exp1 import compose_config  # noqa: E402
from experiments.exp2_arm import ARM_KEYS  # noqa: E402
from utils.run_metadata import differing_keys  # noqa: E402

BUDGETS = {"swimmer-swimmer15": 500_000, "hopper-hop": 500_000, "h1-reach-v0": 2_000_000, "h1-run-v0": 2_000_000}


def _parse(cmd):
    tokens = shlex.split(cmd.split(" > ")[0])
    overrides = [tokens[i + 1] for i, t in enumerate(tokens) if t == "--overrides"]
    flag = lambda name: tokens[tokens.index(name) + 1]
    return {"experiment": flag("--experiment"), "config": flag("--config_name"), "overrides": overrides,
            "ckpt": flag("--checkpoint_dir"), "interval": int(flag("--checkpoint_interval")),
            "log": cmd.split(" > ")[1].split(" ")[0]}


class Exp12GridTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ckpt, self.results = os.path.join(self.root, "ckpt"), os.path.join(self.root, "results")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _grid(self, injection_m=None):
        manifests, status = {}, {}
        gm.add_exp12_grid(manifests, status, self.ckpt, self.results, injection_m)
        return manifests, status

    def test_the_195_run_grid(self):
        jobs = [_parse(c) for c in self._grid()[0]["exp12_exp1_jobs.txt"]]
        self.assertEqual(len(jobs), 195)
        self.assertEqual(len({j["ckpt"] for j in jobs}), 195)
        self.assertEqual(len({j["log"] for j in jobs}), 195)
        for j in jobs:
            self.assertEqual((j["experiment"], j["config"]), ("exp1", "base_exp12"))
            self.assertTrue(os.path.isabs(j["ckpt"]) and os.path.isabs(j["log"]))
            self.assertTrue(j["ckpt"].startswith(self.ckpt))
        keys = {tuple(sorted(o for o in j["overrides"] if o.split("=")[0] in
                             ("critic_num_blocks", "critic_hidden_dim", "env_name", "seed"))) for j in jobs}
        self.assertEqual(len(keys), 195)

    def test_every_job_composes_with_the_methodology_budget(self):
        seen = set()
        for j in map(_parse, self._grid()[0]["exp12_exp1_jobs.txt"]):
            env = next(o.split("=", 1)[1] for o in j["overrides"] if o.startswith("env_name="))
            arch = tuple(o for o in j["overrides"] if o.startswith("critic_"))
            if (env, arch) in seen:  # seeds do not change the budget
                continue
            seen.add((env, arch))
            cfg = compose_config(CONFIG_PATH, "base_exp12", j["overrides"])
            budget = BUDGETS.get(env, 1_000_000)
            self.assertEqual(int(cfg.num_env_steps), budget, env)
            self.assertEqual(int(cfg.num_interaction_steps), budget // 2)
            self.assertEqual(j["interval"], budget // 2 // 20)  # one save per probe check
            self.assertEqual(cfg.run_role, "confirmatory")
            self.assertEqual(cfg.results_root, self.results)
            self.assertEqual(int(cfg.updates_per_interaction_step), 2)
        self.assertEqual(len(seen), 39)

    def test_done_and_resume_classification(self):
        jobs = self._grid()[0]["exp12_exp1_jobs.txt"]
        done, started = _parse(jobs[0])["ckpt"], _parse(jobs[1])["ckpt"]
        os.makedirs(done)
        Path(done, "DONE").write_text("{}")
        os.makedirs(os.path.join(started, "state"))
        Path(started, "state", "LATEST").write_text("step_000000100_x")
        manifests, status = self._grid()
        ckpts = [_parse(c)["ckpt"] for c in manifests["exp12_exp1_jobs.txt"]]
        self.assertEqual(len(ckpts), 194)
        self.assertNotIn(done, ckpts)
        self.assertIn(started, ckpts)  # resumes
        self.assertEqual((status["exp1 done"], status["exp1 resume"]), ([done], [started]))

    def _fork(self, arch, env, seed, device):
        run_dir, _ = gm.exp12_paths(self.ckpt, arch, env, seed)
        os.makedirs(os.path.join(run_dir, "fork"))
        Path(run_dir, "fork", "fork.json").write_text(json.dumps({"fork_step": 1, "device": {
            "platform": "gpu", "device_kind": device, "device_count": 1}}))
        Path(run_dir, "fork", "FORK_READY").write_text("x")
        return run_dir

    def test_arm_jobs_only_for_completed_forks_grouped_by_device(self):
        a100 = self._fork("D6W1536", "dog-run", 3, "NVIDIA A100-SXM4-40GB")
        a40 = self._fork("D4W1024", "myo-reach", 1, "NVIDIA A40")
        self._fork("D2W512", "dog-run", 3, "NVIDIA A100-SXM4-40GB")  # the default critic never forks
        manifests, status = self._grid()
        self.assertEqual([k for k in manifests if k.startswith("exp2_arms")], [])  # m not frozen yet
        self.assertEqual(len(status["exp2 arm waiting for --injection-m"]), 2)
        manifests, status = self._grid("half")
        self.assertEqual(sorted(k for k in manifests if k.startswith("exp2_arms")),
                         ["exp2_arms_NVIDIA_A100_SXM4_40GB.txt", "exp2_arms_NVIDIA_A40.txt"])
        arm = _parse(manifests["exp2_arms_NVIDIA_A100_SXM4_40GB.txt"][0])
        self.assertEqual(arm["experiment"], "exp2_arm")
        self.assertIn(f"fork.source={a100}", arm["overrides"])
        self.assertIn("injection.m=half", arm["overrides"])
        self.assertTrue(arm["ckpt"].startswith(os.path.join(self.ckpt, "exp2_arm")))
        self.assertIn(f"fork.source={a40}", manifests["exp2_arms_NVIDIA_A40.txt"][0])
        # The arm's config equals its parent's except the three arm keys (exp2_arm refuses otherwise).
        parent = next(_parse(c) for c in manifests["exp12_exp1_jobs.txt"] if _parse(c)["ckpt"] == a100)
        to_dict = lambda o: omegaconf.OmegaConf.to_container(compose_config(CONFIG_PATH, "base_exp12", o),
                                                            resolve=True)
        self.assertEqual(set(differing_keys(to_dict(parent["overrides"]), to_dict(arm["overrides"]))), ARM_KEYS)
        Path(arm["ckpt"]).mkdir(parents=True)
        Path(arm["ckpt"], "DONE").write_text("{}")
        manifests, _ = self._grid("half")
        self.assertNotIn("exp2_arms_NVIDIA_A100_SXM4_40GB.txt", manifests)
        with self.assertRaises(ValueError):
            self._grid("most")

    def test_manifests_never_share_a_checkpoint_dir(self):
        self._fork("D6W1536", "dog-run", 3, "GPU A")
        manifests, _ = self._grid("last")
        paths = []
        for name, jobs in manifests.items():
            path = os.path.join(self.root, name)
            Path(path).write_text("\n".join(jobs) + "\n")
            paths.append(path)
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "check_manifest_overlap.py"), *paths],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_cli_writes_the_manifests(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            gm.main(["--grid", "exp12", "--ckpt-root", self.ckpt, "--results-root", self.results])
            self.assertEqual(len(Path(self.root, "exp12_exp1_jobs.txt").read_text().splitlines()), 195)
            with self.assertRaises(ValueError):
                gm.main(["--grid", "exp12", "--ckpt-root", "relative/path", "--results-root", self.results])
        finally:
            os.chdir(cwd)


class LauncherTest(unittest.TestCase):
    def test_claim_launcher_dry_run_on_exp12_manifests(self):
        root = tempfile.mkdtemp()
        ckpt = os.path.join(root, "ckpt")
        manifests, _ = {}, {}
        gm.add_exp12_grid(manifests, {}, ckpt, os.path.join(root, "results"))
        phase = os.path.join(root, "exp12_exp1_jobs.txt")
        Path(phase).write_text("\n".join(manifests["exp12_exp1_jobs.txt"][:2]) + "\n")
        log = os.path.join(root, "launcher.log")
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "claim_launcher.py"), "--concurrency", "1",
                            "--num-gpus", "1", "--dry-run", "--max-seconds", "4", "--log-file", log,
                            "--phase-files", phase], capture_output=True, text=True, cwd=str(REPO), timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("would run: python -u run.py --experiment exp1", r.stdout)
        first = _parse(manifests["exp12_exp1_jobs.txt"][0])["ckpt"]
        self.assertIn(f"claimed {first}", Path(log).read_text())
        self.assertFalse(Path(first, "claim").exists())  # dry run releases its claims
        shutil.rmtree(root, ignore_errors=True)


class PreflightTest(unittest.TestCase):
    TINY = ["critic_num_blocks=1", "critic_hidden_dim=8", "actor_num_blocks=1", "actor_hidden_dim=8",
            "env_name=hopper-hop", "env=dmc_medium", "seed=999", "env.max_episode_steps=40"]

    def _preflight(self, *extra, overrides=()):
        d = tempfile.mkdtemp()
        cmd = [sys.executable, str(REPO / "scripts" / "preflight_checkpoint_check.py"), "--experiment", "exp1",
               "--checkpoint_dir", d, *extra]
        for o in [*self.TINY, *overrides]:
            cmd += ["--override", o]
        env = dict(os.environ, JAX_PLATFORMS="cpu", MUJOCO_GL="disable", WANDB_MODE="disabled")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=600)
        shutil.rmtree(d, ignore_errors=True)
        return r

    def test_pass_with_fork_and_injected_arm(self):
        r = self._preflight("--with-fork", overrides=["fork.architectures=[D1W8]"])
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn("=== PASS ===", r.stdout)
        self.assertRegex(r.stdout, re.compile(r"Check 1: pass=True .*check1_precision=highest"))

    def test_fail_when_the_architecture_does_not_fork(self):
        r = self._preflight("--with-fork")  # D1W8 is not in fork.architectures
        self.assertEqual(r.returncode, 1)
        self.assertIn("did not fork", r.stdout)


if __name__ == "__main__":
    unittest.main()
