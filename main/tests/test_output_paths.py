"""Output locations: absolute-path enforcement at the checkpoint boundary (A1),
launch-directory independence of every output (A2), and validated,
configurable roots for Angles 1/2A/2B/2C (A3)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO / "configs")


def _make_agent(normalize_observation):
    from scale_rl.agents import create_agent

    cfg = OmegaConf.create({
        "agent_type": "sac", "seed": 0, "num_train_envs": 1, "max_episode_steps": 100,
        "normalize_observation": normalize_observation, "actor_block_type": "residual",
        "actor_num_blocks": 1, "actor_hidden_dim": 8, "actor_learning_rate": 1e-4,
        "actor_weight_decay": 1e-2, "critic_block_type": "residual", "critic_num_blocks": 1,
        "critic_hidden_dim": 8, "critic_learning_rate": 1e-4, "critic_weight_decay": 1e-2,
        "critic_use_cdq": False, "temp_target_entropy": None, "temp_target_entropy_coef": -0.5,
        "temp_initial_value": 0.01, "temp_learning_rate": 1e-4, "temp_weight_decay": 0.0,
        "target_tau": 0.005, "gamma": 0.99, "n_step": 1, "mixed_precision": False,
        "actor_sparsity": 0.0, "critic_sparsity": 0.0,
    })
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return create_agent(observation_space=space, action_space=space, cfg=cfg)


class CheckpointBoundaryRejectsRelativePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.addCleanup(os.chdir, original_cwd)

    def test_save_and_load_reject_relative_paths_before_touching_disk(self):
        for normalize in (False, True):
            agent = _make_agent(normalize)
            for method in (agent.save_checkpoint, agent.load_checkpoint):
                with self.subTest(normalize=normalize, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "checkpoint_dir must be an absolute path"):
                        method("relative/ckpt")
                    self.assertEqual(os.listdir(self.tmpdir), [])

    def test_absolute_paths_still_round_trip(self):
        agent = _make_agent(True)
        ckpt = os.path.join(self.tmpdir, "ckpt")
        agent.save_checkpoint(ckpt)
        _make_agent(True).load_checkpoint(ckpt)


_STUB_WANDB = '''
import json, os
class _Run:
    name = "stub-run"
    id = "stub-id"
    summary = {}
    def log(self, *a, **k): pass
    def finish(self): pass
run = None
def init(*a, **kw):
    global run
    with open(os.environ["STUB_WANDB_RECORD"], "a") as f:
        f.write(json.dumps({"dir": kw.get("dir")}) + "\\n")
    run = _Run()
    return run
def log(*a, **k): pass
'''


class LaunchDirectoryIndependenceTest(unittest.TestCase):
    """Runs one real Angle 1 job (run.py, default output roots) from a copy of
    the repository twice: launched from the copy's root, then from an unrelated
    directory. Every file it writes must land at the same absolute path."""

    def setUp(self):
        self.tmpdir = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.repo = Path(self.tmpdir) / "repo"
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "."], capture_output=True, text=True, check=True,
        ).stdout.split("\0")
        for rel in filter(None, tracked):
            src = REPO / rel
            if src.is_file():
                dst = self.repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        self.stub_dir = Path(self.tmpdir) / "stub"
        self.stub_dir.mkdir()
        (self.stub_dir / "wandb.py").write_text(_STUB_WANDB)
        self.elsewhere = Path(self.tmpdir) / "elsewhere"
        self.elsewhere.mkdir()

    def _snapshot(self):
        # Orbax names the files inside agent_ckpt/ by content hash.
        return {
            str(p).split("/agent_ckpt/")[0] + "/agent_ckpt/" if "/agent_ckpt/" in str(p) else str(p)
            for root in (self.repo, self.elsewhere, Path(self.tmpdir) / "ckpt")
            for p in root.rglob("*") if p.is_file()
        }

    def _run_from(self, cwd):
        record = Path(self.tmpdir) / "wandb_record.jsonl"
        record.unlink(missing_ok=True)
        before = self._snapshot()
        overrides = [
            "env_name=cheetah-run", "seed=1", "actor_num_blocks=1", "actor_hidden_dim=8",
            "critic_num_blocks=1", "critic_hidden_dim=8", "env.num_env_steps=40",
            "buffer.min_length=5", "buffer.max_length=200", "buffer.sample_batch_size=4",
            "evaluation_per_interaction_step=10", "logging_per_interaction_step=5",
            "num_eval_episodes=1", "critic_degradation=true", "pathology_prop=true",
            "+save_probe_capture_snapshot=true",
        ]
        cmd = [sys.executable, str(self.repo / "run.py"), "--experiment", "angle_1",
               "--config_path", str(self.repo / "configs"),
               "--checkpoint_dir", str(Path(self.tmpdir) / "ckpt"),
               "--checkpoint_interval", "10", "--checkpoint_start_frac", "0"]
        for override in overrides:
            cmd += ["--overrides", override]
        env = {**os.environ, "PYTHONPATH": f"{self.stub_dir}{os.pathsep}{self.repo}",
               "JAX_PLATFORMS": "cpu", "STUB_WANDB_RECORD": str(record),
               "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("WANDB_DIR", None)
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        written = self._snapshot() - before
        wandb_dirs = [json.loads(line)["dir"] for line in record.read_text().splitlines()]
        return written, wandb_dirs

    def test_outputs_do_not_depend_on_the_launch_directory(self):
        from_repo, wandb_from_repo = self._run_from(self.repo)
        for path in from_repo:
            shutil.rmtree(path, ignore_errors=True) if path.endswith("/agent_ckpt/") else os.remove(path)
        from_elsewhere, wandb_from_elsewhere = self._run_from(self.elsewhere)

        self.assertEqual(sorted(from_repo), sorted(from_elsewhere))
        self.assertFalse([p for p in from_elsewhere if p.startswith(str(self.elsewhere))])
        results = str(self.repo / "results")
        for kind in ("metrics", "ledgers", "baseline_calibration_pool"):
            self.assertTrue(any(p.startswith(f"{results}/{kind}/") for p in from_repo), kind)
        self.assertEqual(wandb_from_repo, [str(self.repo)])
        self.assertEqual(wandb_from_elsewhere, [str(self.repo)])


def _compose(config_name, overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        return compose(config_name=config_name, overrides=overrides)


class ConfigurableRootsTest(unittest.TestCase):
    def tearDown(self):
        GlobalHydra.instance().clear()

    def test_default_roots_are_absolute_and_under_the_repo_results(self):
        from experiments.angle_2_a import resolve_angle2a_roots
        from experiments.angle_2b.config import validate_angle2b_config
        from experiments.angle_2c.config import validate_angle2c_config

        results = str(REPO / "results")
        roots_2a = resolve_angle2a_roots(_compose("base_angle2a", []))
        cfg_2b = validate_angle2b_config(_compose("base_angle2b", ["angle_2_b.matchup_names=[matchup_1]"]))
        cfg_2c = validate_angle2c_config(_compose("base_angle2c", ["angle_2_c.matchup_names=[matchup_1]"]))
        self.assertEqual(roots_2a["output_root"], f"{results}/angle_2a")
        self.assertEqual(roots_2a["pool_root"], f"{results}/baseline_calibration_pool")
        self.assertEqual(roots_2a["onset_ledger_root"], f"{results}/ledgers")
        self.assertEqual(cfg_2b.pool_root, f"{results}/baseline_calibration_pool")
        self.assertEqual(cfg_2b.angle_2a_results_root, f"{results}/angle_2a")
        self.assertEqual(cfg_2c.pool_root, f"{results}/baseline_calibration_pool")
        self.assertEqual(cfg_2c.angle_2a_results_root, f"{results}/angle_2a")
        self.assertEqual(cfg_2c.onset_ledger_root, f"{results}/ledgers")

    def test_results_root_and_explicit_roots_are_honored(self):
        from experiments.angle_2_a import resolve_angle2a_roots
        from experiments.angle_2b.config import validate_angle2b_config
        from experiments.angle_2c.config import validate_angle2c_config

        roots_2a = resolve_angle2a_roots(_compose("base_angle2a", ["results_root=/x", "angle_2_a.pool_root=/pool"]))
        self.assertEqual(roots_2a["output_root"], "/x/angle_2a")
        self.assertEqual(roots_2a["r_calibration_root"], "/x/angle_2a_r_calibration")
        self.assertEqual(roots_2a["pool_root"], "/pool")
        cfg_2b = validate_angle2b_config(_compose(
            "base_angle2b", ["angle_2_b.matchup_names=[matchup_1]", "results_root=/x", "angle_2_b.pool_root=/pool"],
        ))
        self.assertEqual((cfg_2b.output_root, cfg_2b.pool_root), ("/x/angle_2b", "/pool"))
        cfg_2c = validate_angle2c_config(_compose(
            "base_angle2c", ["angle_2_c.matchup_names=[matchup_1]", "results_root=/x", "angle_2_c.angle_2a_results_root=/a"],
        ))
        self.assertEqual((cfg_2c.angle_2a_results_root, cfg_2c.pool_root), ("/a", "/x/baseline_calibration_pool"))

    def test_relative_roots_are_rejected_at_startup(self):
        from experiments.angle_2_a import resolve_angle2a_roots
        from experiments.angle_2b.config import validate_angle2b_config
        from experiments.angle_2c.config import validate_angle2c_config

        cases = [
            (resolve_angle2a_roots, "base_angle2a", ["angle_2_a.pool_root=results/pool"], "angle_2_a.pool_root"),
            (resolve_angle2a_roots, "base_angle2a", ["results_root=results"], "results_root"),
            (validate_angle2b_config, "base_angle2b",
             ["angle_2_b.matchup_names=[matchup_1]", "angle_2_b.pool_root=results/pool"], "angle_2_b.pool_root"),
            (validate_angle2b_config, "base_angle2b",
             ["angle_2_b.matchup_names=[matchup_1]", "angle_2_b.angle_2a_results_root=results/angle_2a"],
             "angle_2_b.angle_2a_results_root"),
            (validate_angle2c_config, "base_angle2c",
             ["angle_2_c.matchup_names=[matchup_1]", "angle_2_c.onset_ledger_root=results/ledgers"],
             "angle_2_c.onset_ledger_root"),
        ]
        for fn, config_name, overrides, field in cases:
            with self.subTest(field=field, overrides=overrides):
                with self.assertRaisesRegex(ValueError, f"{field} must be an absolute path"):
                    fn(_compose(config_name, overrides))

    def test_angle_1_rejects_a_relative_results_root_before_training(self):
        from experiments.angle_1 import run

        GlobalHydra.instance().clear()
        with self.assertRaisesRegex(ValueError, "results_root must be an absolute path"):
            run({
                "experiment": "angle_1", "config_path": CONFIG_PATH, "config_name": "base_sac",
                "overrides": ["env_name=cheetah-run", "results_root=results"],
                "checkpoint_dir": None, "checkpoint_interval": 10, "checkpoint_start_frac": 0.0,
            })

    def test_manifest_default_commands_are_unchanged_and_relative_root_is_rejected(self):
        import generate_manifest as gm

        jobs, status = [], {}
        gm.add_grid(jobs, status)
        self.assertEqual(len(jobs) + sum(len(v) for k, v in status.items() if k not in ("fresh", "resume")), 150)
        self.assertFalse(any("results_root" in job for job in jobs))
        with self.assertRaisesRegex(ValueError, "--results-root must be an absolute path"):
            gm.main(["--results-root", "results"])


if __name__ == "__main__":
    unittest.main()
