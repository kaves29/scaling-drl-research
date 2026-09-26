"""generate_manifest.py's job classification, against a replica of the
"official-angle-1" Lightning Studio as of 2026-09-26: 11 finished D2W512
runs (from before the DONE marker existed), 2 interrupted ones (dog-trot
seed 5, humanoid-run seed 3), and nothing else started."""

import os
import pickle
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

import generate_manifest as gm

FINISHED = [("dog-run", s) for s in range(1, 6)] + [("dog-trot", s) for s in range(1, 5)] + [
    ("humanoid-run", 1), ("humanoid-run", 2),
]
PARTIAL = [("dog-trot", 5), ("humanoid-run", 3)]
DMC_HARD_ENV_STEPS = 1_000_000


def _ckpt_dir(env, seed):
    return Path("angle1_prod") / "D2W512" / env / f"seed_{seed}"


def _write_checkpoint(env, seed, meta_step, last_env_step):
    d = _ckpt_dir(env, seed)
    (d / "logs").mkdir(parents=True)
    (d / "agent_ckpt").mkdir()
    for name in ("buffer_state.pkl", "obs_rms.pkl", "probe_capture_state.pkl"):
        (d / name).write_bytes(b"")
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump({"interaction_step": meta_step, "update_step": 0, "update_counter": 0}, f)
    pd.DataFrame({"env_step": range(0, last_env_step + 1, 4000)}).to_csv(
        d / "logs" / f"{env}_CD2_CW512_AD1_AW128_seed{seed}.csv", index=False,
    )


def _write_finished_artifacts(env, seed, pool_root, snapshot=True):
    metrics = Path("results/metrics/angle_1/D2W512") / env / f"angle_1_D2W512_{env}_seed{seed}.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"interaction_step": range(0, 500_001, 2000), "env_step": range(0, 1_000_001, 4000)}).to_csv(
        metrics, index=False,
    )
    if snapshot:
        snap = Path(pool_root) / env / "D2W512" / f"seed{seed}" / "baseline_pool" / "checkpoints" / "pool"
        snap.mkdir(parents=True)
        (snap / "probe_capture_state.pkl").write_bytes(b"")


class GenerateManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.pool_root = str(Path(self.tmpdir) / "pool")
        patcher = mock.patch.object(gm, "POOL_STORAGE_ROOT", self.pool_root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _generate(self):
        with redirect_stdout(StringIO()):
            gm.main()
        return {
            name: Path(name).read_text().splitlines() for name in ("job_list.txt", "job_list_pool.txt")
        }

    def _commands_by_ckpt(self, lines):
        return {line.split("--checkpoint_dir ")[1].split()[0]: line for line in lines}

    def test_official_angle_1_replica(self):
        clean = self._generate()
        clean_grid = self._commands_by_ckpt(clean["job_list.txt"])

        for env, seed in FINISHED:
            _write_checkpoint(env, seed, meta_step=487_539, last_env_step=DMC_HARD_ENV_STEPS)
            _write_finished_artifacts(env, seed, self.pool_root)
        for env, seed in PARTIAL:
            _write_checkpoint(env, seed, meta_step=312_525, last_env_step=625_048)

        status = {}
        gm.add_grid([], status)
        self.assertEqual(len(status["done_legacy"]), 11)
        self.assertEqual(len(status["resume"]), 2)
        self.assertEqual(len(status["fresh"]), 150 - 13)
        self.assertNotIn("review", status)

        after = self._generate()
        grid = self._commands_by_ckpt(after["job_list.txt"])
        finished_dirs = {os.path.abspath(_ckpt_dir(e, s)) for e, s in FINISHED}
        partial_dirs = {os.path.abspath(_ckpt_dir(e, s)) for e, s in PARTIAL}

        self.assertEqual(len(grid), 139)
        self.assertEqual(set(grid), set(clean_grid) - finished_dirs)
        for d in partial_dirs:
            # Unchanged command: run.py resumes from meta.pkl at the same path.
            self.assertEqual(grid[d], clean_grid[d])
        self.assertEqual(after["job_list_pool.txt"], clean["job_list_pool.txt"])

    def test_finished_run_missing_an_artifact_is_never_resumed(self):
        _write_checkpoint("dog-run", 1, meta_step=487_539, last_env_step=DMC_HARD_ENV_STEPS)
        _write_finished_artifacts("dog-run", 1, self.pool_root, snapshot=False)
        state, reason = gm.classify(
            os.path.abspath(_ckpt_dir("dog-run", 1)), "angle_1", "D2W512", "dog-run", 1,
            DMC_HARD_ENV_STEPS, snapshot=True,
        )
        self.assertEqual(state, "review")
        self.assertIn("pool snapshot", reason)

        shutil.rmtree("results")
        state, reason = gm.classify(
            os.path.abspath(_ckpt_dir("dog-run", 1)), "angle_1", "D2W512", "dog-run", 1,
            DMC_HARD_ENV_STEPS, snapshot=False,
        )
        self.assertEqual(state, "review")
        self.assertIn("metrics CSV", reason)

    def test_done_marker_and_unstarted_dirs(self):
        _write_checkpoint("dog-run", 1, meta_step=100, last_env_step=200)
        (_ckpt_dir("dog-run", 1) / gm.DONE_MARKER).write_text("{}")
        args = ("angle_1", "D2W512", "dog-run", 1, DMC_HARD_ENV_STEPS)
        self.assertEqual(gm.classify(os.path.abspath(_ckpt_dir("dog-run", 1)), *args, snapshot=True)[0], "done")

        started = _ckpt_dir("dog-run", 2)
        (started / "logs").mkdir(parents=True)
        self.assertEqual(gm.classify(os.path.abspath(started), *args, snapshot=True)[0], "review")
        self.assertEqual(gm.classify(os.path.abspath(_ckpt_dir("dog-run", 3)), *args, snapshot=True)[0], "fresh")


if __name__ == "__main__":
    unittest.main()
