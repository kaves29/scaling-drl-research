"""Evidence observer delegation and immutability without running GPU calibration."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.capture_exp12_range_evidence import capture_result, main


class RangeCaptureTest(unittest.TestCase):
    def test_delegates_once_and_returns_the_identical_result(self):
        result = {
            "fresh": {
                "losses": np.arange(5000, dtype=np.float32).reshape(5, 1000),
                "score": np.arange(5, dtype=np.float32),
                "offset": np.zeros(5, np.float32),
            }
        }
        original = mock.Mock(return_value=result)
        args = (object(), object())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = capture_result(root, "D4W1536", original, *args, named="unchanged")
            self.assertIs(actual, result)
            original.assert_called_once_with(*args, named="unchanged")
            with np.load(root / "range_D4W1536_pool_25600_curves.npz") as saved:
                for field, values in result["fresh"].items():
                    np.testing.assert_array_equal(saved[f"fresh_{field}"], values)

    def test_refuses_to_replace_retained_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "range_D4W1536_pool_25600_curves.npz"
            path.write_bytes(b"prior evidence")
            with self.assertRaises(FileExistsError):
                capture_result(
                    root, "D4W1536", lambda: {"fresh": {"score": np.ones(5)}}
                )
            self.assertEqual(path.read_bytes(), b"prior evidence")

    def test_probe_failure_is_not_converted_to_a_result(self):
        with tempfile.TemporaryDirectory() as directory:
            original = mock.Mock(side_effect=RuntimeError("fitting failed"))
            with self.assertRaisesRegex(RuntimeError, "fitting failed"):
                capture_result(Path(directory), "D4W1536", original)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_live_cpu_probe_arrays_are_unchanged_by_retention(self):
        import jax
        import jax.numpy as jnp
        import optax
        from experiments.exp12.probe import ProbeConfig, run_probe
        from scale_rl.agents.sac.sac_network import SACCritic

        rng = np.random.default_rng(731)
        buffer = SimpleNamespace(
            _num_in_buffer=10,
            _observations=rng.normal(size=(10, 3)).astype(np.float32),
            _actions=rng.uniform(-1, 1, (10, 2)).astype(np.float32),
        )
        network = SACCritic("simba", 1, 8, jnp.float32)
        params = network.init(
            jax.random.PRNGKey(1), buffer._observations[:1], buffer._actions[:1]
        )["params"]
        args = (
            SimpleNamespace(),
            buffer,
            network,
            {"fresh": (network, params)},
            optax.adamw(1e-4, weight_decay=0.01),
            990,
            0,
            ProbeConfig(5, 3, 20, 16, 4, 1e5, 8),
        )
        reference = run_probe(*args)
        with tempfile.TemporaryDirectory() as directory:
            observed = capture_result(Path(directory), "fixture", run_probe, *args)
            for field in reference["fresh"]:
                np.testing.assert_array_equal(
                    observed["fresh"][field], reference["fresh"][field]
                )

    def test_mocked_full_workflow_preserves_the_approved_case_and_checksums(self):
        from experiments.exp1 import compose_config

        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        cfg = compose_config(
            config_dir,
            "base_exp12",
            [
                "env_name=hopper-hop",
                "env=dmc_medium",
                "seed=990",
                "run_role=dev",
                "critic_num_blocks=4",
                "critic_hidden_dim=1536",
            ],
        )
        trainer = SimpleNamespace(
            cfg=cfg,
            close=mock.Mock(),
            _sac_agent=SimpleNamespace(
                critic=SimpleNamespace(network_def=object(), params=object())
            ),
            buffer=SimpleNamespace(
                _num_in_buffer=5000,
                _observations=np.zeros((5000, 3), np.float32),
                _actions=np.zeros((5000, 2), np.float32),
            ),
            agent=SimpleNamespace(
                obs_rms=SimpleNamespace(mean=np.zeros(3), var=np.ones(3), count=5000),
                epsilon=1e-8,
            ),
        )
        result = {
            "fresh": {
                "score": np.full(5, 0.475, np.float32),
                "b": np.full(5, 0.5, np.float32),
                "final_loss": np.full(5, 0.025, np.float32),
                "offset": np.zeros(5, np.float32),
                "losses": np.zeros((5, 1000), np.float32),
            }
        }
        code = {"commit": "a" * 40, "dirty": False}

        def fake_fit(*args):
            from experiments.exp12 import probe

            for r in range(5):
                probe._base_targets(
                    None,
                    np.array([0, r], np.uint32),
                    np.zeros((25600, 3), np.float32),
                    np.zeros((25600, 2), np.float32),
                    2560,
                    1e5,
                )
            return result

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "capture"
            with mock.patch(
                "utils.run_metadata.code_version", return_value=code
            ), mock.patch(
                "experiments.exp12.precision.runtime_info",
                return_value={
                    "platform": "gpu",
                    "device_kind": "NVIDIA A100-SXM4-40GB",
                },
            ), mock.patch(
                "scripts.probe_fresh_checks._fresh_trainer", return_value=trainer
            ) as factory, mock.patch(
                "scripts.probe_fresh_checks._stub_wandb"
            ), mock.patch(
                "experiments.exp12.probe.run_probe", side_effect=fake_fit
            ) as fit, mock.patch(
                "experiments.exp12.probe._base_targets",
                return_value=np.zeros(25600, np.float32),
            ):
                self.assertEqual(
                    main(
                        [
                            "--out-dir",
                            str(out),
                            "--arch",
                            "D4W1536",
                            "--expected-commit",
                            code["commit"],
                        ]
                    ),
                    0,
                )
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[5:7], (990, 0))
            self.assertEqual(
                (factory.call_args.args[0].env, *factory.call_args.args[1:]),
                ("hopper-hop", 4, 1536, 990),
            )
            self.assertEqual(factory.call_args.args[0].override, [])
            metadata = json.loads((out / "capture_metadata.json").read_text())
            for name, checksum in metadata["files_sha256"].items():
                self.assertEqual(
                    hashlib.sha256((out / name).read_bytes()).hexdigest(), checksum
                )
            with np.load(out / "warmup_replay.npz") as replay:
                np.testing.assert_array_equal(
                    replay["observations"], trainer.buffer._observations
                )
                self.assertEqual(float(replay["obs_rms_count"]), 5000)
            self.assertEqual(len(list(out.glob("round_*_teacher_pool.npz"))), 5)
            trainer.close.assert_called_once()

    def test_cpu_cannot_enter_full_setting_gpu_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "capture"
            with mock.patch(
                "utils.run_metadata.code_version",
                return_value={"commit": "a" * 40, "dirty": False},
            ), mock.patch(
                "experiments.exp12.precision.runtime_info",
                return_value={"platform": "cpu", "device_kind": "cpu"},
            ), mock.patch(
                "scripts.probe_fresh_checks._fresh_trainer"
            ) as trainer:
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(
                        [
                            "--out-dir",
                            str(out),
                            "--arch",
                            "D4W1536",
                            "--expected-commit",
                            "a" * 40,
                        ]
                    )
                trainer.assert_not_called()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
