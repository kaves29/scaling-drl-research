"""Small real CPU contracts for the previously unexecuted fitting/checkpoint path.

Synthetic narrow networks/replay are unit fixtures, never qualification evidence.
Production scientific settings are not edited. Run in the pinned Delta CPU env.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Contract(unittest.TestCase):
    def test_real_ten_target_fit_curve_contract_and_mutations(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from types import SimpleNamespace as NS
        from experiments.exp12.probe import ProbeConfig
        from experiments.exp12.run_probes import run_probe_networks
        from scale_rl.agents.sac.sac_network import SACCritic
        from scripts.exp96_multigpu_qualification import validate_probe_result

        self.assertEqual(jax.default_backend(), "cpu")
        net = SACCritic("residual", 1, 8, jnp.float32)
        obs = np.random.default_rng(17).normal(size=(32, 3)).astype(np.float32)
        act = np.zeros((32, 2), np.float32)
        current = net.init(
            jax.random.PRNGKey(1),
            observations=jnp.asarray(obs[:1]),
            actions=jnp.asarray(act[:1]),
        )["params"]
        fresh = net.init(
            jax.random.PRNGKey(2),
            observations=jnp.asarray(obs[:1]),
            actions=jnp.asarray(act[:1]),
        )["params"]
        tx = optax.adamw(1e-4, weight_decay=1e-2)
        result = run_probe_networks(
            NS(),
            NS(_observations=obs, _actions=act, _num_in_buffer=32),
            net,
            {"current": (net, current), "fresh": (net, fresh)},
            tx,
            tx,
            991,
            50,
            ProbeConfig(10, 2000, 50, 32, 4, 1e5, 8),
        )
        validate_probe_result(result)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "curves.npz"
            np.savez_compressed(
                p,
                **{
                    f"{name}_{key}": v
                    for name, items in result.items()
                    for key, v in items.items()
                },
            )
            with np.load(p) as data:
                restored = {
                    name: {
                        key[len(name) + 1 :]: data[key]
                        for key in data.files
                        if key.startswith(name + "_")
                    }
                    for name in ("current", "fresh")
                }
            validate_probe_result(restored)
        result["current"]["losses"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_probe_result(result)
        result["current"]["losses"] = np.zeros((10, 1999))
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_probe_result(result)

    def test_startup_worker_real_kernel_cpu_fixture(self):
        import jax
        from experiments.exp12 import precision
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from scripts import exp96_multigpu as m
        from scripts import exp96_multigpu_qualification as q

        self.assertEqual(jax.default_backend(), "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "startup"
            original_write = m.write

            def write(path, value, **kwargs):
                original_write(path, value, **kwargs)
                if Path(path).name == "ready.json":
                    original_write(root / "EXIT.json", dict(synthetic=True))

            with patch.dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="GPU-fixture",
                EXP96_ASSIGNED_CPU_IDS="0,1",
            ), patch.object(
                q, "cuda_visible_identity", return_value=dict(uuid="GPU-fixture")
            ), patch.object(
                os, "sched_getaffinity", return_value={0, 1}
            ), patch.object(
                jax, "default_backend", return_value="gpu"
            ), patch.object(
                jax, "devices", return_value=[NS()]
            ), patch.object(
                m, "checked", return_value="unit-fixture"
            ), patch.object(
                m, "write", side_effect=write
            ), patch.object(
                precision,
                "runtime_info",
                return_value=dict(hardware_mocked=True, actual_backend="cpu"),
            ):
                q.gpu_startup_worker(root)
            receipt = m.read(root / "ready.json")
            self.assertTrue(receipt["kernel_finite_fp32"])
            self.assertTrue(receipt["runtime"]["hardware_mocked"])
            self.assertEqual(receipt["source"], "unit-fixture")

    def test_real_trainer_checkpoint_roundtrip_and_next_update(self):
        import jax
        from experiments.exp1 import compose_config
        from experiments.exp12.state import state_differences
        from experiments.exp12.trainer import Exp12Trainer
        from scripts.exp96_multigpu_qualification import engineering_dimensions

        self.assertEqual(jax.default_backend(), "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cfg = compose_config(
                str(Path(__file__).resolve().parents[1] / "configs"),
                "base_exp12_ten",
                [
                    "seed=991",
                    "run_role=dev",
                    "env_name=dog-run",
                    "env=dmc_hard",
                    "critic_num_blocks=1",
                    "critic_hidden_dim=8",
                    "actor_hidden_dim=8",
                    "buffer.max_length=64",
                    "buffer.min_length=16",
                    "buffer.sample_batch_size=4",
                    f"results_root={root}",
                ],
            )
            trainer = Exp12Trainer(cfg, str(root / "trainer"))
            try:
                self.assertEqual(
                    engineering_dimensions(trainer.train_env, trainer.buffer), (223, 38)
                )
                trainer._attach_logger()
                trainer.observations, _ = trainer.train_env.reset()
                trainer.eval_env.reset()
                trainer.train(20)
                a = trainer.save(root / "before")
                trainer.restore(a, new_wandb_run=True)
                b = trainer.save(root / "after")
                self.assertEqual(state_differences(a, b), [])
                prior = trainer.update_step
                trainer.train(21)
                self.assertGreater(trainer.update_step, prior)
                self.assertEqual(trainer.interaction_step, 21)
            finally:
                trainer.close()


if __name__ == "__main__":
    unittest.main()
