"""Production curve IO with canonical probe settings and a tiny CPU network.

Synthetic replay is engineering input, not a natural fork or positive control.
Full source/population certification is covered by test_exp12_completeness.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from analysis import exp12_validation as validation
from experiments import exp2_arm
from experiments.exp1 import compose_config
from experiments.exp12 import ledger, probe
from experiments.exp12.run_probes import RunProbes
from tests.test_exp12_method_c import LinearCritic


class IntegratedMethodCArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cfg = compose_config(
            str(Path(__file__).resolve().parents[1] / "configs"), "base_exp12", []
        )
        cls.config = OmegaConf.to_container(cls.cfg, resolve=True)
        cls.probe_cfg = probe.probe_config(cls.cfg)
        assert (cls.probe_cfg.rounds, cls.probe_cfg.steps) == (5, 2000)
        assert cls.cfg.testing.force_trigger_check is None
        assert cls.cfg.injection.m is None
        net = LinearCritic()
        observations = np.arange(64, dtype=np.float32).reshape(32, 2) / 65
        actions = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
        params = net.init(
            jax.random.PRNGKey(17), jnp.asarray(observations[:1]), jnp.asarray(actions[:1])
        )["params"]
        trainer = SimpleNamespace(
            cfg=cls.cfg,
            agent=SimpleNamespace(),
            buffer=SimpleNamespace(
                _observations=observations, _actions=actions, _num_in_buffer=32
            ),
            _sac_agent=SimpleNamespace(critic=SimpleNamespace(network_def=net, params=params)),
        )
        # Supply the real runner's dependencies without building/training an RL agent.
        probes = RunProbes.__new__(RunProbes)
        probes.trainer = trainer
        probes.cfg = cls.probe_cfg
        probes.seed = int(cls.cfg.seed)
        probes.dir = cls.root / "run/probes"
        probes.critic_def = net
        probes.fresh = params
        probes.tx = probe.critic_optimizer(cls.cfg.agent)
        probes.injected_tx = probes.tx  # No injected network in this curve-IO check.
        result = probes._probe(1, {"current": (net, params), "fresh": (net, params)})
        cls.record = {
            "check_index": 1,
            "interaction_step": int(cls.cfg.num_interaction_steps) // cls.probe_cfg.checks,
            **{
                f"score_{name}_r{r}": float(value)
                for name in result
                for r, value in enumerate(result[name]["score"])
            },
        }
        identity = dict(
            run_key="synthetic_method_c_io",
            run_role="dev",
            architecture="D1-linear-test-only",
            environment="synthetic-replay",
            seed=int(cls.cfg.seed),
            budget_env_steps=int(cls.cfg.num_env_steps),
            num_interaction_steps=int(cls.cfg.num_interaction_steps),
            num_checks=cls.probe_cfg.checks,
        )
        cls.directory = ledger.write_run(
            identity, [cls.record], None, "running", probes.dir, str(cls.root / "results")
        )
        cls.check2 = exp2_arm.check2(trainer, probes, {"fork_check_index": 1}, params)
        cls.check2_path = probes.dir / "check2_curves.npz"
        (cls.root / "config.json").write_text(json.dumps(cls.config, indent=2))
        (cls.root / "check2.json").write_text(json.dumps(cls.check2, indent=2))
        (cls.root / "record.json").write_text(json.dumps(cls.record, indent=2))

    @classmethod
    def tearDownClass(cls):
        retained = os.environ.get("EXP2_METHOD_C_TEST_ARTIFACTS")
        if retained:
            destination = Path(retained)
            shutil.copytree(cls.root, destination)  # Refuse to overwrite earlier evidence.
            hashes = {
                str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            }
            (destination / "hashes.json").write_text(json.dumps(hashes, indent=2))
        cls.temp.cleanup()

    def validate_parent_curves(self, config=None):
        validation._curves(
            validation.Snapshot(), self.directory / "probe_curves.npz", [1], False,
            [self.record], config if config is not None else self.config,
        )

    def test_exact_produced_saved_and_consolidated_curves_certify(self):
        self.validate_parent_curves()
        with np.load(self.root / "run/probes/check_01.npz") as raw, np.load(
            self.directory / "probe_curves.npz"
        ) as consolidated:
            for name in ("current", "fresh"):
                self.assertEqual(raw[f"{name}_losses"].shape, (5, 2000))
                self.assertTrue(np.isfinite(raw[f"{name}_losses"]).all())
            for key in raw.files:
                np.testing.assert_array_equal(raw[key], consolidated[f"check_01/{key}"])

    def test_real_check2_artifact_certifies_without_forcing_a_rescue(self):
        validation._check2(
            validation.Snapshot(), self.check2, self.check2_path, 1, int(self.cfg.seed),
            self.config, False, "synthetic-curve-io",
        )
        self.assertFalse(self.check2["pass"])
        self.assertEqual(self.check2["paired_difference_rounds"], [0.0] * 5)
        with np.load(self.check2_path) as data:
            for name in ("control", "injected", "fresh"):
                self.assertEqual(data[f"{name}_losses"].shape, (5, 2000))

    def test_exact_artifact_rejects_wrong_or_ambiguous_declared_budget(self):
        for value in (1000, 1999, "2000", None):
            wrong = {**self.config, "probe": {**self.config["probe"], "steps": value}}
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate_parent_curves(wrong)
