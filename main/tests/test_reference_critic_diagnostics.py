"""CPU tensors and tiny networks only; no RL training or probe fitting."""

import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from scale_rl.diagnostics.critic_metrics import (
    ENV_ROOT,
    PREFIX,
    CriticMeasurements,
    activity,
    aggregate_activity,
    capture_critic,
    from_environment,
    mask_transition,
    rank_metrics,
)
from scale_rl.agents.sac.sac_network import SACCritic
from scale_rl.networks.trainer import Trainer

FIXTURES = Path(__file__).parent / "reference_critic_fixtures"
REFERENCES = json.loads((FIXTURES / "formulas.json").read_text())


def source_function(name, namespace):
    source = next(r["source"] for r in REFERENCES["functions"] if r["name"] == name)
    exec(
        compile(textwrap.dedent(source), "<pinned-reference-formula>", "exec"),
        namespace,
    )
    return namespace[name]


class Tensor(np.ndarray):
    """Minimal NumPy backend for original LoP arithmetic, not a Torch runtime."""

    @property
    def device(self):
        return "cpu"

    def to(self, dtype):
        return self.astype(dtype)


TORCH_FORMULA_BACKEND = SimpleNamespace(
    Tensor=Tensor,
    tensor=lambda x, **kw: np.asarray(x, dtype=kw.get("dtype")).view(Tensor),
    sum=np.sum,
    abs=np.abs,
    log=np.log,
    float32=np.float32,
    int32=np.int32,
    sort=lambda x: (np.sort(x), None),
    flip=lambda x, dims: np.flip(x, axis=dims),
)


def make_critic(block="residual"):
    net = SACCritic(block, 1, 4, jnp.float32)
    batch = {
        "observation": np.arange(24, dtype=np.float32).reshape(8, 3) / 8,
        "action": np.zeros((8, 1), np.float32),
    }
    params = net.init(jax.random.PRNGKey(3), batch["observation"], batch["action"])[
        "params"
    ]
    critic = Trainer(
        network_def=net,
        params=params,
        tx=optax.adamw(1e-4),
        opt_state=optax.adamw(1e-4).init(params),
    )
    return critic, batch


def digest(tree):
    h = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten(tree)
    h.update(str(structure).encode())
    for leaf in leaves:
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class FormulaTest(unittest.TestCase):
    def test_redo_formula_on_identical_inputs(self):
        reference = source_function("estimate_neuron_score", {"jnp": jnp})
        owner = SimpleNamespace(sub_mean_score=False)
        for x in (
            np.zeros((4, 3)),
            np.ones((4, 3)),
            np.array([[0, -1, 2], [0, 3, -4]]),
            np.arange(24).reshape(2, 3, 4),
        ):
            expected = np.asarray(reference(owner, jnp.asarray(x, dtype=jnp.float32)))
            actual = activity(x)
            np.testing.assert_allclose(
                actual["relative_activity"], expected, rtol=2e-7, atol=0
            )
            for tau in (0, 0.1, 0.2):
                np.testing.assert_array_equal(
                    actual["masks"][f"relative_tau_{tau:g}"],
                    expected <= np.float32(tau),
                )

    def test_exact_zero_is_separate_from_small_nonzero_or_negative_activity(self):
        result = activity([[0, 1e-12, -2], [0, -1e-12, -2]])
        np.testing.assert_array_equal(
            result["masks"]["exact_zero"], [True, False, False]
        )
        np.testing.assert_array_equal(
            result["masks"]["relative_tau_0.1"], [True, True, False]
        )

    def test_threshold_is_inclusive_and_zero_denominator_is_defined(self):
        result = activity([[0, 2], [0, 2]], thresholds=(0, 2))
        np.testing.assert_array_equal(result["masks"]["relative_tau_2"], [True, True])
        zero = activity(np.zeros((3, 4)))
        np.testing.assert_array_equal(zero["relative_activity"], np.zeros(4))
        self.assertTrue(all(m.all() for m in zero["masks"].values()))

    def test_unit_and_layer_aggregate_weighting_are_distinct(self):
        result = aggregate_activity(
            {"a": activity(np.zeros((2, 1))), "b": activity(np.ones((2, 3)))}
        )
        self.assertEqual(result["exact_zero/unit_weighted_percent"], 25)
        self.assertEqual(result["exact_zero/layer_mean_percent"], 50)

    def test_rank_known_zero_constant_rank_one_and_orthogonal(self):
        for x, expected in (
            (np.zeros((3, 4)), 0),
            (np.ones((3, 4)), 1),
            (np.outer([1, 2, 3], [2, -1]), 1),
            (np.eye(4), 4),
        ):
            r = rank_metrics(x)
            self.assertAlmostEqual(r["stable_rank"], expected, places=12)
            self.assertAlmostEqual(
                r["entropy_effective_rank_lop"], max(1, expected), places=12
            )
            self.assertEqual(r["squared_energy99_count_lop"], max(1, expected))
        self.assertEqual(
            rank_metrics(np.zeros((3, 4)))["singular_sum99_count_analysis64"], 1
        )

    def test_rank_definitions_and_lop_reference_are_not_interchanged(self):
        singular = np.array([10, 1, 1], np.float32).view(Tensor)
        namespace = {"torch": TORCH_FORMULA_BACKEND, "np": np}
        entropy = source_function("compute_effective_rank", namespace)(singular)
        energy = source_function("compute_approximate_rank", namespace)(singular)
        absolute = source_function("compute_abs_approximate_rank", namespace)(singular)
        actual = rank_metrics(np.diag(singular))
        self.assertAlmostEqual(
            actual["entropy_effective_rank_lop"], float(entropy), places=6
        )
        self.assertEqual(actual["squared_energy99_count_lop"], int(energy))
        self.assertEqual(actual["singular_sum99_count_analysis64"], int(absolute))
        self.assertNotAlmostEqual(actual["stable_rank"], float(entropy), places=2)
        r = rank_metrics(np.diag([100, 1, 1]))
        self.assertEqual(r["singular_sum99_count_analysis64"], 2)
        self.assertEqual(r["squared_energy99_count_lop"], 1)

    def test_centering_batch_ceiling_and_no_width_normalization(self):
        self.assertEqual(rank_metrics(np.ones((3, 8)), center=True)["stable_rank"], 0)
        self.assertEqual(rank_metrics(np.eye(3, 8))["batch_ceiling"], 3)
        self.assertEqual(rank_metrics(np.eye(3, 8), center=True)["batch_ceiling"], 2)
        self.assertEqual(rank_metrics(np.eye(3, 8))["stable_rank"], 3)

    def test_rank_scale_invariance_extreme_finite_inputs(self):
        for scale in (1e-300, 1e300):
            self.assertAlmostEqual(rank_metrics(scale * np.eye(3))["stable_rank"], 3)

    def test_bad_inputs_rejected_not_fabricated_as_zero(self):
        for x in ([], [1, 2], [[np.nan]], np.empty((0, 3)), [[np.inf]]):
            for fn in (rank_metrics, activity):
                with self.assertRaises(ValueError):
                    fn(x)
        for tau in ((-0.1,), (0.1, 0.1), (), (float("nan"),)):
            with self.assertRaises(ValueError):
                activity(np.ones((2, 2)), tau)

    def test_persistence_known_masks_and_empty_denominators(self):
        transition = mask_transition(
            np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)
        )
        self.assertEqual(transition["persistent_count"], 1)
        self.assertEqual(transition["reactivated_count"], 1)
        self.assertEqual(transition["newly_dormant_count"], 1)
        self.assertEqual(transition["persistent_fraction_of_previous"], 0.5)
        self.assertAlmostEqual(transition["jaccard"], 1 / 3)
        empty = mask_transition(np.zeros(4, bool), np.zeros(4, bool))
        self.assertIsNone(empty["jaccard"])
        self.assertIsNone(empty["reactivated_fraction_of_previous"])
        for a, b in ((np.zeros(3, bool), np.zeros(4, bool)), ([0, 1], [1, 0])):
            with self.assertRaises(ValueError):
                mask_transition(a, b)


class CaptureTest(unittest.TestCase):
    def test_actual_post_relu_capture_residual_and_mlp(self):
        for block in ("residual", "mlp"):
            critic, batch = make_critic(block)
            # All expanded/first hidden affines are negative, so genuine
            # ReLU sites are zero even though their preactivations are nonzero.
            params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), critic.params)
            encoder = params["encoder"]
            if block == "residual":
                encoder["ResidualBlock_0"]["Dense_0"]["bias"] = -jnp.ones(16)
            else:
                encoder["MLPBlock_0"]["Dense_0"]["bias"] = -jnp.ones(4)
            critic = critic.replace(params=params)
            _, populations, features = capture_critic(critic, batch)
            self.assertEqual(features.shape, (8, 4))
            self.assertTrue(
                all(np.all(x == 0) for x in populations["post_relu"].values())
            )
            affines = [
                x
                for name, x in populations["dense_affine"].items()
                if ("ResidualBlock_0/Dense_0" in name or "MLPBlock_0/Dense_0" in name)
            ]
            self.assertTrue(affines and np.all(affines[0] == -1))
            self.assertEqual(activity(affines[0])["masks"]["exact_zero"].sum(), 0)

    def test_observation_collection_does_not_change_original_outputs_or_params(self):
        from scale_rl.agents.sac import sac_network

        repo = Path(__file__).resolve().parents[2]
        original_layers = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                "f597ee9:main/scale_rl/networks/layers.py",
            ],
            text=True,
        )
        namespace = {}
        exec(compile(original_layers, "<original-layers>", "exec"), namespace)
        original_network = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                "f597ee9:main/scale_rl/agents/sac/sac_network.py",
            ],
            text=True,
        )
        # Import original network then replace its imported block symbols with
        # the actual archived block classes; no production monkeypatching.
        original = {}
        exec(compile(original_network, "<original-network>", "exec"), original)
        original.update({k: namespace[k] for k in ("MLPBlock", "ResidualBlock")})
        for block in ("residual", "mlp"):
            critic, batch = make_critic(block)
            old = original["SACCritic"](block, 1, 4, jnp.float32)
            old_params = old.init(
                jax.random.PRNGKey(3), batch["observation"], batch["action"]
            )["params"]
            self.assertEqual(digest(old_params), digest(critic.params))
            expected = old.apply(
                {"params": old_params}, batch["observation"], batch["action"]
            )
            captured, _, _ = capture_critic(critic, batch)
            np.testing.assert_array_equal(captured, expected)
            np.testing.assert_array_equal(
                critic(
                    **{"observations": batch["observation"], "actions": batch["action"]}
                ),
                expected,
            )
            self.assertEqual(type(critic.network_def), sac_network.SACCritic)
            from scale_rl.agents.sac.sac_update import get_critic_with_metrics

            original_metrics = get_critic_with_metrics(
                None, None, critic.replace(network_def=old), batch, False
            )
            current_metrics = get_critic_with_metrics(None, None, critic, batch, False)
            self.assertEqual(original_metrics.keys(), current_metrics.keys())
            for name in original_metrics:
                np.testing.assert_array_equal(
                    original_metrics[name], current_metrics[name]
                )

    def test_capture_does_not_mutate_params_opt_state_normalizer_or_rng(self):
        from tests.test_sac_agent_checkpoint import _make_cfg, _make_spaces
        from scale_rl.agents import create_agent

        cfg = _make_cfg()
        cfg.normalize_observation = True
        observation_space, action_space = _make_spaces()
        agent = create_agent(
            observation_space=observation_space, action_space=action_space, cfg=cfg
        )
        core = agent.agent
        raw = {
            "observation": np.ones((4, 4), np.float32),
            "action": np.zeros((4, 2), np.float32),
        }
        batch = {
            "observation": agent._normalize(raw["observation"]),
            "action": raw["action"],
        }
        before = digest(
            (
                core.critic.params,
                core.critic.opt_state,
                core._rng,
                core.actor.params,
                (
                    core.target_critic.params
                    if hasattr(core, "target_critic")
                    else core._target_critic.params
                ),
                agent.obs_rms.mean,
                agent.obs_rms.var,
                agent.obs_rms.count,
            )
        )
        np_before, py_before = pickle.dumps(np.random.get_state()), pickle.dumps(
            random.getstate()
        )
        with tempfile.TemporaryDirectory() as tmp:
            observer = CriticMeasurements(
                Path(tmp) / "new", freeze_first=True, compute_rank=True
            )
            record = observer.record(
                core.critic, batch, interaction_step=20, optimizer_step=17
            )
            self.assertEqual(
                record["inputs"]["current_replay"]["sha256"],
                record["inputs"]["frozen_preprocessed"]["sha256"],
            )
        after = digest(
            (
                core.critic.params,
                core.critic.opt_state,
                core._rng,
                core.actor.params,
                core._target_critic.params,
                agent.obs_rms.mean,
                agent.obs_rms.var,
                agent.obs_rms.count,
            )
        )
        self.assertEqual(before, after)
        self.assertEqual(np_before, pickle.dumps(np.random.get_state()))
        self.assertEqual(py_before, pickle.dumps(random.getstate()))

    def test_observer_frozen_stream_masks_and_explicit_legacy_aliases(self):
        critic, batch = make_critic()
        with tempfile.TemporaryDirectory() as tmp:
            observer = CriticMeasurements(
                Path(tmp) / "new", freeze_first=True, compute_rank=True
            )
            legacy = {"train/critic_srank": 3, "train/critic_gnorm": 2.5}
            first = observer.record(
                critic,
                batch,
                interaction_step=20,
                optimizer_step=17,
                legacy_metrics=legacy,
            )
            changed = {k: v + 2 for k, v in batch.items()}
            second = observer.record(
                critic,
                changed,
                interaction_step=40,
                optimizer_step=17,
                legacy_metrics=legacy,
            )
            self.assertNotEqual(
                first["inputs"]["current_replay"]["sha256"],
                second["inputs"]["current_replay"]["sha256"],
            )
            self.assertEqual(
                first["inputs"]["frozen_preprocessed"],
                second["inputs"]["frozen_preprocessed"],
            )
            self.assertEqual(second["previous_interaction_step"], 20)
            self.assertTrue(any("/transition_v1_new/" in k for k in second["metrics"]))
            alias = f"{PREFIX}/current_replay/encoder_output_uncentered/singular_sum99_count_historical"
            self.assertEqual(second["metrics"][alias], 3)
            with np.load(
                observer.output / second["mask_file"], allow_pickle=False
            ) as stored:
                self.assertTrue(all(stored[k].dtype == bool for k in stored.files))
                self.assertEqual(set(stored.files), set(second["masks"]))
            with self.assertRaises(ValueError):
                observer.frozen["observation"][0, 0] = 99
            with self.assertRaises(ValueError):
                observer.frozen["observation"].flags.writeable = True
            self.assertEqual(
                legacy, {"train/critic_srank": 3, "train/critic_gnorm": 2.5}
            )

    def test_disabled_path_and_optional_rank_no_work(self):
        with mock.patch.dict(os.environ, {ENV_ROOT: ""}), mock.patch.object(
            Path, "mkdir"
        ) as mkdir:
            self.assertIsNone(from_environment("/tmp/unopened"))
            mkdir.assert_not_called()
        critic, batch = make_critic()
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "numpy.linalg.svd", side_effect=AssertionError("SVD disabled")
        ):
            observer = CriticMeasurements(
                Path(tmp) / "new", compute_rank=False, every=2
            )
            record = observer.record(
                critic, batch, interaction_step=20, optimizer_step=17
            )
            self.assertFalse(record["rank_enabled"])
            with mock.patch(
                "scale_rl.diagnostics.critic_metrics.capture_critic",
                side_effect=AssertionError("cadence skip"),
            ):
                self.assertIsNone(
                    observer.record(
                        critic, batch, interaction_step=40, optimizer_step=17
                    )
                )

    def test_invalid_stream_or_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                CriticMeasurements(Path(tmp) / "out", every=0)
            with self.assertRaises(ValueError):
                CriticMeasurements(
                    Path(tmp) / "out",
                    frozen_batch={"observation": [[0]], "action": [[0], [1]]},
                )
            with mock.patch.dict(os.environ, {ENV_ROOT: tmp + "/run/sub"}):
                with self.assertRaises(ValueError):
                    from_environment(tmp + "/run")
        critic, batch = make_critic()
        with self.assertRaisesRegex(ValueError, "matching"):
            capture_critic(critic, {**batch, "action": batch["action"][:1]})


class LoggingIntegrationTest(unittest.TestCase):
    def test_existing_logging_blocks_reuse_one_preprocessed_batch(self):
        # Execute only real logging AST blocks; no environment steps, SAC
        # updates, or training loop. Covers enabled and disabled wiring.
        source = (
            Path(__file__).parents[1] / "experiments/exp12/trainer.py"
        ).read_text()
        tree = ast.parse(source)
        trainer = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Exp12Trainer"
        )
        function = next(
            n
            for n in trainer.body
            if isinstance(n, ast.FunctionDef) and n.name == "train"
        )
        loop = next(n for n in function.body if isinstance(n, ast.For))
        blocks = [
            n
            for n in loop.body
            if isinstance(n, ast.If)
            and "logging_per_interaction_step" in ast.unparse(n.test)
        ]
        self.assertEqual(len(blocks), 2)
        code = compile(
            ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
            "<actual-logging-blocks>",
            "exec",
        )
        for enabled in (False, True):
            batch = {
                "observation": np.ones((2, 3), np.float32),
                "action": np.zeros((2, 1), np.float32),
            }

            def get_metrics(step, actual):
                self.assertIs(actual, batch)
                actual["observation"] = (
                    actual["observation"] / 2
                )  # wrapper preprocessing
                return {"train/critic_srank": 2}

            observer = mock.Mock() if enabled else None
            logger = mock.Mock()
            logger.average_meter_dict.averages.return_value = {
                "train/critic_srank": 2,
                "train/critic_gnorm": 3.0,
            }
            logger.media_dict = {}
            obj = SimpleNamespace(
                buffer=mock.Mock(),
                agent=SimpleNamespace(get_metrics=get_metrics),
                update_step=17,
                interaction_step=2000,
                pending=mock.Mock(),
                diagnostics=mock.Mock(),
                logger=logger,
                metrics_rows=[],
                critic_measurements=observer,
                _sac_agent=SimpleNamespace(critic="sentinel"),
            )
            obj.buffer.sample.return_value = batch
            obj.diagnostics.window_metrics.return_value = {}
            cfg = SimpleNamespace(
                logging_per_interaction_step=2000, action_repeat=2, num_train_envs=1
            )
            exec(
                code,
                {
                    "self": obj,
                    "cfg": cfg,
                    "interaction_step": 2000,
                    "training_state": mock.Mock(),
                },
            )
            obj.buffer.sample.assert_called_once_with()
            self.assertEqual(
                obj.metrics_rows,
                [
                    {
                        "env_step": 4000,
                        "train/critic_srank": 2,
                        "train/critic_gnorm": 3.0,
                    }
                ],
            )
            if enabled:
                observer.record.assert_called_once()
                self.assertIs(observer.record.call_args.args[1], batch)
                np.testing.assert_array_equal(
                    observer.record.call_args.args[1]["observation"],
                    np.full((2, 3), 0.5),
                )

    def test_real_legacy_metrics_unchanged_when_custom_collection_is_requested(self):
        from scale_rl.agents.sac.sac_update import get_critic_with_metrics

        critic, batch = make_critic()
        before = get_critic_with_metrics(None, None, critic, batch, False)
        capture_critic(critic, batch)
        after = get_critic_with_metrics(None, None, critic, batch, False)
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(any(k.startswith(PREFIX) for k in after))


class ReferenceProvenanceTest(unittest.TestCase):
    def test_scaling_definitions_are_inherited_in_campaign_and_current(self):
        repo = Path(__file__).resolve().parents[2]
        for revision in ("f597ee9", None):
            if revision:
                source = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "show",
                        revision + ":main/scale_rl/networks/metrics.py",
                    ],
                    text=True,
                )
            else:
                source = (repo / "main/scale_rl/networks/metrics.py").read_text()
            for name, expected in REFERENCES["scaling_formula_ast_sha256"].items():
                node = next(
                    n
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                # EchoCritic adds CPU device placement around the inherited
                # rank calculation; the mathematical body is unchanged.
                node.body = [
                    inner
                    for n in node.body
                    for inner in (n.body if isinstance(n, ast.With) else [n])
                ]
                self.assertEqual(
                    hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest(),
                    expected,
                )


class GradientArrayTest(unittest.TestCase):
    def test_per_layer_raw_update_probe_arrays_have_separate_names(self):
        from scale_rl.diagnostics.critic_metrics import available_gradient_arrays

        arrays = {
            "encoder": {"Dense_0": {"kernel": np.array([3.0, 4.0])}},
            "predictor": {"Dense_0": {"bias": np.array([12.0])}},
        }
        before = digest(arrays)
        for kind in (
            "raw_td_gradient_pre_adamw",
            "optimizer_update_array",
            "probe_gradient",
        ):
            result = available_gradient_arrays(arrays, kind=kind)
            self.assertEqual(
                result[f"{PREFIX}/available_arrays/{kind}/global/l2_analysis64"], 13
            )
            self.assertEqual(
                result[
                    f"{PREFIX}/available_arrays/{kind}/encoder/Dense_0/l2_analysis64"
                ],
                5,
            )
        self.assertEqual(before, digest(arrays))
        with self.assertRaises(ValueError):
            available_gradient_arrays(arrays, kind="gradient")

    def test_original_and_observable_block_gradients_match_without_updates(self):
        # A single synthetic scalar derivative; no optimizer step or fit.
        from scale_rl.networks.layers import ResidualBlock

        repo = Path(__file__).resolve().parents[2]
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "show",
                "f597ee9:main/scale_rl/networks/layers.py",
            ],
            text=True,
        )
        old = {}
        exec(compile(source, "<original-blocks>", "exec"), old)
        new_block, old_block = ResidualBlock(4, jnp.float32), old["ResidualBlock"](
            4, jnp.float32
        )
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 3
        params = old_block.init(jax.random.PRNGKey(1), x)["params"]
        old_grad = jax.grad(lambda p: jnp.sum(old_block.apply({"params": p}, x) ** 2))(
            params
        )
        new_grad = jax.grad(lambda p: jnp.sum(new_block.apply({"params": p}, x) ** 2))(
            params
        )
        self.assertEqual(digest(old_grad), digest(new_grad))


class ReferenceZeroPolicyTest(unittest.TestCase):
    def test_lop_zero_fallback_is_explicit_and_retained(self):
        namespace = {"torch": TORCH_FORMULA_BACKEND, "np": np}
        zeros = np.zeros(3, np.float32).view(Tensor)
        with np.errstate(divide="ignore", invalid="ignore"):
            for name, key in (
                ("compute_effective_rank", "entropy_effective_rank_lop"),
                ("compute_approximate_rank", "squared_energy99_count_lop"),
                ("compute_abs_approximate_rank", "singular_sum99_count_analysis64"),
            ):
                expected = source_function(name, namespace)(zeros)
                self.assertEqual(float(expected), 1)
                self.assertEqual(rank_metrics(np.zeros((3, 3)))[key], expected)
        self.assertEqual(rank_metrics(np.zeros((3, 3)))["stable_rank"], 0)
