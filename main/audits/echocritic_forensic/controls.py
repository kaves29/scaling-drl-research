"""Synthetic CPU controls against an isolated immutable-revision source copy.

No original parent, replay, parameter checkpoint or production probe is rerun.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


def tree_hash(tree):
    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest.update(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


class ScalarCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        bias = self.param("bias", nn.initializers.zeros, ())
        return jnp.broadcast_to(bias, (observations.shape[0], 1))


def scalar_controls(probe):
    network = ScalarCritic()
    observations = jnp.zeros((4, 1), jnp.float32)
    actions = jnp.zeros((4, 1), jnp.float32)
    params = {"bias": jnp.asarray(2.0, jnp.float32)}
    base = jnp.asarray([-2.0, -1.0, 1.0, 2.0], jnp.float32)
    indices = jnp.arange(4, dtype=jnp.int32).reshape(4, 1)
    tx_zero = optax.adamw(learning_rate=0.0, weight_decay=0.01)
    before = tree_hash(params)
    losses, final, offset = probe._fit(
        network, tx_zero, params, observations, actions, base, indices, 2
    )
    np.testing.assert_array_equal(losses, [4.0, 1.0, 1.0, 4.0])
    assert float(final) == 2.5 and float(offset) == 2.0 and tree_hash(params) == before
    baseline = float(jnp.var(base))
    assert baseline - float(final) == 0.0

    tx = probe.critic_optimizer(
        SimpleNamespace(critic_learning_rate=1e-4, critic_weight_decay=0.01)
    )
    one_base = jnp.ones(4, jnp.float32)
    one_idx = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
    one_losses, one_final, _ = probe._fit(
        network, tx, params, observations, actions, one_base, one_idx, 2
    )
    # Independent first-step AdamW equation: g=-2, bias-corrected moments,
    # eps outside sqrt, and decay on bias. Quantize only the resulting FP32 value.
    ideal_updated_bias = 2.0 - 1e-4 * (-2.0 / (2.0 + 1e-8) + 0.01 * 2.0)
    rounded_bias = np.float32(ideal_updated_bias)
    expected_final = np.float32(np.float32(rounded_bias - np.float32(3.0)) ** 2)
    np.testing.assert_array_equal(one_losses, [1.0])
    np.testing.assert_array_equal(one_final, expected_final)
    assert float(one_final) < 1.0

    # A changing-minibatch source fit versus a separately serialized carry.
    key = jax.random.PRNGKey(73)
    first = jax.random.randint(key, (1000, 4), 0, 4)
    second = jax.random.randint(jax.random.fold_in(key, 1), (1000, 4), 0, 4)
    idx = jnp.concatenate((first, second))
    source_losses, source_final, _ = probe._fit(
        network, tx, params, observations, actions, base, idx, 2
    )
    targets = params["bias"] + base

    @jax.jit
    def segment(p, state, batches):
        def step(carry, batch):
            p, state = carry

            def objective(candidate):
                return jnp.mean(
                    (
                        network.apply(
                            {"params": candidate}, observations[batch], actions[batch]
                        ).reshape(-1)
                        - targets[batch]
                    )
                    ** 2
                )

            loss, gradient = jax.value_and_grad(objective)(p)
            updates, state = tx.update(gradient, state, p)
            return (optax.apply_updates(p, updates), state), loss

        return jax.lax.scan(step, (p, state), batches)

    (middle, state), losses1 = segment(params, tx.init(params), first)
    restored = jax.tree_util.tree_map(
        jnp.asarray, pickle.loads(pickle.dumps(jax.device_get((middle, state))))
    )
    (end, _), losses2 = segment(*restored, second)
    pool_final = jnp.mean(
        (network.apply({"params": end}, observations, actions).reshape(-1) - targets)
        ** 2
    )
    np.testing.assert_array_equal(jnp.concatenate((losses1, losses2)), source_losses)
    np.testing.assert_array_equal(pool_final, source_final)
    (reset_end, _), _ = segment(middle, tx.init(middle), second)
    assert tree_hash(reset_end) != tree_hash(end), "reset negative control must differ"

    # Own-mean centering removes mean from initial residuals, not AdamW decay.
    translation = []
    for bias in (0.0, 2000.0):
        p = {"bias": jnp.asarray(bias, jnp.float32)}
        centred = jnp.asarray([-0.25, 0.25, -0.25, 0.25], jnp.float32)
        _, after, _ = probe._fit(
            network, tx, p, observations, actions, centred, one_idx, 2
        )
        translation.append(float(after))
    assert translation[1] > translation[0]

    cancellation = []
    for bias in (0.0, 2000.0, float(2**24)):
        p = {"bias": jnp.asarray(bias, jnp.float32)}
        b = jnp.asarray([-0.25, 0.25, -0.25, 0.25], jnp.float32)
        _, measured, _ = probe._fit(
            network, tx_zero, p, observations, actions, b, one_idx, 2
        )
        literal = np.asarray(np.float32(bias) + np.asarray(b), np.float32)
        cancellation.append(
            {
                "bias": bias,
                "source_zero_lr_pool_mse": float(measured),
                "base_variance": float(jnp.var(b)),
                "materialized_target_variance": float(np.var(literal)),
                "centred_real_arithmetic_mse": 0.0625,
            }
        )
    defaults = inspect.signature(optax.adamw).parameters
    return {
        "zero_lr": {
            "preupdate_minibatch_losses": np.asarray(losses).tolist(),
            "initial_pool_mse": 2.5,
            "final_pool_mse": float(final),
            "baseline": baseline,
            "score": 0.0,
            "params_unchanged": True,
        },
        "one_update": {
            "analytic_fp32_final_mse": float(expected_final),
            "source_final_mse": float(one_final),
            "preupdate_minibatch_loss": float(one_losses[0]),
        },
        "segmentation": {
            "steps": 2000,
            "losses_and_final_mse_bitwise_equal": True,
            "optimizer_reset_negative_control_differs": True,
        },
        "own_offset_weight_decay": {
            "zero_bias_final": translation[0],
            "large_bias_final": translation[1],
            "same_initial_mse": 0.0625,
        },
        "float32_translation": cancellation,
        "optimizer": {
            "name": "optax.adamw",
            "version": importlib.metadata.version("optax"),
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            **{k: defaults[k].default for k in ("b1", "b2", "eps", "eps_root")},
            "mask": None,
        },
    }


def analytic_params(net, seed, observations, actions, trained):
    params = jax.tree_util.tree_map(
        jnp.zeros_like,
        net.init(jax.random.PRNGKey(seed), observations=observations, actions=actions)[
            "params"
        ],
    )

    def scales(node):
        for key, value in node.items():
            if isinstance(value, dict):
                scales(value)
            elif key == "scale":
                node[key] = jnp.ones_like(value)

    scales(params)
    params["predictor"]["Dense_0"]["bias"] = jnp.asarray(
        [4.0 if trained else 1.0], jnp.float32
    )
    if trained:
        kernel = params["encoder"]["Dense_0"]["kernel"]
        params["encoder"]["Dense_0"]["kernel"] = kernel.at[0].set(
            jnp.asarray([1.0, -1.0, 1.0, -1.0])
        )
        params["predictor"]["Dense_0"]["kernel"] = jnp.asarray(
            [[1.0], [0.0], [0.0], [0.0]], jnp.float32
        )
    return params


def route_control(source_root, output, seed, probe):
    from experiments.exp1 import compose_config
    from experiments.exp12.run_probes import RunProbes
    from scale_rl.agents import create_agent
    from gymnasium.spaces import Box

    cfg = compose_config(
        str(source_root / "configs"),
        "base_exp12",
        [
            f"seed={seed}",
            "run_role=dev",
            "env_name=hopper-hop",
            "env=dmc_medium",
            "actor_num_blocks=1",
            "actor_hidden_dim=4",
            "critic_num_blocks=1",
            "critic_hidden_dim=4",
            "env.num_env_steps=80",
            "buffer.min_length=2",
            "buffer.max_length=8",
            "buffer.sample_batch_size=4",
            "probe.rounds=2",
            "probe.checks=2",
            "probe.steps=2000",
            "probe.pool_size=8",
            "probe.batch_size=4",
            "probe.eval_chunk=4",
            "trigger.resamples=32",
            "num_eval_episodes=1",
        ],
    )
    obs_space = Box(-np.inf, np.inf, (3,), dtype=np.float32)
    act_space = Box(-1.0, 1.0, (1,), dtype=np.float32)
    agent = create_agent(
        observation_space=obs_space, action_space=act_space, cfg=cfg.agent
    )
    core = agent.agent
    agent.obs_rms.mean = np.zeros(3, np.float32)
    agent.obs_rms.var = np.full(3, 4.0, np.float32)
    agent.obs_rms.count = 100
    raw_obs = np.asarray(
        [[-2.0, i, -i] if i % 2 == 0 else [2.0, i, -i] for i in range(8)], np.float32
    )
    raw_act = np.linspace(-1, 1, 8, dtype=np.float32).reshape(8, 1)
    buffer = SimpleNamespace(_observations=raw_obs, _actions=raw_act, _num_in_buffer=8)
    net = core.critic.network_def
    normalized = jnp.asarray(agent._normalize(raw_obs))
    fresh = analytic_params(net, seed, normalized, jnp.asarray(raw_act), False)
    trained = analytic_params(net, seed, normalized, jnp.asarray(raw_act), True)
    initial_prediction = np.asarray(
        net.apply({"params": trained}, normalized, raw_act)
    ).reshape(-1)
    s = np.asarray(normalized[:, 0], np.float64)
    expected = 4.0 + s / np.sqrt(s * s + 1e-6)
    np.testing.assert_allclose(
        initial_prediction, expected, rtol=0, atol=8 * np.finfo(np.float32).eps * 5
    )
    assert initial_prediction.min() > 2.9
    np.testing.assert_array_equal(
        net.apply({"params": fresh}, normalized, raw_act), np.ones((8, 1), np.float32)
    )
    core._critic = core._critic.replace(
        params=fresh, opt_state=core._critic.tx.init(fresh)
    )
    trainer = SimpleNamespace(
        cfg=cfg,
        agent=agent,
        _sac_agent=core,
        buffer=buffer,
        extra_state={},
        interaction_step=2,
    )
    numpy_before, python_before = pickle.dumps(np.random.get_state()), pickle.dumps(
        random.getstate()
    )
    buffer_before, normalization_before = tree_hash((raw_obs, raw_act)), tree_hash(
        (agent.obs_rms.mean, agent.obs_rms.var, agent.obs_rms.count)
    )
    calls = []
    original_fit = probe._fit

    def observed(*args, **kwargs):
        calls.append(
            {
                "params": tree_hash(args[2]),
                "obs": np.asarray(args[3]).copy(),
                "act": np.asarray(args[4]).copy(),
                "base": np.asarray(args[5]).copy(),
                "indices": np.asarray(args[6]).copy(),
            }
        )
        return original_fit(*args, **kwargs)

    run_dir = output / f"seed_{seed}"
    run_dir.mkdir()
    with mock.patch.object(probe, "_fit", observed):
        rp = RunProbes(trainer, str(run_dir))
        rp.capture_fresh(trainer)
        fresh_before = file_hashes(rp.fresh_path)
        fresh_hash = tree_hash(rp.fresh)
        # Deliberately altered "trained" parameters, no parent training performed.
        state = core._critic.tx.init(trained)
        adam = state[0]._replace(
            count=jnp.asarray(17, jnp.int32),
            mu=jax.tree_util.tree_map(lambda x: jnp.full_like(x, 0.125), state[0].mu),
            nu=jax.tree_util.tree_map(lambda x: jnp.full_like(x, 0.25), state[0].nu),
        )
        core._critic = core._critic.replace(
            params=trained, opt_state=(adam, *state[1:])
        )
        checkpoint = run_dir / "agent_checkpoint"
        checkpoint.mkdir()
        agent.save_checkpoint(str(checkpoint))
        restored = create_agent(
            observation_space=obs_space, action_space=act_space, cfg=cfg.agent
        )
        restored.load_checkpoint(str(checkpoint))
        assert tree_hash(restored.agent.critic.params) == tree_hash(trained)
        assert tree_hash(restored.agent.critic.opt_state) == tree_hash(
            core.critic.opt_state
        )
        assert (
            tree_hash(
                (restored.obs_rms.mean, restored.obs_rms.var, restored.obs_rms.count)
            )
            == normalization_before
        )
        trainer.agent, trainer._sac_agent = restored, restored.agent
        rp = RunProbes(trainer, str(run_dir))
        assert tree_hash(rp.fresh) == fresh_hash
        rp.capture_fresh(trainer)
        assert file_hashes(rp.fresh_path) == fresh_before
        optimizer_before = tree_hash(trainer._sac_agent.critic.opt_state)
        trainer.interaction_step = 20
        rp.maybe_check(trainer)
        assert tree_hash(trainer._sac_agent.critic.params) == tree_hash(trained)
        assert tree_hash(trainer._sac_agent.critic.opt_state) == optimizer_before
        assert (
            tree_hash(rp.fresh) == fresh_hash
            and file_hashes(rp.fresh_path) == fresh_before
        )
        rp.write_csv()
    assert len(calls) == 6
    check_calls = calls[2:]
    for round_index in range(2):
        current, reference = check_calls[2 * round_index : 2 * round_index + 2]
        assert (
            current["params"] == tree_hash(trained)
            and reference["params"] == fresh_hash
        )
        for key in ("obs", "act", "base", "indices"):
            np.testing.assert_array_equal(current[key], reference[key])
        pool_idx = np.random.default_rng([seed, 0x50524F42, 1, round_index]).integers(
            0, 8, size=8
        )
        np.testing.assert_array_equal(
            current["obs"], np.asarray(agent._normalize(raw_obs[pool_idx]), np.float32)
        )
        np.testing.assert_array_equal(current["act"], raw_act[pool_idx])
        _, idx_key = jax.random.split(probe.probe_key(seed, 1, round_index))
        np.testing.assert_array_equal(
            current["indices"][:1000], jax.random.randint(idx_key, (1000, 4), 0, 8)
        )
        np.testing.assert_array_equal(
            current["indices"][1000:],
            jax.random.randint(jax.random.fold_in(idx_key, 1), (1000, 4), 0, 8),
        )
    critics = {"current": (net, trained), "fresh": (net, fresh)}
    repeat1 = probe.run_probe(restored, buffer, net, critics, rp.tx, seed, 1, rp.cfg)
    repeat2 = probe.run_probe(restored, buffer, net, critics, rp.tx, seed, 1, rp.cfg)
    assert tree_hash(repeat1) == tree_hash(repeat2)
    changed = probe.run_probe(restored, buffer, net, critics, rp.tx, seed, 2, rp.cfg)
    assert tree_hash(repeat1) != tree_hash(changed)
    assert tree_hash((raw_obs, raw_act)) == buffer_before
    assert (
        tree_hash((restored.obs_rms.mean, restored.obs_rms.var, restored.obs_rms.count))
        == normalization_before
    )
    assert (
        pickle.dumps(np.random.get_state()) == numpy_before
        and pickle.dumps(random.getstate()) == python_before
    )
    return {
        "seed": seed,
        "checkpoint_params_optimizer_normalizer_roundtrip": True,
        "trained_prediction": initial_prediction.tolist(),
        "fresh_prediction": 1.0,
        "fresh_hash_unchanged": True,
        "training_params_optimizer_unchanged_by_probe": True,
        "same_inputs_base_and_indices_within_pair": True,
        "normalization_once_matches_expected": True,
        "first_and_followup_1000_indices_match": True,
        "fixed_task_repeat_bitwise": True,
        "changed_check_changes_result": True,
        "global_rng_and_replay_unchanged": True,
        "fresh_scores_check1": np.asarray(repeat1["fresh"]["score"]).tolist(),
        "fresh_scores_check2": np.asarray(changed["fresh"]["score"]).tolist(),
    }


def dormancy_control():
    from scale_rl.agents.sac.sac_network import SACCritic
    from scale_rl.agents.sac.sac_update import get_critic_with_metrics
    from scale_rl.networks.trainer import Trainer
    from scale_rl.networks.metrics import flatten_dict

    net = SACCritic("residual", 1, 4, jnp.float32)
    obs, act = jnp.ones((8, 3)), jnp.ones((8, 1))
    params = analytic_params(net, 11, obs, act, False)
    params["encoder"]["Dense_0"]["bias"] = jnp.asarray([1.0, -1.0, 1.0, -1.0])
    params["encoder"]["ResidualBlock_0"]["Dense_0"]["bias"] = -jnp.ones(16)
    params["encoder"]["ResidualBlock_0"]["Dense_1"]["bias"] = jnp.ones(4)
    state = Trainer(network_def=net, params=params, tx=optax.adamw(1e-4))
    info = get_critic_with_metrics(
        jax.random.PRNGKey(0), None, state, {"observation": obs, "action": act}, False
    )
    _, intermediate = net.apply(
        {"params": params}, obs, act, capture_intermediates=True
    )
    flat = flatten_dict(intermediate)
    key = next(k for k in flat if "ResidualBlock_0_Dense_0___call__" in k)
    pre = np.asarray(flat[key])
    assert np.all(pre == -1.0) and np.all(np.maximum(pre, 0.0) == 0.0)
    assert float(info["train/critic_DR0.1"]) == 0.0
    assert float(info["train/critic_DR0.2"]) == 0.0
    return {
        "historical_dense_preactivation_dormancy_percent": 0.0,
        "post_relu_expansion_inactive_percent": 100.0,
        "interpretation": "Different measured populations; low historical DR does not establish post-ReLU activity",
        "actual_campaign_post_relu_activity": "unavailable without original activations/parameters",
    }


def warmup_flow_control(probe):
    """Four synthetic transitions through original trainer; zero SAC updates."""
    from gymnasium.spaces import Box
    from experiments.exp12.trainer import Exp12Trainer
    from scale_rl.agents.wrappers.normalization import ObservationNormalizer
    from scale_rl.buffers.numpy_buffer import NpyUniformBuffer

    space = Box(-100.0, 100.0, (1,), dtype=np.float32)

    class UnderlyingAgent:
        _observation_space = space

        def sample_actions(self, **kwargs):
            return np.zeros((1, 1), np.float32)

        def update_many(self, *args, **kwargs):
            raise AssertionError("SAC updates are forbidden in this fixture")

    class Environment:
        action_space = Box(-1.0, 1.0, (1, 1), dtype=np.float32)
        count = 0

        def step(self, actions):
            self.count += 1
            return (
                np.asarray([[self.count]], np.float32),
                np.zeros(1),
                np.zeros(1, bool),
                np.zeros(1, bool),
                {},
            )

    trainer = Exp12Trainer.__new__(Exp12Trainer)
    trainer.cfg = SimpleNamespace(
        num_train_envs=1,
        actor_grad_cosine_every=1,
        updates_per_interaction_step=1,
        logging_per_interaction_step=100,
        evaluation_per_interaction_step=100,
    )
    trainer.agent = ObservationNormalizer(UnderlyingAgent())
    trainer.buffer = NpyUniformBuffer(space, space, 1, 0.99, 8, 5, 1, 2)
    trainer.buffer.reset()
    trainer.train_env = Environment()
    trainer.interaction_step = trainer.update_step = trainer.update_counter = 0
    trainer.observations = np.zeros((1, 1), np.float32)
    trainer.timestep = {"next_observation": trainer.observations.copy()}
    trainer.train(4)
    np.testing.assert_array_equal(trainer.buffer._observations[:4, 0], np.arange(4))
    np.testing.assert_array_equal(
        trainer.buffer._next_observations[:4, 0], np.arange(1, 5)
    )
    assert trainer.update_step == 0 and trainer.buffer._num_in_buffer == 4
    normalization = tree_hash(
        (
            trainer.agent.obs_rms.mean,
            trainer.agent.obs_rms.var,
            trainer.agent.obs_rms.count,
        )
    )
    rng = np.random.default_rng(19)
    expected_idx = np.random.default_rng(19).integers(0, 4, size=8)
    obs, act = probe.sample_pool(trainer.agent, trainer.buffer, rng, 8)
    np.testing.assert_array_equal(
        obs,
        np.asarray(
            trainer.agent._normalize(trainer.buffer._observations[expected_idx]),
            np.float32,
        ),
    )
    assert normalization == tree_hash(
        (
            trainer.agent.obs_rms.mean,
            trainer.agent.obs_rms.var,
            trainer.agent.obs_rms.count,
        )
    )
    return {
        "transitions": 4,
        "sac_updates": 0,
        "raw_observations_preserved_in_replay": True,
        "pool_normalization_exactly_once": True,
        "pool_sampling_does_not_update_statistics": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--flow-only", action="store_true")
    args = parser.parse_args()
    for protected in (
        args.source_root.resolve().parent,
        Path(__file__).resolve().parents[3],
    ):
        if args.out.resolve() == protected or protected in args.out.resolve().parents:
            raise ValueError(
                "control outputs must be outside source snapshots and repository"
            )
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_root))
    from experiments.exp12 import probe
    from experiments.exp12.precision import set_matmul_precision, runtime_info

    set_matmul_precision()
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU controls only")
    if args.flow_only:
        result = warmup_flow_control(probe)
        (args.out / "warmup_flow.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    results = {
        "evidence_type": "synthetic CPU controls, not original campaign data",
        "revision": args.revision,
        "runtime": runtime_info(),
        "warmup_flow": warmup_flow_control(probe),
        "scalar": scalar_controls(probe),
        "routes": [
            route_control(args.source_root, args.out, seed, probe) for seed in (1, 3, 8)
        ],
        "dormancy": dormancy_control(),
    }
    (args.out / "controls.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False)
    )
    print(
        json.dumps(
            {
                "revision": args.revision,
                "routes_passed": len(results["routes"]),
                "scalar_controls": "passed",
                "dormancy": results["dormancy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
