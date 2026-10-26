"""CPU-only alternative-probe tests; tiny networks, never RL training."""

from copy import deepcopy
from unittest.mock import patch

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diagnostics import sequential_probe as s


class Tiny(nn.Module):
    width: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        return nn.Dense(1)(nn.tanh(nn.Dense(self.width)(x)))


@pytest.fixture
def package(tmp_path):
    net = Tiny()
    obs = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0], [2.0, -1.0]], np.float32)
    act = np.array([[0.0], [1.0], [2.0], [-1.0]], np.float32)
    fresh = net.init(
        jax.random.PRNGKey(4),
        observations=jnp.asarray(obs[:1]),
        actions=jnp.asarray(act[:1]),
    )["params"]
    bank = tmp_path / "bank"
    ref = tmp_path / "ref"
    rms = {
        "mean": np.array([0.1, 0.2]),
        "var": np.array([1.0, 2.0]),
        "count": 4,
        "epsilon": 1e-8,
    }
    s.create_bank(
        bank,
        observations=obs,
        actions=act,
        num_filled=4,
        warmup=4,
        rms=rms,
        environment="synthetic",
        seed=3,
        teacher=Tiny(),
        teacher_architecture="fixed-tiny2",
        pool_size=4,
        source="synthetic-fixture",
    )
    s.create_reference(ref, bank, net, fresh, architecture="tiny2", sac_updates=0)
    return bank, ref, net, fresh, obs, act, rms


def run(p, out, **kw):
    bank, ref, net, fresh, *_ = p
    return s.evaluate(
        enabled=True,
        bank=bank,
        reference=ref,
        network=net,
        current=fresh,
        fresh=fresh,
        checkpoint=0,
        settings=s.Settings(batch_size=2, chunk=2),
        output=out,
        **kw,
    )


def test_disabled_reads_nothing():
    with patch.object(s, "verify", side_effect=AssertionError):
        assert s.evaluate() is None


def test_bank_determinism_diversity_and_no_mutation(package, tmp_path):
    bank, ref, net, fresh, obs, act, rms = package
    before = deepcopy((obs, act, rms, np.random.get_state()))
    other = tmp_path / "other"
    s.create_bank(
        other,
        observations=obs,
        actions=act,
        num_filled=4,
        warmup=4,
        rms=rms,
        environment="synthetic",
        seed=3,
        teacher=Tiny(),
        teacher_architecture="fixed-tiny2",
        pool_size=4,
        source="synthetic-fixture",
    )
    with np.load(bank / "bank.npz") as a, np.load(other / "bank.npz") as b:
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])
        assert a["base"].shape == (8, 4) and a["base"].dtype == np.float32
        assert s.verify(bank)["distinct_base_arrays"] == 8
    np.testing.assert_array_equal(obs, before[0])
    np.testing.assert_array_equal(act, before[1])
    np.testing.assert_array_equal(rms["mean"], before[2]["mean"])
    np.testing.assert_array_equal(np.random.get_state()[1], before[3][1])
    # Different learner size uses exactly the same teacher bank and inputs.
    larger = Tiny(3)
    params = larger.init(
        jax.random.PRNGKey(8),
        observations=jnp.asarray(obs[:1]),
        actions=jnp.asarray(act[:1]),
    )["params"]
    s.create_reference(
        tmp_path / "large", bank, larger, params, architecture="tiny3", sac_updates=0
    )
    assert s.verify(tmp_path / "large")["bank_hash"] == s.verify(ref)["bank_hash"]


def test_all_eight_identities_counts_resume_and_artifacts(package, tmp_path):
    out = tmp_path / "evaluation"
    before = s.serialization.to_bytes(package[3])
    rng = deepcopy(np.random.get_state())
    results = run(package, out)
    assert [r["K"] for r in results] == list(range(1, 9))
    for k in range(1, 9):
        phase = out / f"phase_{k:02d}"
        receipt = s.verify(phase)
        assert receipt["optimizer_initializations"] == {"A": 1, "B": 1, "C": 1}
        assert receipt["updates_per_arm"] == 2000
        with np.load(phase / "A.npz") as a, np.load(phase / "B.npz") as b, np.load(
            phase / "C.npz"
        ) as c:
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
            for other in (b, c):
                np.testing.assert_array_equal(a["targets"], other["targets"])
                np.testing.assert_array_equal(a["indices"], other["indices"])
            assert a["losses"].shape == (2000,) and a["pool_losses"].shape == (5,)
            assert a["pool_losses"][-1] == pytest.approx(
                float(a["final_mse"]), abs=1e-7
            )
            if k == 1:
                for name in b.files:
                    np.testing.assert_array_equal(b[name], c[name])
    assert s.serialization.to_bytes(package[3]) == before
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1])
    with patch.object(s, "_fit_literal", side_effect=AssertionError("must not refit")):
        assert run(package, out) == results


def test_partial_failure_preserves_completed_phase(package, tmp_path):
    out = tmp_path / "eval"
    original = s._fit_literal
    calls = []

    def fail(*args):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("injected")
        return original(*args)

    with patch.object(s, "_fit_literal", side_effect=fail):
        with pytest.raises(RuntimeError, match="injected"):
            run(package, out)
    first = (out / "phase_01/receipt.json").read_bytes()
    assert not (out / "phase_02").exists() and list(out.glob("failure_02*"))
    result = run(package, out)
    assert len(result) == 8 and (out / "phase_01/receipt.json").read_bytes() == first
    (out / "phase_03/A.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        run(package, out)


def test_parameter_carry_and_optimizer_reset(package, tmp_path):
    original = s._fit_literal
    starts = []
    ends = []

    def watch(network, tx, params, *args):
        starts.append(s.serialization.to_bytes(params))
        result = original(network, tx, params, *args)
        ends.append(s.serialization.to_bytes(result[0]))
        return result

    with patch.object(s, "_fit_literal", side_effect=watch):
        run(package, tmp_path / "out")
    fresh = s.serialization.to_bytes(package[3])
    for k in range(8):
        assert starts[k * 3 + 2] == fresh
        if k:
            assert starts[k * 3] == ends[(k - 1) * 3]
            assert starts[k * 3 + 1] == ends[(k - 1) * 3 + 1]


def test_hand_computed_signs():
    v = s.scores(0.5, {"A": 0.4, "B": 0.2, "C": 0.1})
    assert v["C_minus_A"] == pytest.approx(0.3)
    assert v["B_minus_A"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "steps,batch,chunk", [(1000, 2, 2), (2000, 0, 2), (2000, 2, 3)]
)
def test_configuration_rejection(steps, batch, chunk):
    with pytest.raises(ValueError):
        s.Settings(steps=steps, batch_size=batch, chunk=chunk).validate(4)


def test_reference_rejects_trained_initialization(package, tmp_path):
    bank, ref, net, fresh, *_ = package
    with pytest.raises(ValueError):
        s.create_reference(
            tmp_path / "bad", bank, net, fresh, architecture="tiny", sac_updates=1
        )
    with pytest.raises(FileExistsError):
        s.create_reference(ref, bank, net, fresh, architecture="tiny", sac_updates=0)


def test_cross_checkpoint_literal_target_reuse_and_distinct_current(package, tmp_path):
    bank, ref, net, fresh, *_ = package
    before = (ref / "targets.npz").read_bytes()
    current = jax.tree_util.tree_map(lambda p: p + jnp.float32(0.03), fresh)
    saved = s.serialization.to_bytes(current)
    s.evaluate(
        enabled=True,
        bank=bank,
        reference=ref,
        network=net,
        current=current,
        fresh=fresh,
        checkpoint=50000,
        settings=s.Settings(batch_size=2, chunk=2),
        output=tmp_path / "late",
    )
    run(package, tmp_path / "initial")
    assert (ref / "targets.npz").read_bytes() == before
    assert s.serialization.to_bytes(current) == saved
    for k in range(1, 9):
        with np.load(tmp_path / f"late/phase_{k:02d}/A.npz") as a, np.load(
            tmp_path / f"initial/phase_{k:02d}/C.npz"
        ) as c:
            np.testing.assert_array_equal(a["targets"], c["targets"])
            np.testing.assert_array_equal(a["indices"], c["indices"])


def test_source_b_and_literal_variance_are_separate(package):
    bank, ref, *_ = package
    with np.load(bank / "bank.npz") as a, np.load(ref / "targets.npz") as b:
        np.testing.assert_array_equal(
            b["variances"], np.asarray([jnp.var(jnp.asarray(t)) for t in a["base"]])
        )
        assert "literal_variances" in b


@pytest.mark.parametrize("what", ["bank", "fresh", "settings", "checkpoint"])
def test_resume_contract_rejection(package, tmp_path, what):
    out = tmp_path / "out"
    run(package, out)
    bank, ref, net, fresh, *_ = package
    options = dict(
        enabled=True,
        bank=bank,
        reference=ref,
        network=net,
        current=fresh,
        fresh=fresh,
        checkpoint=0,
        settings=s.Settings(batch_size=2, chunk=2),
        output=out,
    )
    if what == "bank":
        (bank / "bank.npz").write_bytes(b"changed")
    elif what == "fresh":
        options["fresh"] = jax.tree_util.tree_map(lambda p: p + 1, fresh)
    elif what == "settings":
        options["settings"] = s.Settings(batch_size=2, chunk=2, learning_rate=0.002)
    else:
        options["checkpoint"] = 1
    with pytest.raises(ValueError):
        s.evaluate(**options)


def test_actual_optimizer_resets_with_direct_jax_reference(package):
    from types import SimpleNamespace
    from experiments.exp12.probe import critic_optimizer
    import optax

    bank, ref, net, fresh, *_ = package
    tx = critic_optimizer(
        SimpleNamespace(critic_learning_rate=1e-4, critic_weight_decay=0.01)
    )
    with np.load(bank / "bank.npz") as a, np.load(ref / "targets.npz") as b:
        obs, act, target = map(
            jnp.asarray, [a["observations"], a["actions"], b["targets"][0]]
        )
    idx = jnp.zeros((2000, 2), jnp.int32)
    fitted, _, final, _, _ = s._fit_literal(net, tx, fresh, obs, act, target, idx, 2)

    def loss(p, i):
        return jnp.mean(
            (net.apply({"params": p}, obs[i], act[i]).reshape(-1) - target[i]) ** 2
        )

    def step(carry, i):
        p, state = carry
        grads = jax.grad(loss)(p, i)
        updates, state = tx.update(grads, state, p)
        return (optax.apply_updates(p, updates), state), None

    (expected, _), _ = jax.lax.scan(step, (fresh, tx.init(fresh)), idx)
    for a, b in zip(
        jax.tree_util.tree_leaves(fitted), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_array_equal(a, b)
    assert np.isfinite(final)


def test_disabled_hook_and_enabled_instance_hook_restore():
    from types import SimpleNamespace
    from diagnostics.sequential_hooks import Hooks

    class Trainer:
        def _evaluate(self, x):
            return x + 1

    class Probes:
        fresh = "saved-tree"

        def capture_fresh(self, t):
            return "original"

    t, p = Trainer(), Probes()
    before = (dict(t.__dict__), dict(p.__dict__))
    with Hooks(t, p, SimpleNamespace(enabled=False)):
        assert t._evaluate(2) == 3 and p.capture_fresh(t) == "original"
    assert (t.__dict__, p.__dict__) == before
    calls = []
    o = SimpleNamespace(
        enabled=True,
        fresh=None,
        initialize=lambda *a: calls.append("init"),
        observe=lambda *a: calls.append("evaluate"),
    )
    with Hooks(t, p, o):
        assert p.capture_fresh(t) == "original"
        assert t._evaluate(2) == 3
    assert calls == ["init", "evaluate"]
    assert (t.__dict__, p.__dict__) == before


def test_original_probe_outputs_unchanged_when_disabled(package):
    from types import SimpleNamespace
    from diagnostics.sequential_hooks import Hooks
    from experiments.exp12.probe import _fit, critic_optimizer

    bank, ref, net, fresh, *_ = package
    with np.load(bank / "bank.npz") as a:
        obs, act, base = map(
            jnp.asarray, [a["observations"], a["actions"], a["base"][0]]
        )
    tx = critic_optimizer(
        SimpleNamespace(critic_learning_rate=1e-4, critic_weight_decay=0.01)
    )
    indices = jnp.zeros((2000, 2), jnp.int32)
    before = _fit(net, tx, fresh, obs, act, base, indices, 2)
    # Objects deliberately unusable: disabled hooks must not inspect them.
    with Hooks(None, None, SimpleNamespace(enabled=False)):
        assert s.evaluate(enabled=False) is None
        after = _fit(net, tx, fresh, obs, act, base, indices, 2)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_seed_independent_bank_and_float_hashes(package, tmp_path):
    bank, ref, net, fresh, obs, act, rms = package
    other = tmp_path / "newseed"
    s.create_bank(
        other,
        observations=obs,
        actions=act,
        num_filled=4,
        warmup=4,
        rms=rms,
        environment="synthetic",
        seed=4,
        teacher=Tiny(),
        teacher_architecture="fixed-tiny2",
        pool_size=4,
        source="synthetic-fixture",
    )
    assert (
        s.verify(bank)["array_hashes"]["base"]
        != s.verify(other)["array_hashes"]["base"]
    )
    assert s.array_hash(np.zeros(2, np.float32)) != s.array_hash(
        np.zeros(2, np.float64)
    )


def test_observer_initial_capture_and_restore(package, tmp_path):
    from types import SimpleNamespace as N
    from diagnostics.sequential_hooks import Observer

    bank, ref, net, fresh, *_ = package
    cfg = N(
        seed=3,
        env_name="synthetic",
        agent=N(
            critic_use_cdq=False,
            critic_sparsity=0,
            critic_learning_rate=1e-4,
            critic_weight_decay=0.01,
        ),
        probe=N(steps=2000, batch_size=2, eval_chunk=2, pool_size=4, target_scale=1e5),
    )
    t = N(
        update_step=0,
        interaction_step=4,
        cfg=cfg,
        _sac_agent=N(critic=N(network_def=net, params=fresh)),
    )
    options = dict(
        enabled=True,
        bank=bank,
        reference=ref,
        output=tmp_path / "observe",
        teacher=Tiny(),
        teacher_architecture="fixed-tiny2",
        architecture="tiny2",
        environment="synthetic",
        seed=3,
        source="synthetic-fixture",
    )
    o = Observer(**options)
    assert len(o.initialize(t, fresh)) == 8
    resumed = Observer(**options)
    t.update_step = 10
    resumed.restore(t, fresh)
    with patch.object(
        s, "_fit_literal", side_effect=AssertionError("must load committed phases")
    ):
        assert len(resumed.observe(t)) == 8
    with pytest.raises(ValueError, match="before SAC"):
        Observer(**options).initialize(t, fresh)


def test_fail_closed_bank_inputs(package, tmp_path):
    _, _, _, _, obs, act, rms = package
    with pytest.raises(ValueError, match="warmup"):
        s.create_bank(
            tmp_path / "too_early",
            observations=obs,
            actions=act,
            num_filled=2,
            warmup=4,
            rms=rms,
            environment="synthetic",
            seed=3,
            teacher=Tiny(),
            teacher_architecture="fixed",
            pool_size=4,
            source="fixture",
        )
    bad = obs.copy()
    bad[:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s.create_bank(
            tmp_path / "bad",
            observations=bad,
            actions=act,
            num_filled=4,
            warmup=4,
            rms=rms,
            environment="synthetic",
            seed=3,
            teacher=Tiny(),
            teacher_architecture="fixed",
            pool_size=4,
            source="fixture",
        )


def test_saved_teacher_reconstructs_base_and_uses_declared_chunk(package):
    bank, _, net, _, *_ = package
    from experiments.exp12.probe import _base_targets

    with np.load(bank / "bank.npz") as a:
        obs, act, keys, base = a["observations"], a["actions"], a["keys"], a["base"]
    expected_trees = [
        net.init(
            jnp.asarray(key),
            observations=jnp.asarray(obs[:1]),
            actions=jnp.asarray(act[:1]),
        )["params"]
        for key in keys
    ]
    trees = s.serialization.from_bytes(
        expected_trees, (bank / "teachers.msgpack").read_bytes()
    )
    chunk = s.verify(bank)["teacher_eval_chunk"]
    for key, params, target in zip(keys, trees, base):
        reconstructed, parameters = s._teacher_values(
            net, jnp.asarray(key), jnp.asarray(obs), jnp.asarray(act), chunk, 1e5
        )
        np.testing.assert_array_equal(target, reconstructed)
        for saved, rebuilt in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(parameters)
        ):
            np.testing.assert_array_equal(saved, rebuilt)
        np.testing.assert_array_equal(
            target,
            np.asarray(
                _base_targets(
                    net,
                    jnp.asarray(key),
                    jnp.asarray(obs),
                    jnp.asarray(act),
                    chunk,
                    1e5,
                )
            ),
        )


@pytest.mark.parametrize("width", [4, 8, 12])
def test_real_repository_tiny_sac_critic_end_to_end(package, tmp_path, width):
    from scale_rl.agents.sac.sac_network import SACCritic

    bank, _, _, _, obs, act, *_ = package
    net = SACCritic(
        block_type="residual", num_blocks=1, hidden_dim=width, dtype=jnp.float32
    )
    fresh = net.init(
        jax.random.PRNGKey(71),
        observations=jnp.asarray(obs[:1]),
        actions=jnp.asarray(act[:1]),
    )["params"]
    ref = tmp_path / f"sac_ref_{width}"
    s.create_reference(
        ref, bank, net, fresh, architecture=f"tiny-SAC-W{width}", sac_updates=0
    )
    out = tmp_path / f"sac_out_{width}"
    values = s.evaluate(
        enabled=True,
        bank=bank,
        reference=ref,
        output=out,
        network=net,
        current=fresh,
        fresh=fresh,
        checkpoint=0,
        settings=s.Settings(batch_size=2, chunk=2),
    )
    assert len(values) == 8 and all(v["B_minus_A"] == 0 for v in values)
    assert values[0]["C_minus_A"] == 0
    assert s.load_json(out / "trajectory.json")["seed"] == 3


def test_milestones_match_prefix_fits(package):
    from types import SimpleNamespace
    from experiments.exp12.probe import critic_optimizer, _fit

    bank, ref, net, fresh, *_ = package
    with np.load(bank / "bank.npz") as a, np.load(ref / "targets.npz") as b:
        obs, act, target = map(
            jnp.asarray, [a["observations"], a["actions"], b["targets"][0]]
        )
    tx = critic_optimizer(
        SimpleNamespace(critic_learning_rate=1e-4, critic_weight_decay=0.01)
    )
    indices = jnp.zeros((2000, 2), jnp.int32)
    _, _, _, trace, _ = s._fit_literal(net, tx, fresh, obs, act, target, indices, 2)
    for slot, updates in enumerate([100, 500, 1000, 2000], 1):
        # Source primitive with explicit zero offset keeps this literal target.
        _, error, _ = _fit(
            net,
            tx,
            fresh,
            obs,
            act,
            target,
            indices[:updates],
            2,
            offset=jnp.float32(0),
        )
        actual = jnp.mean((trace[slot] - target) ** 2)
        np.testing.assert_array_equal(actual, error)


def test_resume_hash_is_independent_of_dictionary_insertion_order(package, tmp_path):
    bank, ref, net, fresh, *_ = package

    def reordered(tree):
        if isinstance(tree, dict):
            return {k: reordered(tree[k]) for k in reversed(list(tree))}
        return tree

    other = reordered(fresh)
    assert s.tree_hash(fresh) == s.tree_hash(other)
    out = tmp_path / "out"
    expected = run(package, out)
    with patch.object(s, "_fit_literal", side_effect=AssertionError("must not refit")):
        assert (
            s.evaluate(
                enabled=True,
                bank=bank,
                reference=ref,
                network=net,
                current=other,
                fresh=other,
                checkpoint=0,
                settings=s.Settings(batch_size=2, chunk=2),
                output=out,
            )
            == expected
        )


@pytest.mark.parametrize("field", ["seed", "environment", "twin", "sparse"])
def test_observer_restore_fails_before_model_work_on_wrong_config(
    package, tmp_path, field
):
    from types import SimpleNamespace as N
    from diagnostics.sequential_hooks import Observer

    bank, ref, net, fresh, *_ = package
    cfg = N(
        seed=3, env_name="synthetic", agent=N(critic_use_cdq=False, critic_sparsity=0)
    )
    if field == "seed":
        cfg.seed = 4
    elif field == "environment":
        cfg.env_name = "different"
    elif field == "twin":
        cfg.agent.critic_use_cdq = True
    else:
        cfg.agent.critic_sparsity = 0.5
    observer = Observer(
        enabled=True,
        bank=bank,
        reference=ref,
        output=tmp_path / "out",
        teacher=Tiny(),
        teacher_architecture="fixed-tiny2",
        architecture="tiny2",
        environment="synthetic",
        seed=3,
        source="synthetic-fixture",
    )
    trainer = N(cfg=cfg)
    with pytest.raises(ValueError, match="resume"):
        observer.restore(trainer, fresh)
