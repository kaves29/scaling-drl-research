"""Execute original pure routing/arithmetic with controlled stand-ins, never fits."""

import ast
from copy import deepcopy
import hashlib
import subprocess
from types import SimpleNamespace

import numpy as np

from audits.echocritic_forensic.evidence import CAMPAIGN


def original_source(path):
    return subprocess.check_output(["git", "show", f"{CAMPAIGN}:{path}"], text=True)


def function(source, name, namespace):
    """Extract one pinned Git-source function; imports/decorators are not executed."""
    nodes = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(nodes) != 1:
        raise ValueError("function name must identify exactly one source node")
    node = deepcopy(nodes[0])
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(module, "<pinned original source>", "exec"), scope)
    return scope[name]


def structural_checks(source):
    """Check actual original control-flow statements, not a reimplemented fitter."""
    nodes = {
        n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
    }
    bodies = {name: ast.unparse(node) for name, node in nodes.items()}
    expected = {
        "own_offset": (
            "_fit",
            ["offset = jnp.mean(predict(params))", "targets = offset + base_targets"],
        ),
        "fresh_optimizer_state_continuous_scan": (
            "_fit",
            [
                "jax.lax.scan(step, (params, tx.init(params)), idx)",
                "tx.update(grads, opt_state, p)",
            ],
        ),
        "postfit_full_pool": (
            "_fit",
            ["final = jnp.mean((predict(fitted) - targets) ** 2)"],
        ),
        "one_normalization": ("sample_pool", ["obs = agent._normalize(obs)"]),
        "teacher_own_architecture_new_parameters": (
            "_base_targets",
            [
                "network_def.init(key, observations=obs[:1], actions=act[:1])",
                "jnp.sin(scale * f)",
            ],
        ),
        "paired_input_target_indices": (
            "probe_round",
            [
                "_base_targets(target_def, target_key, obs, act, cfg.eval_chunk, cfg.target_scale)",
                "_fit(network_def, critic_tx, params, obs, act, base, idx, cfg.eval_chunk, **shared)",
                "'score': b - final",
            ],
        ),
        "continuous_2000_indices": (
            "probe_round",
            [
                "jax.random.fold_in(idx_key, 1)",
                "idx = jnp.concatenate((first, second), axis=0)",
            ],
        ),
        "dedicated_check_round_streams": (
            "run_probe",
            ["probe_rng(seed, check_index, r)", "probe_key(seed, check_index, r)"],
        ),
        "configured_adamw": (
            "critic_optimizer",
            [
                "optax.adamw(learning_rate=agent_cfg.critic_learning_rate, weight_decay=agent_cfg.critic_weight_decay)"
            ],
        ),
    }
    rows = []
    for name, (fname, fragments) in expected.items():
        passed = all(fragment in bodies.get(fname, "") for fragment in fragments)
        if name == "one_normalization":
            passed &= bodies[fname].count("agent._normalize(") == 1
        rows.append(
            {
                "check": name,
                "passed": bool(passed),
                "function": fname,
                "line": nodes[fname].lineno,
                "end_line": nodes[fname].end_lineno,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "scope": "source structure only; original GPU/runtime values not certified",
            }
        )
    return rows


def controlled_routing(source):
    """Actual source control flow; NumPy arrays and fake fitter, no NN execution."""

    class Agent:
        def __init__(self):
            self.count = 0
            self.training_rng = np.random.default_rng(729)
            self.rms = {"mean": np.array([1.0, 2.0]), "var": np.array([4.0, 9.0])}

        def _normalize(self, obs):
            self.count += 1
            return (obs - self.rms["mean"]) / np.sqrt(self.rms["var"] + 1e-8)

    agent = Agent()
    rng_state = deepcopy(agent.training_rng.bit_generator.state)
    rms = deepcopy(agent.rms)
    buffer = SimpleNamespace(
        _num_in_buffer=4,
        _observations=np.arange(8).reshape(4, 2),
        _actions=np.arange(4).reshape(4, 1),
    )
    sample = function(source, "sample_pool", {"jnp": np})
    obs, act = sample(agent, buffer, np.random.default_rng(91), 4)
    expected_idx = np.random.default_rng(91).integers(0, 4, size=4)
    np.testing.assert_array_equal(
        obs,
        (buffer._observations[expected_idx] - rms["mean"]) / np.sqrt(rms["var"] + 1e-8),
    )
    np.testing.assert_array_equal(act, buffer._actions[expected_idx])
    assert agent.count == 1 and agent.training_rng.bit_generator.state == rng_state
    for key in rms:
        np.testing.assert_array_equal(rms[key], agent.rms[key])

    calls, teachers = [], []
    base = np.array([0.1, -0.2, 0.4, 0.0], dtype=np.float32)

    def teacher(*args):
        teachers.append(args)
        return base

    def fake_fit(network, tx, params, o, a, target, idx, chunk, **kwargs):
        calls.append((network, tx, params, o, a, target, idx, kwargs))
        # Sentinel return only. No gradient, optimizer or parameter operation.
        return np.zeros(len(idx)), params["sentinel_mse"], params["sentinel_offset"]

    class Random:
        @staticmethod
        def split(key):
            return key + 1, key + 2

        @staticmethod
        def fold_in(key, value):
            return key * 13 + value

        @staticmethod
        def randint(key, shape, low, high):
            return np.random.default_rng(key).integers(low, high, shape)

    cfg = SimpleNamespace(
        steps=2000, batch_size=2, pool_size=4, eval_chunk=2, target_scale=100000
    )
    params = {
        "current": {"sentinel_mse": 3.0, "sentinel_offset": 2.0},
        "fresh": {"sentinel_mse": 0.2, "sentinel_offset": -0.1},
    }
    before = deepcopy(params)
    critics = {name: ("same-network", p) for name, p in params.items()}
    namespace = {
        "jax": SimpleNamespace(random=Random),
        "jnp": np,
        "ORIGINAL_FIT_STEPS": 1000,
        "_base_targets": teacher,
        "_fit": fake_fit,
        "_unpack": function(source, "_unpack", {}),
    }
    result = function(source, "probe_round", namespace)(
        "teacher-network", critics, "same-tx", obs, act, 19, cfg
    )
    assert len(teachers) == 1 and len(calls) == 2
    for i in (3, 4, 5, 6):
        assert calls[0][i] is calls[1][i]
    assert calls[0][6].shape == (2000, 2)
    assert calls[0][2] is params["current"] and calls[1][2] is params["fresh"]
    assert params == before and calls[0][7] == calls[1][7] == {}
    assert float(result["current"]["score"]) < float(result["fresh"]["score"])
    return {
        "one_normalization": True,
        "rms_and_training_rng_unchanged": True,
        "one_teacher_per_pair": True,
        "same_array_objects_within_pair": True,
        "2000_index_rows": True,
        "distinct_parameter_routing": True,
        "parameters_unchanged": True,
        "scope": "original control flow with fake fitter/teacher; not a numerical fit or original runtime certification",
    }
