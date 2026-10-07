"""CPU-only independent scalar/statistical references and approved 195-command census.

Run from main/: python scripts/methodology_cpu_reference.py
Does not train, access a cluster, submit jobs, or select scientific thresholds.
The binomial tail is an illustration under an IID assumption, not a new gate.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("EXP12_JAX_CACHE_DIR", "off")

import json
import math
import sys
import tempfile
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
import numpy as np
import jax
import jax.numpy as jnp
import generate_manifest as gm
import shlex

CONFIG_PATH = str(MAIN / "configs")


def _parse(command):
    tokens = shlex.split(command.split(" > ")[0])
    return {
        "overrides": [
            tokens[i + 1] for i, t in enumerate(tokens) if t == "--overrides"
        ],
        "interval": int(tokens[tokens.index("--checkpoint_interval") + 1]),
    }


from experiments.exp1 import compose_config
from experiments.exp12 import trigger
from scale_rl.agents.sac.sac_update import update_temperature


class TemperatureReference:
    params = {"log_alpha": jnp.log(jnp.float32(0.01))}

    def apply(self, variables):
        return jnp.exp(variables["params"]["log_alpha"])

    def apply_gradient(self, fn):
        (loss, info), gradient = jax.value_and_grad(fn, has_aux=True)(self.params)
        self.gradient = float(gradient["log_alpha"])
        return self, {**info, "grad_norm": jnp.abs(gradient["log_alpha"])}


def main():
    out = {}
    for hstar in (-1.0, 1.0):
        temp, info = update_temperature(TemperatureReference(), jnp.float32(0), hstar)
        expected = float(jnp.exp(temp.params["log_alpha"])) * (0 - hstar)
        assert temp.gradient == expected
        out[f"temperature_gradient_H0_target{hstar:+g}"] = temp.gradient
    # Independent row sorting and direct average of the middle three values.
    x = np.array([-0.2, 0.04, 0.13, 0.31, 1.4], np.float32).astype(np.float64)
    indices = np.random.default_rng([17, trigger.BOOT_STREAM, 8]).integers(
        0, 5, (10000, 5)
    )
    reference = np.percentile(np.sort(x[indices], axis=1)[:, 1:4].mean(1), [2.5, 97.5])
    actual = trigger.bootstrap_interval(x, 17, 8, 10000, 0.95)
    np.testing.assert_array_equal(actual, reference)
    out["paired_iqm_percentile_reference"] = list(actual)
    out["binomial_illustration_p_ge13_n100_p05"] = sum(
        math.comb(100, k) * 0.05**k * 0.95 ** (100 - k) for k in range(13, 101)
    )
    archs = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D4W1536": (4, 1536)}
    envs = {
        "dog-run": 1000000,
        "dog-trot": 1000000,
        "humanoid-run": 1000000,
        "humanoid-walk": 1000000,
        "humanoid-stand": 1000000,
        "swimmer-swimmer15": 500000,
        "hopper-hop": 500000,
        "myo-key-turn": 1000000,
        "myo-pen-twirl": 1000000,
        "myo-pose-hard": 1000000,
        "myo-reach": 1000000,
        "h1-reach-v0": 2000000,
        "h1-run-v0": 2000000,
    }
    seen = set()
    with tempfile.TemporaryDirectory() as root:
        manifests = {}
        gm.add_exp12_grid(manifests, {}, root + "/ckpt", root + "/results")
        assert len(manifests["exp12_exp1_jobs.txt"]) == 195
        assert not any(k.startswith("exp2_arms") for k in manifests)
        for command in manifests["exp12_exp1_jobs.txt"]:
            job = _parse(command)
            cfg = compose_config(CONFIG_PATH, "base_exp12", job["overrides"])
            dims = (int(cfg.critic_num_blocks), int(cfg.critic_hidden_dim))
            arch = next(a for a, d in archs.items() if d == dims)
            key = (arch, cfg.env_name, int(cfg.seed))
            assert key not in seen
            seen.add(key)
            assert (cfg.actor_num_blocks, cfg.actor_hidden_dim) == (1, 128)
            assert (
                cfg.num_env_steps == envs[cfg.env_name]
                and cfg.num_interaction_steps == envs[cfg.env_name] // 2
            )
            assert cfg.action_repeat == 2 and cfg.updates_per_interaction_step == 2
            assert cfg.agent.critic_use_cdq == cfg.env_name.startswith("h1-")
            assert (
                cfg.probe.rounds == 5
                and cfg.probe.steps == 1000
                and cfg.probe.pool_size == 25600
                and cfg.probe.batch_size == 256
            )
            assert (
                cfg.trigger.resamples == 10000
                and cfg.trigger.confidence == 0.95
                and cfg.trigger.consecutive_checks == 2
                and cfg.trigger.null_threshold == 0
            )
            assert (
                cfg.fork.horizon_fraction == 0.25
                and cfg.fork.max_total_fraction == 1.2
                and cfg.fork.eval_every_fraction == 0.01
                and cfg.fork.eval_episodes == 10
            )
            assert cfg.injection.m is None and cfg.testing.force_trigger_check is None
            assert (
                cfg.agent.temp_target_entropy_coef == -0.5
                and cfg.agent.temp_initial_value == 0.01
            )
            assert (
                cfg.agent.actor_learning_rate == 1e-4
                and cfg.agent.critic_learning_rate == 1e-4
                and cfg.agent.temp_learning_rate == 1e-4
            )
            assert (
                cfg.agent.actor_weight_decay == 0.01
                and cfg.agent.critic_weight_decay == 0.01
                and cfg.agent.temp_weight_decay == 0
            )
            assert cfg.agent.target_tau == 0.005 and cfg.buffer.sample_batch_size == 256
            assert cfg.buffer.max_length == 1000000 and cfg.buffer.min_length == 5000
            assert job["interval"] == envs[cfg.env_name] // 2 // 20
        assert seen == {(a, e, s) for a in archs for e in envs for s in range(1, 6)}
    out["parent_commands"] = len(seen)
    out["architecture_environment_configs"] = 39
    out["eligibility_candidates"] = 130
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
