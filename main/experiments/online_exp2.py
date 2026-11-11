"""Opt-in online reference-policy Exp2 prototype; no production defaults."""

import hashlib
import io
import json
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from omegaconf import OmegaConf

from experiments.exp1 import compose_config
from experiments.exp12.envs import create_eval_env
from experiments.exp12.online_exp2 import (
    Shadow, gradient_cosine, paired_actor_intervention, paired_shadow_update,
)
from experiments.exp12.precision import set_matmul_precision
from experiments.exp12.probe import critic_optimizer, probe_config, summarize
from experiments.exp12.run_probes import run_probe_networks
from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.registry import register_experiment
from scale_rl.agents import create_agent
from utils.atomic_io import atomic_write_bytes, atomic_write_text
from utils.paths import require_absolute


def validate(cfg):
    p = cfg.online_exp2
    if p.get("protocol") == "revised_v1":
        return _validate_revised(cfg)
    pilot = bool(p.get("pilot_enabled", False))
    needed = ("environments", "training_seeds", "reference_critic", "training_interactions",
              "checkpoint_steps", "probe_checkpoints", "evaluation_episodes", "evaluation_mode",
              "intervention_updates", "analysis_aggregation", "resource_cap_gpu_hours")
    if pilot:
        needed = tuple(x for x in needed if x not in ("reference_critic", "intervention_updates")) + (
            "reference_regime", "diagnostic_horizon", "measurement_interactions")
    missing = [name for name in needed if p.get(name) is None]
    if missing:
        raise ValueError("online Exp2 protocol has unresolved fields: " + ", ".join(missing))
    if pilot:
        from experiments.exp12.online_exp2_diagnostic import reference_roles
        reference_roles(p.reference_regime)
        measurements = [int(x) for x in p.measurement_interactions]
        if (cfg.run_role != "pilot" or p.dev_architectures is not None
                or p.reference_critic is not None or p.intervention_updates is not None
                or int(p.diagnostic_horizon) != 10000
                or measurements != sorted(set(measurements))
                or not {0, 5000, 10000}.issubset(measurements)
                or measurements[-1] != 10000 or float(p.resource_cap_gpu_hours) <= 0):
            raise ValueError("invalid online Exp2 pilot protocol")
        if int(p.diagnostic_save_every) < 1:
            raise ValueError("diagnostic_save_every must be positive")
    elif cfg.run_role != "dev" or float(p.resource_cap_gpu_hours) != 0:
        raise ValueError("online Exp2 is a CPU development prototype, not a production launch")
    if not p.enabled or p.initialization != "fresh":
        raise ValueError("only opt-in fresh initialization is implemented")
    if cfg.env_name not in p.environments or cfg.seed not in p.training_seeds:
        raise ValueError("environment/seed outside the selected online Exp2 protocol")
    if not pilot and (set(p.reference_critic) != {"blocks", "width"} or any(
        int(v) <= 0 for v in p.reference_critic.values()
    )):
        raise ValueError("reference critic must specify positive blocks and width")
    if (p.evaluation_mode not in ("deterministic", "stochastic") or
            (not pilot and int(p.intervention_updates) != 1)):
        raise ValueError("invalid evaluation mode or development actor steps")
    if p.analysis_aggregation != "mean_checkpoint_means":
        raise ValueError("unsupported run-level aggregation")
    steps = int(p.training_interactions)
    checks = [int(x) for x in p.checkpoint_steps]
    probes = [int(x) for x in p.probe_checkpoints]
    if (steps < 1 or not checks or int(p.evaluation_episodes) < 1 or float(p.resource_cap_gpu_hours) < 0
            or checks != sorted(set(checks)) or any(x < 1 or x > steps for x in checks)
            or any(x not in checks for x in probes)):
        raise ValueError("invalid online Exp2 schedule or resource cap")
    if p.dev_architectures is None and (
        (cfg.probe.rounds, cfg.probe.steps, cfg.probe.pool_size, cfg.probe.batch_size,
         cfg.probe.eval_chunk) != (10, 2000, 25600, 256, 2560)
    ):
        raise ValueError("full-size shadow runs must use the selected Exp1 assay")
    return steps, checks, probes


def _validate_revised(cfg):
    from experiments.exp12.revised_exp2_population import checkpoint_interactions
    p = cfg.online_exp2
    needed = ("environments", "training_seeds", "reference_regime",
              "training_interactions", "checkpoint_steps", "probe_checkpoints",
              "diagnostic_raw_steps", "diagnostic_save_every", "resource_cap_gpu_hours")
    missing = [name for name in needed if p.get(name) is None]
    if missing:
        raise ValueError("revised online Exp2 pilot has unresolved fields: " + ", ".join(missing))
    if (cfg.run_role != "pilot" or not p.enabled or p.initialization != "fresh"
            or p.reference_critic is not None or p.intervention_updates is not None
            or p.evaluation_mode is not None or p.evaluation_episodes is not None):
        raise ValueError("invalid revised online Exp2 pilot identity")
    if int(p.training_interactions) != 500_000 or int(p.diagnostic_raw_steps) != 10_000:
        raise ValueError("revised pilot requires 1,000,000 raw steps at action repeat two")
    steps = int(p.training_interactions)
    checks = [int(x) for x in p.checkpoint_steps]
    probes = [int(x) for x in p.probe_checkpoints]
    expected = checkpoint_interactions()
    if (checks != expected or any(x not in checks for x in probes)
            or int(p.diagnostic_save_every) < 1 or int(p.diagnostic_save_every) > 10_000):
        raise ValueError("invalid revised pilot checkpoint schedule")
    if cfg.env_name not in p.environments or cfg.seed not in p.training_seeds:
        raise ValueError("environment/seed outside the revised online Exp2 population")
    if p.dev_architectures is None and (
        (cfg.probe.rounds, cfg.probe.steps, cfg.probe.pool_size, cfg.probe.batch_size,
         cfg.probe.eval_chunk) != (10, 2000, 25600, 256, 2560)
    ):
        raise ValueError("full-size revised pilot must use the selected Exp1 assay")
    if float(p.resource_cap_gpu_hours) <= 0:
        raise ValueError("revised pilot needs a positive resource cap")
    return steps, checks, probes


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def aggregate_run(rows, checkpoints):
    """Equal-checkpoint mean of episode-level large-minus-small return contrasts."""
    means = []
    for step in checkpoints:
        values = [row["advantage_large"] for row in rows
                  if row.get("type") == "evaluation" and row.get("interaction_step") == step]
        if not values or not np.all(np.isfinite(values)):
            raise ValueError(f"missing or nonfinite Exp2 evaluation at checkpoint {step}")
        means.append(float(np.mean(values)))
    if not means:
        raise ValueError("analysis requires at least one checkpoint")
    return {"checkpoint_means": dict(zip(checkpoints, means)),
            "mean_checkpoint_means": float(np.mean(means)), "independent_units": 1}


class OnlineExp2:
    def __init__(self, cfg, root):
        cfg = deepcopy(cfg)
        self.last_step, self.checks, self.probes = validate(cfg)
        self.pilot = bool(cfg.online_exp2.get("pilot_enabled", False))
        self.revised = cfg.online_exp2.get("protocol") == "revised_v1"
        self._pilot_started_at = time.monotonic()
        self._pilot_prior_device_seconds = 0.
        self.cfg = cfg
        self.root = Path(require_absolute(root, "checkpoint_dir"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = _sha(OmegaConf.to_yaml(cfg, resolve=True).encode())
        self.source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
        source_root = Path(__file__).resolve().parents[1]
        source_files = ("experiments/exp1.py", "experiments/online_exp2.py", "experiments/exp12/online_exp2.py",
                        "experiments/exp12/online_exp2_diagnostic.py",
                        "experiments/exp12/online_exp2_pilot.py",
                        "experiments/exp12/revised_exp2_pilot.py",
                        "experiments/exp12/revised_exp2_population.py",
                        "experiments/exp12/trainer.py", "scale_rl/buffers/numpy_buffer.py",
                        "scale_rl/agents/sac/sac_agent.py",
                        "scale_rl/agents/wrappers/normalization.py",
                        "scale_rl/envs/wrappers/repeat_action.py",
                        "scale_rl/agents/sac/sac_update.py", "experiments/exp12/probe.py")
        self.source_hash = _sha(b"".join((name.encode() + b"\0" +
                                          (source_root / name).read_bytes()) for name in source_files))
        OmegaConf.set_struct(cfg, False)
        if self.pilot:
            from experiments.exp12.online_exp2_diagnostic import reference_roles
            reference_arch, _, _ = reference_roles(cfg.online_exp2.reference_regime)
            cfg.agent.critic_num_blocks, cfg.agent.critic_hidden_dim = reference_arch
        else:
            cfg.agent.critic_num_blocks = int(cfg.online_exp2.reference_critic.blocks)
            cfg.agent.critic_hidden_dim = int(cfg.online_exp2.reference_critic.width)
        cfg.num_interaction_steps = self.last_step
        OmegaConf.set_struct(cfg, True)
        np.random.seed(int(cfg.seed))
        random.seed(int(cfg.seed))
        self.trainer = Exp12Trainer(cfg, str(self.root))
        self.key = jax.random.fold_in(jax.random.PRNGKey(int(cfg.seed)), 0xE220)
        self.diagnostic_rng = np.random.default_rng([int(cfg.seed), 0xE221])
        self.shadows = {}
        architectures = cfg.online_exp2.dev_architectures
        if architectures is None:
            architectures = (("small", 2, 512), ("large", 4, 1536))
        if [entry[0] for entry in architectures] != ["small", "large"]:
            raise ValueError("development architectures must retain small/large labels")
        for index, (name, blocks, width) in enumerate(architectures):
            agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=True))
            agent_cfg.critic_num_blocks = blocks
            agent_cfg.critic_hidden_dim = width
            agent_cfg.seed = int(cfg.seed) + 100001 + index
            agent = create_agent(self.trainer.train_env.observation_space,
                                 self.trainer.train_env.action_space, agent_cfg)
            sac = getattr(agent, "agent", agent)
            self.shadows[name] = Shadow(sac.critic, sac._target_critic, sac.critic.params)
        self.rows = []
        self.fresh_done = False

    def _next_key(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def _device_seconds(self):
        devices = jax.local_device_count() if jax.default_backend() == "gpu" else 1
        return self._pilot_prior_device_seconds + (time.monotonic() - self._pilot_started_at) * devices

    def _check_resource_cap(self):
        if (self.pilot and self._device_seconds() >
                float(self.cfg.online_exp2.resource_cap_gpu_hours) * 3600):
            raise RuntimeError("online Exp2 pilot resource cap exceeded")

    def _before_update(self, trainer, batches, indices):
        self._shadow_updates(trainer, batches, indices, None)

    def _after_update(self, trainer, batches, indices, snapshots):
        self._shadow_updates(trainer, batches, indices, snapshots)

    def _shadow_updates(self, trainer, batches, indices, snapshots):
        # The pilot uses pre-update policy states returned by the source SAC scan.
        for i, sampled_indices in enumerate(indices):
            raw = {name: value[i] for name, value in batches.items()}
            normalized = dict(raw)
            if snapshots is None and hasattr(trainer.agent, "_normalize"):
                normalized["observation"] = trainer.agent._normalize(raw["observation"])
                normalized["next_observation"] = trainer.agent._normalize(raw["next_observation"])
            actor = trainer._sac_agent.actor
            temperature = trainer._sac_agent.temperature
            if snapshots is not None:
                actor = actor.replace(params=jax.tree.map(lambda x: x[i], snapshots["actor"]))
                temperature = temperature.replace(
                    params=jax.tree.map(lambda x: x[i], snapshots["temperature"]))
            self.shadows, result = paired_shadow_update(
                actor, temperature, self.shadows,
                normalized, self._next_key(), float(trainer.cfg.agent.gamma),
                int(trainer.cfg.agent.n_step), float(trainer.cfg.agent.target_tau),
                bool(trainer.cfg.agent.critic_use_cdq))
            if not all(np.all(np.isfinite(np.asarray(value))) for value in
                       (result["next_actions"], result["next_log_probs"],
                        *[metrics["loss"] for metrics in result["critics"].values()])):
                raise ValueError("nonfinite shadow update")
            payload = b"".join(np.asarray(normalized[name]).tobytes() for name in sorted(normalized))
            self.rows.append({"type": "update", "interaction_step": trainer.interaction_step,
                              "raw_env_steps": trainer.interaction_step * int(trainer.cfg.action_repeat),
                              "reference_update_step": trainer.update_step + i,
                              "reference_actor_snapshot_sha256": _sha(flax.serialization.to_bytes(actor.params)),
                              "indices": np.asarray(sampled_indices).tolist(),
                              "batch_sha256": _sha(payload),
                              "next_actions_sha256": _sha(np.asarray(result["next_actions"]).tobytes()),
                              "next_log_probs_sha256": _sha(np.asarray(result["next_log_probs"]).tobytes()),
                              "small_loss": float(result["critics"]["small"]["loss"]),
                              "large_loss": float(result["critics"]["large"]["loss"]),
                              "small_target_q_mean": float(result["critics"]["small"]["q_target_mean"]),
                              "large_target_q_mean": float(result["critics"]["large"]["q_target_mean"])})

    def _probe(self, check):
        cfg = probe_config(self.cfg)
        for name, shadow in self.shadows.items():
            critic = shadow.critic
            critics = {"fresh": (critic.network_def, shadow.fresh_params)}
            if check:
                critics["current"] = (critic.network_def, critic.params)
            tx = critic_optimizer(self.cfg.agent)
            result = run_probe_networks(self.trainer.agent, self.trainer.buffer, critic.network_def,
                                        critics, tx, tx, int(self.cfg.seed) + 200002,
                                        check, cfg)
            if not all(np.all(np.isfinite(values)) for fields in result.values()
                       for values in fields.values()):
                raise ValueError("nonfinite plasticity probe")
            output = self.root / "probes" / f"{name}_{check:04d}.npz"
            output.parent.mkdir(exist_ok=True)
            with io.BytesIO() as stream:
                np.savez_compressed(stream, **{f"{role}_{field}": values
                                             for role, fields in result.items()
                                             for field, values in fields.items()})
                atomic_write_bytes(output, stream.getvalue())
            row = {"type": "probe", "check": check, "architecture": name,
                   "artifact": str(output), "sha256": _sha(output.read_bytes())}
            if check:
                row["plasticity_loss_iqm"] = float(summarize(result)["loss_iqm"])
            self.rows.append(row)

    def _evaluate(self, actor, seed):
        env_cfg = OmegaConf.to_container(self.cfg.env, resolve=True)
        env_cfg["seed"] = int(seed)
        env = create_eval_env(**env_cfg)
        key = jax.random.PRNGKey(int(seed))
        total, length, success, terminated, truncated = 0., 0, 0., False, False
        try:
            observation, _ = env.reset()
            while not (terminated or truncated):
                if hasattr(self.trainer.agent, "_normalize"):
                    observation = self.trainer.agent._normalize(observation)
                key, action_key = jax.random.split(key)
                mode = 0. if self.cfg.online_exp2.evaluation_mode == "deterministic" else 1.
                action = np.asarray(actor(observations=jnp.asarray(observation), temperature=mode).sample(seed=action_key))
                observation, reward, term, trunc, info = env.step(action)
                total += float(reward[0])
                length += 1
                terminated, truncated = bool(term[0]), bool(trunc[0])
                if "success" in info:
                    success = max(success, float(info["success"][0]))
                elif "final_info" in info and info["final_info"][0] is not None:
                    success = max(success, float(info["final_info"][0].get("success", 0.)))
        finally:
            env.close()
        return {"return": total, "length": length, "success": success,
                "terminated": terminated, "truncated": truncated, "seed": int(seed)}

    def _checkpoint(self, trainer):
        if self.revised:
            from experiments.exp12.revised_exp2_pilot import revised_pilot_checkpoint
            revised_pilot_checkpoint(self, trainer, bool(self.cfg.get("wandb_enabled", True)))
            return
        if self.pilot:
            from experiments.exp12.online_exp2_pilot import pilot_checkpoint
            pilot_checkpoint(self, trainer)
            return
        batch_size = int(trainer.cfg.buffer.sample_batch_size)
        indices = self.diagnostic_rng.integers(0, trainer.buffer._num_in_buffer, size=batch_size)
        observations = trainer.buffer._observations[indices]
        if hasattr(trainer.agent, "_normalize"):
            observations = trainer.agent._normalize(observations)
        actor = trainer._sac_agent.actor
        clones, records = paired_actor_intervention(
            actor, trainer._sac_agent.temperature, self.shadows, observations,
            self._next_key(), bool(trainer.cfg.agent.critic_use_cdq))
        if not all(np.all(np.isfinite(np.asarray(leaf))) for values in records.values()
                   for field in ("gradient", "parameter_delta")
                   for leaf in jax.tree.leaves(values[field])):
            raise ValueError("nonfinite actor intervention")
        step = trainer.interaction_step
        artifact = self.root / "interventions" / f"step_{step:08d}.npz"
        artifact.parent.mkdir(exist_ok=True)
        action_indices = self.diagnostic_rng.integers(0, trainer.buffer._num_in_buffer, size=batch_size)
        action_obs = trainer.buffer._observations[action_indices]
        if hasattr(trainer.agent, "_normalize"):
            action_obs = trainer.agent._normalize(action_obs)
        action_key = self._next_key()
        action_arrays = {name: np.asarray(policy(observations=jnp.asarray(action_obs), temperature=0.).sample(
            seed=action_key)) for name, policy in {"baseline": actor, **clones}.items()}
        if not all(np.all(np.isfinite(value)) for value in action_arrays.values()):
            raise ValueError("nonfinite diagnostic actions")
        with io.BytesIO() as stream:
            np.savez_compressed(stream, observation_indices=indices, action_observation_indices=action_indices,
                                **{f"gradient_{name}": np.asarray(ravel_pytree(values["gradient"])[0])
                                   for name, values in records.items()},
                                **{f"delta_{name}": np.asarray(ravel_pytree(values["parameter_delta"])[0])
                                   for name, values in records.items()},
                                **{f"actions_{name}": value for name, value in action_arrays.items()})
            atomic_write_bytes(artifact, stream.getvalue())
        for episode in range(int(self.cfg.online_exp2.evaluation_episodes)):
            seed = int(self.cfg.seed) * 100000 + step * 100 + episode
            evaluations = {"baseline": self._evaluate(actor, seed),
                           "small": self._evaluate(clones["small"], seed),
                           "large": self._evaluate(clones["large"], seed)}
            self.rows.append({"type": "evaluation", "interaction_step": step, "episode": episode,
                              "policies": evaluations,
                              "delta_small": evaluations["small"]["return"] - evaluations["baseline"]["return"],
                              "delta_large": evaluations["large"]["return"] - evaluations["baseline"]["return"],
                              "advantage_large": evaluations["large"]["return"] - evaluations["small"]["return"]})
        self.rows.append({"type": "intervention", "interaction_step": step,
                          "raw_env_steps": step * int(trainer.cfg.action_repeat),
                          "observation_indices": indices.tolist(),
                          "artifact": str(artifact), "sha256": _sha(artifact.read_bytes()),
                          "reference_update_step": trainer.update_step,
                          "small_updates": self.shadows["small"].updates,
                          "large_updates": self.shadows["large"].updates,
                          "small_action_change": float(np.linalg.norm(
                              action_arrays["small"] - action_arrays["baseline"], axis=-1).mean()),
                          "large_action_change": float(np.linalg.norm(
                              action_arrays["large"] - action_arrays["baseline"], axis=-1).mean()),
                          "gradient_cosine": gradient_cosine(records["small"]["gradient"],
                                                             records["large"]["gradient"]),
                          **{f"{name}_{field}": float(values[field]) for name, values in records.items()
                             for field in ("loss", "entropy", "gradient_norm", "delta_norm")}})
        if step in self.probes:
            self._probe(step)
        self.save()

    def _pilot_analysis(self):
        """Keep the independent run separate from its repeated measurements."""
        rows = [row for row in self.rows if row.get("type") == "pilot_evaluation"]
        means = {}
        for horizon in self.cfg.online_exp2.measurement_interactions:
            if int(horizon) == 0:
                continue
            checkpoint_means = {}
            for step in self.checks:
                values = [row["advantage_large"] for row in rows
                          if row["interaction_step"] == step
                          and row["diagnostic_interactions"] == horizon]
                if len(values) != int(self.cfg.online_exp2.evaluation_episodes) or not np.all(np.isfinite(values)):
                    raise ValueError(f"incomplete pilot evaluation: checkpoint {step}, horizon {horizon}")
                checkpoint_means[step] = float(np.mean(values))
            means[int(horizon)] = {"checkpoint_means": checkpoint_means,
                                   "mean_checkpoint_means": float(np.mean(list(checkpoint_means.values())))}
        return {"reference_regime": self.cfg.online_exp2.reference_regime,
                "environment": self.cfg.env_name, "training_seed": int(self.cfg.seed),
                "independent_units": 1,
                "horizons": means}

    def _save_shadow(self, path):
        from flax.serialization import to_bytes
        state = {name: {"critic": shadow.critic, "target": shadow.target,
                        "fresh_params": shadow.fresh_params, "updates": shadow.updates}
                 for name, shadow in self.shadows.items()}
        data = to_bytes({"shadows": state, "key": self.key,
                         "diagnostic_rng": json.dumps(self.diagnostic_rng.bit_generator.state)})
        atomic_write_bytes(path / "online_exp2.msgpack", data)
        atomic_write_text(path / "online_exp2_manifest.json", json.dumps({
            "protocol": "online-reference-policy-v1", "config_sha256": self.config_hash,
            "source_commit": self.source_commit, "source_sha256": self.source_hash,
            "payload_sha256": _sha(data), "interaction_step": self.trainer.interaction_step,
        }, sort_keys=True))

    def save(self):
        if self.pilot:
            self.trainer.extra_state["online_exp2_device_seconds"] = self._device_seconds()
        self.trainer.extra_state["online_exp2_rows"] = self.rows
        self.trainer.extra_state["online_exp2_fresh_done"] = self.fresh_done
        self.trainer.save(self.root / "state", before_commit=self._save_shadow)
        atomic_write_text(self.root / "online_exp2.jsonl", "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in self.rows))

    def restore(self, path):
        from flax.serialization import from_bytes
        manifest = json.loads((path / "online_exp2_manifest.json").read_text())
        data = (path / "online_exp2.msgpack").read_bytes()
        if (manifest["config_sha256"] != self.config_hash or
                manifest["source_commit"] != self.source_commit or
                manifest["source_sha256"] != self.source_hash or
                manifest["payload_sha256"] != _sha(data)):
            raise ValueError("online Exp2 checkpoint identity mismatch")
        self.trainer.restore(path)
        target = {name: {"critic": shadow.critic, "target": shadow.target,
                         "fresh_params": shadow.fresh_params, "updates": shadow.updates}
                  for name, shadow in self.shadows.items()}
        restored = from_bytes({"shadows": target, "key": self.key,
                               "diagnostic_rng": json.dumps(self.diagnostic_rng.bit_generator.state)}, data)
        self.shadows = {name: Shadow(**value) for name, value in restored["shadows"].items()}
        self.key = restored["key"]
        self.diagnostic_rng.bit_generator.state = json.loads(restored["diagnostic_rng"])
        self.rows = self.trainer.extra_state["online_exp2_rows"]
        self.fresh_done = self.trainer.extra_state["online_exp2_fresh_done"]
        if self.pilot:
            self._pilot_prior_device_seconds = float(
                self.trainer.extra_state.get("online_exp2_device_seconds", 0.))
            self._pilot_started_at = time.monotonic()

    def run(self):
        latest = latest_state_dir(self.root / "state")
        if latest is None:
            self.trainer.start()
        else:
            self.restore(latest)
            self._check_resource_cap()
            if self.pilot and self.trainer.extra_state.get("online_exp2_pending_checkpoint"):
                self._checkpoint(self.trainer)
        self._check_resource_cap()

        def fresh(trainer):
            if not self.fresh_done:
                if not self.revised:
                    self._probe(0)
                self.fresh_done = True

        def after(trainer):
            self._check_resource_cap()
            if trainer.interaction_step in self.checks and not any(
                (row.get("type") == "intervention"
                 or row.get("type") == "revised_pilot_checkpoint")
                and row.get("interaction_step", row.get("checkpoint")) == trainer.interaction_step
                for row in self.rows
            ):
                self._checkpoint(trainer)
            elif (self.revised
                  and trainer.interaction_step % int(
                      self.cfg.online_exp2.save_every_interaction_steps) == 0):
                self.save()

        try:
            self.trainer.train(self.last_step, after_step=after, before_first_update=fresh,
                               before_updates=None if self.pilot else self._before_update,
                               after_updates=self._after_update if self.pilot else None)
            self.save()
            if self.revised:
                from experiments.exp12.revised_exp2_pilot import revised_analysis
                analysis = revised_analysis(self)
            else:
                analysis = (self._pilot_analysis() if self.pilot
                            else aggregate_run(self.rows, self.checks))
            atomic_write_text(self.root / "analysis.json", json.dumps(
                analysis, sort_keys=True, allow_nan=False))
        except BaseException:
            # Preserve the latest training state for restart; branch progress is
            # written by the checkpoint runner before any branch work starts.
            self.save()
            raise
        finally:
            self.trainer.close()


@register_experiment("online_exp2_prototype")
def run(args):
    set_matmul_precision()
    selected = args.get("config_name")
    cfg = compose_config(args["config_path"],
                         "base_online_exp2" if selected in (None, "base_sac") else selected,
                         args.get("overrides", []), allow_pilot=True)
    if not cfg.online_exp2.get("pilot_enabled", False) and jax.default_backend() != "cpu":
        raise ValueError("online Exp2 development prototype is restricted to CPU")
    OnlineExp2(cfg, args["checkpoint_dir"]).run()
