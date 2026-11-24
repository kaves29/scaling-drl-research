"""Run short, paired mechanism diagnostics using trusted Exp1/Exp2 checkpoints."""

import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from flax.serialization import from_bytes
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from experiments.exp12 import exp3_mechanisms as m
from experiments.exp12.envs import env_restore_state, restore_env
from experiments.exp12.online_exp2_pilot import _branch_env, _seed, frozen_normalizer
from experiments.exp12.precision import set_matmul_precision
from experiments.exp12.replay_exp2_pilot import ReplayExp2Pilot, compose_replay_config
from experiments.exp12.replay_exp2_contract import admit_parent, validate_parent_config
from experiments.exp12.state import latest_state_dir
from scale_rl.agents import create_agent
from utils.atomic_io import atomic_write_text

REGIMES = {"small": "reference_small", "large": "reference_large"}


def normalize(transform, x):
    return transform(x) if transform else x


def action_q(critic, cdq, obs, action):
    q = critic(observations=obs, actions=action)
    return jnp.minimum(*q).reshape(-1) if cdq else q.reshape(-1)


def alignment(critic_normalizer, actor_normalizer, example):
    """Convert the actor's normalized inputs to the critic's native coordinates."""
    zeros, ones = np.zeros_like(example), np.ones_like(example)
    a0, a1 = (normalize(actor_normalizer, x) for x in (zeros, ones))
    c0, c1 = (normalize(critic_normalizer, x) for x in (zeros, ones))
    scale = (c1 - c0) / (a1 - a0)
    offset = c0 - a0 * scale
    if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(offset)):
        raise ValueError("invalid normalizer bridge")
    return jnp.asarray(scale), jnp.asarray(offset)


def bridge(critic, scale, offset):
    def evaluate(*, observations, actions):
        return critic(observations=observations * scale + offset, actions=actions)

    return evaluate


def load_exp1_agent(manifest, identity, observation_space, action_space):
    state, parent = admit_parent(manifest, identity)
    cfg = compose_replay_config(identity)
    OmegaConf.set_struct(cfg, False)
    cfg.agent.critic_num_blocks, cfg.agent.critic_hidden_dim = (
        (2, 512) if identity["reference_regime"] == "reference_small" else (4, 1536)
    )
    metadata = json.loads((state.parent.parent / "run_metadata.json").read_text())
    validate_parent_config(
        metadata["resolved_config"], OmegaConf.to_container(cfg, resolve=True)
    )
    agent = create_agent(observation_space, action_space, cfg.agent)
    agent.load_checkpoint(str(state))
    return agent, parent


def load_shadows(reference, directory, expected_commit):
    """Read a trusted replay-pilot artifact without attaching its W&B run."""
    path = latest_state_dir(directory / "state")
    if path is None:
        raise ValueError(f"missing Exp2 checkpoint: {directory}")
    meta = json.loads((path / "online_exp2_manifest.json").read_text())
    payload = (path / "online_exp2.msgpack").read_bytes()
    if (
        meta["protocol"] != "exp1-final-replay-v1"
        or meta["source_commit"] != expected_commit
        or meta["parent"] != reference.parent
        or meta["config_sha256"] != reference.config_hash
        or meta["source_sha256"] != reference.source_hash
        or meta["payload_sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        raise ValueError(
            "Exp2 checkpoint differs from pinned source, parent or protocol"
        )
    target = {
        name: dict(
            critic=s.critic,
            target=s.target,
            fresh_params=s.fresh_params,
            updates=s.updates,
        )
        for name, s in reference.shadows.items()
    }
    data = from_bytes(
        dict(
            shadows=target,
            key=reference.key,
            diagnostic_rng=json.dumps(reference.diagnostic_rng.bit_generator.state),
        ),
        payload,
    )
    checkpoint = meta["offline_critic_updates"]
    if (
        type(checkpoint) is not int
        or checkpoint <= 0
        or any(
            int(shadow["updates"]) != checkpoint for shadow in data["shadows"].values()
        )
    ):
        raise ValueError("Exp2 update counters disagree")
    return data["shadows"], checkpoint


@jax.jit
def sample(actor, obs, key):
    dist = actor(observations=obs)
    a = dist.sample(seed=key)
    return a, dist.log_prob(a)


def rollout(
    env, actor, normalizer, obs, key, horizon, *, first=None, gamma=1.0, temperature=0.0
):
    total = 0.0
    for step in range(horizon):
        key, choice = jax.random.split(key)
        if step == 0 and first is not None:
            action, entropy_reward = first, 0.0
        else:
            a, logp = sample(actor, normalize(normalizer, obs), choice)
            action = np.asarray(a)
            entropy_reward = (
                -temperature * float(np.asarray(logp).reshape(-1)[0])
                if temperature
                else 0.0
            )
        obs, reward, term, trunc, _ = env.step(action)
        total += gamma**step * (float(reward[0]) + entropy_reward)
        if term[0] or trunc[0]:
            break
    if not np.isfinite(total):
        raise ValueError("nonfinite rollout return")
    return total


class EvaluationEnv:
    """Reuse a simulator and cache exact seeded reset states across actor arms."""

    def __init__(self, config, seed):
        self.config = config
        self.env = _branch_env(config, seed)
        self.starts = {}

    @property
    def envs(self):
        return self.env.envs

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, *, seed=None):
        if seed is None:
            return self.env.reset()
        if seed not in self.starts:
            with closing(_branch_env(self.config, seed)) as source:
                obs, _ = source.reset()
                self.starts[seed] = obs.copy(), env_restore_state(source)
        obs, state = self.starts[seed]
        self.env.reset()
        restore_env(self.env, state)
        return obs.copy(), {}

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def evaluate(env, actor, normalizer, seeds, horizon):
    return [
        rollout(
            env,
            actor,
            normalizer,
            env.reset(seed=seed)[0],
            jax.random.PRNGKey(seed),
            horizon,
        )
        for seed in seeds
    ]


def match_kl(actor, proposal, obs, target):
    """Find a reachable target along one update direction, to 10% tolerance."""

    def at(fraction):
        candidate = m.scale_actor_displacement(actor, proposal, fraction)
        return candidate, float(m.actor_policy_kl(actor, candidate, obs))

    lo, hi = 0.0, 1.0
    candidate, kl = at(hi)
    while kl < target and hi < 4:
        hi *= 2
        candidate, kl = at(hi)
    if not np.isfinite(kl) or kl < target:
        return None
    for _ in range(14):
        mid = (lo + hi) / 2
        candidate, kl = at(mid)
        if abs(kl - target) <= 0.1 * target:
            return candidate, kl, mid
        if kl < target:
            lo = mid
        else:
            hi = mid
    return None


def probe_ranking(
    env, actor, normalizer, critics, cdq, seeds, horizon, gamma, alpha, *, warmup=20
):
    rows = []
    # Gather naturally visited states. Replay observations do not encode simulator state.
    for seed in seeds[:3]:
        obs, _ = env.reset(seed=seed)
        key = jax.random.PRNGKey(seed)
        for _ in range(warmup):
            key, next_key = jax.random.split(key)
            obs, _, term, trunc, _ = env.step(
                np.asarray(sample(actor, normalize(normalizer, obs), next_key)[0])
            )
            if term[0] or trunc[0]:
                break
        if term[0] or trunc[0]:
            continue
        snapshot = env_restore_state(env)
        z = jnp.asarray(normalize(normalizer, obs))
        anchor = sample(actor, z, key)[0]
        directions = {
            name: jax.grad(lambda a: action_q(critic, cdq, z, a).sum())(anchor)
            for name, critic in critics.items()
        }
        for name, vector in directions.items():
            norm = float(jnp.linalg.norm(vector))
            if norm < 1e-8:
                continue
            unit = vector / norm
            distance = min(
                0.05,
                float(
                    jnp.min((1.0 - jnp.abs(anchor)) / jnp.maximum(jnp.abs(unit), 1e-12))
                )
                * 0.95,
            )
            if distance < 1e-5:
                continue
            actions = [anchor - distance * unit, anchor + distance * unit]
            scores = {}
            for critic_name, critic in critics.items():
                q = [float(action_q(critic, cdq, z, a)[0]) for a in actions]
                scores[critic_name] = q[1] - q[0]
            actual = []
            for trial in range(4):
                continuation = jax.random.fold_in(key, trial + 100)
                pair = []
                for a in actions:
                    env.reset()  # Clear the vector wrapper's pending autoreset.
                    restore_env(env, snapshot)
                    pair.append(
                        rollout(
                            env,
                            actor,
                            normalizer,
                            obs,
                            continuation,
                            horizon,
                            first=np.asarray(a),
                            gamma=gamma,
                            temperature=alpha,
                        )
                    )
                actual.append(pair[1] - pair[0])
            rows.append(
                dict(
                    seed=seed,
                    direction=name,
                    distance=distance,
                    critic_differences=scores,
                    soft_return_differences=actual,
                )
            )
    return rows


def run(args):
    set_matmul_precision()
    if getattr(args, "require_a100_40", False):
        devices = jax.devices()
        if (
            len(devices) != 1
            or devices[0].platform != "gpu"
            or (
                "A100" not in devices[0].device_kind
                or "40GB" not in devices[0].device_kind
            )
        ):
            raise ValueError(f"expected one A100 40 GB device, found {devices}")
    seed = args.seed
    rng = np.random.default_rng(seed)
    identity = dict(
        environment=args.env, seed=seed, reference_regime=REGIMES[args.reference]
    )
    cfg = compose_replay_config(identity)
    with TemporaryDirectory(prefix="exp3-") as tmp, ExitStack() as stack:
        ref = ReplayExp2Pilot(cfg, str(Path(tmp) / "ref"), args.manifest, identity)
        stack.callback(ref.trainer.close)
        actor = ref.trainer._sac_agent.actor
        temp = ref.trainer._sac_agent.temperature
        alpha = float(temp())
        cdq = bool(ref.cfg.agent.critic_use_cdq)
        normalizer, _ = frozen_normalizer(ref.trainer.agent)
        buf = ref.trainer.buffer
        size = min(512, 2 * (len(buf) // 2))
        if size < 4:
            raise ValueError("need disjoint proposal and holdout replay samples")
        permutation = rng.choice(len(buf), size=size, replace=False)
        batch, hold = [
            jnp.asarray(normalize(normalizer, buf._observations[i]))
            for i in np.split(permutation, 2)
        ]
        provenance = {"reference_parent": ref.parent}
        if args.exp2_root:
            shadows, checkpoint = load_shadows(ref, args.exp2_root, args.exp2_commit)
            critics = {name: part["critic"] for name, part in shadows.items()}
            provenance.update(
                source="exp2_shared_replay",
                critic_updates=checkpoint,
                checkpoint=str(args.exp2_root),
                checkpoint_source=args.exp2_commit,
            )
        else:
            other_name = "large" if args.reference == "small" else "small"
            other_id = dict(identity, reference_regime=REGIMES[other_name])
            other_agent, other_parent = load_exp1_agent(
                args.manifest,
                other_id,
                ref.trainer.train_env.observation_space,
                ref.trainer.train_env.action_space,
            )
            other_norm, _ = frozen_normalizer(other_agent)
            scale, offset = alignment(other_norm, normalizer, buf._observations[0])
            other_sac = getattr(other_agent, "agent", other_agent)
            critics = {
                args.reference: ref.trainer._sac_agent.critic,
                other_name: bridge(other_sac.critic, scale, offset),
            }
            provenance.update(
                source="exp1_native_training_histories", other_parent=other_parent
            )
            del other_agent, other_sac
        env = stack.enter_context(closing(EvaluationEnv(ref.cfg, seed)))
        seeds = [_seed("exp3-eval", args.env, seed, i) for i in range(args.episodes)]
        baseline = evaluate(env, actor, normalizer, seeds, args.horizon)
        key = jax.random.PRNGKey(_seed("exp3-update", args.env, seed))
        rows = []
        requested = set(args.pilots)
        gradients = {}
        standard = {}
        widths = (
            {"original": m.reset_actor_optimizer(actor), "wider": m.widen_actor(actor)}
            if 6 in requested
            else {}
        )
        if requested & {1, 2, 3, 5, 6}:
            for name, critic in critics.items():
                if 5 in requested:
                    gradients[name] = m.actor_gradient_components(
                        actor, critic, batch, key, critic_use_cdq=cdq
                    )
                if not requested & {1, 2, 3, 5}:
                    if 6 in requested:
                        run_width(
                            rows,
                            env,
                            actor,
                            widths,
                            critic,
                            temp,
                            batch,
                            hold,
                            key,
                            cdq,
                            normalizer,
                            seeds,
                            args.horizon,
                            baseline,
                            name,
                        )
                    continue
                proposal, _ = m.propose_actor_update(
                    actor, critic, temp, batch, key, critic_use_cdq=cdq
                )
                for pilot in sorted(requested & {2, 3, 5}):
                    if name not in standard:
                        matched_record(
                            rows,
                            env,
                            actor,
                            proposal,
                            hold,
                            normalizer,
                            seeds,
                            args.horizon,
                            baseline,
                            pilot,
                            name,
                            "standard",
                            0.001,
                        )
                        standard[name] = rows[-1]
                    else:
                        rows.append(dict(standard[name], pilot=pilot))
                if 1 in requested:
                    for fraction in (0.25, 0.5, 1.0, 2.0):
                        candidate = m.scale_actor_displacement(
                            actor, proposal, fraction
                        )
                        record(
                            rows,
                            env,
                            actor,
                            candidate,
                            hold,
                            normalizer,
                            seeds,
                            args.horizon,
                            baseline,
                            1,
                            name,
                            f"fraction_{fraction}",
                        )
                if 2 in requested:
                    matched_record(
                        rows,
                        env,
                        actor,
                        proposal,
                        hold,
                        normalizer,
                        seeds,
                        args.horizon,
                        baseline,
                        2,
                        name,
                        "standard",
                        0.005,
                    )
                if 3 in requested:
                    penalty, _ = m.propose_actor_update(
                        actor,
                        critic,
                        temp,
                        batch,
                        key,
                        critic_use_cdq=cdq,
                        saturation_penalty=1.0,
                        saturation_threshold=0.95,
                    )
                    matched_record(
                        rows,
                        env,
                        actor,
                        penalty,
                        hold,
                        normalizer,
                        seeds,
                        args.horizon,
                        baseline,
                        3,
                        name,
                        "boundary_penalty",
                        0.001,
                    )
                if 6 in requested:
                    run_width(
                        rows,
                        env,
                        actor,
                        widths,
                        critic,
                        temp,
                        batch,
                        hold,
                        key,
                        cdq,
                        normalizer,
                        seeds,
                        args.horizon,
                        baseline,
                        name,
                    )
        if 5 in requested:
            weight = m.matched_entropy_weight(
                gradients["small"][0], gradients["large"][0], alpha
            )
            for name, critic in critics.items():
                reward_grad, entropy_grad = gradients[name]
                balance = m.gradient_balance(reward_grad, entropy_grad, alpha)
                rows.append(
                    dict(pilot=5, critic=name, gradient_balance=serializable(balance))
                )
                if name == "large":
                    proposal, _ = m.propose_actor_update(
                        actor,
                        critic,
                        temp,
                        batch,
                        key,
                        critic_use_cdq=cdq,
                        entropy_weight=weight,
                    )
                    matched_record(
                        rows,
                        env,
                        actor,
                        proposal,
                        hold,
                        normalizer,
                        seeds,
                        args.horizon,
                        baseline,
                        5,
                        name,
                        "matched_balance",
                        0.001,
                        entropy_weight=float(weight),
                    )
        ranking = (
            probe_ranking(
                env,
                actor,
                normalizer,
                critics,
                cdq,
                seeds,
                args.horizon,
                float(ref.cfg.agent.gamma),
                alpha,
            )
            if 2 in requested
            else []
        )
        feedback = []
        if 4 in requested:
            if not args.exp2_root:
                raise ValueError(
                    "pilot 4 requires paired Exp2 critics with a common normalizer"
                )
            from experiments.exp12.exp3_feedback import run_feedback

            feedback = run_feedback(
                ref,
                shadows,
                actor,
                normalizer,
                env,
                seeds,
                args.horizon,
                args.feedback_steps,
                evaluate,
            )
            for row in feedback:
                row["return_change"] = np.subtract(row["returns"], baseline).tolist()
        return dict(
            schema="exp3-mechanisms-v1",
            identity=identity,
            provenance=provenance,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            evaluation_seeds=seeds,
            horizon=args.horizon,
            baseline_returns=baseline,
            actor_updates=rows,
            action_rankings=ranking,
            feedback=feedback,
        )


def serializable(values):
    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in values.items()
    }


def record(
    rows,
    env,
    actor,
    model,
    hold,
    normalizer,
    seeds,
    horizon,
    baseline,
    pilot,
    critic,
    arm,
    **extras,
):
    kl = float(m.actor_policy_kl(actor, model, hold))
    actions, _ = sample(model, hold, jax.random.PRNGKey(769))
    returns = evaluate(env, model, normalizer, seeds, horizon)
    rows.append(
        dict(
            pilot=pilot,
            critic=critic,
            arm=arm,
            kl=kl,
            saturation=float(jnp.mean(jnp.abs(actions) > 0.99)),
            returns=returns,
            return_change=np.subtract(returns, baseline).tolist(),
            **extras,
        )
    )


def matched_record(
    rows,
    env,
    actor,
    proposal,
    hold,
    normalizer,
    seeds,
    horizon,
    baseline,
    pilot,
    critic,
    arm,
    target,
    **extras,
):
    match = match_kl(actor, proposal, hold, target)
    if match is None:
        rows.append(
            dict(
                pilot=pilot,
                critic=critic,
                arm=arm,
                target_kl=target,
                status="unreachable_kl",
                **extras,
            )
        )
        return
    model, _, fraction = match
    record(
        rows,
        env,
        actor,
        model,
        hold,
        normalizer,
        seeds,
        horizon,
        baseline,
        pilot,
        critic,
        arm,
        target_kl=target,
        fraction=fraction,
        **extras,
    )


def run_width(
    rows,
    env,
    actor,
    wide_actors,
    critic,
    temp,
    batch,
    hold,
    key,
    cdq,
    normalizer,
    seeds,
    horizon,
    baseline,
    name,
):
    # The cosine uses the Q component so entropy cannot create the measured conflict.
    for label, start in wide_actors.items():
        dist, original = (
            start(observations=hold).distribution,
            actor(observations=hold).distribution,
        )
        np.testing.assert_allclose(dist.mean(), original.mean(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            dist.stddev(), original.stddev(), rtol=1e-4, atol=1e-5
        )
        left, right = jnp.split(batch, 2)
        ga, _ = m.actor_gradient_components(
            start, critic, left, key, critic_use_cdq=cdq
        )
        gb, _ = m.actor_gradient_components(
            start, critic, right, key, critic_use_cdq=cdq
        )
        cosine = serializable(m.gradient_balance(ga, gb, 1.0))["cosine"]
        model, _ = m.propose_actor_update(
            start, critic, temp, batch, key, critic_use_cdq=cdq
        )
        matched_record(
            rows,
            env,
            start,
            model,
            hold,
            normalizer,
            seeds,
            horizon,
            baseline,
            6,
            name,
            label,
            0.001,
            gradient_cosine=cosine,
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", choices=REGIMES, default="small")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilots", type=int, choices=range(1, 7), nargs="+", default=[1, 2, 3, 5, 6]
    )
    parser.add_argument("--exp2-root", type=Path)
    parser.add_argument("--exp2-commit")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--horizon", type=int, default=500, help="Evaluation policy interactions"
    )
    parser.add_argument("--feedback-steps", type=int, default=256)
    parser.add_argument("--require-a100-40", action="store_true")
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.horizon < 1
        or args.feedback_steps < 1
        or (args.exp2_root is None) != (args.exp2_commit is None)
    ):
        parser.error(
            "use positive budgets and provide the Exp2 path and commit together"
        )
    if 4 in args.pilots and not args.exp2_root:
        parser.error("pilot 4 requires a paired Exp2 checkpoint")
    if args.output.exists():
        parser.error("output already exists")
    print(f"Exp3 worker PID {os.getpid()} writing {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.output, json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
