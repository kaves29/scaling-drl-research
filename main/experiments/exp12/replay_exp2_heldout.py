"""Separate held-out replay protocol; launch only after GPU qualification."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.flatten_util import ravel_pytree

from experiments.exp12.heldout_data import collect_reference_test
from experiments.exp12.heldout_td import td_residuals
from experiments.exp12.online_exp2 import gradient_cosine, paired_actor_intervention
from experiments.exp12.online_exp2_pilot import _hash
from experiments.exp12.replay_exp2_contract import checkpoints
from experiments.exp12.replay_exp2_pilot import ReplayExp2Pilot, compose_replay_config
from experiments.exp12.precision import set_matmul_precision
from utils.atomic_io import atomic_write_bytes


def prepare_output(root, identity, parent_manifest, checkpoint_count):
    """Do not touch an old pilot or resume a different held-out run."""
    from utils.atomic_io import atomic_write_text
    root = Path(root)
    marker = root / 'heldout_protocol.json'
    binding = {'protocol': 'exp2-heldout-v1', 'identity': identity,
               'parent_manifest_sha256': hashlib.sha256(Path(parent_manifest).read_bytes()).hexdigest(),
               'checkpoint_count': checkpoint_count}
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != binding:
            raise ValueError('output directory belongs to another run or protocol')
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker, json.dumps(binding, sort_keys=True))


class HeldoutReplayPilot(ReplayExp2Pilot):
    protocol = 'exp2-heldout-v1'
    validated_branch_raw_steps = 5000

    def __init__(self, cfg, root, parent_manifest, identity, checkpoint_count=15):
        prepare_output(root, identity, parent_manifest, checkpoint_count)
        if (Path(root) / 'state' / 'LATEST').exists() and not (Path(root) / 'heldout_reference.npz').is_file():
            raise ValueError('cannot resume without the original held-out dataset')
        cfg.online_exp2.branch_raw_steps = 5000
        cfg.online_exp2.measurement_raw_steps = [0, 2500, 5000]
        super().__init__(cfg, root, parent_manifest, identity)
        self.checks = self.probes = checkpoints(self.update_budget, checkpoint_count)
        self.heldout_path = self.root / 'heldout_reference.npz'
        self.heldout_sha = collect_reference_test(cfg, self.trainer, self.parent, self.heldout_path)
        binding = json.dumps({'protocol': 'exp2-heldout-v1', 'dataset_sha256': self.heldout_sha,
                              'checkpoints': self.checks}, sort_keys=True).encode()
        self.config_hash = hashlib.sha256(self.config_hash.encode() + binding).hexdigest()
        source_dir = Path(__file__).parent
        self.source_hash = hashlib.sha256(self.source_hash.encode() + b''.join(
            (source_dir / filename).read_bytes() for filename in
            ('replay_exp2_heldout.py', 'heldout_data.py', 'heldout_returns.py', 'heldout_td.py'))).hexdigest()

    def _save_shadow(self, path):
        super()._save_shadow(path)
        from utils.atomic_io import atomic_write_text
        manifest_path = path / 'online_exp2_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(heldout_sha256=self.heldout_sha, checkpoints=self.checks)
        atomic_write_text(manifest_path, json.dumps(manifest, sort_keys=True))

    def _checkpoint(self, trainer):
        checkpoint = trainer.update_step
        directory = self.root / 'heldout_diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'update_{checkpoint:08d}.npz'
        checkpoint_identity = _hash({'actor': trainer._sac_agent.actor,
                                     'small': self.shadows['small'].critic,
                                     'large': self.shadows['large'].critic,
                                     'small_target': self.shadows['small'].target,
                                     'large_target': self.shadows['large'].target,
                                     'temperature': trainer._sac_agent.temperature,
                                     'config': self.config_hash})
        if not path.exists():
            with np.load(self.heldout_path, allow_pickle=False) as data:
                obs = np.asarray(data['observation'])
                actions = np.asarray(data['action'])
                target = np.asarray(data['reward_to_go'])
                complete = np.asarray(data['complete'])
                test_batch = {k: np.asarray(data[k]) for k in
                              ('observation', 'action', 'reward', 'terminated', 'next_observation')}
            if hasattr(trainer.agent, '_normalize'):
                obs = trainer.agent._normalize(obs)
            predictions = {}
            for name, shadow in self.shadows.items():
                chunks = []
                for start in range(0, len(obs), 256):
                    q = shadow.critic(observations=jnp.asarray(obs[start:start+256]),
                                      actions=jnp.asarray(actions[start:start+256]))
                    if trainer.cfg.agent.critic_use_cdq:
                        q = jnp.minimum(*q)
                    chunks.append(np.asarray(q).reshape(-1))
                predictions[name] = np.concatenate(chunks)
                if not np.all(np.isfinite(predictions[name])):
                    raise ValueError('nonfinite held-out critic predictions')
            # One shared, deterministic held-out batch and sampling key per checkpoint.
            seed_bytes = hashlib.sha256(f'{self.heldout_sha}:{checkpoint}'.encode()).digest()
            seed = int.from_bytes(seed_bytes[:4], 'little')
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(obs), min(256, len(obs)), replace=False)
            if int(trainer.cfg.n_step) != 1:
                raise ValueError('held-out transition comparison requires n_step=1')
            replay = trainer.buffer
            train_indices = rng.choice(replay._num_in_buffer, min(256, replay._num_in_buffer), replace=False)
            train_batch = {k: np.asarray(getattr(replay, name)[train_indices]) for k, name in {
                'observation': '_observations', 'action': '_actions', 'reward': '_rewards',
                'terminated': '_terminateds', 'next_observation': '_next_observations'}.items()}
            test_batch = {k: v[indices] for k, v in test_batch.items()}
            for batch in (train_batch, test_batch):
                if hasattr(trainer.agent, '_normalize'):
                    batch['observation'] = trainer.agent._normalize(batch['observation'])
                    batch['next_observation'] = trainer.agent._normalize(batch['next_observation'])
            train_res = td_residuals(trainer._sac_agent.actor, trainer._sac_agent.temperature,
                                     self.shadows, train_batch, jax.random.PRNGKey(seed),
                                     float(trainer.cfg.gamma), 1, bool(trainer.cfg.agent.critic_use_cdq))
            test_res = td_residuals(trainer._sac_agent.actor, trainer._sac_agent.temperature,
                                    self.shadows, test_batch, jax.random.PRNGKey(seed),
                                    float(trainer.cfg.gamma), 1, bool(trainer.cfg.agent.critic_use_cdq))
            _, records = paired_actor_intervention(
                trainer._sac_agent.actor, trainer._sac_agent.temperature, self.shadows,
                obs[indices], jax.random.PRNGKey(seed), bool(trainer.cfg.agent.critic_use_cdq))
            arrays = {'dataset_sha256': np.asarray(self.heldout_sha), 'indices': indices,
                      'train_indices': train_indices,
                      'reward_to_go': target, 'complete': complete,
                      'checkpoint_identity': np.asarray(checkpoint_identity),
                      'checkpoint': np.asarray(checkpoint)}
            metrics = {'heldout/gradient_cosine': gradient_cosine(
                records['small']['gradient'], records['large']['gradient']),
                'heldout/update_cosine': gradient_cosine(
                    records['small']['parameter_delta'], records['large']['parameter_delta'])}
            for name in ('small', 'large'):
                arrays[name + '_train_td_residual'] = train_res[name]
                arrays[name + '_heldout_td_residual'] = test_res[name]
                metrics['heldout/' + name + '_train_td_mse'] = float(np.mean(train_res[name] ** 2))
                metrics['heldout/' + name + '_test_td_mse'] = float(np.mean(test_res[name] ** 2))
                arrays['prediction_' + name] = predictions[name]
                for field in ('gradient', 'parameter_delta'):
                    arrays[name + '_' + field] = np.asarray(ravel_pytree(records[name][field])[0])
                    if not np.all(np.isfinite(arrays[name + '_' + field])):
                        raise ValueError('nonfinite held-out actor diagnostic')
                metrics['heldout/' + name + '_gradient_norm'] = float(records[name]['gradient_norm'])
                metrics['heldout/' + name + '_delta_norm'] = float(records[name]['delta_norm'])
                if complete.any():
                    metrics['heldout/' + name + '_reward_return_mse_complete'] = float(
                        np.mean((predictions[name][complete] - target[complete]) ** 2))
            arrays['metrics_json'] = np.asarray(json.dumps(metrics, allow_nan=False))
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            atomic_write_bytes(path, buffer.getvalue())
        else:
            with np.load(path, allow_pickle=False) as data:
                if str(data['dataset_sha256']) != self.heldout_sha:
                    raise ValueError('held-out diagnostic dataset mismatch')
                if str(data['checkpoint_identity']) != checkpoint_identity:
                    raise ValueError('held-out diagnostic checkpoint state mismatch')
                metrics = json.loads(str(data['metrics_json']))
        if wandb.run is not None:
            wandb.log({**{k: v for k, v in metrics.items() if v is not None},
                       'heldout/critic_updates': checkpoint})
        super()._checkpoint(trainer)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity', type=json.loads, required=True)
    parser.add_argument('--checkpoint-count', type=int, choices=(15, 20), default=15)
    args = parser.parse_args()
    set_matmul_precision()
    cfg = compose_replay_config(args.identity)
    HeldoutReplayPilot(cfg, str(args.output.resolve()), args.parent_manifest,
                       args.identity, args.checkpoint_count).run()


if __name__ == '__main__':
    main()
