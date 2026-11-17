"""Resumable fixed-Exp1-replay pilot; reference collection is never advanced.

Checkpoints are offline critic UPDATE counts. ``raw_env_steps`` in the reused
branch machinery is an internal checkpoint key, not a claim of new reference
interactions. Every output records that axis explicitly.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import flax.serialization
import numpy as np
import wandb
from omegaconf import OmegaConf

from experiments.exp1 import compose_config
from experiments.online_exp2 import OnlineExp2
from experiments.exp12.replay_exp2_contract import admit_parent, checkpoints, validate_parent_config
from experiments.exp12.state import latest_state_dir, load_buffer, load_meta
from experiments.exp12.precision import set_matmul_precision
from experiments.exp12.official_exp1 import ENVIRONMENTS
from utils.atomic_io import atomic_write_text


def compose_replay_config(identity):
    """Use the source Exp1 environment profile; parent parity is checked on restore."""
    environment = identity["environment"]
    group = dict(ENVIRONMENTS)[environment]
    return compose_config(str(Path(__file__).parents[2] / "configs"),
                          "online_exp2_pilot_" + identity["reference_regime"],
                          ["env=" + group, "env_name=" + environment,
                           "seed=" + str(identity["seed"])], allow_pilot=True)


class ReplayExp2Pilot(OnlineExp2):
    def __init__(self, cfg, root, parent_manifest, identity):
        state, parent = admit_parent(parent_manifest, identity)
        meta = load_meta(state)
        if meta.get("extra_state", {}).get("injection") is not None:
            raise ValueError("injected parent layout requires explicit compatibility review")
        budget = meta.get("update_step")
        if isinstance(budget, np.integer):
            budget = int(budget)
        self.update_budget = budget
        schedule = checkpoints(budget)
        super().__init__(cfg, root)
        metadata = json.loads((state.parent.parent / "run_metadata.json").read_text())
        validate_parent_config(metadata.get("resolved_config"),
                               OmegaConf.to_container(self.cfg, resolve=True))
        self.checkpoint_axis = "critic_updates"
        self.parent = parent
        self.parent_state = state
        self.checks = self.probes = schedule
        # Bind the new protocol and exact parent to existing durable restore checks.
        binding = json.dumps({"protocol": "exp1-final-replay-v1", "parent": parent,
                              "checkpoints": schedule}, sort_keys=True).encode()
        self.config_hash = hashlib.sha256(self.config_hash.encode() + binding).hexdigest()
        self.source_hash = hashlib.sha256(self.source_hash.encode() +
                                         Path(__file__).read_bytes()).hexdigest()
        # Restoring the model wrapper also restores the parent's normalizer. We do
        # not restore its logger, environments, episode windows or online progress.
        self.trainer.agent.load_checkpoint(str(state))
        load_buffer(self.trainer.buffer, state)
        if len(self.trainer.buffer) != 500000 or not self.trainer.buffer.can_sample():
            raise ValueError("parent replay endpoint/count incompatible")
        self.trainer.observations, _ = self.trainer.train_env.reset()
        self.trainer.eval_env.reset()
        self.trainer.timestep = None
        self.trainer.update_counter = meta["update_counter"]
        self.trainer.extra_state["replay_exp2_parent"] = parent
        atomic_write_text(self.root / "parent.json", json.dumps({
            **parent, "protocol": "exp1-final-replay-v1", "checkpoint_axis": "critic_updates",
            "critic_update_budget": budget, "checkpoints": schedule}, indent=2))

    def _save_shadow(self, path):
        super()._save_shadow(path)
        manifest_path = path / "online_exp2_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(protocol="exp1-final-replay-v1", parent=self.parent,
                        checkpoint_axis="critic_updates", offline_critic_updates=self.trainer.update_step)
        atomic_write_text(manifest_path, json.dumps(manifest, sort_keys=True))

    def run(self):
        latest = latest_state_dir(self.root / "state")
        if latest is None:
            self.trainer._attach_logger()
        else:
            self.restore(latest)
        trainer = self.trainer
        if wandb.run is not None:
            wandb.config.update({"protocol": "exp1-final-replay-v1", "parent": self.parent,
                                "checkpoint_axis": "critic_updates", "critic_update_budget": self.update_budget},
                               allow_val_change=True)
        from experiments.exp12.online_exp2_pilot import _reference_state
        fixed_reference = _reference_state(trainer)
        try:
            if not self.fresh_done:
                self._probe(0)
                self.fresh_done = True
                self.save()
            if trainer.extra_state.get("online_exp2_pending_checkpoint"):
                self._checkpoint(trainer)
            fixed_actor_hash = hashlib.sha256(
                flax.serialization.to_bytes(trainer._sac_agent.actor.params)
            ).hexdigest()
            last_save = time.monotonic()
            while trainer.update_step < self.update_budget:
                self._check_resource_cap()
                batch, indices = trainer.buffer.sample(return_indices=True)
                # Reuse paired SAC TD targets with a fixed parent actor and temp.
                # _shadow_updates applies the parent's frozen observation normalizer.
                self._shadow_updates(trainer, {k: v[None] for k, v in batch.items()},
                                     indices[None], None, compiled=True,
                                     fixed_actor_hash=fixed_actor_hash)
                trainer.update_step += 1
                if trainer.update_step in self.checks:
                    self._checkpoint(trainer)
                    if _reference_state(trainer) != fixed_reference:
                        raise ValueError("offline training changed fixed parent reference")
                if trainer.update_step % 1000 == 0:
                    if wandb.run is not None:
                        wandb.log({"offline/critic_updates": trainer.update_step,
                                   "offline/fraction_complete": trainer.update_step / self.update_budget})
                    if time.monotonic() - last_save >= 120:
                        self.save()
                        last_save = time.monotonic()
            self.save()
            analysis = self._pilot_analysis()
            analysis.update(protocol="exp1-final-replay-v1", parent=self.parent,
                            checkpoint_axis="critic_updates", critic_update_budget=self.update_budget)
            atomic_write_text(self.root / "analysis.json", json.dumps(analysis, allow_nan=False))
        finally:
            trainer.close()
            if wandb.run is not None:
                wandb.finish()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=json.loads, required=True)
    args = parser.parse_args()
    identity = args.identity
    set_matmul_precision()
    cfg = compose_replay_config(identity)
    ReplayExp2Pilot(cfg, str(args.output.resolve()), args.parent_manifest, identity).run()


if __name__ == "__main__":
    main()
