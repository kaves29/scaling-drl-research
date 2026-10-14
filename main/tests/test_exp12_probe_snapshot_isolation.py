"""The plasticity fit must leave the live trainer state untouched."""

import hashlib
import json
import pickle
import random
import tempfile
import unittest
from unittest import mock

import jax
import numpy as np

from tests.exp12_helpers import compose, patch_wandb, tiny_overrides
from experiments.exp12.envs import env_restore_state
from experiments.exp12.run_probes import RunProbes
from experiments.exp12.trainer import Exp12Trainer


def _digest(value):
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, (np.ndarray, np.generic, jax.Array)):
            array = np.asarray(jax.device_get(item))
            h.update(b"array\0" + array.dtype.str.encode() + b"\0")
            h.update(json.dumps(array.shape).encode() + b"\0" + array.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict\0")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b"\0")
            for child in item:
                add(child)
        else:
            h.update(b"scalar\0" + pickle.dumps(item, protocol=4))

    add(value)
    return h.hexdigest()


def _live_state(trainer):
    agent = trainer._sac_agent
    buffer = trainer.buffer
    network = {"actor": agent._actor, "critic": agent._critic,
               "target_critic": agent._target_critic, "temperature": agent._temperature,
               "rng": agent._rng, "churn_ref_batch": agent.churn_ref_batch}
    state = {
        "network": {str(path): _digest(leaf)
                    for path, leaf in jax.tree_util.tree_leaves_with_path(network)},
        "normalizer": _digest(trainer.agent.obs_rms.__dict__),
        "replay_counters": _digest((buffer._num_in_buffer, buffer._current_idx,
                                     list(buffer._n_step_transitions))),
        "numpy_rng": _digest(np.random.get_state()),
        "python_rng": _digest(random.getstate()),
        "train_env": _digest(env_restore_state(trainer.train_env)),
        "eval_env": _digest(env_restore_state(trainer.eval_env)),
        "interaction_step": trainer.interaction_step,
        "update_step": trainer.update_step,
        "update_counter": trainer.update_counter,
    }
    for key in ("_observations", "_actions", "_rewards", "_terminateds", "_truncateds",
                "_next_observations"):
        state["replay" + key] = _digest(getattr(buffer, key)[:buffer._num_in_buffer])
    return state


class ProbeSnapshotIsolationTest(unittest.TestCase):
    def test_fresh_and_trained_probe_leave_live_state_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = compose(tiny_overrides(steps=400))
            with mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
                patchers = patch_wandb()
                try:
                    trainer = Exp12Trainer(cfg, tmp)
                    trainer.start()
                    probes = RunProbes(trainer, tmp)

                    def fresh(t):
                        before = _live_state(t)
                        probes.capture_fresh(t)
                        self.assertEqual(before, _live_state(t))

                    def trained(t):
                        if t.interaction_step == 20:
                            before = _live_state(t)
                            probes.maybe_check(t)
                            self.assertEqual(before, _live_state(t))

                    trainer.train(20, before_first_update=fresh, after_step=trained)
                    self.assertEqual([r["check_index"] for r in probes.records], [0, 1])
                    trainer.close()
                finally:
                    for patcher in patchers:
                        patcher.stop()
