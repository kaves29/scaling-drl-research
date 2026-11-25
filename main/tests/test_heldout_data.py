import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from experiments.exp12 import heldout_data


class Distribution:
    def sample(self, seed):
        return np.array([[0.25]])


class FakeEnv:
    def __init__(self):
        self.step_count = 0
    def reset(self):
        return np.array([[0.0]]), {}
    def step(self, action):
        self.step_count += 1
        done = self.step_count % 2 == 0
        return np.array([[self.step_count]], dtype=float), np.array([1.0]), np.array([done]), np.array([False]), {}
    def close(self):
        pass


def test_data_are_fixed_and_parent_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(heldout_data, '_branch_env', lambda cfg, seed: FakeEnv())
    monkeypatch.setattr(heldout_data, 'limit_raw_steps', lambda env, remaining: None)
    monkeypatch.setattr(heldout_data, 'executed_raw_steps', lambda env: 1)
    cfg = OmegaConf.create({'env_name': 'dog-run', 'seed': 1, 'gamma': 0.5,
                            'online_exp2': {'reference_regime': 'reference_small'}})
    trainer = SimpleNamespace(agent=object(), _sac_agent=SimpleNamespace(actor=lambda **kw: Distribution()))
    path = tmp_path / 'heldout.npz'
    state = np.random.get_state()
    py_state = random.getstate()
    digest = heldout_data.collect_reference_test(cfg, trainer, {'id': 1}, path, raw_steps=5)
    assert digest == heldout_data.collect_reference_test(cfg, trainer, {'id': 1}, path, raw_steps=5)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    with np.load(path, allow_pickle=False) as data:
        assert sum(data['raw_steps']) == 5
        np.testing.assert_array_equal(data['complete'], [True, True, True, True, False])
        assert json.loads(str(data['identity_json']))['parent'] == {'id': 1}
    with pytest.raises(ValueError, match='identity'):
        heldout_data.collect_reference_test(cfg, trainer, {'id': 2}, path, raw_steps=5)
