import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.exp12.online_exp2_pilot import _hash, _reference_state
from experiments.exp12.replay_exp2_heldout import HeldoutReplayPilot, prepare_output
from experiments.online_exp2 import OnlineExp2
from tests.test_revised_exp2_runner import patches, revised_cfg


def test_output_guard_preserves_prior_pilot(tmp_path):
    manifest = tmp_path / 'parents.json'
    manifest.write_text('{}')
    output = tmp_path / 'old-pilot'
    output.mkdir()
    (output / 'parent.json').write_text('old')
    with pytest.raises(ValueError, match='another run'):
        prepare_output(output, {'seed': 1}, manifest, 15)
    assert (output / 'parent.json').read_text() == 'old'
    new = tmp_path / 'new-pilot'
    prepare_output(new, {'seed': 1}, manifest, 15)
    prepare_output(new, {'seed': 1}, manifest, 15)
    with pytest.raises(ValueError, match='another run'):
        prepare_output(new, {'seed': 2}, manifest, 15)


def test_heldout_checkpoint_is_read_only_and_reusable(tmp_path):
    with patches():
        runner = object.__new__(HeldoutReplayPilot)
        OnlineExp2.__init__(runner, revised_cfg(), str(tmp_path))
        trainer = runner.trainer
        trainer.start()
        for i in range(8):
            trainer.buffer.add({'observation': np.full((1, 3), i / 10, np.float32),
                                'next_observation': np.full((1, 3), (i + 1) / 10, np.float32),
                                'action': np.zeros((1, 2), np.float32), 'reward': np.array([1.], np.float32),
                                'terminated': np.array([False]), 'truncated': np.array([False])})
        runner.heldout_path = Path(tmp_path) / 'test.npz'
        np.savez(runner.heldout_path, observation=np.full((5, 3), .3, np.float32),
                 next_observation=np.full((5, 3), .4, np.float32), action=np.zeros((5, 2), np.float32),
                 reward=np.ones(5), terminated=np.zeros(5), reward_to_go=np.ones(5), complete=np.ones(5, bool))
        runner.heldout_sha = hashlib.sha256(runner.heldout_path.read_bytes()).hexdigest()
        before = _reference_state(trainer)
        shadows = {k: _hash(v.critic) for k, v in runner.shadows.items()}
        with patch('experiments.exp12.replay_exp2_pilot.ReplayExp2Pilot._checkpoint'):
            runner._checkpoint(trainer)
            runner._checkpoint(trainer)
        assert before == _reference_state(trainer)
        assert shadows == {k: _hash(v.critic) for k, v in runner.shadows.items()}
        path = Path(tmp_path) / 'heldout_diagnostics/update_00000000.npz'
        assert path.exists()
        runner.heldout_sha = 'wrong'
        with patch('experiments.exp12.replay_exp2_pilot.ReplayExp2Pilot._checkpoint'):
            with pytest.raises(ValueError, match='dataset mismatch'):
                runner._checkpoint(trainer)
        trainer.close()


def test_short_checkpoint_runs_all_branches_and_probes(tmp_path):
    import pickle
    from experiments.exp12.online_exp2_pilot import CONDITIONS

    with patches():
        runner = object.__new__(HeldoutReplayPilot)
        OnlineExp2.__init__(runner, revised_cfg(), str(tmp_path))
        trainer = runner.trainer
        trainer.start()
        for i in range(8):
            trainer.buffer.add({'observation': np.full((1, 3), i / 10, np.float32),
                                'next_observation': np.full((1, 3), (i + 1) / 10, np.float32),
                                'action': np.zeros((1, 2), np.float32), 'reward': np.array([1.], np.float32),
                                'terminated': np.array([False]), 'truncated': np.array([False])})
        runner.parent = {'fixture': True}
        runner.update_budget = 1
        runner.checkpoint_axis = 'critic_updates'
        runner.checks = runner.probes = [1]
        trainer.update_step = 1
        runner.heldout_path = Path(tmp_path) / 'test.npz'
        np.savez(runner.heldout_path, observation=np.full((5, 3), .3, np.float32),
                 next_observation=np.full((5, 3), .4, np.float32), action=np.zeros((5, 2), np.float32),
                 reward=np.ones(5), terminated=np.zeros(5), reward_to_go=np.ones(5), complete=np.ones(5, bool))
        runner.heldout_sha = hashlib.sha256(runner.heldout_path.read_bytes()).hexdigest()
        before = _reference_state(trainer)
        runner._checkpoint(trainer)
        state = pickle.loads((Path(tmp_path) / 'diagnostics/update_00000001/progress.pkl').read_bytes())
        assert set(state['branches']) == {a + '_' + b for a, b in CONDITIONS}
        assert state['probe_done']
        assert all(branch['raw_env_steps'] == 10 for branch in state['branches'].values())
        assert before == _reference_state(trainer)
        runner._checkpoint(trainer)
        assert before == _reference_state(trainer)
        trainer.close()
