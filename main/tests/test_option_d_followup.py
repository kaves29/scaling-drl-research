"""CPU-only checks for ten distinct teacher streams and fail-closed D joining."""

import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.probe import _base_targets, probe_key
from scripts.sci_investigation.combine_option_d import combine


class _RandomLinearTeacher:
    def init(self, key, observations, actions):
        return {"params": jax.random.normal(key, (observations.shape[1] + actions.shape[1],))}

    def apply(self, variables, observations, actions):
        return jnp.concatenate((observations, actions), axis=1) @ variables["params"]


class OptionDTest(unittest.TestCase):
    def test_ten_rounds_have_distinct_target_functions_and_pool_streams(self):
        teacher = _RandomLinearTeacher()
        obs = jnp.arange(40, dtype=jnp.float32).reshape(20, 2) / 7
        act = jnp.ones((20, 1), dtype=jnp.float32)
        keys, values = [], []
        for rnd in range(10):
            key, idx_key = jax.random.split(probe_key(991, 1, rnd))
            keys.append((tuple(np.asarray(key)), tuple(np.asarray(idx_key))))
            values.append(np.asarray(_base_targets(teacher, key, obs, act, 5, 1e5)))
        self.assertEqual(len(set(keys)), 10)
        self.assertTrue(all(not np.array_equal(values[0], values[i]) for i in range(1, 10)))

    def test_combiner_requires_complete_matched_pairs_and_keeps_trigger_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            first, extra = Path(directory) / "first", Path(directory) / "extra"
            first.mkdir(); extra.mkdir()
            for root, commit, start in ((first, "a" * 40, 0), (extra, "b" * 40, 5)):
                (root / "provenance.json").write_text(json.dumps({
                    "code": {"commit": commit, "dirty": False}, "mode": "null",
                    "arch": "D4W1024", "env": "dog-run", "seed": 991,
                    "round_start": start, "rounds": 5}))
                for pair in range(2):
                    row = {"pair": pair, "check_index": pair + 1, "seed": 991,
                           "null_2000": {"loss_rounds": [.01] * 5}}
                    if start:
                        row["round_indices"] = [5, 6, 7, 8, 9]
                        row["target_keys"] = [np.asarray(jax.random.split(
                            probe_key(991, pair + 1, r))[0]).tolist() for r in range(5, 10)]
                    (root / f"pair_{pair:03d}.json").write_text(json.dumps(row))
            out = combine(first, extra, "a" * 40, "b" * 40, pairs=2)
            self.assertEqual(out["our_existing_trigger"]["fired"], 2)
            self.assertEqual(len(out["per_pair"][0]["loss_rounds"]), 10)
            broken = json.loads((extra / "pair_001.json").read_text())
            broken["target_keys"][0] = broken["target_keys"][1]
            (extra / "pair_001.json").write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "target keys"):
                combine(first, extra, "a" * 40, "b" * 40, pairs=2)


if __name__ == "__main__":
    unittest.main()
