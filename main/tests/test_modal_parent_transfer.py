"""Parent selection must not upload unrelated or unfinished states."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "infra/modal/transfer_parents.py"
spec = importlib.util.spec_from_file_location("transfer_parents", SCRIPT)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class ParentTransferTest(unittest.TestCase):
    def make_population(self, base):
        for env, seeds in transfer.POPULATION.items():
            for seed in seeds:
                for architecture in ("D2W512", "D4W1536"):
                    run = base / architecture / env / f"seed_{seed}"
                    state = run / "state/step_000500000_test"
                    (state / "agent_ckpt").mkdir(parents=True)
                    for name in ("meta.pkl", "buffer.npz", "buffer_meta.pkl", "obs_rms.pkl", "agent_ckpt/checkpoint"):
                        (state / name).write_bytes(b"fixture")
                    (run / "state/LATEST").write_text(state.name)
                    (run / "DONE").write_text(json.dumps({"interaction_step": 500000}))
                    (run / "run_metadata.json").write_text(json.dumps({
                        "identity": dict(architecture=architecture, environment=env, seed=seed),
                        "code": {"commit": transfer.SOURCE}, "config_hash": "fixture"}))

    def test_selects_exact_population_and_hashes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_population(base)
            (base / "unrelated-secret").write_text("do not upload")
            runs, files = transfer.inspect(base)
            self.assertEqual(len(runs), 20)
            self.assertEqual(len(files), 160)
            self.assertFalse(any("unrelated" in relative for _, relative, _ in files))

    def test_rejects_incomplete_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_population(base)
            (base / "D2W512/dog-run/seed_1/DONE").write_text('{"interaction_step": 499999}')
            with self.assertRaisesRegex(ValueError, "incomplete parent"):
                transfer.inspect(base)

    def test_rejects_escaped_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_population(base)
            (base / "D2W512/dog-run/seed_1/state/LATEST").write_text("../other")
            with self.assertRaisesRegex(ValueError, "invalid state pointer"):
                transfer.inspect(base)


if __name__ == "__main__":
    unittest.main()
