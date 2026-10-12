"""CPU checks for the disposable two-node Delta preflight verifier."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.exp12_dynamic_delta_preflight import _item, verify
from scripts.exp12_dynamic_scheduler import Ledger


class DeltaPreflightVerifierTest(unittest.TestCase):
    def test_one_claim_and_link_preserve_old_owner_on_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"job_id": "123", "token": "first", "host": "node-a", "pid": 1}
            self.assertTrue(
                Ledger(root, query=lambda _: True).claim(
                    _item(root), original, lambda _: None
                )
            )
            for rank, host in enumerate(("node-a", "node-b")):
                (root / f"link_source_{rank}").write_text(f"token-{rank}")
                (root / f"result_{rank}.json").write_text(
                    json.dumps(
                        {
                            "rank": rank,
                            "host": host,
                            "job_id": "123",
                            "linked": rank == 0,
                            "claimed": rank == 0,
                            "device_count": 4,
                            "cpu_affinity": 16,
                        }
                    )
                )
            (root / "atomic_link_winner").hardlink_to(root / "link_source_0")
            verify(root)
            self.assertEqual(
                json.loads((root / "verdict.json").read_text())["verdict"], "PASS"
            )
            retired = list(
                (Ledger(root)._row_dir(_item(root)) / "retired_claims").iterdir()
            )
            self.assertEqual(len(retired), 1)
            self.assertEqual(
                json.loads((retired[0] / "owner.json").read_text()), original
            )

    def test_same_node_or_two_winners_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in (0, 1):
                (root / f"result_{rank}.json").write_text(
                    json.dumps(
                        {
                            "rank": rank,
                            "host": "same-node",
                            "job_id": "123",
                            "linked": True,
                            "claimed": True,
                            "device_count": 4,
                            "cpu_affinity": 16,
                        }
                    )
                )
            with self.assertRaisesRegex(ValueError, "binding or shared atomic"):
                verify(root)
            self.assertFalse((root / "verdict.json").exists())


if __name__ == "__main__":
    unittest.main()
