import unittest

from experiments.exp12.replay_exp2_qualification import validate_evidence


class QualificationEvidenceTests(unittest.TestCase):
    def test_requires_all_four_full_branches_and_both_probes(self):
        conditions = {name: {"raw_env_steps": 10000} for name in (
            "small_frozen", "small_learning", "large_frozen", "large_learning")}
        rows = [{"type": "pilot_diagnostic", "checkpoint": 10, "branch_counters": conditions},
                *({"type": "probe", "check": 10, "architecture": arch} for arch in ("small", "large"))]
        validate_evidence(rows, 10, 10000)
        with self.assertRaises(ValueError):
            validate_evidence(rows[:-1], 10, 10000)
        conditions["small_learning"]["raw_env_steps"] = 9999
        with self.assertRaises(ValueError):
            validate_evidence(rows, 10, 10000)
