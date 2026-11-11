"""Concurrency admission requires measured exact-source GPU evidence."""

import unittest
from experiments.exp12.revised_exp2_scheduler import concurrency, pair_is_qualified
from experiments.exp12.revised_exp2_population import population


class SchedulerTest(unittest.TestCase):
    def test_population(self):
        rows = population()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({tuple(sorted(row.items())) for row in rows}), 20)

    def test_measured_memory_and_exact_identity_required(self):
        gpu = {"uuid": "GPU-test", "free_mib": "15000"}
        receipt = {
            "source_commit": "source",
            "full_branch_horizon": True,
            "plasticity_probe": True,
            "restart": True,
            "concurrent_pass": True,
            "devices": {"GPU-test": {"jobs": 2, "concurrent_peak_mib": 14000}},
        }
        self.assertEqual(concurrency(gpu, 2, None, "source"), 1)
        self.assertEqual(concurrency(gpu, 2, receipt, "changed-source"), 1)
        self.assertEqual(concurrency(gpu, 2, receipt, "source"), 2)
        receipt["devices"]["GPU-test"]["concurrent_peak_mib"] = 16000
        self.assertEqual(concurrency(gpu, 2, receipt, "source"), 1)
        with self.assertRaises(ValueError):
            concurrency(gpu, 3, receipt, "source")

    def test_different_device_or_incomplete_path_falls_back(self):
        gpu = {"uuid": "GPU-test", "free_mib": "15000"}
        for missing in (
            "full_branch_horizon",
            "plasticity_probe",
            "restart",
            "concurrent_pass",
        ):
            receipt = {
                "source_commit": "source",
                "full_branch_horizon": True,
                "plasticity_probe": True,
                "restart": True,
                "concurrent_pass": True,
                "devices": {"GPU-test": {"jobs": 2, "concurrent_peak_mib": 10000}},
            }
            receipt[missing] = False
            self.assertEqual(concurrency(gpu, 2, receipt, "source"), 1)
        self.assertEqual(
            concurrency(
                {"uuid": "different", "free_mib": "15000"}, 2, receipt, "source"
            ),
            1,
        )


class WorkloadAdmissionTest(unittest.TestCase):
    def test_only_measured_pair_can_share_gpu(self):
        gpu = {"uuid": "GPU-test"}
        small = {"environment": "dog-run", "reference_regime": "reference_small"}
        large = {"environment": "dog-run", "reference_regime": "reference_large"}
        receipt = {"devices": {"GPU-test": {"qualified_workloads": [[small, large]]}}}
        self.assertTrue(pair_is_qualified(gpu, [small], large, receipt))
        self.assertTrue(pair_is_qualified(gpu, [large], small, receipt))
        self.assertFalse(pair_is_qualified(gpu, [large], large, receipt))
        self.assertFalse(pair_is_qualified(gpu, [small], small, receipt))
        myo = dict(large, environment="myo-key-turn")
        self.assertFalse(pair_is_qualified(gpu, [small], myo, receipt))
        self.assertFalse(
            pair_is_qualified({"uuid": "unmeasured"}, [small], large, receipt)
        )
        self.assertFalse(pair_is_qualified(gpu, [small], large, None))
