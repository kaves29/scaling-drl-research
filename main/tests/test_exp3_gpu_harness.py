"""CPU self-checks of the Exp3 GPU harness logic. Thresholds are not chosen anywhere here.

Run from main/: python -m unittest tests.test_exp3_gpu_harness -v
"""

import sys
import unittest
from pathlib import Path

import jax
import numpy as np

from experiments.exp3 import gpu_checks
from tests.exp12_helpers import compose, tiny_overrides

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp3_gpu_validation as harness  # noqa: E402


class MeasurementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = compose(tiny_overrides())
        cls.u = gpu_checks.make_agent(cfg.agent, 5, 2)
        cls.i = gpu_checks.make_agent(cfg.agent, 5, 2)
        cls.i._critic = cls.i._critic.replace(params=jax.tree_util.tree_map(lambda x: x * 1.01, cls.i._critic.params))
        cls.batch = gpu_checks.synthetic_inputs(0, 6, 5, 2)
        cls.key = jax.random.PRNGKey(0)

    def test_difference_reports_without_judging(self):
        d = gpu_checks.difference([1.0, 2.0], [1.0, 2.5])
        self.assertEqual((d["max_abs"], d["max_rel"], d["bitwise_equal"]), (0.5, 0.2, False))
        self.assertTrue(gpu_checks.difference([0.0], [0.0])["bitwise_equal"])
        self.assertTrue(gpu_checks.difference([np.nan], [0.0])["nonfinite"])
        with self.assertRaises(ValueError):
            gpu_checks.difference([1.0], [1.0, 2.0])

    def test_oracle_measurements_are_finite_and_small_on_cpu_fp32(self):
        with jax.default_matmul_precision("highest"):
            m = gpu_checks.oracle_measurements(self.u, self.i, self.batch, self.key, False)
        for name in ("per_state_jacobian_row_vs_row_grad", "panel_gradient_update_vs_update_actor"):
            self.assertFalse(m[name]["nonfinite"], name)
            self.assertLess(m[name]["max_rel"], 1e-3, name)  # CPU self-check of the harness, not a GPU criterion
        self.assertTrue(m["critic_fit_finite"])

    def test_repeat_is_bitwise_and_sensitivity_covers_every_field(self):
        self.assertTrue(gpu_checks.repeat_bitwise(self.u, self.i, self.batch["observation"], self.key, False))
        s = gpu_checks.tf32_sensitivity(self.u, self.i, self.batch["observation"], self.key, False)
        fields = gpu_checks.measurement_fields(self.u, self.i, self.batch["observation"], self.key, False)
        self.assertEqual(set(s), set(fields))
        self.assertIn("sign_flips", s["dq_da_cosine"])


class VerdictTest(unittest.TestCase):
    def report(self, repeat=True, **suite):
        row = {"tests_run": 5, "failures": 0, "errors": 0, "skips": 0, "passed": True, **suite}
        return {"exactness": {"tensorfloat32": row}, "measurements": {"repeat_bitwise": {"highest": True,
                                                                                         "tensorfloat32": repeat}}}

    def test_verdicts(self):
        v = harness.verdict
        self.assertEqual(v(self.report(), False, False), "EXACTNESS_PASS")
        self.assertEqual(v(self.report(failures=1, passed=False), False, False), "EXACTNESS_FAIL")
        self.assertEqual(v(self.report(repeat=False, failures=1, passed=False), False, False), "NONDETERMINISTIC")
        self.assertEqual(v(self.report(skips=1, passed=False), False, False), "INCOMPLETE")
        self.assertEqual(v(self.report(), False, True), "INCOMPLETE")
        self.assertEqual(v(self.report(), True, False), "CPU_DRY_RUN_EXACTNESS_PASS")
        self.assertNotIn(v(self.report(), True, False), harness.EXIT)

    def test_exactness_names_resolve(self):
        import unittest as u

        suite = u.defaultTestLoader.loadTestsFromNames(harness.EXACTNESS)
        self.assertGreater(suite.countTestCases(), 20)

    def test_required_choices_have_no_defaults(self):
        with self.assertRaises(SystemExit):
            harness.parse(["--expected-commit", "x", "--out", "/tmp/x"])


if __name__ == "__main__":
    unittest.main()
