"""The unattended GPU jobs' report helpers: the unittest log parser and the A2/B2 range criterion (amendment (w))."""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import exp12_reports as R  # noqa: E402

FAKE_TESTS = textwrap.dedent('''
    import sys, unittest, warnings

    class T(unittest.TestCase):
        def test_a_ok(self):
            pass

        def test_b_status_after_output(self):
            print("progress line on stderr while the test runs", file=sys.stderr)
            warnings.warn("a warning printed between the test name and its status")

        def test_c_subtests(self):
            for blocks in (2, 4):
                for m in ("last", "all"):
                    with self.subTest(blocks=blocks, m=m):
                        self.assertLess(blocks, 3)

        def test_d_docstring(self):
            """A docstring shown on its own line."""

        def test_e_fail(self):
            self.assertEqual(1, 2)

        def test_f_error(self):
            raise RuntimeError("boom")

        @unittest.skip("needs humanoid_bench")
        def test_g_skip(self):
            pass

        def test_h_stdout(self):
            print(" total params: 1.54K")  # stdout is block-buffered when redirected, as on Delta
            print("more output", file=sys.stderr)
''')


def _run_fake():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "test_fake.py").write_text(FAKE_TESTS)
        r = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_fake"], cwd=d, capture_output=True,
                           text=True)
        return r.stderr + r.stdout  # stdout (buffered) lands after the run, as in A1.log


class UnittestLogTest(unittest.TestCase):
    def test_every_test_and_every_failing_subtest_is_listed_with_its_status(self):
        parsed = R.parse_unittest_log(_run_fake())
        status = dict(parsed["tests"])
        self.assertEqual(parsed["ran"], 8)
        self.assertEqual(len(parsed["tests"]), 8)
        self.assertEqual(status, {
            "test_fake.T.test_a_ok": "PASS", "test_fake.T.test_b_status_after_output": "PASS",
            "test_fake.T.test_c_subtests": "FAIL", "test_fake.T.test_d_docstring": "PASS",
            "test_fake.T.test_e_fail": "FAIL", "test_fake.T.test_f_error": "ERROR", "test_fake.T.test_g_skip": "SKIP",
            "test_fake.T.test_h_stdout": "PASS"})
        self.assertEqual([p for _, p in parsed["failures"]["test_fake.T.test_c_subtests"]],
                         ["blocks=4, m='last'", "blocks=4, m='all'"])
        self.assertEqual(parsed["warnings"], [])
        text = "\n".join(R.format_unittest(parsed))
        self.assertIn("failing subtests: 2", text)
        self.assertIn("FAIL     test_fake.T.test_e_fail", text)

    def test_the_block_a_shape_failures_counted_per_subtest(self):
        # The A1 log of job 22667743 in miniature: 'ok' pushed to a later line by output, 3 subtest
        # failures of one test, and 'FAILED (failures=4)' counting each subtest.
        log = textwrap.dedent("""\
            test_p (m.P.test_p) ... ok
            test_q (m.P.test_q) ... some warning text
            ok
            test_inj (m.I.test_inj) ...
              test_inj (m.I.test_inj) (blocks=2, m='last') ... FAIL
              test_inj (m.I.test_inj) (blocks=2, m='half') ... FAIL
              test_inj (m.I.test_inj) (blocks=4, m='all') ... FAIL
            test_kl (m.K.test_kl) ... FAIL
            ======================================================================
            FAIL: test_inj (m.I.test_inj) (blocks=2, m='last')
            ----------------------------------------------------------------------
            Traceback (most recent call last):
            AssertionError: 4.8846006e-05 not less than or equal to 1.1936714145122096e-05
            ======================================================================
            FAIL: test_inj (m.I.test_inj) (blocks=2, m='half')
            ======================================================================
            FAIL: test_inj (m.I.test_inj) (blocks=4, m='all')
            ======================================================================
            FAIL: test_kl (m.K.test_kl)
            ----------------------------------------------------------------------
            Ran 4 tests in 290.027s

            FAILED (failures=4)
             total params: 1.54K
        """)
        parsed = R.parse_unittest_log(log)
        self.assertEqual(parsed["tests"], [("m.P.test_p", "PASS"), ("m.P.test_q", "PASS"), ("m.I.test_inj", "FAIL"),
                                           ("m.K.test_kl", "FAIL")])
        self.assertEqual(len(parsed["failures"]["m.I.test_inj"]), 3)
        self.assertEqual(parsed["warnings"], [])

    def test_missing_statuses_are_flagged_not_guessed(self):
        log = "test_a (m.T.test_a) ... ok\ntest_b (m.T.test_b) ... \n"  # killed mid-run
        parsed = R.parse_unittest_log(log)
        self.assertEqual(dict(parsed["tests"])["m.T.test_b"], "UNKNOWN")
        self.assertTrue(any("did not finish" in w for w in parsed["warnings"]))


class RangeCriterionTest(unittest.TestCase):
    """Amendment (w): P/b >= 0.9 at the configured pool, every size; the 10-90% rule is information only."""

    BLOCK_A = {"D2W512": 0.9899232321227599, "D4W1024": 0.9966177937101619, "D6W1536": 0.9930110983133777}

    def _write(self, d, ratios):
        for arch, ratio in ratios.items():
            rows = [{"arch": arch, "pool_size": pool, "is_configured_pool": pool == 25600, "score_over_b": ratio,
                     "score_iqm": ratio * 0.5, "b_iqm": 0.5, "score_std": 0.01, "score_range": 0.02,
                     "final_loss_rounds": [0.004, 0.002, 0.004, 0.02, 0.001],
                     "within_10_90_pct_of_b": 0.1 <= ratio <= 0.9} for pool in (1600, 25600)]
            Path(d, f"range_{arch}.json").write_text(json.dumps(rows))

    def test_block_a_values_pass_the_new_criterion_and_fail_the_old_rule(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, self.BLOCK_A)
            c = R.range_criterion(d, list(self.BLOCK_A))
            self.assertTrue(c["pass"])
            self.assertEqual(set(c["old_rule_per_size"].values()), {False})
            self.assertEqual(R.main(["range-check", d, *self.BLOCK_A]), 0)

    def test_a_size_below_0_9_or_missing_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, {**self.BLOCK_A, "D4W1024": 0.899})
            self.assertFalse(R.range_criterion(d, list(self.BLOCK_A))["pass"])
            self.assertEqual(R.main(["range-check", d, *self.BLOCK_A]), 1)
        with tempfile.TemporaryDirectory() as d:
            self._write(d, {"D2W512": 0.99})
            c = R.range_criterion(d, list(self.BLOCK_A))
            self.assertFalse(c["pass"])
            self.assertEqual(c["missing"], ["D4W1024", "D6W1536"])


if __name__ == "__main__":
    unittest.main()
