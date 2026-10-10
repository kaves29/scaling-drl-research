"""Verdict logic of gpu_resume_probe on synthetic reports (no training).

Run from main/: python -m unittest scripts.sci_investigation.test_gpu_resume_probe
"""

import copy
import unittest

from scripts.sci_investigation import gpu_resume_probe as g


def child(exit_code=0, done=True, backend="gpu"):
    return {"exit": exit_code, "done": done, "backend": backend, "device_kinds": [], "log": "x.log", "seconds": 1.0}


def good():
    return {"reference": child(), "reference_repeat": child(), "reference_repeat_differences": [],
            "cases": {str(s): {"crash": child(g.CRASH_EXIT, False), "resume": child(), "differences": [],
                               "differences_vs_repeat": []} for s in g.CRASH_STEPS}}


class ClassifyTest(unittest.TestCase):
    def verdict(self, report, require="gpu"):
        return g.classify(report, require)[0]

    def test_clean_report_passes(self):
        self.assertEqual(g.classify(good(), "gpu"), (g.PASS, []))

    def test_resume_difference_is_a_resume_defect(self):
        r = good()
        r["cases"]["60"]["differences"] = ["agent:critic/params"]
        self.assertEqual(self.verdict(r), g.RESUME_DEFECT)

    def test_failed_relaunch_is_a_resume_defect(self):
        for exit_code, done in ((1, False), (0, False)):
            r = good()
            r["cases"]["5"]["resume"] = child(exit_code, done)
            self.assertEqual(self.verdict(r), g.RESUME_DEFECT)

    def test_differing_references_are_nondeterminism_not_a_resume_defect(self):
        r = good()
        r["reference_repeat_differences"] = ["agent:critic/params"]
        r["cases"]["95"]["differences"] = ["agent:critic/params"]
        self.assertEqual(self.verdict(r), g.NONDETERMINISTIC)

    def test_harness_failures_are_incomplete_and_take_precedence(self):
        mutations = {
            "cpu fallback": lambda r: r["cases"]["60"]["resume"].update(backend="cpu"),
            "no backend record": lambda r: r["reference"].update(backend=None),
            "reference failed": lambda r: r["reference"].update(exit=1, done=False),
            "repeat failed": lambda r: r["reference_repeat"].update(exit=1, done=False),
            "crash did not fire": lambda r: r["cases"]["5"]["crash"].update(exit=0, done=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                r = good()
                r["reference_repeat_differences"] = ["agent:x"]  # would otherwise read as nondeterminism
                mutate(r)
                self.assertEqual(self.verdict(r), g.INCOMPLETE)

    def test_backend_not_enforced_without_requirement(self):
        r = good()
        for c in [r["reference"], r["reference_repeat"]]:
            c["backend"] = "cpu"
        self.assertEqual(self.verdict(r, None), g.PASS)

    def test_exit_codes_are_distinct(self):
        self.assertEqual(sorted(g.EXIT.values()), [0, 1, 2, 3])

    def test_classify_does_not_mutate(self):
        r = good()
        r["cases"]["5"]["differences"] = ["buffer:obs"]
        before = copy.deepcopy(r)
        g.classify(r, "gpu")
        self.assertEqual(r, before)


class CliTest(unittest.TestCase):
    def test_out_must_be_new_absolute(self):
        with self.assertRaises(SystemExit):
            g.main(["--out", "relative/dir"])
        with self.assertRaises(SystemExit):
            g.main(["--out", "/"])


if __name__ == "__main__":
    unittest.main()
