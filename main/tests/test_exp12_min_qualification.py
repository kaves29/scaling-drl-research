"""CPU tests for the reduced-queue qualification tooling (no GPU, no Slurm submission)."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))

from scripts import exp12_numeric_bounds as bounds  # noqa: E402
from scripts import exp12_qualification_commands as commands  # noqa: E402
from scripts import exp12_shared_fs_claim_check as claim_check  # noqa: E402
from scripts.exp12_dynamic_scheduler import Ledger, _read  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
NO_JAX = (
    "import sys, runpy; sys.modules['jax'] = None; sys.argv = ['x'] + sys.argv[1:]; "
    f"runpy.run_path({str(MAIN / 'scripts/exp12_shared_fs_claim_check.py')!r}, run_name='__main__')"
)


def _workers(root, items):
    """Two concurrent worker processes that cannot import JAX."""
    procs = [subprocess.Popen([sys.executable, "-c", NO_JAX, "worker", "--root", str(root),
                               "--rank", str(rank), "--items", str(items)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for rank in (0, 1)]
    return [(p.wait(timeout=300), p.stderr.read()) for p in procs]


class SharedFilesystemClaimCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.root = cls.tmp / "race"
        cls.root.mkdir()
        cls.items = 150
        cls.exits = _workers(cls.root, cls.items)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _copy(self, name):
        copy = self.tmp / name
        inodes = {}

        def copy_file(source, destination):
            stat = os.stat(source)
            key = (stat.st_dev, stat.st_ino)
            if key in inodes:
                os.link(inodes[key], destination)
                return destination
            result = shutil.copy2(source, destination)
            inodes[key] = destination
            return result

        shutil.copytree(self.root, copy, copy_function=copy_file)
        return copy

    def test_workers_run_without_jax(self):
        for code, err in self.exits:
            self.assertEqual(code, 0, err[-2000:])

    def test_every_row_has_exactly_one_winner_and_recovery_preserves_the_old_claim(self):
        report = claim_check.verify(self._copy("ok"), require_distinct_hosts=False)
        self.assertEqual(report["verdict"], "PASS", report["failures"])
        self.assertGreater(report["race_overlap_seconds"], 0)
        results = [_read(self.root / f"result_{r}.json") for r in (0, 1)]
        for index in range(self.items):
            self.assertEqual(sum(r["rows"][index]["claimed"] for r in results), 1)
            self.assertEqual(sum(r["rows"][index]["linked"] for r in results), 1)

    def test_handoff_rows_are_finished_by_their_owner_and_never_reclaimed(self):
        report = claim_check.verify(self._copy("handoff"), require_distinct_hosts=False)
        self.assertEqual(report["handoff_rows"], self.items // claim_check.HANDOFF_FRACTION)
        self.assertTrue(all(n > 0 for n in report["handoff_attempts"]))
        self.assertEqual(len(report["finish_guard_conflicts"]), 2)
        for rank in (0, 1):
            handoff = _read(self.root / f"result_{rank}.json")["handoff"]
            self.assertEqual(handoff["wrongly_claimed"], [])
            self.assertEqual(handoff["finished"], list(range(rank, report["handoff_rows"], 2)))

    def test_a_reclaimed_finished_row_or_a_missing_completion_is_detected(self):
        copy = self._copy("reclaimed")
        path = copy / "result_1.json"
        result = json.loads(path.read_text())
        result["handoff"]["wrongly_claimed"] = [0]
        path.write_text(json.dumps(result))
        (Ledger(copy)._row_dir(claim_check._handoff_item(copy, 3)) / "completed.json").unlink()
        failures = claim_check.verify(copy, require_distinct_hosts=False)["failures"]
        self.assertTrue(any("claimed finished rows [0]" in f for f in failures), failures)
        self.assertIn("handoff row 3: not completed", failures)

    def test_one_host_is_never_a_pass(self):
        report = claim_check.verify(self._copy("same_host"))
        self.assertEqual(report["verdict"], "FAIL")
        self.assertTrue(any("same host" in f for f in report["failures"]))
        self.assertFalse((self.tmp / "same_host" / "verdict.json").exists())  # verify never writes

    def test_a_double_claim_is_detected(self):
        copy = self._copy("double")
        path = copy / "result_1.json"
        result = json.loads(path.read_text())
        loser = next(row for row in result["rows"] if not row["claimed"])
        loser["claimed"] = True
        path.write_text(json.dumps(result))
        report = claim_check.verify(copy, require_distinct_hosts=False)
        self.assertIn(f"row {loser['index']}: 2 ledger claim winners", report["failures"])

    def test_a_wrong_link_owner_a_left_guard_and_a_temporary_file_are_detected(self):
        copy = self._copy("damaged")
        winner = copy / "links" / "winner_0003"
        winner.unlink()
        loser = 1 - next(r for r in (0, 1)
                         if _read(copy / f"result_{r}.json")["rows"][3]["linked"])
        os.link(copy / "links" / f"source_0003_{loser}", winner)
        row = Ledger(copy)._row_dir(claim_check._item(copy, 5))
        (row / "transition_guard").mkdir()
        (copy / "stray.json.abc.tmp").write_text("")
        failures = claim_check.verify(copy, require_distinct_hosts=False)["failures"]
        self.assertTrue(any(f.startswith("row 3:") and "hard link" in f for f in failures), failures)
        self.assertIn("row 5: a transition guard was left behind", failures)
        self.assertTrue(any("temporary files" in f for f in failures))

    def test_different_filesystems_fail(self):
        copy = self._copy("mounts")
        first_path = copy / "ready_0.json"
        first = json.loads(first_path.read_text())
        first["mount"] = first["mount"] or {
            "fstype": "lustre",
            "device": "shared",
            "mount_point": "/",
            "options": "",
        }
        first_path.write_text(json.dumps(first))
        path = copy / "ready_1.json"
        ready = json.loads(path.read_text())
        ready["mount"] = {
            **(ready["mount"] or {}),
            "fstype": "tmpfs",
            "device": "other",
            "mount_point": "/",
            "options": "",
        }
        path.write_text(json.dumps(ready))
        report = claim_check.verify(copy, require_distinct_hosts=False)
        self.assertIn(
            "the two hosts mount the root from different filesystems",
            report["failures"],
        )

    def test_mismatched_item_counts_are_refused(self):
        root = self.tmp / "mismatch"
        root.mkdir()
        procs = [subprocess.Popen([sys.executable, "-c", NO_JAX, "worker", "--root", str(root),
                                   "--rank", str(rank), "--items", str(items)],
                                  stderr=subprocess.PIPE, text=True)
                 for rank, items in ((0, 5), (1, 6))]
        errors = [p.communicate(timeout=120)[1] for p in procs]
        self.assertTrue(all(p.returncode != 0 for p in procs))
        self.assertTrue(any("different --items" in e for e in errors), errors)

    def test_the_cpu_wrapper_is_valid_bash_and_never_requests_a_gpu(self):
        text = (MAIN / "scripts/sbatch_exp12_shared_fs_claim_check.sh").read_text()
        subprocess.run(["bash", "-n", str(MAIN / "scripts/sbatch_exp12_shared_fs_claim_check.sh")], check=True)
        self.assertNotRegex(text, r"--gpus|gpus-per|#SBATCH --partition=gpu|import jax")
        self.assertIn("exp12_check_checkout", text)
        self.assertNotIn("\nsbatch ", text)


class QualificationCommandsTest(unittest.TestCase):
    def _render(self, **kw):
        return commands.render(SHA, "/w/checkout", "/w/qual", "/w/hb_env", **kw)

    def test_six_cold_cases_with_sized_walltimes_and_priority_first(self):
        text = self._render()
        cases = re.findall(r"IDENTITY_CASE=(\S+) IDENTITY_CACHE_MODE=cold", text)
        self.assertEqual(len(cases), 6)
        self.assertEqual(set(cases), {"D4W1024_dog-run", "D4W1536_dog-run", "D4W1024_myo-key-turn",
                                      "D4W1536_myo-key-turn", "D4W1024_h1-run-v0", "D4W1536_h1-run-v0"})
        self.assertEqual(cases[:2], list(commands.PRIORITY))
        times = dict(commands.IDENTITY_CASES)
        for case in cases:
            limit = "03:00:00" if "h1-run" in case else "01:30:00"
            self.assertEqual(times[case], limit)
            self.assertLess(commands._walltime_minutes(times[case]), 4 * 60)
        self.assertEqual(text.count("scripts/sbatch_exp12_identity_case.sh"), 6)
        self.assertEqual(text.count("--cpus-per-task=16 --mem=64G"), 7)
        self.assertIn("scripts/sbatch_exp12_numeric_bounds.sh", text)
        self.assertIn(f"--time={commands.NUMERICS_TIME}", text)
        self.assertNotIn("claim_check", text)

    def test_generated_script_is_valid_bash_and_guards_source_and_outputs(self):
        text = self._render(cpu_partition="cpu", cpu_account="acct-cpu")
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as f:
            f.write(text)
            f.flush()
            subprocess.run(["bash", "-n", f.name], check=True)
        self.assertIn('test "$(git -C "$FINAL_CHECKOUT" rev-parse HEAD)" = "$FINAL_SHA"', text)
        self.assertEqual(len(re.findall(r"^submit_once ", text, re.M)), 8)
        self.assertIn("--partition=cpu", text)
        self.assertIn("scripts/sbatch_exp12_shared_fs_claim_check.sh", text)

    def test_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            commands.render("abc", "/c", "/q", "/h")
        with self.assertRaises(ValueError):
            commands.render(SHA, "relative", "/q", "/h")
        with self.assertRaises(ValueError):
            commands.render(SHA, "/c", "/q", "/h", cpu_partition="cpu")

    def test_the_tool_writes_a_file_and_cannot_submit(self):
        source = (MAIN / "scripts/exp12_qualification_commands.py").read_text()
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "submit.sh"
            self.assertEqual(commands.main(["--final-sha", SHA, "--checkout", "/c", "--qual-root", "/q",
                                            "--hb-env", "/h", "--out", str(out)]), 0)
            self.assertTrue(out.read_text().startswith("#!/usr/bin/env bash"))
            with self.assertRaises(FileExistsError):
                commands.main(["--final-sha", SHA, "--checkout", "/c", "--qual-root", "/q",
                               "--hb-env", "/h", "--out", str(out)])


class NumericBoundsPortTest(unittest.TestCase):
    def test_numeric_bound_rule_rounds_ten_times_the_maximum_up(self):
        from tests.test_exp12_review_blockers import COMMIT, _numerics

        def deviations(mode, rows):
            return [{**r, "deviation": d} for r, d in zip(rows, (1.1e-7, 2.3e-7, 2.3e-7))]

        with tempfile.TemporaryDirectory() as directory:
            report = bounds.propose(_numerics(Path(directory) / "job", edit=deviations), COMMIT)
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["status"], "PROPOSED_NOT_APPLIED")
        self.assertEqual(report["bounds"]["diagnostics_update/tf32"]["proposed_bound"], 3e-06)

    def test_one_digit_rounding(self):
        self.assertEqual(bounds.one_digit_up(0.0), 0.0)
        self.assertEqual(bounds.one_digit_up(3e-6), 3e-6)
        self.assertEqual(bounds.one_digit_up(3.01e-6), 4e-6)

    def test_names_match_the_unset_tolerances_which_stay_unset(self):
        sys.path.insert(0, str(MAIN / "tests"))
        from exp12_helpers import GPU_TOLERANCES

        self.assertEqual(set(bounds.NAMES), set(GPU_TOLERANCES))
        self.assertTrue(all(value is None for value in GPU_TOLERANCES.values()))

    def test_the_job_runs_existing_tests_in_both_modes_and_never_submits(self):
        path = MAIN / "scripts/sbatch_exp12_numeric_bounds.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        names = re.findall(r"(tests\.test_exp12_\w+\.\w+\.\w+)", text)
        self.assertEqual(len(names), 3)
        loader = unittest.TestLoader()
        for name in names:
            self.assertEqual(loader.loadTestsFromName(name).countTestCases(), 1, name)
        self.assertIn("XLA_FLAGS=--xla_gpu_deterministic_ops=true", text)
        self.assertIn("unset JAX_PLATFORM_NAME JAX_DEFAULT_MATMUL_PRECISION NVIDIA_TF32_OVERRIDE XLA_FLAGS", text)
        self.assertNotIn("\nsbatch ", text)


class ResumePerturbationStudyTest(unittest.TestCase):
    def test_perturbation_moves_every_float32_leaf_by_one_ulp_or_the_relative_scale(self):
        import numpy as np

        from scripts import exp12_resume_perturbation_study as study

        tree = {"w": np.linspace(-2, 3, 50, dtype=np.float32).reshape(5, 10),
                "step": np.int32(7), "b": np.array([0.5, -0.25], np.float32)}
        moved = study._perturb(tree, np.random.default_rng(0))
        for key in ("w", "b"):
            a, m = tree[key], np.asarray(moved[key])
            self.assertEqual(m.dtype, np.float32)
            ulps = np.abs(m.view(np.int32).astype(np.int64) - a.view(np.int32).astype(np.int64))
            nonzero = a != 0
            self.assertTrue(np.all(ulps[nonzero] == 1), key)
        self.assertEqual(int(moved["step"]), 7)
        scaled = np.asarray(study._perturb(tree, np.random.default_rng(1), 1e-3)["w"])
        ratio = scaled[tree["w"] != 0] / tree["w"][tree["w"] != 0]
        np.testing.assert_allclose(np.abs(ratio - 1), 1e-3, rtol=1e-3)
        self.assertTrue(np.any(ratio > 1) and np.any(ratio < 1))

    def test_summary_pairs_by_seed(self):
        from scripts import exp12_resume_perturbation_study as study

        rows = [{"seed": s,
                 "uninterrupted": {"post_resume_eval_return": float(s), "post_resume_probe_loss_iqm": 1.0},
                 "perturbed": {"post_resume_eval_return": s + 0.5, "post_resume_probe_loss_iqm": 1.0,
                               "critic_relative_distance": 0.1}} for s in range(6)]
        rows[0]["exact"] = {"critic_relative_distance": 0.0, "post_resume_eval_return": 0.0}
        summary = study.summarize(rows, resamples=200)
        returns = summary["post_resume_eval_return"]
        self.assertAlmostEqual(returns["paired_difference_mean"], 0.5)
        self.assertEqual(returns["paired_difference_ci95"], [0.5, 0.5])
        self.assertEqual(summary["post_resume_probe_loss_iqm"]["seeds_changed"], 0)
        self.assertTrue(summary["exact_resumes_identical"])
        self.assertEqual(summary["exact_resume_count"], 1)


class HumanoidBenchRunnerTest(unittest.TestCase):
    PATH = MAIN / "scripts/run_exp12_hb_cpu_restore_checks.sh"

    def test_runner_is_valid_bash_and_every_listed_test_exists(self):
        subprocess.run(["bash", "-n", str(self.PATH)], check=True)
        text = self.PATH.read_text()
        block = text[text.index("tests=(") + 7:text.index(")", text.index("tests=("))]
        names = block.split()
        self.assertIn("tests.test_exp12_hb_cpu_restore", names)
        loader = unittest.TestLoader()
        for name in names:
            self.assertGreater(loader.loadTestsFromName(name).countTestCases(), 0, name)
        self.assertIn("JAX_PLATFORMS=cpu", text)
        self.assertNotRegex(text, r"\bsbatch\b|\bsrun\b")

    def test_runner_refuses_the_wrong_commit_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            result = subprocess.run(["bash", str(self.PATH)], capture_output=True, text=True,
                                    env=dict(os.environ, EXPECTED_COMMIT=SHA, HB_ENV="/nonexistent",
                                             HB_CHECK_ROOT=str(root)))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
