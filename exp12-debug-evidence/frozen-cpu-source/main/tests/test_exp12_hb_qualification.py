"""CPU checks for Block HB bookkeeping; these do not claim CUDA qualification."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_exp12_hb_environment import HB_COMMIT, validate
from scripts.check_exp12_hb_status import check


class HbQualificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        (self.out / "status.tsv").write_text("tests\t0\t1s\nspeed_h1-run-v0\t0\t1s\n")
        (self.out / "tests.log").write_text("Ran 1 test in 1.0s\n\nOK\n")
        self.speed = [
            dict(
                arch="D2W512",
                train_it_per_s_probes_off=1.0,
                probe_check_s=1.0,
                probe_overhead_pct_of_wallclock=1.0,
                peak_device_bytes=1,
            )
        ]
        self.write_speed()

    def tearDown(self):
        self.temp.cleanup()

    def write_speed(self):
        (self.out / "speed_h1-run-v0.json").write_text(json.dumps(self.speed))

    def test_complete_bookkeeping_passes(self):
        self.assertTrue(check(self.out, [], ["D2W512"])["pass"])

    def test_each_required_failure_and_timeout_propagates(self):
        for name in ("tests", "speed_h1-run-v0"):
            for code in ("1", "124", "137"):
                with self.subTest(name=name, code=code):
                    (self.out / "status.tsv").write_text(
                        "".join(
                            f"{n}\t{code if n==name else 0}\t1s\n"
                            for n in ("tests", "speed_h1-run-v0")
                        )
                    )
                    self.assertFalse(check(self.out, [], ["D2W512"])["pass"])

    def test_missing_duplicate_and_extra_steps_fail(self):
        for text in (
            "tests\t0\t1s\n",
            "tests\t0\t1s\ntests\t0\t1s\n",
            "tests\t0\t1s\nspeed_h1-run-v0\t0\t1s\nextra\t0\t1s\n",
        ):
            (self.out / "status.tsv").write_text(text)
            self.assertFalse(check(self.out, [], ["D2W512"])["pass"])

    def test_skips_and_no_tests_fail_qualification(self):
        for log in (
            "Ran 1 test in 1s\nOK (skipped=1)\n",
            "Ran 0 tests in 0s\nOK\n",
            "",
        ):
            (self.out / "tests.log").write_text(log)
            self.assertFalse(check(self.out, [], ["D2W512"])["pass"])

    def test_missing_nonfinite_and_duplicate_profile_values_fail(self):
        for value in (None, float("nan"), float("inf"), -1):
            self.speed[0]["peak_device_bytes"] = value
            self.write_speed()
            self.assertFalse(check(self.out, [], ["D2W512"])["pass"])
        self.speed[0]["peak_device_bytes"] = 1
        self.speed.append(self.speed[0])
        self.write_speed()
        self.assertFalse(check(self.out, [], ["D2W512"])["pass"])

    def test_identity_requires_pass_and_both_1000_step_snapshots(self):
        (self.out / "identity").mkdir()
        rows = (self.out / "status.tsv").read_text()
        for env in ("dog-run", "myo-key-turn", "h1-run-v0"):
            name = f"D4W1024_{env}"
            rows += f"identity_{name}\t0\t1s\n"
            (self.out / "identity" / f"{name}.json").write_text(
                json.dumps(
                    dict(
                        **{"pass": True},
                        differences=[],
                        fork_step=12000,
                        interaction_step=13000,
                        identity_interaction_step=13000,
                    )
                )
            )
        (self.out / "status.tsv").write_text(rows)
        self.assertTrue(check(self.out, ["4:1024"], ["D2W512"])["pass"])
        path = self.out / "identity/D4W1024_h1-run-v0.json"
        data = json.loads(path.read_text())
        for field, value in [
            ("pass", False),
            ("interaction_step", 12999),
            ("identity_interaction_step", 12999),
            ("differences", ["changed"]),
        ]:
            changed = {**data, field: value}
            path.write_text(json.dumps(changed))
            self.assertFalse(check(self.out, ["4:1024"], ["D2W512"])["pass"])

    def test_actual_shell_exit_propagates_required_failures(self):
        driver = Path(__file__).resolve().parents[1] / "scripts/sbatch_exp12_hb.sh"
        checker = driver.with_name("check_exp12_hb_status.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            shutil.copy(driver, root / "scripts" / driver.name)
            shutil.copy(checker, root / "scripts" / checker.name)
            (root / "configs").mkdir()
            (root / "configs/base_exp12.yaml").touch()
            (root / "run.py").touch()
            (root / "bin").mkdir()
            (root / "hb/bin").mkdir(parents=True)
            git = root / "bin/git"
            git.write_text(
                '#!/bin/sh\nif [ "$1" = rev-parse ]; then echo "$FAKE_HEAD"; fi\n'
            )
            git.chmod(0o755)
            timeout = root / "bin/timeout"
            timeout.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
            timeout.chmod(0o755)
            fake = (
                "#!"
                + sys.executable
                + "\n"
                + r"""
import json, os, pathlib, sys
args=sys.argv[1:]
if args[0].endswith("check_exp12_hb_status.py") or args[0] == "-c":
    os.execv(sys.executable, [sys.executable, *args])
elif args[0].endswith("check_exp12_hb_environment.py"):
    pathlib.Path(args[args.index("--out")+1]).write_text("{}")
elif args[:2] == ["-m", "unittest"]:
    print("Ran 1 test in 1.0s\n\n" + ("OK (skipped=1)" if os.environ.get("FAKE_SKIP") else "OK"))
    sys.exit(int(os.environ.get("FAKE_TEST_CODE", "0")))
elif args[0].endswith("profile_exp12.py"):
    if not os.environ.get("FAKE_MISSING_PROFILE"):
        pathlib.Path(args[args.index("--out")+1]).write_text(json.dumps([
            dict(arch="D2W512",train_it_per_s_probes_off=1,probe_check_s=1,
                 probe_overhead_pct_of_wallclock=1,peak_device_bytes=1)]))
else:
    sys.exit(99)
"""
            )
            for path in (root / "bin/python", root / "hb/bin/python"):
                path.write_text(fake)
                path.chmod(0o755)
            hooks = root / "hooks.sh"
            env = {
                **os.environ,
                "PATH": str(root / "bin") + ":" + os.environ["PATH"],
                "HB_ENV": str(root / "hb"),
                "EXPECTED_COMMIT": "1" * 40,
                "EXPECTED_GPU_MODEL": "SYNTHETIC",
                "FAKE_HEAD": "1" * 40,
                "HB_TEST_HOOKS": str(hooks),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
            }
            env.pop("SLURM_JOB_ID", None)
            cases = [
                ({}, 0),
                ({"FAKE_TEST_CODE": "7"}, 1),
                ({"FAKE_TEST_CODE": "124"}, 1),
                ({"FAKE_MISSING_PROFILE": "1"}, 1),
                ({"FAKE_SKIP": "1"}, 1),
                ({"FAKE_HEAD": "2" * 40}, 2),
            ]
            for index, (change, code) in enumerate(cases):
                out = root / f"case_{index}"
                hooks.write_text(
                    f'OUT="{out}"\nIDENTITY_ARCHS=""\nSPEED_ARCHS="D2W512"\n'
                )
                result = subprocess.run(
                    ["bash", str(root / "scripts" / driver.name)],
                    env={**env, **change},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                with self.subTest(change=change):
                    self.assertEqual(result.returncode, code, result.stdout)
                    if code != 2:
                        qualification = json.loads(
                            (out / "qualification.json").read_text()
                        )
                        self.assertEqual(qualification["pass"], code == 0)

    def test_environment_contract(self):
        main = {
            "devices": [dict(platform="gpu", device_kind="SYNTHETIC")],
            "packages": {"jax": "1"},
            "executable": "/main/python",
            "prefix": "/main",
        }
        hb = {
            **copy.deepcopy(main),
            "executable": "/hb/python",
            "prefix": "/hb",
            "hb_commit": HB_COMMIT,
            "hb_dirty": False,
        }
        validate(main, "SYNTHETIC")
        validate(hb, "SYNTHETIC", main)
        for field, value in [
            ("devices", [dict(platform="cpu", device_kind="SYNTHETIC")]),
            ("devices", [dict(platform="gpu", device_kind="wrong")]),
            ("packages", {"jax": "2"}),
            ("prefix", "/main"),
            ("hb_commit", "wrong"),
            ("hb_dirty", True),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate({**hb, field: value}, "SYNTHETIC", main)

    def test_slurm_spool_copy_uses_submit_checkout(self):
        """Execute the complete copied script as Slurm does, stopping at module setup.

        No Slurm, cluster files, conda or workloads are invoked. The module stand-in
        records the cwd reached by the real script and deliberately stops setup.
        """
        driver = Path(__file__).resolve().parents[1] / "scripts/sbatch_exp12_hb.sh"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            checkout = root / "final checkout/main"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / "configs").mkdir()
            for name in (
                "run.py",
                "configs/base_exp12.yaml",
                "scripts/check_exp12_hb_status.py",
            ):
                (checkout / name).touch()
            spool = root / "slurmd/job123/slurm_script"
            spool.parent.mkdir(parents=True)
            shutil.copy(driver, spool)
            spool.chmod(0o755)
            (root / "bin").mkdir()
            module = root / "bin/module"
            module.write_text('#!/bin/sh\npwd -P > "$CWD_RECORD"\nexit 73\n')
            module.chmod(0o755)
            record = root / "cwd.txt"
            env = {
                **os.environ,
                "PATH": str(root / "bin") + ":" + os.environ["PATH"],
                "SLURM_JOB_ID": "123",
                "SLURM_SUBMIT_DIR": str(checkout),
                "EXPECTED_COMMIT": "1" * 40,
                "EXPECTED_GPU_MODEL": "SYNTHETIC",
                "HB_ENV": str(root / "unused-hb"),
                "CWD_RECORD": str(record),
            }
            env.pop("HB_TEST_HOOKS", None)
            env.pop("BASH_ENV", None)
            # Default Slurm cwd, and an unrelated initial cwd (--chdir).
            for cwd in (checkout, root):
                result = subprocess.run(
                    [str(spool)],
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(record.read_text().strip(), str(checkout))
                record.unlink()
            for submit_dir in ("", str(root), str(root / "missing")):
                with self.subTest(submit_dir=submit_dir):
                    result = subprocess.run(
                        [str(spool)],
                        cwd=checkout,
                        env={**env, "SLURM_SUBMIT_DIR": submit_dir},
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode, 2, result.stdout + result.stderr
                    )
                    self.assertFalse(
                        record.exists(), "invalid checkout reached cluster setup"
                    )
