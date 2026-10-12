"""Read-only, lightweight pilot inspection.

Fixtures are real tiny Exp1 runs (no trigger, natural fork, forced fork) produced by the
production code. The inspector is then checked (a) in-process, (b) in a subprocess whose
import system refuses JAX, pandas, the `experiments` package and other heavy frameworks,
and (c) helper-by-helper against the production implementations it re-states.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_helpers import CONFIG_PATH, compose, patch_wandb, tiny_overrides  # noqa: E402
from test_exp12_fork import fork_overrides, run_exp1  # noqa: E402

from scripts import exp12_pilot_inspect as light  # noqa: E402

MAIN = Path(__file__).resolve().parents[1]
PILOT_COMMIT = "7d0883cfa99d27fed062d77a68cbbfa27e560061"
OLD_COMMIT = "b4a90cb"  # Block B source: scientific paths differ from today's
BLOCKED = ("jax", "jaxlib", "flax", "optax", "orbax", "mujoco", "dm_control", "wandb", "pandas",
           "experiments", "scale_rl", "gymnasium", "gym", "torch", "tensorflow", "humanoid_bench",
           "myosuite")


def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def natural_overrides(results, threshold):
    return tiny_overrides("hopper-hop", "dmc_medium", seed=102, steps=400, extra=[
        "run_role=dev", "fork.architectures=[D1W8]", "env.max_episode_steps=40",
        "fork.eval_episodes=1", f"trigger.null_threshold={threshold}", f"results_root={results}"])


def git_has(commit):
    return not subprocess.run(["git", "-C", str(MAIN), "cat-file", "-e", commit + "^{commit}"],
                              capture_output=True).returncode


class Fixtures:
    root = None

    @classmethod
    def build(cls):
        if cls.root is not None:
            return cls.root
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        patchers = patch_wandb()
        try:
            run_exp1(str(root / "none" / "checkpoints" / "run"), natural_overrides(root / "none" / "results", 1e9))
            run_exp1(str(root / "natural" / "checkpoints" / "run"),
                     natural_overrides(root / "natural" / "results", -1e9))
            run_exp1(str(root / "forced" / "checkpoints" / "run"), fork_overrides(root / "forced" / "results"))
        finally:
            for p in patchers:
                p.stop()
        cls.root = root
        return root


def tearDownModule():
    if Fixtures.root is not None:
        Fixtures._tmp.cleanup()


def with_commit(source, destination, commit):
    """A copy of a fixture whose provenance names another clean commit."""
    shutil.copytree(source, destination)
    path = destination / "checkpoints" / "run" / "run_metadata.json"
    data = json.loads(path.read_text())
    data["code"] = {"commit": commit, "dirty": False}
    path.write_text(json.dumps(data))
    return destination


class VerdictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = Fixtures.build()

    def inspect(self, name, *extra):
        out = Path(tempfile.mkdtemp()) / "report.json"
        root = self.fixtures / name
        before = tree_digest(root)
        with self.assertRaises(SystemExit):
            light.main(["--root", str(root), "--out", str(root / "inside.json")])
        light.main(["--root", str(root), "--out", str(out), *extra])
        self.assertEqual(tree_digest(root), before)  # read-only
        return json.loads(out.read_text())

    def test_no_natural_trigger_prescribes_stop_and_consult(self):
        report = self.inspect("none")
        self.assertEqual(report["verdict"], "NO_NATURAL_TRIGGER")
        self.assertTrue(report["horizon"]["complete"])
        self.assertEqual(report["trigger"]["fired_checks"], [])
        self.assertIsNone(report["trigger"]["recomputed_f_star"])
        self.assertIsNone(report["trigger"]["recorded_f_star"])
        self.assertFalse(report["fork"]["fork_dir_exists"])
        self.assertIn("stop and consult", report["next_step"])

    def test_natural_fork_statistics_match_the_saved_production_records(self):
        report = self.inspect("natural")
        self.assertEqual(report["verdict"], "NATURAL_FORK_AVAILABLE", report["problems"])
        self.assertEqual(report["trigger"]["recomputed_f_star"], {"check_index": 2, "interaction_step": 40})
        self.assertEqual(report["trigger"]["recorded_f_star"], report["trigger"]["recomputed_f_star"])
        self.assertTrue(report["fork"]["natural_fork_valid"])
        self.assertEqual(report["fork"]["plan"]["fork_check_index"], 2)
        self.assertEqual(report["unsupported_pickled_classes"], [])
        # Every recomputed bound equals the production run's recorded value.
        import csv

        run = self.fixtures / "natural" / "checkpoints" / "run"
        with open(run / "probes" / "probe_checks.csv") as stream:
            recorded = {int(float(r["check_index"])): r for r in csv.DictReader(stream)}
        for row in report["trigger"]["checks"][1:]:
            self.assertEqual(row["ci_low"], float(recorded[row["check_index"]]["ci_low"]))
            self.assertEqual(row["triggered_recorded"], row["fired_recomputed"])

    def test_forced_fork_is_never_eligible(self):
        report = self.inspect("forced")
        self.assertEqual(report["verdict"], "FORCED_FORK_NOT_ELIGIBLE")
        self.assertTrue(report["fork"]["fork_ready"])
        self.assertTrue(report["trigger"]["forced_checks"])

    def test_tampered_trigger_record_is_inconsistent(self):
        tmp = Path(tempfile.mkdtemp())
        copy = shutil.copytree(self.fixtures / "natural", tmp / "tampered")
        csv_path = copy / "checkpoints" / "run" / "probes" / "probe_checks.csv"
        text = csv_path.read_text().splitlines()
        header = text[0].split(",")
        cells = text[3].split(",")
        cells[header.index("ci_low")] = "123.0"
        text[3] = ",".join(cells)
        csv_path.write_text("\n".join(text) + "\n")
        out = tmp / "report.json"
        light.main(["--root", str(copy), "--out", str(out)])
        report = json.loads(out.read_text())
        self.assertEqual(report["verdict"], "INCONSISTENT_EVIDENCE")
        self.assertTrue(any("ci_low" in p for p in report["problems"]))


@unittest.skipUnless(git_has(PILOT_COMMIT) and git_has(OLD_COMMIT), "historical commits not in this clone")
class CompatibilityPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = Fixtures.build()
        cls.head = subprocess.check_output(["git", "-C", str(MAIN), "rev-parse", "HEAD"], text=True).strip()

    def test_pilot_commit_certificate_is_compatible_and_verifiable(self):
        tmp = Path(tempfile.mkdtemp())
        root = with_commit(self.fixtures / "natural", tmp / "pilot", PILOT_COMMIT)
        out, cert = tmp / "report.json", tmp / "compat.json"
        light.main(["--root", str(root), "--out", str(out), "--final-sha", self.head,
                    "--compat-out", str(cert)])
        report = json.loads(out.read_text())
        self.assertEqual(report["source_compatibility"]["verdict"], "COMPATIBLE")
        self.assertEqual(report["source_compatibility"]["differing"], ["main/experiments/exp2_arm.py"])
        digest = hashlib.sha256(cert.read_bytes()).hexdigest()
        self.assertEqual(report["source_compatibility"]["certificate_sha256"], digest)
        light.compatibility.verify(cert, digest, PILOT_COMMIT, self.head)  # the scheduler's check

    def test_old_source_is_incompatible(self):
        tmp = Path(tempfile.mkdtemp())
        old = subprocess.check_output(["git", "-C", str(MAIN), "rev-parse", OLD_COMMIT], text=True).strip()
        root = with_commit(self.fixtures / "natural", tmp / "old", old)
        out = tmp / "report.json"
        light.main(["--root", str(root), "--out", str(out), "--final-sha", self.head])
        report = json.loads(out.read_text())
        self.assertEqual(report["source_compatibility"]["verdict"], "INCOMPATIBLE")
        self.assertIn("main/experiments/exp12/probe.py", report["source_compatibility"]["unexplained_differences"])


class LightweightIsolationTest(unittest.TestCase):
    """A subprocess whose import system refuses every heavy framework still inspects correctly."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = Fixtures.build()
        cls.site = Path(tempfile.mkdtemp())
        (cls.site / "sitecustomize.py").write_text(textwrap.dedent(f"""
            import sys
            BLOCKED = {BLOCKED!r}

            class _Refuse:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in BLOCKED:
                        raise ImportError("blocked heavy import: " + name)
                    return None

            sys.meta_path.insert(0, _Refuse())
        """))

    def run_blocked(self, name, *extra):
        out = Path(tempfile.mkdtemp()) / "report.json"
        env = dict(os.environ, PYTHONPATH=str(self.site))
        result = subprocess.run(
            [sys.executable, "-X", "importtime", str(MAIN / "scripts" / "exp12_pilot_inspect.py"),
             "--root", str(self.fixtures / name), "--out", str(out), *extra],
            cwd=str(MAIN), env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        imported = {line.split("|")[-1].strip().split(".")[0] for line in result.stderr.splitlines()
                    if line.startswith("import time:")}
        self.assertFalse(imported & set(BLOCKED), imported & set(BLOCKED))
        return json.loads(out.read_text())

    def test_verdicts_without_heavy_frameworks(self):
        self.assertEqual(self.run_blocked("none")["verdict"], "NO_NATURAL_TRIGGER")
        report = self.run_blocked("natural")
        self.assertEqual(report["verdict"], "NATURAL_FORK_AVAILABLE", report["problems"])
        self.assertTrue(report["fork"]["natural_fork_valid"])
        self.assertEqual(self.run_blocked("forced")["verdict"], "FORCED_FORK_NOT_ELIGIBLE")


class ProductionEquivalenceTest(unittest.TestCase):
    """Each helper the inspector re-states equals the production implementation."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = Fixtures.build()

    def test_helpers(self):
        import numpy as np
        from experiments.exp12 import fork, probe, state
        from scale_rl.common.logger import get_architecture_id

        rng = np.random.default_rng(0)
        for _ in range(20):
            values = rng.normal(size=5).astype(np.float32)
            self.assertEqual(light.iqm(values), probe.iqm(values))
        self.assertEqual(light.check_steps(500_000, 20), probe.check_steps(500_000, 20))
        with self.assertRaises(ValueError):
            light.check_steps(10, 3)
        cfg = compose(["env_name=dog-run", "env=dmc_hard", "critic_num_blocks=4", "critic_hidden_dim=1536",
                       "run_role=dev", "seed=102"])
        for k in range(1, 20):
            step = k * int(cfg.num_interaction_steps) // 20
            self.assertEqual(light.fork_plan(cfg, step, k), fork.fork_plan(cfg, step, k))
        self.assertEqual(light.architecture_id(cfg), get_architecture_id(cfg))
        import omegaconf

        self.assertEqual(
            omegaconf.OmegaConf.to_container(light.compose_config(["env_name=dog-run", "env=dmc_hard",
                                                                   "critic_num_blocks=4", "critic_hidden_dim=1536",
                                                                   "run_role=dev", "seed=102"]), resolve=True),
            omegaconf.OmegaConf.to_container(cfg, resolve=True))
        run = self.fixtures / "natural" / "checkpoints" / "run"
        self.assertEqual(light.latest_state_dir(run / "state"), state.latest_state_dir(run / "state"))

    def test_meta_reader_and_natural_fork_validation_agree_with_production(self):
        from omegaconf import OmegaConf
        from experiments.exp12 import fork
        from experiments.exp12.state import latest_state_dir, load_meta
        from scripts.positive_control import _validate_natural_fork, _setting_errors

        run = self.fixtures / "natural" / "checkpoints" / "run"
        cfg = OmegaConf.create(json.loads((run / "run_metadata.json").read_text())["resolved_config"])
        state_dir = latest_state_dir(run / "fork" / "state")
        production_meta = load_meta(state_dir)
        raw, opaque = light.read_meta(state_dir)
        self.assertEqual(opaque, [])
        self.assertEqual(light._plain(raw["extra_state"]), light._plain(production_meta["extra_state"]))
        plan = fork.read_fork(run)
        plan_light = {k: v for k, v in json.loads((run / "fork" / "fork.json").read_text()).items()
                      if k not in ("run_dir", "fresh_critic_dir", "device")}
        self.assertEqual(plan_light, plan)
        _validate_natural_fork(cfg, plan, production_meta)  # both accept
        light.validate_natural_fork(cfg, plan, {"interaction_step": light._plain(raw["interaction_step"]),
                                                "extra_state": light._plain(raw["extra_state"])})
        for field, value in (("fork_step", plan["fork_step"] + 20), ("arm_end_step", 1),
                             ("fork_check_index", 3)):
            bad = dict(plan, **{field: value})
            with self.assertRaises(ValueError):
                _validate_natural_fork(cfg, bad, production_meta)
            with self.assertRaises(ValueError):
                light.validate_natural_fork(cfg, bad, {"interaction_step": light._plain(raw["interaction_step"]),
                                                       "extra_state": light._plain(raw["extra_state"])})
        self.assertEqual(light.setting_errors(cfg), _setting_errors(cfg, False))
        canonical = compose(["env_name=dog-run", "env=dmc_hard", "critic_num_blocks=4", "critic_hidden_dim=1536",
                             "run_role=dev", "seed=102"])
        self.assertEqual(light.setting_errors(canonical), _setting_errors(canonical, False))
        self.assertEqual(light.setting_errors(canonical), [])


if __name__ == "__main__":
    unittest.main()


class TrajectoryAnalysisTest(unittest.TestCase):
    """scripts/exp12_trigger_trajectory.py: descriptive classes and an end-to-end run."""

    @staticmethod
    def rows(values):
        out = []
        for k, (iqm, low, high) in enumerate(values, start=1):
            out.append({"check_index": k, "eligible": True, "loss_iqm": iqm, "ci_low": low, "ci_high": high,
                        "ci_width": high - low, "boundary_gap": low, "fires_registered_rule": low > 0,
                        "near_boundary": low <= 0 and -low <= 0.25 * (high - low)})
        return out

    def test_classes(self):
        from scripts import exp12_trigger_trajectory as t

        self.assertEqual(t.classify(self.rows([(-0.2, -0.3, -0.1)] * 19), [], 0.0)[0], "NEGATIVE")
        self.assertEqual(t.classify(self.rows([(0.02, -0.01, 0.06)] * 19), [], 0.0)[0], "POSITIVE_UNDERPOWERED")
        wide = [(0.01 * (-1) ** k, -0.2, 0.2) for k in range(19)]
        self.assertEqual(t.classify(self.rows(wide), [], 0.0)[0], "NEAR_ZERO_UNCERTAIN")
        mixed = [(-0.2, -0.3, -0.1)] * 10 + [(0.05, 0.01, 0.1)] * 9
        self.assertEqual(t.classify(self.rows(mixed), [], 0.0)[0], "UNDETERMINED")
        self.assertEqual(t.classify(self.rows([(-0.2, -0.3, -0.1)] * 19), ["check 4: missing"], 0.0)[0],
                         "UNDETERMINED")

    def test_end_to_end_on_a_real_run_without_heavy_imports(self):
        fixtures = Fixtures.build()
        tmp = Path(tempfile.mkdtemp())
        report = tmp / "report.json"
        light.main(["--root", str(fixtures / "none"), "--out", str(report)])
        site = Path(tempfile.mkdtemp())
        (site / "sitecustomize.py").write_text(textwrap.dedent(f"""
            import sys
            BLOCKED = {BLOCKED + ("matplotlib", "scipy")!r}
            class _Refuse:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in BLOCKED:
                        raise ImportError("blocked: " + name)
            sys.meta_path.insert(0, _Refuse())
        """))
        before = tree_digest(fixtures / "none")
        result = subprocess.run([sys.executable, str(MAIN / "scripts" / "exp12_trigger_trajectory.py"),
                                 "--report", str(report), "--out", str(tmp / "traj")],
                                env=dict(os.environ, PYTHONPATH=str(site)), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(tree_digest(fixtures / "none"), before)
        summary = json.loads((tmp / "traj" / "summary.json").read_text())
        recomputed = json.loads(report.read_text())["trigger"]["checks"]
        self.assertEqual(summary["statistics"]["eligible_checks"], 19)
        self.assertEqual(summary["statistics"]["max_lower_bound"],
                         max(r["ci_low"] for r in recomputed if 1 <= r["check_index"] <= 19))
        self.assertTrue((tmp / "traj" / "trajectory.html").read_text().startswith("<!doctype html>"))
        import csv

        with open(tmp / "traj" / "trajectory.csv") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 21)
