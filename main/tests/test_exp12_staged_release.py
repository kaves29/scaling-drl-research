"""CPU checks for per-run releases, staged (A2) eligibility, allocation shapes, the
null-threshold computation and science-source compatibility certificates."""

import hashlib
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import exp12_dynamic_scheduler as scheduler
from scripts import exp12_launch_plan as launch_plan
from scripts import exp12_source_compatibility as compatibility

COMMIT = "a" * 40
ALL_GATES = set(launch_plan.GATES["parents"])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_item(arch, environment, kind="exp1", group="dmc_hard", seed=1, root="/x"):
    return {
        "id": f"{kind}_{arch}_{environment}_{seed}",
        "kind": kind,
        "row": {
            "architecture": arch,
            "environment": environment,
            "group": group,
            "seed": seed,
            "budget": 1_000_000,
            "run_key": f"exp1_{arch}_{environment}_seed{seed}",
            "checkpoint_dir": f"{root}/{arch}/{environment}/{seed}",
            "config_hash": "c",
            "overrides": [f"arch={arch}"],
            "command": ["python", "run.py", "--experiment", "exp1"],
        },
    }


def state(**changes):
    base = {
        "gates": set(scheduler.GLOBAL_GATES),
        "identity": set(),
        "a2": False,
        "fresh_null": None,
        "hopper": None,
        "exp2": None,
        "releases": [],
    }
    base.update(changes)
    return base


class EligibilityTest(unittest.TestCase):
    def test_default_protocol_needs_all_six_identity_cases_for_every_run(self):
        d2 = run_item("D2W512", "dog-run")
        st = state(identity={("D4W1024", "dmc"), ("D4W1536", "dmc")})
        self.assertTrue(any(b.startswith("identity:") for b in scheduler.run_blockers(d2, st)))
        st["identity"] = set(launch_plan.IDENTITY_REQUIRED)
        self.assertEqual(scheduler.run_blockers(d2, st), [])

    def test_positive_control_and_global_gates_block_everything_even_under_a2(self):
        st = state(a2=True, gates=set(scheduler.GLOBAL_GATES) - {"positive_control"})
        for arch in ("D2W512", "D4W1024", "D4W1536"):
            self.assertIn(
                "gate:positive_control",
                scheduler.run_blockers(run_item(arch, "dog-run"), st),
            )

    def test_a2_stages_identity_by_suite_and_exempts_the_non_forking_size(self):
        st = state(a2=True, fresh_null={"decision": "keep_zero"})
        self.assertEqual(scheduler.run_blockers(run_item("D2W512", "h1-run-v0"), st), [])
        dmc = {("D4W1024", "dmc"), ("D4W1536", "dmc")}
        st["identity"] = {("D4W1536", "dmc")}
        self.assertTrue(scheduler.run_blockers(run_item("D4W1536", "dog-run"), st))
        st["identity"] = dmc
        self.assertEqual(scheduler.run_blockers(run_item("D4W1536", "dog-run"), st), [])
        self.assertEqual(scheduler.run_blockers(run_item("D4W1024", "dog-run"), st), [])
        for env in ("myo-key-turn", "h1-run-v0"):
            self.assertTrue(scheduler.run_blockers(run_item("D4W1536", env), st))

    def test_failed_gates_hold_only_their_own_cells(self):
        st = state(identity=set(launch_plan.IDENTITY_REQUIRED))
        self.assertEqual(
            scheduler.run_blockers(run_item("D4W1024", "dog-run"), st),
            ["fresh_null_disposition"],
        )
        self.assertEqual(
            scheduler.run_blockers(run_item("D4W1536", "hopper-hop"), st),
            ["hopper_range_disposition"],
        )
        for item in (
            run_item("D2W512", "hopper-hop"),
            run_item("D4W1536", "dog-run"),
            run_item("D2W512", "dog-run"),
        ):
            self.assertEqual(scheduler.run_blockers(item, st), [])
        st["hopper"] = "exclude"
        self.assertEqual(
            scheduler.run_blockers(run_item("D4W1536", "hopper-hop"), st),
            ["hopper_range_excluded"],
        )
        st["hopper"] = "release"
        self.assertEqual(scheduler.run_blockers(run_item("D4W1536", "hopper-hop"), st), [])

    def test_arms_wait_for_exp2_release(self):
        st = state(identity=set(launch_plan.IDENTITY_REQUIRED))
        arm = run_item("D4W1536", "dog-run", kind="exp2_injected")
        self.assertEqual(scheduler.run_blockers(arm, st, exp2_active=False), ["exp2_release"])
        self.assertEqual(scheduler.run_blockers(arm, st, exp2_active=True), [])


class EffectiveRowsAndPriorityTest(unittest.TestCase):
    def test_threshold_release_changes_only_d4w1024_rows(self):
        parents = [run_item(a, "dog-run") for a in ("D2W512", "D4W1024", "D4W1536")]
        queue = {"commit": COMMIT, "plan_sha256": "p", "parents": parents}
        with mock.patch.object(scheduler, "config_hash", side_effect=lambda o: "h" + str(len(o))):
            same = scheduler.effective_parents(queue, state(fresh_null={"decision": "keep_zero"}))
            self.assertEqual(same, parents)
            release = {"decision": "adopt_threshold", "override": "trigger.null_threshold=0.0125"}
            changed = scheduler.effective_parents(queue, state(fresh_null=release))
        self.assertEqual(changed[0], parents[0])
        self.assertEqual(changed[2], parents[2])
        row = changed[1]["row"]
        self.assertEqual(row["release_overrides"], ["trigger.null_threshold=0.0125"])
        self.assertEqual(row["overrides"][-1], "trigger.null_threshold=0.0125")
        self.assertEqual(row["command"][-2:], ["--overrides", "trigger.null_threshold=0.0125"])
        self.assertNotEqual(changed[1]["id"], parents[1]["id"])

    def test_real_threshold_override_changes_the_resolved_config_hash(self):
        from generate_manifest import EXP12_ARCHS, exp12_overrides

        base = exp12_overrides(EXP12_ARCHS[1], "dog-run", "dmc_hard", 1, "/r")
        self.assertNotEqual(
            launch_plan.config_hash(base),
            launch_plan.config_hash(base + ["trigger.null_threshold=0.0125"]),
        )

    def test_arms_first_then_longest_remaining_work(self):
        admission = {
            "architectures": {
                a: {
                    "estimated_minutes_by_group": {g: m for g in scheduler.GROUPS},
                    "estimated_arm_minutes_by_group": {g: 30 for g in scheduler.GROUPS},
                }
                for a, m in (("D2W512", 100), ("D4W1024", 200), ("D4W1536", 300))
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            items = [
                run_item("D2W512", "dog-run", root=directory),
                run_item("D4W1536", "dog-run", root=directory),
                run_item("D4W1024", "dog-run", kind="exp2_injected", root=directory),
                run_item("D4W1024", "dog-run", seed=2, root=directory),
            ]
            # A half-finished D4W1536 run has 150 minutes left: below D4W1024's 200.
            resumed = Path(items[1]["row"]["checkpoint_dir"]) / "state"
            resumed.mkdir(parents=True)
            (resumed / "LATEST").write_text("step_000250000_abcd1234")
            order = sorted(items, key=lambda i: scheduler.priority(i, admission))
        self.assertEqual(
            [(i["kind"], i["row"]["architecture"]) for i in order],
            [
                ("exp2_injected", "D4W1024"),
                ("exp1", "D4W1024"),
                ("exp1", "D4W1536"),
                ("exp1", "D2W512"),
            ],
        )


class AllocationShapeTest(unittest.TestCase):
    def test_legacy_exact_and_generic_bounded(self):
        shape = scheduler.allocation_shape
        self.assertEqual(shape("A", {"nodes": 1, "gpus_per_node": 4, "wall_minutes": 1800}), (1, 1800))
        with self.assertRaisesRegex(ValueError, "exactly"):
            shape("B", {"nodes": 3, "gpus_per_node": 4, "wall_minutes": 2880})
        for nodes in (2, 3, 4):
            self.assertEqual(
                shape("C16", {"nodes": nodes, "gpus_per_node": 4, "wall_minutes": 2880}),
                (nodes, 2880),
            )
        for bad in ({"nodes": 5, "wall_minutes": 60}, {"nodes": 2, "wall_minutes": 2881}):
            with self.assertRaisesRegex(ValueError, "generic allocation"):
                shape("C", {**bad, "gpus_per_node": 4})
        with self.assertRaisesRegex(ValueError, "name"):
            shape("bad name", {"nodes": 1, "gpus_per_node": 4, "wall_minutes": 60})

    def test_delta_partitions_gpu_counts(self):
        shape = scheduler.allocation_shape
        self.assertEqual(shape("B8", {"partition": "gpuA100x8", "nodes": 1, "gpus_per_node": 8,
                                      "wall_minutes": 2880}), (1, 2880))
        with self.assertRaisesRegex(ValueError, "whole A100 nodes"):
            shape("B8", {"partition": "gpuA100x8", "nodes": 1, "gpus_per_node": 4, "wall_minutes": 60})
        with self.assertRaisesRegex(ValueError, "whole A100 nodes"):
            shape("X", {"partition": "gpuA40x4", "nodes": 1, "gpus_per_node": 4, "wall_minutes": 60})
        with self.assertRaisesRegex(ValueError, "generic allocation"):
            shape("B8", {"partition": "gpuA100x8", "nodes": 3, "gpus_per_node": 8, "wall_minutes": 60})
        with self.assertRaisesRegex(ValueError, "legacy"):
            shape("B", {"partition": "gpuA100x8", "nodes": 2, "gpus_per_node": 8, "wall_minutes": 2880})
        self.assertEqual(shape("A35", {"nodes": 1, "gpus_per_node": 4, "wall_minutes": 2100}), (1, 2100))

    def test_numeric_bound_rule_rounds_ten_times_the_maximum_up(self):
        from scripts import exp12_numeric_bounds as bounds

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "n.jsonl"
            rows = [{"name": n, "deviation": d} for n in bounds.NAMES for d in (1.1e-7, 2.3e-7)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = bounds.propose([path])
            self.assertTrue(report["complete"])
            self.assertEqual(report["bounds"]["diagnostics_update/tf32"]["proposed_bound"], 3e-06)
            path.write_text(json.dumps(rows[0]) + "\n")
            self.assertFalse(bounds.propose([path])["complete"])

    def test_medium_dmc_group_requires_and_uses_its_own_estimate(self):
        """hopper-hop/swimmer rows (dmc_medium) previously had no admissible estimate."""
        profile = {
            "estimated_minutes_by_group": {g: 10 * (i + 1) for i, g in enumerate(scheduler.GROUPS)}
        }
        hopper = run_item("D2W512", "hopper-hop", group="dmc_medium")
        self.assertEqual(scheduler.estimated_minutes(hopper, profile), 20)


class ReleaseValidationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.queue_root = self.root / "queue"
        self.queue_root.mkdir()
        (self.queue_root / "queue.json").write_text("{}")
        self.queue = {"commit": COMMIT, "plan": "p", "plan_sha256": "p", "parents": []}
        self.approval = {
            "release_mode": "per_run",
            "gates": {g: {"status": launch_plan.DEFERRED} for g in ALL_GATES},
        }
        patcher = mock.patch.object(scheduler, "_plan_approval", side_effect=lambda q: self.approval)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)

    def decision(self, release):
        return {
            "approved": True,
            "expected_commit": COMMIT,
            "queue_sha256": _sha(self.queue_root / "queue.json"),
            "decision_reference": "owner",
            "release": release,
        }

    def validate(self, release):
        return scheduler.validate_release(self.decision(release), self.queue, self.queue_root)

    def test_binding_and_deferral_are_required(self):
        bad = self.decision({"type": "hopper_range", "decision": "release"})
        bad["expected_commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "source- and queue-bound"):
            scheduler.validate_release(bad, self.queue, self.queue_root)
        self.validate({"type": "hopper_range", "decision": "release"})
        self.approval["gates"]["hopper_range_disposition"] = {"status": "APPROVED"}
        with self.assertRaisesRegex(ValueError, "not deferred"):
            self.validate({"type": "hopper_range", "decision": "release"})
        self.approval["release_mode"] = "all_gates"
        with self.assertRaisesRegex(ValueError, "not deferred"):
            self.validate({"type": "fresh_null", "decision": "keep_zero"})

    def test_a2_requires_an_amendment_reference(self):
        with self.assertRaisesRegex(ValueError, "A2"):
            self.validate({"type": "a2_staged_identity"})
        self.validate({"type": "a2_staged_identity", "amendment_reference": "A2 approved"})

    def test_threshold_release_must_equal_the_computed_report(self):
        report = self.root / "threshold.json"
        report.write_text(
            json.dumps(
                {
                    "status": "COMPUTED_NOT_ACTIVATED",
                    "architecture": "D4W1024",
                    "selection": "pooled",
                    "selected_threshold": 0.0125,
                    "override": "trigger.null_threshold=0.0125",
                }
            )
        )
        release = {
            "type": "fresh_null",
            "decision": "adopt_threshold",
            "selection": "pooled",
            "override": "trigger.null_threshold=0.0125",
            "computation": {"path": str(report), "sha256": _sha(report)},
        }
        self.validate(release)
        for field, value in (("override", "trigger.null_threshold=0.02"), ("selection", "s990")):
            with self.assertRaisesRegex(ValueError, "differs from the computed"):
                self.validate({**release, field: value})
        with self.assertRaisesRegex(ValueError, "keep_zero or adopt"):
            self.validate({"type": "fresh_null", "decision": "keep_zero", "override": "x"})

    def test_published_releases_are_immutable_and_reverified(self):
        with mock.patch.object(scheduler, "load_queue", return_value=self.queue):
            decision = self.root / "hopper.json"
            decision.write_text(json.dumps(self.decision({"type": "hopper_range", "decision": "exclude"})))
            target = scheduler.publish_release(self.queue_root, decision)
            self.assertEqual(target.name, "hopper_range.json")
            again = self.root / "hopper2.json"
            again.write_text(json.dumps(self.decision({"type": "hopper_range", "decision": "release"})))
            with self.assertRaisesRegex(ValueError, "immutable"):
                scheduler.publish_release(self.queue_root, again)
        st = scheduler.release_state(self.queue_root, self.queue)
        self.assertEqual(st["hopper"], "exclude")
        # A published record whose queue binding no longer holds is refused on read.
        (self.queue_root / "queue.json").write_text('{"changed": true}')
        scheduler._RELEASE_CACHE.clear()
        with self.assertRaisesRegex(ValueError, "queue-bound"):
            scheduler.release_state(self.queue_root, self.queue)

    def test_positive_control_needs_same_source_or_a_verified_certificate(self):
        report = self.root / "pc.json"
        body = {
            "status": "m_chosen",
            "chosen_m": "half",
            "allow_any_setting": False,
            "trigger_check": {"forced": False},
            "analysis_code": {"commit": COMMIT, "dirty": False},
            "source_code": {"commit": "b" * 40, "dirty": False},
        }
        report.write_text(json.dumps(body))
        evidence = {"path": str(report), "sha256": _sha(report)}
        with self.assertRaisesRegex(ValueError, "does not qualify"):
            scheduler.validate_positive_control(evidence, COMMIT)
        with mock.patch.object(compatibility, "verify") as verify:
            certificate = {"path": "/c.json", "sha256": "d"}
            scheduler.validate_positive_control(
                {**evidence, "source_compatibility": certificate}, COMMIT, require_m="half"
            )
            verify.assert_called_once_with("/c.json", "d", "b" * 40, COMMIT)
            with self.assertRaisesRegex(ValueError, "does not qualify"):
                scheduler.validate_positive_control(
                    {**evidence, "source_compatibility": certificate}, COMMIT, require_m="last"
                )


class LaunchPlanReleaseModeTest(unittest.TestCase):
    def test_deferral_only_in_per_run_mode_and_array_paths_refuse(self):
        approval = {
            "approved": True,
            "stage": "parents",
            "expected_commit": COMMIT,
            "gates": {g: {"status": launch_plan.DEFERRED} for g in ALL_GATES},
            "resources": {"gpus": 1, "cpus": 16, "memory_gb": 64, "wall_minutes": 60, "concurrency": 8},
        }
        with self.assertRaisesRegex(ValueError, "unresolved mandatory gate"):
            launch_plan.approve(approval, "parents", COMMIT)
        approval["release_mode"] = "per_run"
        self.assertEqual(launch_plan.approve(approval, "parents", COMMIT)["concurrency"], 8)
        with tempfile.TemporaryDirectory() as directory:
            approval_path = Path(directory) / "approval.json"
            approval_path.write_text(json.dumps(approval))
            plan = Path(directory) / "plan.json"
            plan.write_text(
                json.dumps({"stage": "parents", "approval": str(approval_path), "root": directory})
            )
            with self.assertRaisesRegex(ValueError, "dynamic scheduler"):
                launch_plan.execute(plan, 0)
            with mock.patch.object(launch_plan, "submission_command", return_value=None):
                with self.assertRaisesRegex(ValueError, "dynamic scheduler"):
                    launch_plan.prepare_batch_schedule(plan, Path(directory) / "s.json")
        approval["stage"] = "pilot"
        with self.assertRaisesRegex(ValueError, "release mode"):
            launch_plan.approve(approval, "pilot", COMMIT)


def _null_rows(seed, count, offset=0):
    from omegaconf import OmegaConf
    from experiments.exp1 import compose_config
    from experiments.exp12.probe import iqm
    from experiments.exp12.trigger import bootstrap_interval, trigger_config, triggered
    from generate_manifest import EXP12_ARCHS, exp12_overrides
    from utils.run_metadata import _strip_locations

    cfg = compose_config(
        str(launch_plan.MAIN / "configs"),
        "base_exp12",
        exp12_overrides(EXP12_ARCHS[1], "dog-run", "dmc_hard", seed, "/r"),
    )
    tc = trigger_config(cfg)
    provenance = {
        "arch": "D4W1024",
        "env": "dog-run",
        "seed": seed,
        "resolved_config": _strip_locations(OmegaConf.to_container(cfg, resolve=True)),
        "code": {"commit": COMMIT, "dirty": False},
        "runtime": {},
    }
    rng = np.random.default_rng(seed)
    rows = []
    for pair in range(count):
        fresh = rng.normal(0.45, 0.02, 5).astype(np.float32)
        current = (fresh - rng.normal(0.002 + offset, 0.004, 5)).astype(np.float32)
        loss = fresh - current
        low, high = bootstrap_interval(loss, seed, 1 + pair, tc.resamples, tc.confidence)
        rows.append(
            {
                "arch": "D4W1024", "env": "dog-run", "seed": seed, "pair": pair,
                "check_index": 1 + pair, "provenance": provenance,
                "score_fresh_rounds": fresh.tolist(), "score_current_rounds": current.tolist(),
                "b_rounds": [0.5] * 5, "loss_rounds": loss.tolist(), "loss_iqm": iqm(loss),
                "valid": True, "ci_low": low, "ci_high": high, "resamples": tc.resamples,
                "confidence": tc.confidence, "null_threshold": tc.null_threshold,
                "fired": triggered(low, tc.null_threshold),
            }
        )
    return rows


class NullThresholdTest(unittest.TestCase):
    def test_pooled_percentile_is_recomputed_never_activated(self):
        from scripts import exp12_null_threshold as null_threshold

        with tempfile.TemporaryDirectory() as directory:
            files = []
            for label, seed in (("s990", 990), ("s991", 991)):
                path = Path(directory) / f"{label}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in _null_rows(seed, 12)))
                files.append((label, str(path), _sha(path)))
            report = null_threshold.compute(files, "pooled")
            losses = [
                json.loads(line)["loss_iqm"]
                for _, path, _ in files
                for line in Path(path).read_text().splitlines()
            ]
            self.assertEqual(report["status"], "COMPUTED_NOT_ACTIVATED")
            self.assertEqual(report["selected_threshold"], float(np.percentile(losses, 95)))
            self.assertEqual(report["override"], f"trigger.null_threshold={report['selected_threshold']!r}")
            self.assertEqual(report["pooled"]["pairs"], 24)
            self.assertEqual(
                report["pooled"]["fired"],
                report["per_sample"]["s990"]["fired"] + report["per_sample"]["s991"]["fired"],
            )
            self.assertEqual(
                null_threshold.compute(files, "s991")["selected_threshold"],
                report["per_sample"]["s991"]["p95_of_L"],
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                null_threshold.compute([(files[0][0], files[0][1], "0" * 64)], "pooled")
            tampered = Path(directory) / "tampered.jsonl"
            rows = _null_rows(992, 3)
            rows[1]["fired"] = not rows[1]["fired"]
            tampered.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "inconsistent fired"):
                null_threshold.compute([("t", str(tampered), _sha(tampered))], "pooled")


class SourceCompatibilityTest(unittest.TestCase):
    PILOT, HEAD = "7d0883cfa99d27fed062d77a68cbbfa27e560061", "1ea37e49a9279bc75e8390f2e702b746fff2eed8"

    def setUp(self):
        for commit in (self.PILOT, self.HEAD):
            if subprocess.run(
                ["git", "-C", str(compatibility.REPO), "cat-file", "-e", commit + "^{commit}"],
                capture_output=True,
            ).returncode:
                self.skipTest("pilot/head commits are not in this clone")

    def test_pilot_and_head_differ_only_by_the_arm_source_guard(self):
        strict = compatibility.certify(self.PILOT, self.HEAD)
        self.assertEqual(strict["verdict"], "INCOMPATIBLE")
        self.assertEqual(strict["unexplained_differences"], ["main/experiments/exp2_arm.py"])
        allowed = compatibility.certify(self.PILOT, self.HEAD, ["main/experiments/exp2_arm.py"])
        self.assertEqual(allowed["verdict"], "COMPATIBLE")
        for path in (
            "main/experiments/exp12/probe.py",
            "main/experiments/exp12/trigger.py",
            "main/experiments/exp12/injection.py",
            "main/experiments/exp12/fork.py",
            "main/experiments/exp12/m_selection.py",
            "main/experiments/exp1.py",
            "main/scripts/positive_control.py",
            "main/configs/base_exp12.yaml",
        ):
            self.assertNotIn(path, allowed["differing"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            path.write_text(json.dumps(allowed, indent=2, sort_keys=True) + "\n")
            compatibility.verify(path, _sha(path), self.PILOT, self.HEAD)
            with self.assertRaisesRegex(ValueError, "other commits"):
                compatibility.verify(path, _sha(path), self.HEAD, self.PILOT)
            forged = dict(allowed, identical_files=allowed["identical_files"] + 1)
            path.write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, "does not match git"):
                compatibility.verify(path, _sha(path), self.PILOT, self.HEAD)
        with self.assertRaisesRegex(ValueError, "outside the scientific scope"):
            compatibility.certify(self.PILOT, self.HEAD, ["main/docs/x.md"])


if __name__ == "__main__":
    unittest.main()


class PerRunQueueIntegrationTest(unittest.TestCase):
    """Real plan (195 composed configs), real queue, real releases, real census."""

    HEAD = SourceCompatibilityTest.HEAD
    PILOT = SourceCompatibilityTest.PILOT

    def setUp(self):
        SourceCompatibilityTest.setUp(self)

    def test_staged_census_follows_published_releases(self):
        from analysis.exp12_validation import suite

        commit = self.HEAD
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approval = {
                "approved": True,
                "stage": "parents",
                "expected_commit": commit,
                "release_mode": "per_run",
                "gates": {g: {"status": launch_plan.DEFERRED} for g in ALL_GATES},
                "resources": {"gpus": 1, "cpus": 16, "memory_gb": 64, "wall_minutes": 2880, "concurrency": 8},
            }
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval))
            plan_path = root / "plan.json"
            launch_plan.prepare("parents", root / "campaign", commit, approval_path, plan_path)
            queue_root = root / "queue"
            scheduler.prepare(plan_path, queue_root)
            queue_sha = _sha(queue_root / "queue.json")

            def publish(name, release):
                path = root / f"decision_{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "approved": True,
                            "expected_commit": commit,
                            "queue_sha256": queue_sha,
                            "decision_reference": f"owner:{name}",
                            "release": release,
                        }
                    )
                )
                return scheduler.publish_release(queue_root, path)

            def census():
                scheduler._RELEASE_CACHE.clear()
                report = scheduler.queue_status(queue_root)
                return report, {r["run_key"]: r for r in report["rows"]}

            report, rows = census()
            self.assertEqual(report["counts"], {"exp1:blocked_by_prerequisite": 195})

            # Positive control from the pilot's earlier commit, with a real certificate.
            certificate = root / "compat.json"
            body = compatibility.certify(self.PILOT, commit, ["main/experiments/exp2_arm.py"])
            certificate.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
            pc = root / "positive_control.json"
            pc.write_text(
                json.dumps(
                    {
                        "status": "m_chosen", "chosen_m": "half", "allow_any_setting": False,
                        "trigger_check": {"forced": False},
                        "source_code": {"commit": self.PILOT, "dirty": False},
                        "analysis_code": {"commit": commit, "dirty": False},
                    }
                )
            )
            pc_evidence = {
                "path": str(pc), "sha256": _sha(pc),
                "source_compatibility": {"path": str(certificate), "sha256": _sha(certificate)},
            }
            for gate in scheduler.GLOBAL_GATES:
                release = {"type": "gate", "gate": gate}
                if gate == "positive_control":
                    release["positive_control"] = pc_evidence
                publish(gate, release)
            report, rows = census()
            self.assertEqual(report["counts"], {"exp1:blocked_by_prerequisite": 195})
            self.assertTrue(all(r["blockers"][0].startswith("identity:") for r in rows.values()))

            # A2 (owner-approved amendment) plus the two DMC identity certificates.
            publish("a2", {"type": "a2_staged_identity", "amendment_reference": "A2"})
            evidence = []
            for arch in ("D4W1024", "D4W1536"):
                cert = root / f"identity_{arch}.json"
                cert.write_text(
                    json.dumps(
                        {
                            "verdict": "PASS", "expected_commit": commit,
                            "qualification": "cuda_identity_fork",
                            "coverage": [{"architecture": arch, "suite": "dmc", "environment": "dog-run"}],
                        }
                    )
                )
                evidence.append({"path": str(cert), "sha256": _sha(cert)})
            publish("identity_dmc", {"type": "identity", "evidence": evidence})
            report, rows = census()
            eligible = {k for k, r in rows.items() if r["status"] == "eligible"}
            expected = {
                k
                for k, r in rows.items()
                if r["architecture"] == "D2W512"
                or (
                    r["architecture"] == "D4W1536"
                    and suite(r["environment"]) == "dmc"
                    and r["environment"] != "hopper-hop"
                )
            }
            self.assertEqual(eligible, expected)
            self.assertEqual(len(eligible), 65 + 30)  # all D2W512 + 6 DMC tasks x 5 D4W1536
            self.assertEqual(
                rows["exp1_D4W1024_dog-run_seed1"]["blockers"], ["fresh_null_disposition"]
            )
            self.assertEqual(
                rows["exp1_D4W1536_hopper-hop_seed1"]["blockers"], ["hopper_range_disposition"]
            )

            publish("null", {"type": "fresh_null", "decision": "keep_zero"})
            publish("hopper", {"type": "hopper_range", "decision": "exclude"})
            report, rows = census()
            self.assertEqual(rows["exp1_D4W1024_dog-run_seed1"]["status"], "eligible")
            self.assertEqual(rows["exp1_D4W1536_hopper-hop_seed3"]["blockers"], ["hopper_range_excluded"])
            self.assertEqual(report["counts"]["exp1:eligible"], 65 + 35 + 30)
            with self.assertRaisesRegex(ValueError, "immutable"):
                publish("null2", {"type": "fresh_null", "decision": "keep_zero"})


class RunLoopGatingTest(unittest.TestCase):
    """The live allocation loop starts only runs whose prerequisites hold."""

    def test_hold_file_prevents_every_start(self):
        started = self._run_loop(hold={"reason": "shared-code defect under review"})
        self.assertEqual(started, [])
        started = self._run_loop(hold_text="not json")
        self.assertEqual(started, [])

    def _run_loop(self, hold=None, hold_text=None):
        """Minimal live loop over four rows; returns the run keys started."""
        parents = [run_item("D2W512", "dog-run"), run_item("D4W1536", "dog-run")]
        queue = {"commit": COMMIT, "plan": "/plan", "plan_sha256": "p", "parents": parents}
        st = state(identity=set(launch_plan.IDENTITY_REQUIRED))
        profile = {
            "cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1, "min_interactions_per_second": 1,
            "estimated_minutes_by_group": {g: 10 for g in scheduler.GROUPS},
            "estimated_arm_minutes_by_group": {g: 5 for g in scheduler.GROUPS},
        }
        admission = {"stop_margin_minutes": 1, "host_headroom_gb": 1, "gpu_headroom_gb": 1,
                     "architectures": {a: profile for a in scheduler.WEIGHTS}}
        approval = {"queue_sha256": "q", "expected_commit": COMMIT, "admission": admission}
        started = []

        class Done:
            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            if hold is not None:
                (Path(directory) / "HOLD.json").write_text(json.dumps(hold))
            if hold_text is not None:
                (Path(directory) / "HOLD.json").write_text(hold_text)
            approval_path = Path(directory) / "a.json"
            approval_path.write_text("{}")

            def fake_sha(path):
                fixed = {"/plan": "p", "/ap": "s"}
                if str(path) in fixed:
                    return fixed[str(path)]
                return "q" if str(path).endswith("queue.json") else _sha(path)

            def fake_read(path):
                if Path(path) == approval_path:
                    return approval
                if Path(path).name == "HOLD.json":
                    return json.loads(Path(path).read_text())
                return {"approval": "/ap", "approval_sha256": "s"}

            env = {"EXP12_APPROVAL_SHA256": _sha(approval_path), "EXPECTED_COMMIT": COMMIT,
                   "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
            patches = [
                mock.patch.dict("os.environ", env),
                mock.patch.object(scheduler, "load_queue", return_value=queue),
                mock.patch.object(scheduler, "_read", side_effect=fake_read),
                mock.patch.object(scheduler, "sha", side_effect=fake_sha),
                mock.patch.object(scheduler, "validate_approval", return_value=({"cpus_per_node": 64, "memory_gb_per_node": 256}, admission)),
                mock.patch.object(scheduler.subprocess, "check_output", return_value=""),
                mock.patch.object(scheduler, "_actual_allocation", return_value=time.time() + 4 * 3600),
                mock.patch.object(scheduler, "gpu_snapshot", return_value=(40960, 0)),
                mock.patch.object(scheduler, "available_host_gb", return_value=200),
                mock.patch.object(scheduler, "_source_preflight"),
                mock.patch.object(scheduler, "release_state", return_value=st),
                mock.patch.object(scheduler, "_start", side_effect=lambda item, token, *_: started.append(item["row"]["run_key"]) or {"item": item, "gpu": token, "process": Done(), "profile": profile}),
                mock.patch.object(scheduler, "_finish", return_value=True),
                mock.patch.object(scheduler.Ledger, "state", autospec=True, side_effect=lambda self, item: "complete" if item["row"]["run_key"] in started else "pending"),
                mock.patch.object(scheduler, "POLL_SECONDS", 0),
                mock.patch.object(scheduler.time, "sleep"),
            ]
            for p in patches:
                p.start()
            try:
                self.assertEqual(scheduler.run(directory, approval_path, "C"), 0)
            finally:
                for p in reversed(patches):
                    p.stop()
        return started

    def test_without_hold_the_same_loop_starts_eligible_runs(self):
        self.assertEqual(sorted(self._run_loop()), ["exp1_D2W512_dog-run_seed1", "exp1_D4W1536_dog-run_seed1"])

    def test_only_released_runs_start_and_conflicting_m_refuses(self):
        parents = [
            run_item("D2W512", "dog-run"),
            run_item("D4W1024", "dog-run"),
            run_item("D4W1536", "hopper-hop", group="dmc_medium"),
            run_item("D4W1536", "dog-run"),
        ]
        queue = {"commit": COMMIT, "plan": "/plan", "plan_sha256": "p", "parents": parents}
        st = state(identity=set(launch_plan.IDENTITY_REQUIRED))  # null + hopper still held
        profile = {
            "cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1, "min_interactions_per_second": 1,
            "estimated_minutes_by_group": {g: 10 for g in scheduler.GROUPS},
            "estimated_arm_minutes_by_group": {g: 5 for g in scheduler.GROUPS},
        }
        approval = {
            "queue_sha256": "q",
            "expected_commit": COMMIT,
            "admission": {"stop_margin_minutes": 1, "host_headroom_gb": 1, "gpu_headroom_gb": 1, "architectures": {a: profile for a in scheduler.WEIGHTS}},
        }
        started = []

        class Done:
            def poll(self):
                return 0

        def fake_start(item, token, *_):
            started.append(item["row"]["run_key"])
            return {"item": item, "gpu": token, "process": Done(), "profile": profile}

        finished = set()

        def fake_sha(path):
            fixed = {"/plan": "p", "/ap": "s"}
            if str(path) in fixed:
                return fixed[str(path)]
            return "q" if str(path).endswith("queue.json") else _sha(path)

        with tempfile.TemporaryDirectory() as directory:
            approval_path = Path(directory) / "a.json"
            approval_path.write_text("{}")
            env = {
                "EXP12_APPROVAL_SHA256": _sha(approval_path),
                "EXPECTED_COMMIT": COMMIT,
                "SLURM_JOB_ID": "123",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            }
            patches = [
                mock.patch.dict("os.environ", env),
                mock.patch.object(scheduler, "load_queue", return_value=queue),
                mock.patch.object(scheduler, "_read", side_effect=lambda p: approval if Path(p) == approval_path else {"approval": "/ap", "approval_sha256": "s"}),
                mock.patch.object(scheduler, "sha", side_effect=fake_sha),
                mock.patch.object(scheduler, "validate_approval", return_value=({"cpus_per_node": 64, "memory_gb_per_node": 256}, approval["admission"])),
                mock.patch.object(scheduler.subprocess, "check_output", return_value=""),
                mock.patch.object(scheduler, "_actual_allocation", return_value=time.time() + 4 * 3600),
                mock.patch.object(scheduler, "gpu_snapshot", return_value=(40960, 0)),
                mock.patch.object(scheduler, "available_host_gb", return_value=200),
                mock.patch.object(scheduler, "_source_preflight"),
                mock.patch.object(scheduler, "release_state", return_value=st),
                mock.patch.object(scheduler, "_start", side_effect=fake_start),
                mock.patch.object(scheduler, "_finish", side_effect=lambda job, *a: finished.add(job["item"]["id"]) or True),
                mock.patch.object(scheduler.Ledger, "state", autospec=True, side_effect=lambda self, item: "complete" if item["id"] in finished else "pending"),
                mock.patch.object(scheduler, "POLL_SECONDS", 0),
                mock.patch.object(scheduler.time, "sleep"),
            ]
            for p in patches:
                p.start()
            try:
                self.assertEqual(scheduler.run(directory, approval_path, "C"), 0)
                self.assertEqual(
                    sorted(started), ["exp1_D2W512_dog-run_seed1", "exp1_D4W1536_dog-run_seed1"]
                )
                approval["exp2"] = {"approved": True, "injection_m": "last"}
                st["exp2"] = {"injection_m": "half"}
                with self.assertRaisesRegex(ValueError, "different m"):
                    scheduler.run(directory, approval_path, "C")
            finally:
                for p in reversed(patches):
                    p.stop()
