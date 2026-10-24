"""Focused infrastructure tests; no scientific simulation or GPU required."""

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m


def resources(minutes=2880):
    return dict(
        partition="gpuA100x8",
        nodes=1,
        gpus=8,
        cpus=128,
        memory_gb=2007,
        wall_minutes=minutes,
    )


def qualification_resources(minutes=15):
    return dict(
        partition="gpuA100x4",
        nodes=1,
        gpus=1,
        cpus=16,
        memory_gb=60,
        wall_minutes=minutes,
    )


class Infrastructure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def budget(self):
        b = m.Budget(self.root / "budget")
        b.initialize("test fixture, not real spending approval")
        return b

    def test_live_shape_and_billing_arithmetic(self):
        m.resources_valid(resources())
        self.assertEqual(m.rate(resources()), 12.042)
        self.assertAlmostEqual(m.rate(resources()) * 48, 578.016)
        for field, value in [
            ("gpus", 9),
            ("cpus", 129),
            ("memory_gb", 2016),
            ("nodes", 2),
            ("partition", "gpuA100x4"),
            ("wall_minutes", 2881),
        ]:
            r = resources()
            r[field] = value
            with self.assertRaises(ValueError):
                m.resources_valid(r)

    def test_packing_limits_and_only_qualified_mixtures(self):
        for composition in m.COMPOSITIONS:
            self.assertTrue(m.composition_allowed(composition))
            self.assertLessEqual(
                sum(n / m.LIMITS[a] for n, a in zip(composition, m.ARCHS)), 1
            )
            self.assertLessEqual(
                sum(n * m.CPUS[a] for n, a in zip(composition, m.ARCHS)), 16
            )
        for bad in [(8, 0, 0), (0, 6, 0), (0, 0, 5), (7, 5, 4), (-1, 0, 0), (1, 3, 1)]:
            self.assertFalse(m.composition_allowed(bad))

    def test_cpu_affinity_partition_is_disjoint_and_bounded(self):
        affinity = set(range(128, 256))
        lanes = m.cpu_partition(affinity, 8)
        self.assertEqual(len(set(sum(lanes, []))), 128)
        self.assertEqual(set(sum(lanes, [])), affinity)
        with self.assertRaises(ValueError):
            m.cpu_partition(range(127), 8)

    def test_budget_bounds_and_duplicate_submission_fence(self):
        b = self.budget()
        b.reserve("q", qualification_resources(15), "qualification")
        b.reserve("first", resources(), "campaign")
        self.assertAlmostEqual(
            sum(x["reserved_su"] for x in m.read(b.path)["entries"].values()), 578.266
        )
        for key, r, cat in [
            ("first", resources(), "campaign"),
            ("again", resources(), "campaign"),
            ("q2", qualification_resources(120), "qualification"),
        ]:
            with self.assertRaises(ValueError):
                b.reserve(key, r, cat)
        self.assertEqual(len(m.read(b.path)["entries"]), 2)

    def test_racing_budget_reservations_cannot_overspend(self):
        b = self.budget()
        won = []

        def reserve(i):
            try:
                b.reserve(str(i), resources(), "campaign")
                won.append(i)
            except ValueError:
                pass

        threads = [threading.Thread(target=reserve, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(won), 1)

    def test_ambiguous_submission_stays_reserved(self):
        b = self.budget()
        b.reserve("q", qualification_resources(15), "qualification")
        with self.assertRaises(ValueError):
            b.submitted("q", "not-a-job")
        with self.assertRaises(ValueError):
            b.settle("q")
        self.assertEqual(m.read(b.path)["entries"]["q"]["status"], "RESERVED")

    def test_budget_releases_only_accounted_terminal_charge(self):
        b = self.budget()
        b.reserve("q", qualification_resources(15), "qualification")
        b.submitted("q", "123")
        with patch.object(
            m, "checked", return_value="123|RUNNING|60|billing=1000,gres/gpu=1|"
        ):
            with self.assertRaises(ValueError):
                b.settle("q")
        with patch.object(
            m, "checked", return_value="123|COMPLETED|600|billing=1000,gres/gpu=1|"
        ):
            result = b.settle("q")
        self.assertAlmostEqual(result["actual_su"], 1 / 6)
        self.assertAlmostEqual(result["actual_gpu_hours"], 1 / 6)

    def test_gpu_inventory_refuses_partial_or_wrong_device(self):
        lines = "\n".join(
            f"{i}, GPU-{i}, NVIDIA A100-SXM4-40GB, 40960" for i in range(8)
        )
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(8))),
            "SLURM_JOB_GPUS": ",".join(map(str, range(8))),
        }
        with patch.dict(m.os.environ, env), patch.object(
            m, "checked", return_value=lines
        ):
            self.assertEqual(len(m.gpu_inventory(8)), 8)
            with patch.dict(m.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(ValueError):
                    m.gpu_inventory(8)
        with patch.dict(m.os.environ, env), patch.object(
            m, "checked", return_value=lines.replace("40GB", "80GB")
        ):
            with self.assertRaises(ValueError):
                m.gpu_inventory(8)

    def test_slurm_gpu_ranges_and_duplicates(self):
        self.assertEqual(m.gpu_tokens("0-7"), list(map(str, range(8))))
        for invalid in ("0,0", "0-100", "3-1"):
            with self.assertRaises(ValueError):
                m.gpu_tokens(invalid)

    def admission(self):
        return {
            "profiles": {
                a: dict(
                    cpus=m.CPUS[a],
                    host_ram_gb=10,
                    gpu_memory_mb=2000,
                    checkpoint_minutes=5,
                    estimated_minutes=200,
                )
                for a in m.ARCHS
            },
            "host_headroom_gb": 16,
            "gpu_headroom_mb": 2048,
            "stop_margin_minutes": 10,
        }

    def job(self, arch, i):
        return dict(
            item={"id": str(i), "row": {"architecture": arch}},
            gpu="GPU-0",
            profile=self.admission()["profiles"][arch],
        )

    def test_admission_counts_validation_and_live_memory(self):
        a = self.admission()
        active = [self.job("D4W1536", i) for i in range(3)]
        item = self.job("D4W1536", 4)["item"]
        args = (
            item,
            "GPU-0",
            active,
            [],
            a,
            resources(),
            time.time() + 3600,
            (40960, 8000),
            100,
        )
        self.assertTrue(m.can_admit(*args))
        active.append(self.job("D4W1536", 3))
        self.assertFalse(m.can_admit(*args))
        active.pop()
        self.assertFalse(
            m.can_admit(
                item,
                "GPU-0",
                active,
                [self.job("D4W1536", 9)],
                a,
                resources(),
                time.time() + 3600,
                (40960, 8000),
                100,
            )
        )
        self.assertFalse(
            m.can_admit(
                item,
                "GPU-0",
                active,
                [],
                a,
                resources(),
                time.time() + 3600,
                (40960, 39000),
                100,
            )
        )
        self.assertFalse(
            m.can_admit(
                item,
                "GPU-0",
                active,
                [],
                a,
                resources(),
                time.time() + 3600,
                (40960, 8000),
                10,
            )
        )

    def test_duplicate_parent_and_too_late_admission_refused(self):
        a = self.admission()
        job = self.job("D2W512", 0)
        self.assertFalse(
            m.can_admit(
                job["item"],
                "GPU-1",
                [job],
                [],
                a,
                resources(),
                time.time() + 3600,
                (40960, 0),
                100,
            )
        )
        self.assertFalse(
            m.can_admit(
                job["item"],
                "GPU-0",
                [],
                [],
                a,
                resources(),
                time.time() + 60,
                (40960, 0),
                100,
            )
        )

    def test_source_fingerprint_matches_frozen_reference_and_rejects_changed_training(
        self,
    ):
        # The frozen smoke stays immutable; later engineering commits must not
        # inherit its qualification merely because this test runs at HEAD.
        self.assertEqual(
            m.training_fingerprint(m.BASE_SMOKE_SHA),
            "718a7ec41f3ed8814472bf1eca125f0bae85c69f76712d1008e585e0bf02acf9",
        )
        receipt, _, _, q = self.fixture_receipt()
        q.update(expected_commit="HEAD", training_fingerprint="changed-source")
        m.write(receipt, q, replace=True)
        with patch.object(
            m,
            "training_fingerprint",
            side_effect=["changed-source", "changed-source", "frozen-source"],
        ):
            with self.assertRaisesRegex(ValueError, "source-bound"):
                m.qualification(receipt, "HEAD")
        with patch.object(m, "checked", return_value=""):
            with self.assertRaises(ValueError):
                m.training_fingerprint("HEAD")

    def test_planned_pause_retires_only_stopped_validated_owner(self):
        from unittest.mock import Mock
        from scripts import exp12_dynamic_scheduler as d

        attempt = self.root / "attempt"
        attempt.mkdir()
        ledger = Mock()
        p = Mock()
        p.poll.return_value = None
        p.pid = 98765
        p.returncode = -15
        job = dict(
            process=p, item={"id": "one"}, owner={"token": "owner"}, attempt=attempt
        )
        with patch.object(
            d, "validate_resume", return_value=True
        ) as validate, patch.object(d, "_finish") as finish, patch.object(
            m.os, "killpg"
        ) as kill:
            m.pause(job, ledger, "a" * 40)
        validate.assert_called_once_with(job["item"], "a" * 40)
        kill.assert_called_once()
        p.wait.assert_called_once()
        ledger.finish.assert_called_once_with(job["item"], job["owner"])
        finish.assert_not_called()
        self.assertTrue(m.read(attempt / "planned_pause.json")["checkpoint_available"])

    def test_planned_pause_invalid_checkpoint_blocks_instead_of_resuming(self):
        from unittest.mock import Mock
        from scripts import exp12_dynamic_scheduler as d

        ledger = Mock()
        p = Mock()
        p.poll.return_value = None
        p.pid = 98765
        job = dict(
            process=p, item={"id": "one"}, owner={"token": "x"}, attempt=self.root
        )
        with patch.object(
            d, "validate_resume", side_effect=ValueError("corrupt")
        ) as validate, patch.object(d, "_finish"), patch.object(m.os, "killpg"):
            m.pause(job, ledger, "a" * 40)
        validate.assert_called_once_with(job["item"], "a" * 40)
        self.assertIn("corrupt", ledger.finish.call_args.kwargs["error"])

    def test_submit_timeout_never_frees_or_retries_reservation(self):
        b = self.budget()
        with patch.object(m, "source"), patch.object(m, "live_request"), patch(
            "scripts.exp96_multigpu_qualification.reusable_checks"
        ) as reuse, patch.object(
            m, "checked", side_effect=TimeoutError("ambiguous sbatch")
        ):
            with self.assertRaises(TimeoutError):
                m.submit(
                    "a" * 40,
                    self.root / "submission",
                    qualification_resources(15),
                    b.root,
                    "qualification",
                    reuse_checks=self.root / "validated-component-fixture.json",
                )
        reuse.assert_called_once_with(
            self.root / "validated-component-fixture.json", "a" * 40
        )
        data = m.read(b.path)
        self.assertEqual(len(data["entries"]), 1)
        self.assertEqual(next(iter(data["entries"].values()))["status"], "RESERVED")

    def fixture_receipt(self):
        smoke = self.root / "original.json"
        m.write(smoke, {"expected_commit": m.BASE_SMOKE_SHA})
        curve = self.root / "curves.bin"
        curve.write_bytes(b"fixture only")
        receipt = self.root / "qualification.json"
        q = dict(
            schema=m.SCHEMA,
            verdict="PASS",
            expected_commit="a" * 40,
            limits=m.LIMITS,
            compositions=[list(c) for c in m.COMPOSITIONS],
            training_fingerprint="fixture",
            original_smoke=str(smoke),
            evidence_sha256={str(curve): m.digest(curve)},
            claims_race_passed=True,
            all_devices_isolated=True,
            production_orchestration_passed=True,
            gpu_startup_passed=True,
            job_id="123",
            resources=qualification_resources(15),
        )
        m.write(receipt, q)
        return receipt, smoke, curve, q

    def test_qualification_requires_exact_source_intact_artifacts_and_accounting(self):
        receipt, smoke, curve, q = self.fixture_receipt()
        with patch.object(m, "BASE_SMOKE_DIGEST", m.digest(smoke)), patch.object(
            m, "training_fingerprint", return_value="fixture"
        ), patch.object(
            m, "checked", return_value="123|COMPLETED|0:0|600|billing=1000,gres/gpu=1|"
        ):
            self.assertEqual(m.qualification(receipt, "a" * 40)["job_id"], "123")
            with self.assertRaises(ValueError):
                m.qualification(receipt, "b" * 40)
            curve.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                m.qualification(receipt, "a" * 40)
        curve.write_bytes(b"fixture only")
        with patch.object(m, "BASE_SMOKE_DIGEST", m.digest(smoke)), patch.object(
            m, "training_fingerprint", return_value="fixture"
        ), patch.object(
            m, "checked", return_value="123|RUNNING|0:0|600|billing=1000,gres/gpu=1|"
        ):
            with self.assertRaises(ValueError):
                m.qualification(receipt, "a" * 40)


if __name__ == "__main__":
    unittest.main()
