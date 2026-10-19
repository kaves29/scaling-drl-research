"""Only regressions needed by the short, component-reusing qualification fix."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m
from scripts import exp96_multigpu_qualification as q


def resources(minutes):
    return dict(
        partition="gpuA100x4",
        nodes=1,
        gpus=1,
        cpus=16,
        memory_gb=60,
        wall_minutes=minutes,
    )


class Continuation(unittest.TestCase):
    def test_vector_geometry_matches_actual_replay_not_batch_dimension(self):
        env = NS(
            observation_space=NS(shape=(1, 223)),
            action_space=NS(shape=(1, 38)),
            single_observation_space=NS(shape=(223,)),
            single_action_space=NS(shape=(38,)),
        )
        replay = NS(
            _observations=NS(shape=(1000000, 223)), _actions=NS(shape=(1000000, 38))
        )
        self.assertEqual(q.engineering_dimensions(env, replay), (223, 38))
        for attribute in ("single_observation_space", "single_action_space"):
            bad = copy.deepcopy(env)
            setattr(bad, attribute, NS(shape=(1,)))
            with self.assertRaises(ValueError):
                q.engineering_dimensions(bad, replay)
        replay._actions.shape = (1000000, 39)
        with self.assertRaises(ValueError):
            q.engineering_dimensions(env, replay)

    def test_short_walltime_and_cumulative_physical_limit(self):
        for minutes in (1, 30, 120):
            m.resources_valid(resources(minutes), qualification=True)
        for minutes in (0, 121, 2880):
            with self.assertRaises(ValueError):
                m.resources_valid(resources(minutes), qualification=True)
        with tempfile.TemporaryDirectory() as tmp:
            b = m.Budget(Path(tmp) / "budget")
            b.initialize("owner authorization, test fixture")
            b.reserve("old", resources(120), "qualification")
            data = m.read(b.path)
            data["entries"]["old"].update(
                resources=dict(
                    partition="gpuA100x8",
                    nodes=1,
                    gpus=8,
                    cpus=128,
                    memory_gb=2007,
                    wall_minutes=15,
                ),
                reserved_su=3.0105,
                physical_gpu_hours=2,
            )
            m.write(b.path, data, replace=True)
            b.submitted("old", "22990946")
            with patch.object(
                m,
                "checked",
                return_value="22990946|FAILED|108|billing=12042,gres/gpu=8|",
            ):
                old = b.settle("old")
            self.assertAlmostEqual(old["actual_gpu_hours"], 0.24)
            self.assertAlmostEqual(old["actual_su"], 0.36126)
            with self.assertRaises(ValueError):
                b.reserve("too-long", resources(106), "qualification")
            b.reserve("rerun", resources(30), "qualification")
            self.assertAlmostEqual(old["actual_gpu_hours"] + 30 / 60, 0.74)

    def test_gate_uses_actual_short_slurm_bound(self):
        receipt = dict(
            schema=m.SCHEMA,
            verdict="PASS",
            expected_commit="new",
            limits=m.LIMITS,
            compositions=[list(c) for c in m.COMPOSITIONS],
            training_fingerprint="same",
            original_smoke="smoke.json",
            evidence_sha256={},
            claims_race_passed=True,
            all_devices_isolated=True,
            production_orchestration_passed=True,
            gpu_startup_passed=True,
            resources=resources(13),
            job_id="123",
        )

        def read(path):
            return (
                {"expected_commit": m.BASE_SMOKE_SHA}
                if str(path) == "smoke.json"
                else receipt
            )

        with patch.object(m, "read", side_effect=read), patch.object(
            m, "digest", return_value=m.BASE_SMOKE_DIGEST
        ), patch.object(m, "training_fingerprint", return_value="same"):
            for state, code, seconds, valid in (
                ("COMPLETED", "0:0", 780, True),
                ("COMPLETED", "0:0", 781, False),
                ("FAILED", "3:0", 60, False),
            ):
                with patch.object(
                    m,
                    "checked",
                    return_value=f"123|{state}|{code}|{seconds}|gres/gpu=1|",
                ):
                    if valid:
                        m.qualification("report.json", "new")
                    else:
                        with self.assertRaises(ValueError):
                            m.qualification("report.json", "new")

    def fixture(self, root):
        evidence = {}
        for i in range(8):
            p = root / "qualification" / "production_assignment" / str(i) / "ready.json"
            m.write(p, dict(pid=i + 1, gpu=f"GPU-{i}", cpu_ids=[i * 16, i * 16 + 1]))
            evidence[str(p)] = m.digest(p)
        for i in range(12):
            m.write(
                root / "qualification" / "claims_race" / f"{i}.json",
                dict(won=i == 0, owner=dict(job_id="22990946", token=str(i))),
            )
        receipt = root / "qualification.json"
        m.write(
            receipt,
            dict(
                expected_commit=q.REUSABLE_SOURCE,
                job_id="22990946",
                verdict="INCOMPLETE",
                production_orchestration_passed=True,
                gpu_startup_passed=True,
                claims_race_passed=True,
                training_fingerprint=m.training_fingerprint("HEAD"),
                limits=m.LIMITS,
                compositions=[list(c) for c in m.COMPOSITIONS],
                evidence_sha256=evidence,
            ),
        )
        return receipt

    def test_passed_components_bind_hashes_and_reject_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = self.fixture(root)
            binding = q.reusable_checks(receipt, "HEAD")
            self.assertEqual(len(binding["evidence_sha256"]), 20)
            q.reusable_checks(receipt, "HEAD", binding)
            path = root / "qualification" / "claims_race" / "1.json"
            x = m.read(path)
            x["won"] = True
            m.write(path, x, replace=True)
            with self.assertRaises(ValueError):
                q.reusable_checks(receipt, "HEAD", binding)

    def test_changed_implementation_or_assignment_cannot_inherit_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = self.fixture(root)
            original = q.implementation_nodes

            def changed(commit, path, names):
                nodes = original(commit, path, names)
                if commit == "HEAD":
                    nodes[names[0]] = "changed"
                return nodes

            with patch.object(q, "implementation_nodes", side_effect=changed):
                with self.assertRaises(ValueError):
                    q.reusable_checks(receipt, "HEAD")
            p = root / "qualification" / "production_assignment" / "1" / "ready.json"
            x = m.read(p)
            x["gpu"] = "GPU-0"
            m.write(p, x, replace=True)
            with self.assertRaises(ValueError):
                q.reusable_checks(receipt, "HEAD")


if __name__ == "__main__":
    unittest.main()
