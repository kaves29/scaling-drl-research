"""New Delta integration only; no repeated scientific/GPU qualification."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import exp96_exp2_delta as delta


class DeltaIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.r = dict(
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=4,
            cpus_per_node=64,
            memory_gb_per_node=240,
            wall_minutes=480,
        )
        self.a = dict(
            expected_commit="a" * 40,
            scheduler_commit="b" * 40,
            queue_sha256="c" * 64,
            resources=self.r,
            funding_path=str(self.root / "funding.json"),
        )
        self.b = dict(
            schema=delta.SCHEMA,
            approved=True,
            decision_reference="test fixture only",
            limit_su=40,
            entries={},
            **{
                k: self.a[k]
                for k in ("expected_commit", "scheduler_commit", "queue_sha256")
            },
        )

    def save(self):
        Path(self.a["funding_path"]).write_text(json.dumps(self.b))

    def test_billing_and_real_node_memory(self):
        self.assertEqual(delta.su_cost(self.r), 32)
        self.assertEqual(delta.su_cost(dict(self.r, nodes=2)), 64)
        one = dict(
            self.r,
            gpus_per_node=1,
            cpus_per_node=16,
            memory_gb_per_node=60,
            wall_minutes=120,
        )
        self.assertEqual(delta.su_cost(one), 2)
        with self.assertRaises(ValueError):
            delta.su_cost(dict(self.r, memory_gb_per_node=256))
        view = "Account=biqc-delta-gpu Partition=gpuA100x4 AllocTRES=cpu=64,mem=240G,billing=4000,gres/gpu=4"
        delta.verify_billing(self.r, view)
        with self.assertRaises(ValueError):
            delta.verify_billing(self.r, view.replace("billing=4000", "billing=8000"))

    def test_unapproved_and_foreign_source_funding_fail(self):
        self.b["approved"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "authorization"):
            delta.funding_gate(self.a)
        self.b["approved"] = True
        self.b["expected_commit"] = "d" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            delta.funding_gate(self.a)

    def test_budget_is_cumulative_and_reservation_is_job_bound(self):
        self.b["entries"] = {
            "one": dict(
                job_id="123", status="SUBMITTED", resources=self.r, reserved_su=32
            )
        }
        self.save()
        delta.funding_gate(self.a, "123")
        with self.assertRaisesRegex(ValueError, "unique"):
            delta.funding_gate(self.a, "124")
        self.b["entries"]["two"] = copy.deepcopy(self.b["entries"]["one"])
        self.b["entries"]["one"]["actual_su"] = 0  # Cannot reclaim unverified credits.
        self.save()
        with self.assertRaisesRegex(ValueError, "ceiling"):
            delta.funding_gate(self.a, "123")

    def test_multigpu_parent_receipt_uses_existing_strict_gate(self):
        from scripts import exp96_exp2_arms as arms

        parent = self.root / "parent.json"
        receipt = self.root / "qualification.json"
        receipt.write_text(json.dumps({"schema": "exp96-multigpu-754-v1"}))
        parent.write_text(
            json.dumps({"gates": {"gpu_smoke": {"evidence": {"path": str(receipt)}}}})
        )
        queue = {"commit": "a" * 40}
        plan = {"approval": str(parent)}
        with mock.patch.object(
            arms.exp96_campaign,
            "load",
            side_effect=[ValueError("legacy gate"), (queue, plan)],
        ), mock.patch("scripts.exp96_multigpu.study") as gate:
            self.assertEqual(arms.qualified_parent(self.root), (queue, plan))
            gate.assert_called_once_with(plan)
        with mock.patch.object(
            arms.exp96_campaign,
            "load",
            side_effect=[ValueError("legacy gate"), (queue, plan)],
        ), mock.patch(
            "scripts.exp96_multigpu.study",
            side_effect=ValueError("actual qualification missing"),
        ):
            with self.assertRaisesRegex(ValueError, "actual qualification"):
                arms.qualified_parent(self.root)

    def test_unknown_receipt_cannot_bypass_parent_gate(self):
        from scripts import exp96_exp2_arms as arms

        p = self.root / "parent.json"
        q = self.root / "q.json"
        q.write_text(json.dumps({"schema": "unknown", "verdict": "PASS"}))
        p.write_text(
            json.dumps({"gates": {"gpu_smoke": {"evidence": {"path": str(q)}}}})
        )
        with mock.patch.object(
            arms.exp96_campaign,
            "load",
            side_effect=[ValueError("legacy gate"), ({}, {"approval": str(p)})],
        ):
            with self.assertRaisesRegex(ValueError, "legacy gate"):
                arms.qualified_parent(self.root)

    def test_isolated_arm_queue_binding_and_output_identity(self):
        from scripts import exp96_exp2_arms as arms
        from scripts import exp12_dynamic_scheduler as durable

        parent = self.root / "parent_queue"
        parent.mkdir()
        separate = self.root / "arms"
        separate.mkdir()
        output = self.root / "new_outputs"
        approval = dict(
            self.a, arm_queue_root=str(separate), arm_output_root=str(output)
        )
        binding = dict(
            parent_queue=str(parent),
            queue_sha256=self.a["queue_sha256"],
            parent_commit=self.a["expected_commit"],
            scheduler_commit=self.a["scheduler_commit"],
            arm_output_root=str(output),
        )
        (separate / "parent_binding.json").write_text(json.dumps(binding))
        self.assertEqual(arms.arm_queue(parent, approval), separate)
        binding["parent_commit"] = "d" * 40
        (separate / "parent_binding.json").write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, "binding"):
            arms.arm_queue(parent, approval)
        row = dict(
            architecture="D4W1024",
            environment="dog-run",
            seed=1,
            group="dmc_hard",
            budget=1000000,
            run_key="exp1_D4W1024_dog-run_seed1",
            checkpoint_dir=str(
                self.root / "old/checkpoints/exp1/D4W1024/dog-run/seed_1"
            ),
            overrides=["env_name=dog-run"],
            config_hash="parenthash",
            study="method-c-ten-96-v1",
        )
        row["command"] = [
            "python",
            "run.py",
            "--experiment",
            "exp1",
            "--checkpoint_dir",
            row["checkpoint_dir"],
        ]
        fork = Path(row["checkpoint_dir"]) / "fork"
        fork.mkdir(parents=True)
        (fork / "fork.json").write_text("{}")
        with mock.patch("scripts.exp96_campaign._config_hash", return_value="armhash"):
            item = durable._arm_item(
                {"id": "parent", "row": row}, "a" * 40, "last", arm_root=output
            )
        self.assertTrue(Path(item["row"]["checkpoint_dir"]).is_relative_to(output))
        self.assertEqual(item["row"]["source_config_hash"], "parenthash")
        self.assertIn(f'results_root={output/"results"}', item["row"]["overrides"])
        self.assertEqual(
            item["row"]["overrides"][-3:],
            [
                f'fork.source={row["checkpoint_dir"]}',
                "fork.arm=injected",
                "injection.m=last",
            ],
        )
        self.assertFalse((parent / "arms").exists())


if __name__ == "__main__":
    unittest.main()
