"""Narrow owner-disposition regression checks; no simulation or GPU work."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp96_multigpu as m
from scripts import exp96_owner_disposition as w


class OwnerDisposition(unittest.TestCase):
    def fixture(self):
        q = dict(
            schema=w.SCHEMA,
            verdict="OWNER_AUTHORIZED_WITH_UNVERIFIED_LARGE_PACKING",
            expected_commit="f" * 40,
            limits=m.LIMITS,
            compositions=[list(c) for c in m.COMPOSITIONS],
            waived_requirements=w.WAIVERS,
            owner_decision="owner",
            owner_decision_sha256="hash",
            profiles={},
        )
        owner = dict(
            authorize_submission=True,
            accept_unverified_large_packing=True,
            limit_su=600,
            verbatim_instructions=["fixture only"],
        )
        return q, owner

    def test_disposition_is_not_a_pass_or_generic_bypass(self):
        q, owner = self.fixture()
        for key, value in (
            ("verdict", "PASS"),
            ("expected_commit", "e" * 40),
            ("waived_requirements", []),
        ):
            bad = copy.deepcopy(q)
            bad[key] = value
            with patch.object(m, "read", side_effect=[bad, owner]), patch.object(
                m, "digest", return_value="hash"
            ):
                with self.assertRaises(ValueError):
                    w.validate("receipt", "f" * 40)
        owner["authorize_submission"] = False
        with patch.object(m, "read", side_effect=[q, owner]), patch.object(
            m, "digest", return_value="hash"
        ):
            with self.assertRaises(ValueError):
                w.validate("receipt", "f" * 40)

    def test_scientific_bytes_cannot_be_waived(self):
        q, owner = self.fixture()
        with patch.object(m, "read", side_effect=[q, owner]), patch.object(
            m, "digest", return_value="hash"
        ), patch.object(m, "training_fingerprint", side_effect=["new", "old"]):
            with self.assertRaisesRegex(ValueError, "scientific source changed"):
                w.validate("receipt", "f" * 40)

    def test_only_gpu_gate_accepts_owner_waiver(self):
        a = dict(
            approved=True,
            expected_commit="f" * 40,
            study_version="study",
            parent_count=96,
            decision_reference="owner",
            gates={
                k: dict(status="APPROVED", decision_reference="owner")
                for k in (
                    "ten_function_method",
                    "scientific_calibration",
                    "gpu_smoke",
                    "allocation_and_storage",
                    "cross_seed_analysis",
                )
            },
        )
        a["gates"]["gpu_smoke"].update(
            status="OWNER_WAIVED", evidence=dict(path="receipt", sha256="hash")
        )
        plan = dict(approval="study", approval_sha256="hash", commit="f" * 40)
        from types import SimpleNamespace

        with patch.object(m, "read", return_value=a), patch.object(
            m, "digest", return_value="hash"
        ), patch.object(
            m, "core", return_value=SimpleNamespace(VERSION="study")
        ), patch.object(
            m,
            "qualification",
            return_value={"verdict": "OWNER_AUTHORIZED_WITH_UNVERIFIED_LARGE_PACKING"},
        ):
            self.assertNotEqual(m.study(plan)["verdict"], "PASS")
            a["gates"]["scientific_calibration"]["status"] = "OWNER_WAIVED"
            with self.assertRaisesRegex(ValueError, "unapproved study gate"):
                m.study(plan)


if __name__ == "__main__":
    unittest.main()
