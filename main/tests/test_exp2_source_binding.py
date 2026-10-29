"""Admission tests: no environment, training or probe execution."""

import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from experiments import exp2_arm
from utils import run_metadata


class ArmSourceBindingTest(unittest.TestCase):
    def check(self, parent_code, current_code, config=None):
        cfg = OmegaConf.create({"seed": 7})
        parent = {"code": parent_code, "resolved_config": config or {"seed": 7}}
        with mock.patch.object(
            exp2_arm, "load_run_metadata", return_value=parent
        ), mock.patch.object(run_metadata, "code_version", return_value=current_code):
            exp2_arm.check_matches_parent(cfg, Path("/unused/parent"))

    def test_exact_clean_source_accepted(self):
        code = {"commit": "a" * 40, "dirty": False}
        self.check(code, code)

    def test_changed_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.check(
                {"commit": "a" * 40, "dirty": False},
                {"commit": "b" * 40, "dirty": False},
            )

    def test_missing_parent_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.check(None, {"commit": "a" * 40, "dirty": False})

    def test_missing_current_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.check(
                {"commit": "a" * 40, "dirty": False}, {"commit": None, "dirty": None}
            )

    def test_equal_unknown_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.check({"commit": None, "dirty": None}, {"commit": None, "dirty": None})

    def test_equal_dirty_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.check(
                {"commit": "a" * 40, "dirty": True}, {"commit": "a" * 40, "dirty": True}
            )

    def test_malformed_or_unverified_sources_rejected(self):
        for code in (
            {},
            {"commit": "a" * 7, "dirty": False},
            {"commit": "z" * 40, "dirty": False},
            {"commit": "a" * 40},
            {"commit": "a" * 40, "dirty": 0},
        ):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "source"):
                self.check(code, code)

    def test_config_mismatch_still_rejected(self):
        code = {"commit": "a" * 40, "dirty": False}
        with self.assertRaisesRegex(ValueError, "differs from the parent"):
            self.check(code, code, {"seed": 8})


if __name__ == "__main__":
    unittest.main(verbosity=2)
