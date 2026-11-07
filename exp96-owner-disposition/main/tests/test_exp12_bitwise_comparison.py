"""Exact-equality gates reject equal values with differing representations."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.exp12 import state
from experiments.exp12.fork import check1
from scripts import compare_identity_fork


class BitwiseComparisonTest(unittest.TestCase):
    def test_dtype_and_signed_zero_are_not_value_equality(self):
        for a, b in (
            (np.array([1], np.float32), np.array([1], np.float64)),
            (np.array([0.0], np.float32), np.array([-0.0], np.float32)),
            (np.array([1], np.int32), np.array([1], np.uint32)),
            (np.array([1], "<f4"), np.array([1], ">f4")),
        ):
            with self.subTest(a=a.dtype, b=b.dtype):
                self.assertTrue(np.array_equal(a, b))
                self.assertFalse(state.bitwise_equal(a, b))
                self.assertFalse(state.bitwise_equal(b, a))

    def test_shape_and_nan_are_rejected_and_layout_is_irrelevant(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertTrue(state.bitwise_equal(values, np.asfortranarray(values)))
        self.assertTrue(state.bitwise_equal(values[:, ::2], values[:, ::2].copy()))
        self.assertFalse(state.bitwise_equal(values, values.reshape(4, 3)))
        self.assertFalse(state.bitwise_equal([np.nan], [np.nan]))
        self.assertFalse(
            state.bitwise_equal(np.array([object()]), np.array([object()]))
        )

    def test_nested_metadata_uses_exact_array_and_scalar_comparison(self):
        for a, b in (
            (np.array([0.0], np.float32), np.array([-0.0], np.float32)),
            (np.float32(1), np.float64(1)),
            (0.0, -0.0),
            (1, 1.0),
            (True, 1),
            (complex(0.0, 0.0), complex(0.0, -0.0)),
            ([1], (1,)),
        ):
            self.assertFalse(state._deep_equal({"x": [{"a": a}]}, {"x": [{"a": b}]}))
        self.assertTrue(state._deep_equal({"x": float("nan")}, {"x": float("nan")}))
        nan_a, nan_b = (
            float(np.array([bits], np.uint64).view(np.float64)[0])
            for bits in (0x7FF8000000000001, 0x7FF8000000000002)
        )
        self.assertTrue(state._deep_equal(nan_a, nan_a))
        self.assertFalse(state._deep_equal(nan_a, nan_b))
        self.assertTrue(state._deep_equal({"x": [1, 2.0]}, {"x": [1, 2.0]}))

    def test_saved_state_components_reject_representation_changes(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = [Path(root) / x for x in ("a", "b")]
            positive = np.array([0.0], np.float32)
            changed = (np.array([-0.0], np.float32), np.array([0.0], np.float64))

            def write(path, value, category):
                path.mkdir(exist_ok=True)
                for name, data in (
                    (
                        "obs_rms.pkl",
                        {"mean": value if category == "obs_rms" else positive},
                    ),
                    (
                        "buffer_meta.pkl",
                        {
                            "transitions": [
                                {
                                    "reward": (
                                        value if category == "buffer_meta" else positive
                                    )
                                }
                            ]
                        },
                    ),
                    ("meta.pkl", {"x": value if category == "meta" else positive}),
                ):
                    (path / name).write_bytes(pickle.dumps(data))
                np.savez(
                    path / "buffer.npz",
                    observations=value if category == "buffer" else positive,
                )

            for category in ("agent", "obs_rms", "buffer", "buffer_meta", "meta"):
                for value in changed:
                    with self.subTest(category=category, dtype=value.dtype):
                        write(dirs[0], positive, category)
                        write(dirs[1], value, category)
                        trees = [
                            {"x": positive},
                            {"x": value if category == "agent" else positive},
                        ]
                        with mock.patch.object(
                            state, "load_agent_tree", side_effect=trees
                        ):
                            diffs = state.state_differences(*dirs)
                        self.assertEqual(len(diffs), 1, diffs)
                        self.assertTrue(diffs[0].startswith(category + ":"), diffs)

    def test_check1_requires_bits_for_control_and_identity_only(self):
        pre = {
            "q": np.array([1.0, 0.0], np.float32),
            "dq_da": np.array([[1.0, 0.0]], np.float32),
        }
        for field in pre:
            altered = {k: v.copy() for k, v in pre.items()}
            altered[field].flat[-1] = -0.0
            self.assertFalse(check1(pre, altered, pre, 64, injected=False)["pass"])
            self.assertFalse(check1(pre, pre, altered, 64, injected=True)["pass"])
            self.assertTrue(check1(pre, altered, pre, 64, injected=True)["pass"])
        self.assertTrue(check1(pre, pre, pre, 64, injected=False)["pass"])

    def test_check1_injected_64_eps_rule_is_unchanged(self):
        pre = {"q": np.ones(1, np.float32), "dq_da": np.ones((1, 1), np.float32)}
        for factor, expected in ((64, True), (65, False)):
            after = {
                k: v + np.float32(factor * np.finfo(np.float32).eps)
                for k, v in pre.items()
            }
            self.assertEqual(
                check1(pre, after, pre, 64, injected=True)["pass"], expected
            )

    def test_identity_comparator_reports_panel_dtype_bits_and_missing_keys(self):
        meta = {
            "extra_state": {"fork": {"fork_step": 12000}},
            "interaction_step": 13000,
        }
        positive = {
            "q": np.array([0.0], np.float32),
            "dq_da": np.zeros((1, 1), np.float32),
        }
        variants = (
            (positive, True),
            ({**positive, "q": np.array([-0.0], np.float32)}, False),
            ({**positive, "q": np.array([0.0], np.float64)}, False),
            ({"q": positive["q"]}, False),
            ({**positive, "extra": positive["q"]}, False),
        )
        for panel, expected in variants:
            with self.subTest(panel=panel):
                with mock.patch.object(
                    compare_identity_fork,
                    "latest_state_dir",
                    return_value=Path("/state"),
                ), mock.patch.object(
                    compare_identity_fork, "state_differences", return_value=[]
                ), mock.patch.object(
                    compare_identity_fork, "load_meta", return_value=meta
                ), mock.patch.object(
                    compare_identity_fork.fork,
                    "load_npz",
                    side_effect=[positive, panel],
                ):
                    result = compare_identity_fork.compare(
                        Path("/control"), Path("/identity")
                    )
                self.assertEqual(result["pass"], expected)
                if not expected:
                    self.assertTrue(
                        all(
                            x.startswith("check1_panel:") for x in result["differences"]
                        )
                    )

    def test_identity_metadata_rejects_signed_zero_dtype_and_nan(self):
        for field in ("probe_records", "post_fork_evals", "fork"):
            for left, right in (
                (0.0, -0.0),
                (np.float32(1), np.float64(1)),
                (float("nan"), float("nan")),
            ):
                with self.subTest(field=field, left=left, right=right):

                    def meta(value):
                        content = (
                            [{"value": value}] if field != "fork" else {"value": value}
                        )
                        return {
                            "extra_state": {field: content},
                            "interaction_step": 13000,
                        }

                    with mock.patch.object(
                        compare_identity_fork,
                        "latest_state_dir",
                        return_value=Path("/state"),
                    ), mock.patch.object(
                        compare_identity_fork, "state_differences", return_value=[]
                    ), mock.patch.object(
                        compare_identity_fork,
                        "load_meta",
                        side_effect=[meta(left), meta(right)],
                    ), mock.patch.object(
                        compare_identity_fork.fork, "load_npz", return_value={}
                    ):
                        result = compare_identity_fork.compare(
                            Path("/control"), Path("/identity")
                        )
                    self.assertFalse(result["pass"])
                    self.assertEqual(result["differences"], ["extra_state:" + field])


if __name__ == "__main__":
    unittest.main()
