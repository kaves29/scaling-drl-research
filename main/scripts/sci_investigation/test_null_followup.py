"""Fixtures in the exact b4a90cb null_mode row format.

Run from main/: python -m unittest scripts.sci_investigation.test_null_followup
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp12.probe import iqm
from experiments.exp12.trigger import bootstrap_interval, triggered
from scripts.sci_investigation import null_followup as nf


def make_row(arch, seed, pair, loss):
    fresh = np.float32(0.49) + np.zeros(5, np.float32)
    current = (fresh - np.asarray(loss, np.float32)).astype(np.float32)
    loss = fresh - current
    low, high = bootstrap_interval(loss, seed, pair + 1, 10_000, 0.95)
    return {"arch": arch, "env": "dog-run", "seed": seed, "pair": pair, "check_index": pair + 1,
            "init_keys": [10_000 + pair, 0, 1], "score_fresh_rounds": fresh.tolist(),
            "score_current_rounds": current.tolist(), "b_rounds": [0.5] * 5, "loss_rounds": loss.tolist(),
            "loss_iqm": iqm(loss), "valid": True, "ci_low": low, "ci_high": high, "resamples": 10_000,
            "confidence": 0.95, "null_threshold": 0.0, "fired": triggered(low, 0.0)}


def write(root, arch, rows, summary_rows=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"null_pairs_{arch}.jsonl").write_text("".join(
        r if isinstance(r, str) else json.dumps(r) + "\n" for r in rows))
    good = [r for r in (summary_rows or rows) if isinstance(r, dict)]
    losses = np.array([r["loss_iqm"] for r in good])
    (root / f"null_summary_{arch}.json").write_text(json.dumps({
        "arch": arch, "env": "dog-run", "null_pairs": len(good),
        "per_check_fire_rate": float(np.mean([r["fired"] for r in good])),
        "fire_rate_exceeds_5pct": bool(np.mean([r["fired"] for r in good]) > 0.05),
        "would_be_null_threshold_p95_of_L": float(np.percentile(losses, 95)),
        "iqm_of_L": iqm(losses), "std_of_L": float(np.std(losses, ddof=1))}))


def dataset(seed, mu, noise_seed):
    rng = np.random.default_rng(noise_seed)
    return [make_row("D4W1024", seed, p, mu[p] + 1e-3 * rng.normal(size=5)) for p in range(100)]


class NullFollowupTest(unittest.TestCase):
    def run_cli(self, original, replication=None):
        out = Path(self.tmp) / "report.json"
        args = ["--original", str(original), "--archs", "D4W1024", "--out", str(out)]
        if replication:
            args += ["--replication", str(replication)]
        code = nf.main(args)
        return code, json.loads(out.read_text())["archs"]["D4W1024"]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_clean_iid_null_passes_integrity_and_shows_no_pair_effect(self):
        write(Path(self.tmp) / "o", "D4W1024", dataset(990, np.zeros(100), 1))
        write(Path(self.tmp) / "r", "D4W1024", dataset(991, np.zeros(100), 2))
        code, e = self.run_cli(Path(self.tmp) / "o", Path(self.tmp) / "r")
        self.assertEqual(code, 0)
        self.assertEqual(e["original"]["integrity_errors"], [])
        self.assertGreater(e["original"]["persistence"]["permutation_p_between_pair_var"], 0.01)
        self.assertGreater(e["replication_comparison"]["permutation_p_one_sided"], 0.01)
        self.assertNotIn("PERSISTENT", e["diagnostic_reading"])

    def test_persistent_pair_effect_is_detected_within_and_across_seeds(self):
        mu = 1e-3 * np.random.default_rng(3).normal(size=100)  # tau = sigma_w
        write(Path(self.tmp) / "o", "D4W1024", dataset(990, mu, 4))
        write(Path(self.tmp) / "r", "D4W1024", dataset(991, mu, 5))
        code, e = self.run_cli(Path(self.tmp) / "o", Path(self.tmp) / "r")
        self.assertEqual(code, 0)
        self.assertLess(e["original"]["persistence"]["permutation_p_between_pair_var"], 0.01)
        self.assertLess(e["replication_comparison"]["permutation_p_one_sided"], 0.01)
        self.assertTrue(e["diagnostic_reading"].startswith("PERSISTENT"))
        self.assertGreater(e["original"]["approved_null_rule"]["fire_rate"], 0.05)
        p95 = e["p95_alternative_REPORT_ONLY"]
        self.assertLessEqual(p95["in_sample_fire_rate"], 0.06)
        self.assertIn("out_of_sample_fire_rate_on_replication", p95)

    def test_corruptions_are_integrity_errors(self):
        base = dataset(990, np.zeros(100), 6)
        cases = {  # name: (rows, substring of the specific error)
            "duplicate": (base + [base[3]], "duplicate pair 3 (first at line 4, identical"),
            "duplicate_different": (base + [make_row("D4W1024", 990, 7, np.full(5, 1e-3))], "DIFFERENT"),
            "missing": (base[:50] + base[51:], "missing pairs [50]"),
            "wrong_seed": (base[:9] + [make_row("D4W1024", 991, 9, base[9]["loss_rounds"])] + base[10:], "pair 9: seed 991"),
            "malformed": (base[:20] + ["{not json\n"] + base[20:], "malformed JSON"),
            "tampered_fired": (base[:5] + [{**base[5], "fired": not base[5]["fired"]}] + base[6:], "pair 5: stored fired"),
            "loss_mismatch": (base[:8] + [{**base[8], "loss_rounds": [0.1] * 5}] + base[9:], "pair 8: loss_rounds"),
            "short_rounds": (base[:2] + [{**base[2], "score_fresh_rounds": [0.4] * 4}] + base[3:], "pair 2: score_fresh_rounds"),
        }
        for name, (rows, expected) in cases.items():
            with self.subTest(name):
                root = Path(self.tmp) / name
                write(root, "D4W1024", rows, summary_rows=base)
                code, e = self.run_cli(root)
                self.assertEqual(code, 2, name)
                self.assertTrue(any(expected in x for x in e["original"]["integrity_errors"]),
                                (name, e["original"]["integrity_errors"]))
                self.assertTrue(e["diagnostic_reading"].startswith("EVIDENCE INVALID"), name)

    def test_summary_mismatch_is_an_integrity_error(self):
        rows = dataset(990, np.zeros(100), 8)
        root = Path(self.tmp) / "s"
        write(root, "D4W1024", rows)
        summary = json.loads((root / "null_summary_D4W1024.json").read_text())
        summary["per_check_fire_rate"] = 0.13
        (root / "null_summary_D4W1024.json").write_text(json.dumps(summary))
        code, e = self.run_cli(root)
        self.assertEqual(code, 2)
        self.assertTrue(any("per_check_fire_rate" in x for x in e["original"]["integrity_errors"]))

    def test_deterministic(self):
        mu = 1e-3 * np.random.default_rng(9).normal(size=100)
        write(Path(self.tmp) / "o", "D4W1024", dataset(990, mu, 10))
        _, a = self.run_cli(Path(self.tmp) / "o")
        _, b = self.run_cli(Path(self.tmp) / "o")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
