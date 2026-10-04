"""Break-and-restore evidence: each mutation must make its invariant test fail.

Not collected by unittest discovery (no test_ prefix). Run:
    python tests/exp12_break_checks.py
Each line reports whether the named test FAILED under the mutation (expected)
and PASSED again without it.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from experiments.exp12 import probe  # noqa: E402

real_probe_round = probe.probe_round
real_fit = probe._fit
real_summarize = probe.summarize


def unpaired_probe_round(target_def, critics, tx, obs, act, key, cfg):
    out = {}
    for i, (name, critic) in enumerate(critics.items()):
        out.update(real_probe_round(target_def, {name: critic}, tx, obs, act, jax.random.fold_in(key, i), cfg))
    return out


def zero_offset_fit(network_def, tx, params, obs, act, base, idx, chunk):
    losses, final, offset = real_fit(network_def, tx, params, obs, act, base - 0.0, idx, chunk)
    # Re-run with a = 0 by shifting the targets back by the critic's own offset.
    losses0, final0, _ = real_fit(network_def, tx, params, obs, act, base - offset, idx, chunk)
    return losses0, final0, jnp.zeros_like(offset)


def flipped_summarize(result, current="current", fresh="fresh"):
    return real_summarize(result, current=fresh, fresh=current)


def no_baseline_probe_round(*args, **kwargs):
    out = real_probe_round(*args, **kwargs)
    for v in out.values():
        v["score"] = -v["final_loss"]
    return out


def always_valid_summarize(result, current="current", fresh="fresh"):
    s = real_summarize(result, current, fresh)
    s["valid"] = True
    return s


MUTATIONS = [
    ("pairing: current/fresh get different targets+minibatches", "probe_round", unpaired_probe_round,
     "tests.test_exp12_probe.ProbePairingTest.test_identical_inputs_targets_and_minibatches"),
    ("offset: a forced to 0 instead of each critic's own mean", "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbePairingTest.test_per_critic_offset_is_each_critics_own_mean"),
    ("offset: a forced to 0 (known-answer test)", "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("baseline b dropped from P", "probe_round", no_baseline_probe_round,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("sign: L = P(current) - P(fresh)", "summarize", flipped_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_sign_convention_plasticity_loss_positive_when_current_is_worse"),
    ("non-finite check treated as valid", "summarize", always_valid_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_non_finite_round_marks_check_invalid"),
]


def _run(test_id):
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(suite)
    return result.wasSuccessful()


def main():
    ok = True
    import tests.test_exp12_probe as test_module

    for label, attr, replacement, test_id in MUTATIONS:
        patches = [mock.patch.object(probe, attr, replacement)]
        if hasattr(test_module, attr):  # names the test module imported directly
            patches.append(mock.patch.object(test_module, attr, replacement))
        for p in patches:
            p.start()
        try:
            broken_passes = _run(test_id)
        finally:
            for p in patches:
                p.stop()
        restored_passes = _run(test_id)
        status = "OK" if (not broken_passes and restored_passes) else "PROBLEM"
        ok &= status == "OK"
        print(f"[{status}] {label}: mutated -> {'PASS' if broken_passes else 'FAIL'}, "
              f"restored -> {'PASS' if restored_passes else 'FAIL'}  ({test_id.split('.')[-1]})")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
