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

from experiments.exp12 import ledger, probe, trigger  # noqa: E402

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


real_load = ledger.load
real_last_eligible = trigger.last_eligible_check


def flipped_triggered(ci_low):
    return bool(np.isfinite(ci_low) and ci_low < 0)


def non_strict_triggered(ci_low):
    return bool(np.isfinite(ci_low) and ci_low >= 0)


def mean_bootstrap_interval(loss_rounds, seed, check_index, reps=trigger.REPS, confidence=trigger.CONFIDENCE):
    x = np.asarray(loss_rounds, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return float("nan"), float("nan")
    rng = np.random.default_rng([seed, trigger.BOOT_STREAM, check_index])
    stats = x[rng.integers(0, x.size, size=(reps, x.size))].mean(1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def keep_dev_load(results_root=None, include_dev=False, require_complete=True):
    return real_load(results_root, include_dev=True, require_complete=require_complete)


def source_mutation(module, attr, old, new):
    """attr rebuilt from module's source with one textual change (for logic written inline)."""
    import inspect

    source = inspect.getsource(module)
    if old not in source:
        raise ValueError(f"mutation anchor not found in {module.__name__}: {old!r}")
    namespace = {"__name__": module.__name__ + "_mutated"}
    exec(compile(source.replace(old, new), module.__file__, "exec"), namespace)
    return namespace[attr]


from experiments.exp12 import trainer as trainer_module  # noqa: E402

PHASE3 = "tests.test_exp12_phase3"
MUTATIONS = [
    ("pairing: current/fresh get different targets+minibatches", probe, "probe_round", unpaired_probe_round,
     "tests.test_exp12_probe.ProbePairingTest.test_identical_inputs_targets_and_minibatches"),
    ("offset: a forced to 0 instead of each critic's own mean", probe, "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbePairingTest.test_per_critic_offset_is_each_critics_own_mean"),
    ("offset: a forced to 0 (known-answer test)", probe, "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("baseline b dropped from P", probe, "probe_round", no_baseline_probe_round,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("sign: L = P(current) - P(fresh)", probe, "summarize", flipped_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_sign_convention_plasticity_loss_positive_when_current_is_worse"),
    ("non-finite check treated as valid", probe, "summarize", always_valid_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_non_finite_round_marks_check_invalid"),
    ("trigger fires on the gain side (upper tail)", trigger, "triggered", flipped_triggered,
     f"{PHASE3}.TriggerTest.test_current_worse_triggers_and_reverse_never_does"),
    ("trigger fires when the lower bound equals 0", trigger, "triggered", non_strict_triggered,
     f"{PHASE3}.TriggerTest.test_edge_cases"),
    ("f*_run allowed at 20/20 (past 95% of budget)", trigger, "last_eligible_check", lambda checks, frac=0.95: checks,
     f"{PHASE3}.TriggerTest.test_f_star_first_eligible_check_only"),
    ("bootstrap statistic is the mean, not the IQM", trigger, "bootstrap_interval", mean_bootstrap_interval,
     f"{PHASE3}.TriggerTest.test_statistic_is_iqm_not_mean"),
    ("dev runs not excluded by default", ledger, "load", keep_dev_load,
     f"{PHASE3}.LedgerTest.test_round_trip_and_dev_excluded_by_default"),
    ("A9: random action only on step 1 (angle_1 rule) instead of until min_length", trainer_module,
     "Exp12Trainer",
     source_mutation(trainer_module, "Exp12Trainer", "if not self.buffer.can_sample():", "if self.timestep is None:"),
     "tests.test_exp12_foundations.SimbaRandomWarmupTest.test_random_until_min_length_then_policy"),
]


def _run(test_id):
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(suite)
    return result.wasSuccessful()


def main():
    ok = True
    import importlib

    for label, module, attr, replacement, test_id in MUTATIONS:
        test_module = importlib.import_module(test_id.rsplit(".", 2)[0])
        patches = [mock.patch.object(module, attr, replacement)]
        if getattr(test_module, attr, None) is getattr(module, attr):  # names a test module imported directly
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
