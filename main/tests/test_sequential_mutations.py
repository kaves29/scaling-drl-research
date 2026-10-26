"""In-memory break/restore checks; never rewrite production source files."""

import inspect
from unittest.mock import patch

import numpy as np
import pytest

from diagnostics import sequential_probe as s
from tests.test_sequential_probe import package as make_package, run


@pytest.fixture
def package(tmp_path):
    return make_package.__wrapped__(tmp_path)


def assert_contract(package, output):
    results = run(package, output)
    previous = None
    first_c = None
    for k, result in enumerate(results, 1):
        with np.load(output / f"phase_{k:02d}" / "A.npz") as a, np.load(
            output / f"phase_{k:02d}" / "B.npz"
        ) as b, np.load(output / f"phase_{k:02d}" / "C.npz") as c:
            for arrays in (a, b, c):
                assert arrays["losses"].shape == (2000,)
                np.testing.assert_array_equal(arrays["targets"], a["targets"])
                np.testing.assert_array_equal(arrays["indices"], a["indices"])
            if previous is not None:
                for arm, arrays in (("A", a), ("B", b)):
                    np.testing.assert_array_equal(
                        arrays["pool_predictions"][0], previous[arm]
                    )
            previous = {
                "A": a["pool_predictions"][-1].copy(),
                "B": b["pool_predictions"][-1].copy(),
            }
            if k == 1:
                first_c = c["pool_predictions"][0].copy()
                for key in b.files:
                    np.testing.assert_array_equal(b[key], c[key])
            else:
                np.testing.assert_array_equal(c["pool_predictions"][0], first_c)
        assert result["C_minus_A"] == float(
            np.float32(result["scores"]["C"]) - np.float32(result["scores"]["A"])
        )


MUTATIONS = [
    (
        "lost_AB_carry",
        'state = {a: fitted[a] for a in ("A", "B")}',
        'state = {a: fresh for a in ("A", "B")}',
    ),
    (
        "C_carries_A",
        'start = fresh if arm == "C" else state[arm]',
        'start = state["A"] if arm == "C" else state[arm]',
    ),
    (
        "unpaired_minibatches",
        "network, tx, start, obs, act, target, idx, settings.chunk",
        "network, tx, start, obs, act, target, idx.at[0,0].set((idx[0,0]+ord(arm))%len(obs)), settings.chunk",
    ),
    (
        "shortened_fit",
        "network, tx, start, obs, act, target, idx, settings.chunk",
        "network, tx, start, obs, act, target, idx[:-1], settings.chunk",
    ),
    (
        "arm_specific_target",
        "network, tx, start, obs, act, target, idx, settings.chunk",
        "network, tx, start, obs, act, target+jnp.float32(ord(arm)-65), idx, settings.chunk",
    ),
    (
        "flipped_contrast",
        'summary.update(scores(variances[k], summary["final_mse"]))',
        'summary.update(scores(variances[k], summary["final_mse"])); summary["C_minus_A"] *= -1',
    ),
]


@pytest.mark.parametrize("name,old,new", MUTATIONS, ids=[x[0] for x in MUTATIONS])
def test_break_then_restore(package, tmp_path, name, old, new):
    source = inspect.getsource(s.evaluate)
    assert source.count(old) == 1
    namespace = dict(vars(s))
    exec(compile(source.replace(old, new), s.__file__, "exec"), namespace)
    with patch.object(s, "evaluate", namespace["evaluate"]):
        with pytest.raises((AssertionError, ValueError)):
            assert_contract(package, tmp_path / (name + "_broken"))
    assert_contract(package, tmp_path / (name + "_restored"))
