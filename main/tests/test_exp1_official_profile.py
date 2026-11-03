"""Population-independent checks for the owner's official protocol counts."""

import numpy as np

from experiments.exp1 import compose_config
from experiments.exp12.probe import iqm, check_steps
from experiments.exp12 import ledger
from tests.exp12_helpers import CONFIG_PATH
from experiments.exp12.official_exp1 import population


def test_official_population_is_legacy_without_all_humanoidbench():
    rows = population()
    assert len(rows) == 330
    assert len({(r["environment"], r["architecture"], r["seed"]) for r in rows}) == 330
    assert len({r["environment"] for r in rows}) == 11
    assert {r["seed"] for r in rows} == set(range(1, 11))
    assert not any(r["environment"].startswith("h1-") for r in rows)


def test_official_profile_has_ten_targets_continuous_2000_and_fifty_checks():
    cfg = compose_config(CONFIG_PATH, "base_exp1_official", [])
    assert (cfg.probe.rounds, cfg.probe.steps, cfg.probe.checks) == (10, 2000, 50)
    assert cfg.probe.target_scale == 100000
    assert cfg.probe.pool_size == 25600
    assert cfg.probe.batch_size == 256
    assert cfg.trigger.consecutive_checks == 2
    steps = check_steps(int(cfg.num_interaction_steps), cfg.probe.checks)
    assert len(steps) == 50 and steps[-1] == cfg.num_interaction_steps


def test_fractional_ten_iqm_retains_historical_five_behavior():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 80.0, 90.0])
    expected = (0.5 * values[2] + values[3:7].sum() + 0.5 * values[7]) / 5
    assert iqm(values) == expected
    assert iqm([1, 2, 3, 4, 100]) == 3


def test_ten_round_ledger_schema_keeps_all_rounds():
    columns = ledger.check_columns(10)
    for name in ("score_current", "score_fresh", "loss"):
        assert all(f"{name}_r{r}" in columns for r in range(10))
    assert columns == list(dict.fromkeys(columns))
