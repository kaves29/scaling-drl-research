"""CPU source-control checks; stand-ins never perform a scientific fit."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from audits.probe_discrepancy.investigate import choose_representatives, validate_pilot
from audits.probe_discrepancy.source_checks import (
    controlled_routing,
    function,
    original_source,
    structural_checks,
)
from audits.echocritic_forensic.evidence import CAMPAIGN


@pytest.fixture(scope="module")
def source():
    return original_source("main/experiments/exp12/probe.py")


def test_pinned_source_guards(source):
    assert all(r["passed"] for r in structural_checks(source))


def test_actual_routing_without_fitting(source):
    result = controlled_routing(source)
    assert all(v for k, v in result.items() if k != "scope")


@pytest.mark.parametrize(
    "before,after,failed",
    [
        ("targets = offset + base_targets", "targets = base_targets", "own_offset"),
        ("tx.init(params)", "tx.init({})", "fresh_optimizer_state_continuous_scan"),
        ("predict(fitted) - targets", "predict(params) - targets", "postfit_full_pool"),
        ("obs = agent._normalize(obs)", "obs = obs", "one_normalization"),
        ("jax.random.fold_in(idx_key, 1)", "idx_key", "continuous_2000_indices"),
        ("jnp.sin(scale * f)", "jnp.sin(f)", "teacher_own_architecture_new_parameters"),
    ],
)
def test_source_guards_detect_changes(source, before, after, failed):
    assert before in source
    checks = {
        r["check"]: r["passed"]
        for r in structural_checks(source.replace(before, after))
    }
    assert not checks[failed]


def test_original_pairing_sign_and_input_immutability(source):
    result = {
        "fresh": {"score": np.array([0.4, 0.3], np.float32)},
        "current": {"score": np.array([0.45, -0.2], np.float32)},
    }
    before = deepcopy(result)
    paired = function(source, "paired_loss", {"np": np})(result)
    assert paired[0] < 0 and paired[1] > 0
    for kind in result:
        np.testing.assert_array_equal(result[kind]["score"], before[kind]["score"])


def test_original_current_critic_selects_live_params():
    source = original_source("main/experiments/exp12/run_probes.py")
    trained, fresh = {"identity": "trained"}, {"identity": "fresh"}
    critic = SimpleNamespace(network_def=object(), params=trained)
    trainer = SimpleNamespace(_sac_agent=SimpleNamespace(critic=critic))
    owner = SimpleNamespace(fresh=fresh)
    fn = function(
        source, "current_critic", {"InjectedSACCritic": type("Injected", (), {})}
    )
    network, params = fn(owner, trainer)
    assert params is trained and params is not fresh and network is critic.network_def


def test_retrieval_selection_deterministic_and_distinct():
    rows = [{"seed": i, "id": str(i), "paired_loss_iqm": i} for i in range(1, 9)]
    typical, exceptional = choose_representatives(rows)
    assert (typical["seed"], exceptional["seed"]) == (4, 1)
    assert choose_representatives(rows[::-1]) == (typical, exceptional)
    with pytest.raises(ValueError):
        choose_representatives(rows[:7])


def test_pilot_validation_refuses_authorization_and_setting_changes():
    runs = []
    for arch, count in [("D4W1536", 8), ("D4W1024", 4), ("D2W512", 2)]:
        for _ in range(count):
            runs.append(
                {
                    "original_run_id": str(len(runs)),
                    "architecture": arch,
                    "training_source": CAMPAIGN,
                    "source_scientific_config": {
                        "probe": {
                            "steps": 2000,
                            "rounds": 10,
                            "checks": 50,
                            "pool_size": 25600,
                        },
                        "num_interaction_steps": 500000,
                        "agent": {
                            "critic_learning_rate": 1e-4,
                            "critic_weight_decay": 0.01,
                        },
                    },
                }
            )
    manifest = {
        "launch_approved": False,
        "executable": False,
        "new_campaign_root": None,
        "runs": runs,
    }
    validate_pilot(manifest)
    for field, value in [
        ("launch_approved", True),
        ("executable", True),
        ("new_campaign_root", "/old/results"),
    ]:
        bad = deepcopy(manifest)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_pilot(bad)
    bad = deepcopy(manifest)
    bad["runs"][0]["source_scientific_config"]["probe"]["steps"] = 1000
    with pytest.raises(ValueError):
        validate_pilot(bad)
