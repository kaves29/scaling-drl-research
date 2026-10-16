"""CPU-only contract tests for the isolated 96-parent Method C campaign."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from scripts import exp96_campaign as campaign
from scripts import exp12_dynamic_scheduler as scheduler


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    return campaign.rows(tmp_path_factory.mktemp("exp96") / "campaign")


def test_exact_96_and_source_bound_identities(population):
    assert len(population) == 96
    assert {r["environment"] for r in population} == {e for e, _ in campaign.ENVS}
    assert {r["architecture"] for r in population} == {"D2W512", "D4W1024", "D4W1536"}
    assert {r["seed"] for r in population} == set(range(1, 9))
    assert {r["group"] for r in population} == {"dmc_hard", "myosuite_simba"}
    assert all(r["study"] == campaign.VERSION and r["budget"] == 1_000_000 for r in population)
    assert all(r["command"].count("--checkpoint_interval") == 1 and
               r["command"][-3:] == ["--checkpoint_interval", "10000", "--checkpoint_start_frac", "0.0"][-3:]
               for r in population)
    assert all("base_exp12_ten" in r["command"] for r in population)
    assert len({scheduler.identity("exp1", r, "a" * 40) for r in population}) == 96
    assert scheduler.identity("exp1", population[0], "a" * 40) != scheduler.identity(
        "exp1", population[0], "b" * 40)


def test_packing_six_four_two_and_mixed_replacement(population, monkeypatch):
    by_arch = {a: next(r for r in population if r["architecture"] == a)
               for a in ("D2W512", "D4W1024", "D4W1536")}
    profiles = {a: {"cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1,
                    "estimated_minutes_by_group": {"dmc_hard": 1, "myosuite_simba": 1},
                    "min_interactions_per_second": .1} for a in by_arch}
    admission = {"packing": "ten_target", "validated_concurrency": True,
                 "max_processes_per_gpu": 6, "stop_margin_minutes": 1,
                 "host_headroom_gb": 1, "gpu_headroom_gb": 1,
                 "architectures": profiles}
    resources = {"cpus_per_node": 16, "memory_gb_per_node": 64}
    import time
    deadline = time.time() + 3600
    def eligible(arch, active):
        row = dict(by_arch[arch], run_key=f"next-{arch}")
        item = {"kind": "exp1", "id": row["run_key"], "row": row}
        return scheduler.can_admit(item, "0", active, admission, resources,
                                   deadline, (40_000, 1000), 64)
    def running(arch, number):
        return {"item": {"id": f"{arch}-{number}", "row": by_arch[arch]},
                "gpu": "0", "profile": profiles[arch], "rate": 2.0}
    small = [running("D2W512", i) for i in range(6)]
    assert not eligible("D2W512", small)
    medium = [running("D4W1024", i) for i in range(4)]
    assert not eligible("D4W1024", medium)
    large = [running("D4W1536", i) for i in range(2)]
    assert not eligible("D4W1536", large)
    mixed = [running("D2W512", 0), running("D4W1024", 0), running("D4W1536", 0)]
    assert not eligible("D2W512", mixed)  # 1/6 + 1/4 + 1/2 + 1/6 > 1
    assert not eligible("D4W1024", mixed)
    assert eligible("D2W512", mixed[:2])  # completed large is replaced immediately
    assert eligible("D4W1536", mixed[2:])  # completed small/medium can be replaced
    admission["validated_concurrency"] = False
    mixed[0]["rate"] = None
    assert not eligible("D2W512", mixed)


def test_approval_fail_closed_on_source_and_smoke(tmp_path):
    evidence = tmp_path / "smoke.json"
    evidence.write_text(json.dumps({"verdict": "PASS", "expected_commit": "a" * 40,
                                    "study_version": campaign.VERSION, "gpu": campaign.GPU}))
    approval = {"approved": True, "expected_commit": "a" * 40, "study_version": campaign.VERSION,
                "decision_reference": "owner-review", "parent_count": 96,
                "gates": {name: {"status": "APPROVED", "decision_reference": "owner-review"}
                          for name in ("ten_function_method", "scientific_calibration",
                                       "gpu_smoke", "allocation_and_storage", "cross_seed_analysis")}}
    approval["gates"]["gpu_smoke"]["evidence"] = {"path": str(evidence), "sha256": campaign.sha(evidence)}
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval))
    plan = {"commit": "a" * 40, "approval": str(path), "approval_sha256": campaign.sha(path)}
    campaign._approval(plan)
    evidence.write_text(evidence.read_text() + " ")
    with pytest.raises(ValueError, match="smoke evidence"):
        campaign._approval(plan)
    evidence.write_text(json.dumps({"verdict": "PASS", "expected_commit": "b" * 40,
                                    "study_version": campaign.VERSION, "gpu": campaign.GPU}))
    approval["gates"]["gpu_smoke"]["evidence"]["sha256"] = campaign.sha(evidence)
    path.write_text(json.dumps(approval))
    plan["approval_sha256"] = campaign.sha(path)
    with pytest.raises(ValueError, match="does not qualify"):
        campaign._approval(plan)


def test_shared_96_queue_freeze_and_tamper_detection(tmp_path, monkeypatch):
    commit = "a" * 40
    evidence = tmp_path / "smoke.json"
    evidence.write_text(json.dumps({"verdict": "PASS", "expected_commit": commit,
                                    "study_version": campaign.VERSION, "gpu": campaign.GPU}))
    gates = {name: {"status": "APPROVED", "decision_reference": "test-only"}
             for name in ("ten_function_method", "scientific_calibration", "gpu_smoke",
                          "allocation_and_storage", "cross_seed_analysis")}
    gates["gpu_smoke"]["evidence"] = {"path": str(evidence), "sha256": campaign.sha(evidence)}
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"approved": True, "expected_commit": commit,
                                    "study_version": campaign.VERSION, "parent_count": 96,
                                    "decision_reference": "test-only", "gates": gates}))
    monkeypatch.setattr(campaign, "_source", lambda _: None)  # no GPU/real owner approval in CPU test
    root, queue_root = tmp_path / "campaign", tmp_path / "queue"
    campaign.prepare(root, commit, approval, queue_root)
    queue, plan = campaign.load(queue_root)
    assert len(queue["parents"]) == 96
    assert len({item["id"] for item in queue["parents"]}) == 96
    assert (root / "source_freeze.json").is_file()
    queue["parents"].append(queue["parents"][0])
    (queue_root / "queue.json").write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="queue differs"):
        campaign.load(queue_root)


def test_allocation_requires_measured_profiles_and_fitting_walltime(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "queue.json").write_text("{}")
    commit = "a" * 40
    profiles = {arch: {"cpus": 2, "host_ram_gb": 4, "gpu_memory_gb": 4,
                       "min_interactions_per_second": .1,
                       "estimated_minutes_by_group": {"dmc_hard": 60, "myosuite_simba": 60}}
                for arch in ("D2W512", "D4W1024", "D4W1536")}
    approval = {"approved": True, "expected_commit": commit,
                "queue_sha256": campaign.sha(queue / "queue.json"),
                "decision_reference": "test-only",
                "resources": {"gpus": 1, "cpus": 16, "memory_gb": 64, "wall_minutes": 90},
                "admission": {"packing": "ten_target", "validated_concurrency": True,
                              "max_processes_per_gpu": 6, "gpu_headroom_gb": 2,
                              "host_headroom_gb": 4, "stop_margin_minutes": 5,
                              "architectures": profiles}}
    assert campaign._allocation(approval, queue, commit)[0]["gpus"] == 1
    approval["admission"]["validated_concurrency"] = False
    with pytest.raises(ValueError, match="passing smoke"):
        campaign._allocation(approval, queue, commit)
    approval["admission"]["validated_concurrency"] = True
    profiles["D4W1536"]["estimated_minutes_by_group"]["dmc_hard"] = 86
    with pytest.raises(ValueError, match="cannot fit"):
        campaign._allocation(approval, queue, commit)


def test_historical_launcher_population_unchanged(tmp_path):
    from scripts.exp12_launch_plan import cells
    assert len(cells(tmp_path, "parents")) == 195


def test_smoke_commands_are_dev_only_and_keep_full_probe_fit(tmp_path):
    from scripts.exp96_gpu_smoke import CASES, command
    assert {c[1] for c in CASES[:3]} == {"D2W512", "D4W1024", "D4W1536"}
    for case in CASES:
        cmd = command(case, tmp_path)
        assert "base_exp12_ten" in cmd
        assert "env.num_env_steps=40000" in cmd  # 20k interactions, action repeat 2
        assert "probe.checks=2" in cmd         # full checks at 10k and 20k
        assert "run_role=dev" in cmd and "fork.enabled=false" in cmd
        assert not any("probe.steps=" in arg or "probe.rounds=" in arg for arg in cmd)


def test_completion_rejects_missing_rounds_curves_and_check_steps(tmp_path):
    import numpy as np
    import pandas as pd
    run = tmp_path / "run"
    curves = run / "probes"
    curves.mkdir(parents=True)
    rows = []
    zero = np.zeros((10, 2000), np.float32)
    for k in range(51):
        row = {"check_index": k, "interaction_step": 5000 if k == 0 else k * 10_000,
               "loss_iqm": 0., "ci_low": 0., "ci_high": 0., "triggered": False, "valid": True}
        row.update({f"{stem}_r{r}": 0. for stem in ("score_current", "score_fresh", "loss")
                    for r in range(10)})
        rows.append(row)
        np.savez_compressed(curves / f"check_{k:02d}.npz",
                            fresh_score=np.zeros(10), current_score=np.zeros(10),
                            fresh_q1_losses=zero, fresh_q2_losses=zero,
                            current_q1_losses=zero, current_q2_losses=zero)
    pd.DataFrame(rows).to_csv(curves / "probe_checks.csv", index=False)
    assert campaign.validate_probe_artifacts(run, {"buffer": {"min_length": 5000}}, 500_000) == 50
    rows[47]["interaction_step"] = 470_001
    pd.DataFrame(rows).to_csv(curves / "probe_checks.csv", index=False)
    with pytest.raises(ValueError, match="mistimed"):
        campaign.validate_probe_artifacts(run, {"buffer": {"min_length": 5000}}, 500_000)
    rows[47]["interaction_step"] = 470_000
    pd.DataFrame(rows).to_csv(curves / "probe_checks.csv", index=False)
    np.savez_compressed(curves / "check_47.npz", fresh_score=np.zeros(10),
                        current_score=np.zeros(9), fresh_q1_losses=zero)
    with pytest.raises(ValueError, match="ten current_score rounds"):
        campaign.validate_probe_artifacts(run, {"buffer": {"min_length": 5000}}, 500_000)


def test_exp2_check_47_boundary_uses_unchanged_horizon():
    from experiments.exp1 import compose_config
    from experiments.exp12 import fork, trigger
    cfg = compose_config(str(campaign.MAIN / "configs"), "base_exp12_ten",
                         ["env_name=dog-run", "env=dmc_hard"])
    assert trigger.last_eligible_check(50, .95) == 47
    plan = fork.fork_plan(cfg, 470_000, 47)
    assert plan["arm_end_step"] == plan["control_end_step"] == 595_000
    assert 480_000 + 125_000 > 600_000


def test_fresh_pair_order_reversal_changes_sign_and_interval():
    import numpy as np
    from experiments.exp12.trigger import bootstrap_interval, triggered
    # These are paired per-function scores, with the same bootstrap indices in
    # each orientation. The test checks the arithmetic, not scientific calibration.
    a = np.array([.1, .2, -.3, .4, .5, -.6, .7, .8, -.9, 1.0], dtype=np.float32)
    b = np.array([.3, .1, -.1, .7, .4, -.2, .9, .7, -.6, 1.2], dtype=np.float32)
    forward = b - a
    backward = a - b
    np.testing.assert_array_equal(forward, -backward)
    lower, upper = bootstrap_interval(forward, 77, 2, 10_000, .95)
    reverse_lower, reverse_upper = bootstrap_interval(backward, 77, 2, 10_000, .95)
    assert (lower, upper) == pytest.approx((-reverse_upper, -reverse_lower), abs=1e-8)
    assert not (triggered(lower, 0) and triggered(reverse_lower, 0))


def test_probe_pair_order_symmetry_with_shared_target_and_pool():
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from experiments.exp12 import probe
    class Tiny(nn.Module):
        @nn.compact
        def __call__(self, observations, actions):
            return nn.Dense(1)(jnp.concatenate((observations, actions), axis=-1))
    net = Tiny()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 16
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    a = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
    b = net.init(jax.random.PRNGKey(2), obs[:1], act[:1])["params"]
    cfg = probe.ProbeConfig(10, 4, 50, 16, 4, 1e5, 8)
    key = probe.probe_key(42, 2, 0)
    forward = probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                                optax.adamw(1e-3), obs, act, key, cfg)
    reverse = probe.probe_round(net, {"current": (net, b), "fresh": (net, a)},
                                optax.adamw(1e-3), obs, act, key, cfg)
    np.testing.assert_array_equal(forward["fresh"]["score"], reverse["current"]["score"])
    np.testing.assert_array_equal(forward["current"]["score"], reverse["fresh"]["score"])
    assert float(forward["fresh"]["score"] - forward["current"]["score"]) == -float(
        reverse["fresh"]["score"] - reverse["current"]["score"])


def test_ten_target_natural_fork_validates_paired_twin_arithmetic():
    import numpy as np
    from experiments.exp1 import compose_config
    from experiments.exp12 import fork, probe, trigger
    from scripts.positive_control import _validate_natural_fork
    cfg = compose_config(str(campaign.MAIN / "configs"), "base_exp12_ten",
                         ["env_name=dog-run", "env=dmc_hard", "seed=1"])
    records = [{"check_index": 0, "interaction_step": 5000}]
    for check in (1, 2):
        current_q1 = np.full(10, .1, np.float32)
        current_q2 = np.full(10, .3, np.float32)
        fresh_q1 = np.full(10, .2, np.float32)
        fresh_q2 = np.full(10, .4, np.float32)
        loss = np.mean([fresh_q1 - current_q1, fresh_q2 - current_q2], axis=0)
        lower, upper = trigger.bootstrap_interval(loss, 1, check, 10_000, .95)
        row = {"check_index": check, "interaction_step": check * 10_000,
               "loss_iqm": probe.iqm(loss), "ci_low": lower, "ci_high": upper,
               "valid": True, "triggered": True}
        for r in range(10):
            row.update({"score_current_r" + str(r): float(np.mean([current_q1[r], current_q2[r]])),
                        "score_fresh_r" + str(r): float(np.mean([fresh_q1[r], fresh_q2[r]])),
                        "loss_r" + str(r): float(loss[r])})
            for q, current, fresh in ((1, current_q1, fresh_q1), (2, current_q2, fresh_q2)):
                row[f"score_current_q{q}_r{r}"] = float(current[r])
                row[f"score_fresh_q{q}_r{r}"] = float(fresh[r])
        records.append(row)
    plan = {**fork.fork_plan(cfg, 20_000, 2), "run_key": "fixture"}
    meta = {"interaction_step": 20_000,
            "extra_state": {"probe_records": records,
                            "f_star": {"check_index": 2, "interaction_step": 20_000}}}
    _validate_natural_fork(cfg, plan, meta)
    records[1]["loss_r0"] += .01
    with pytest.raises(ValueError, match="paired losses differ"):
        _validate_natural_fork(cfg, plan, meta)
