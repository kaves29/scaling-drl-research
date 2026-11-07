"""CPU qualification of the isolated ten-target Exp1 candidate.

Tiny-model tests prove wiring and recovery only; they do not calibrate the
full-width A100 detector or authorize the proposed official study.
"""

import json
import hashlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp12_helpers import CONFIG_PATH, fake_wandb_init, tiny_overrides  # noqa: E402
from experiments.exp1 import compose_config
from experiments import exp1
from experiments.exp12 import fork, ledger, probe, trigger
from experiments.exp12.run_probes import RunProbes
from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from scripts.method_c_ten_reference import bootstrap_iqm10, empirical_iqm10
from scripts import exp12_dynamic_scheduler as scheduler


class TinyCritic(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        return nn.Dense(1)(jnp.concatenate((observations, actions), axis=-1))


def test_v2_config_and_original_five_are_distinct():
    old = compose_config(CONFIG_PATH, "base_exp12", [])
    new = compose_config(CONFIG_PATH, "base_exp12_ten", [])
    assert (old.probe.rounds, old.probe.steps, old.probe.checks) == (5, 2000, 20)
    assert (new.probe.rounds, new.probe.steps, new.probe.checks) == (10, 2000, 50)
    assert new.project_name == "echocritic_method_c_ten_96"
    assert new.trigger.null_threshold == 0
    assert new.trigger.confidence == .95
    assert new.trigger.resamples == 10_000
    assert new.fork.horizon_fraction == old.fork.horizon_fraction == .25
    assert new.fork.max_total_fraction == old.fork.max_total_fraction == 1.2


def test_ten_targets_are_distinct_repeatable_and_paired():
    net = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 16
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    a = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
    b = net.init(jax.random.PRNGKey(2), obs[:1], act[:1])["params"]
    cfg = probe.ProbeConfig(10, 2, 50, 16, 4, 1e5, 8)
    keys, targets = [], []
    for r in range(10):
        key = probe.probe_key(11, 2, r)
        target_key = jax.random.split(key)[0]
        keys.append(tuple(map(int, np.asarray(target_key))))
        targets.append(np.asarray(probe._base_targets(net, target_key, obs, act, 8, 1e5)))
        paired = probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                                  optax.adamw(1e-3), obs, act, key, cfg)
        np.testing.assert_array_equal(paired["current"]["b"], paired["fresh"]["b"])
        assert float(paired["current"]["b"]) == pytest.approx(float(np.var(targets[-1])), rel=1e-6)
        np.testing.assert_array_equal(paired["current"]["losses"].shape, (2,))
    assert len(set(keys)) == 10
    assert all(not np.array_equal(targets[i], targets[j])
               for i in range(10) for j in range(i + 1, 10))
    np.testing.assert_array_equal(targets[0], probe._base_targets(
        net, jax.random.split(probe.probe_key(11, 2, 0))[0], obs, act, 8, 1e5))
    with mock.patch.object(probe, "_fit", wraps=probe._fit) as fitting:
        probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                          optax.adamw(1e-3), obs, act, probe.probe_key(11, 2, 0), cfg)
    assert fitting.call_count == 2
    for position in (3, 4, 5, 6):  # observations, actions, teacher base, minibatch indices
        np.testing.assert_array_equal(fitting.call_args_list[0].args[position],
                                      fitting.call_args_list[1].args[position])


@pytest.mark.parametrize("losses,fires", [
    (np.zeros(10), False), (np.full(10, -1.), False),
    (np.full(10, 1.), True),
    (np.array([0.] * 9 + [1e6]), False),
    (np.array([1.] * 9 + [-1e6]), False),
])
def test_iqm_bootstrap_bounds_and_firing(losses, fires):
    expected = bootstrap_iqm10(losses, 9, 4)
    assert probe.iqm(losses) == pytest.approx(empirical_iqm10(losses))
    low, high = trigger.bootstrap_interval(losses, 9, 4, 10_000, .95)
    assert (low, high) == pytest.approx((expected.ci_low, expected.ci_high))
    assert trigger.triggered(low, 0) == fires
    assert trigger.bootstrap_interval(losses, 9, 4, 10_000, .95) == (low, high)


def test_fifty_interaction_checks_and_unchanged_exp2_boundaries():
    steps = probe.check_steps(500_000, 50)
    assert steps == list(range(10_000, 500_001, 10_000))
    assert [s * 2 for s in steps] == list(range(20_000, 1_000_001, 20_000))
    assert trigger.last_eligible_check(50, .95) == 47
    cfg = compose_config(CONFIG_PATH, "base_exp12_ten", [])
    records = [{"check_index": k, "interaction_step": k * 10_000,
                "triggered": k in (46, 47)} for k in range(1, 51)]
    result = trigger.f_star(records, 50, 2, .95)
    assert result == {"check_index": 47, "interaction_step": 470_000}
    plan = fork.fork_plan(cfg, 470_000, 47)
    assert (plan["horizon_steps"], plan["arm_end_step"], plan["control_end_step"]) == (
        125_000, 595_000, 595_000)
    records[45]["triggered"] = False
    assert trigger.f_star(records, 50, 2, .95) is None
    records[45]["triggered"] = True
    records[46]["triggered"] = False
    records[47]["triggered"] = True
    records[48]["triggered"] = True
    assert trigger.f_star(records, 50, 2, .95) is None


def test_full_2000_step_ten_function_probe_on_cpu():
    net = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 16
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    a = net.init(jax.random.PRNGKey(1), obs[:1], act[:1])["params"]
    b = net.init(jax.random.PRNGKey(2), obs[:1], act[:1])["params"]
    cfg = probe.ProbeConfig(10, 2000, 50, 16, 4, 1e5, 8)
    rounds = [probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                                optax.adamw(1e-3), obs, act, probe.probe_key(99, 2, r), cfg)
              for r in range(10)]
    current = np.asarray([float(x["current"]["score"]) for x in rounds])
    fresh = np.asarray([float(x["fresh"]["score"]) for x in rounds])
    assert all(x["current"]["losses"].shape == (2000,) for x in rounds)
    assert all(np.isfinite(x).all() for x in (current, fresh))
    loss = fresh - current
    assert probe.iqm(loss) == pytest.approx(empirical_iqm10(loss))
    assert np.isfinite(trigger.bootstrap_interval(loss, 99, 2, 10_000, .95)).all()


def test_small_fresh_fresh_cpu_controls_do_not_calibrate_gpu():
    net = TinyCritic()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 16
    act = jnp.zeros((16, 1), dtype=jnp.float32)
    cfg = probe.ProbeConfig(10, 4, 50, 16, 4, 1e5, 8)
    fires = []
    for pair in range(4):
        a = net.init(jax.random.PRNGKey(2 * pair + 1), obs[:1], act[:1])["params"]
        b = net.init(jax.random.PRNGKey(2 * pair + 2), obs[:1], act[:1])["params"]
        losses = []
        for r in range(10):
            outcome = probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                                        optax.adamw(1e-3), obs, act,
                                        probe.probe_key(100 + pair, 2, r), cfg)
            losses.append(float(outcome["fresh"]["score"] - outcome["current"]["score"]))
        low, _ = trigger.bootstrap_interval(losses, 100 + pair, 2, 10_000, .95)
        fires.append(trigger.triggered(low, 0))
    assert len(fires) == 4
    print(f"small-model CPU fresh/fresh firing {sum(fires)}/4 (descriptive only)")


def test_tiny_ten_function_training_checkpoint_restore_and_artifacts(tmp_path):
    overrides = tiny_overrides("hopper-hop", "dmc_medium", steps=40, extra=(
        "probe.checks=4", "probe.rounds=10", "probe.steps=2", "probe.pool_size=16",
        "probe.batch_size=4", "probe.eval_chunk=8", "buffer.min_length=1",
        "logging_per_interaction_step=10", "evaluation_per_interaction_step=40",
        "fork.enabled=false", "run_role=dev", "env.max_episode_steps=20",
    ))
    cfg = compose_config(CONFIG_PATH, "base_exp12_ten", overrides)
    run_dir, state_root = tmp_path / "run", tmp_path / "run/state"
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))
    with mock.patch("wandb.init", side_effect=fake_wandb_init) as init, \
         mock.patch("wandb.log") as log, \
         mock.patch("wandb.define_metric") as define:
        first = Exp12Trainer(cfg, str(run_dir))
        first.start()
        probes = RunProbes(first, str(run_dir))
        first.train(20, after_step=probes.maybe_check, before_first_update=probes.capture_fresh)
        first.save(state_root)
        first.close()
        second = Exp12Trainer(cfg, str(run_dir))
        second.restore(latest_state_dir(state_root))
        restored = RunProbes(second, str(run_dir))
        assert len(restored.records) == 3  # fresh, check 1, check 2
        second.train(40, after_step=restored.maybe_check,
                     before_first_update=restored.capture_fresh)
        second.save(state_root)
        restored.write_csv()
        identity = {"run_key": "tiny-ten", "run_role": "dev", "architecture": "D1W8",
                    "environment": "hopper-hop", "seed": 1, "budget_env_steps": 80,
                    "num_interaction_steps": 40, "num_checks": 4, "probe_rounds": 10,
                    "code_commit": "cpu-test"}
        ledger.write_run(identity, restored.records, restored.f_star, "complete",
                         restored.dir, results_root=str(tmp_path / "results"),
                         metrics_rows=second.metrics_rows, action_repeat=2)
        replay_obs = jnp.asarray(second.agent._normalize(second.buffer._observations[:16]))
        replay_act = jnp.asarray(second.buffer._actions[:16])
        net = TinyCritic()
        tiny_cfg = probe.ProbeConfig(10, 2, 50, 16, 4, 1e5, 8)
        replay_fires = 0
        for pair in range(4):
            a = net.init(jax.random.PRNGKey(pair * 2 + 21), replay_obs[:1], replay_act[:1])["params"]
            b = net.init(jax.random.PRNGKey(pair * 2 + 22), replay_obs[:1], replay_act[:1])["params"]
            losses = []
            for r in range(10):
                value = probe.probe_round(net, {"current": (net, a), "fresh": (net, b)},
                                          optax.adamw(1e-3), replay_obs, replay_act,
                                          probe.probe_key(pair + 150, 4, r), tiny_cfg)
                losses.append(float(value["fresh"]["score"] - value["current"]["score"]))
            low, _ = trigger.bootstrap_interval(losses, pair + 150, 4, 10_000, .95)
            replay_fires += trigger.triggered(low, 0)
        print(f"saved tiny-CPU replay fresh/fresh firing {replay_fires}/4 (descriptive only)")
        second.close()
    assert init.call_args.kwargs["group"] == "hopper-hop_D1W8_UTD2"
    assert init.call_args.kwargs["project"] == "echocritic_method_c_ten_96"
    assert init.call_args.kwargs["job_type"] == "D1W8"
    assert init.call_args.kwargs["name"].endswith("_seed1")
    assert init.call_args.kwargs["config"]["probe"]["rounds"] == 10
    define.assert_called_with("probe/*", step_metric="probe/interaction_step")
    assert all(call.kwargs["step"] % 2 == 0 for call in log.call_args_list if "step" in call.kwargs)
    assert any("train/critic_loss" in call.args[0] for call in log.call_args_list)
    probe_logs = [call.args[0] for call in log.call_args_list if "probe/check_index" in call.args[0]]
    assert [entry["probe/check_index"] for entry in probe_logs] == [0, 1, 2, 3, 4]
    assert [entry["probe/interaction_step"] for entry in probe_logs] == [1, 10, 20, 30, 40]
    assert [entry["probe/env_step"] for entry in probe_logs] == [2, 20, 40, 60, 80]
    assert all({"probe/loss_iqm", "probe/ci_low", "probe/ci_high", "probe/triggered"}
               <= entry.keys() for entry in probe_logs[1:])
    assert all(all(f"probe/{name}_r{r}" in entry for name in
                   ("score_current", "score_fresh", "loss") for r in range(10))
               for entry in probe_logs[1:])
    checks = pd.read_csv(tmp_path / "results/exp12/exp1/runs/tiny-ten/checks.csv")
    assert list(checks.check_index) == [0, 1, 2, 3, 4]
    assert list(checks.interaction_step) == [1, 10, 20, 30, 40]
    assert all(f"loss_r{r}" in checks for r in range(10))
    assert not checks.loc[checks.check_index > 0, [f"loss_r{r}" for r in range(10)]].isna().any().any()
    with np.load(tmp_path / "results/exp12/exp1/runs/tiny-ten/probe_curves.npz") as arrays:
        assert arrays["check_04/current_losses"].shape == (10, 2)
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "probes/probe_checks.csv").exists()
    assert latest_state_dir(state_root).is_dir()


def test_versioned_mixed_packing_replaces_finished_runs_without_duplicates():
    profiles = {arch: {"cpus": 1, "host_ram_gb": 1, "gpu_memory_gb": 1,
                       "estimated_minutes_by_group": {"dmc_hard": 1},
                       "min_interactions_per_second": 1.0}
                for arch in scheduler.WEIGHTS}
    admission = {"packing": "ten_target", "max_processes_per_gpu": 6,
                 "host_headroom_gb": 1, "gpu_headroom_gb": 1,
                 "stop_margin_minutes": 1, "architectures": profiles}
    resources = {"cpus_per_node": 16, "memory_gb_per_node": 64}
    arrivals = [("large", "D4W1536"), ("medium1", "D4W1024"),
                ("medium2", "D4W1024"), ("small1", "D2W512")]
    queue = [{"id": name, "kind": "exp1", "row": {
        "architecture": arch, "group": "dmc_hard"}} for name, arch in arrivals]
    active, completed = [], set()

    def admit():
        for candidate in list(queue):
            if scheduler.can_admit(candidate, "gpu0", active, admission, resources,
                                   time.time() + 3600, (40960, 0), 60):
                active.append({"item": candidate, "gpu": "gpu0",
                               "profile": profiles[candidate["row"]["architecture"]],
                               "rate": 2.0})
                queue.remove(candidate)

    admit()
    assert {x["item"]["id"] for x in active} == {"large", "medium1", "medium2"}
    assert [x["id"] for x in queue] == ["small1"]
    finished = active.pop(1)
    completed.add(finished["item"]["id"])
    admit()
    assert not queue
    assert {x["item"]["id"] for x in active} == {"large", "medium2", "small1"}
    assert len({x["item"]["id"] for x in active} | completed) == 4
    six_small = [{"item": {"id": str(i), "kind": "exp1", "row": {
        "architecture": "D2W512", "group": "dmc_hard"}}, "gpu": "gpu0",
        "profile": profiles["D2W512"], "rate": 2.0} for i in range(6)]
    seventh = {"id": "seventh", "kind": "exp1", "row": {
        "architecture": "D2W512", "group": "dmc_hard"}}
    assert not scheduler.can_admit(seventh, "gpu0", six_small, admission, resources,
                                   time.time() + 3600, (40960, 0), 60)


def test_ten_round_ledger_rejects_missing_round_evidence(tmp_path):
    identity = {"run_key": "incomplete", "run_role": "dev", "architecture": "D1W8",
                "environment": "hopper-hop", "seed": 1, "budget_env_steps": 80,
                "num_interaction_steps": 40, "num_checks": 4, "probe_rounds": 10}
    with pytest.raises(ledger.LedgerSchemaError, match="missing function-level evidence"):
        ledger.write_run(identity, [{"check_index": 0, "interaction_step": 1,
                                     **{f"score_fresh_r{r}": 0.0 for r in range(10)}},
                                    {"check_index": 1, "interaction_step": 10,
                                     **{f"score_fresh_r{r}": 0.0 for r in range(10)}}],
                         None, "running", tmp_path / "probes", str(tmp_path / "results"))


def test_tiny_ten_function_forced_dev_fork_checkpoint_capture(tmp_path):
    overrides = tiny_overrides("hopper-hop", "dmc_medium", steps=100, extra=(
        "probe.checks=4", "probe.rounds=10", "probe.steps=2", "probe.pool_size=16",
        "probe.batch_size=4", "probe.eval_chunk=8", "buffer.min_length=1",
        "logging_per_interaction_step=10", "evaluation_per_interaction_step=40",
        "fork.architectures=[D1W8]", "fork.eval_episodes=1",
        "run_role=dev", "testing.force_trigger_check=2", "env.max_episode_steps=20",
        f"results_root={tmp_path / 'results'}",
    ))
    run = tmp_path / "run"
    with mock.patch("wandb.init", side_effect=fake_wandb_init), \
         mock.patch("wandb.log"), mock.patch("wandb.define_metric"):
        exp1.run({"experiment": "exp1", "config_path": CONFIG_PATH,
                  "config_name": "base_exp12_ten", "overrides": overrides,
                  "checkpoint_dir": str(run), "checkpoint_interval": 25,
                  "checkpoint_start_frac": 0.0})
    plan = fork.read_fork(run)
    assert plan["fork_step"] == 50
    assert plan["horizon_steps"] == 25
    assert plan["arm_end_step"] == 75
    assert (run / "fork/FORK_READY").exists()
    assert latest_state_dir(run / "fork/state").is_dir()
    assert json.loads((run / "DONE").read_text())["interaction_step"] == 100
    checks = pd.read_csv(tmp_path / "results/exp12/exp1/runs/exp1_D1W8_hopper-hop_seed1/checks.csv")
    assert list(checks.check_index) == [0, 1, 2, 3, 4]
    assert checks.loc[checks.check_index == 2, "triggered"].iloc[0]


def test_archived_five_function_pilot_source_and_curves_are_unchanged():
    repo = Path(__file__).resolve().parents[2]
    source = "f8619dd5f494399d9febc79fc9092010234c714d"
    historical = subprocess.check_output(
        ["git", "show", f"{source}:main/configs/base_exp12.yaml"], cwd=repo, text=True)
    assert "rounds: 5" in historical and "checks: 20" in historical
    assert "steps: 2000" in historical
    root = os.environ.get("PILOT_ARCHIVE_ROOT")
    if not root:
        pytest.skip("PILOT_ARCHIVE_ROOT absent: archived curve hashes were not checked")
    expected = {"D2W512": "bcf6532627b598b5f530f45f42041d8f382c480ab365ccdf306987fd5798bffc",
                "D4W1024": "6f3bc9e2efcbd4eac1ae7a85e5ec5e704073bcc061508e90091e876ecd8b0179",
                "D4W1536": "ce4fe11b1bfb5c71c1c82d8ea1da425ca9fb32b1dcecdfe94fb58d4e4f835745"}
    for arch, digest in expected.items():
        path = Path(root) / "results/exp12/exp1/runs" / f"exp1_{arch}_myo-key-turn_seed1/probe_curves.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
