"""Population-independent checks for the owner's official protocol counts."""

import numpy as np
import pytest
import io
from unittest import mock
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from experiments.exp1 import compose_config
from experiments.exp12.probe import iqm, check_steps
from experiments.exp12 import ledger
from tests.exp12_helpers import CONFIG_PATH
from experiments.exp12.official_exp1 import population


def test_all_330_resolved_configs_and_legacy_budgets(tmp_path):
    from scripts.generate_official_exp1_manifest import build

    manifest = build(tmp_path / "campaign")
    assert len(manifest["entries"]) == 330
    assert not manifest["approved"] and not manifest["executable"]
    for row in manifest["entries"]:
        assert row["raw_steps"] == (
            500000
            if row["environment"] in ("hopper-hop", "swimmer-swimmer15")
            else 1000000
        )


def test_official_short_task_captures_fresh_before_same_step_paired_check(tmp_path):
    from tests.exp12_helpers import compose, tiny_overrides, patch_wandb
    from experiments.exp12.trainer import Exp12Trainer
    from experiments.exp12.run_probes import RunProbes

    cfg = compose(
        tiny_overrides(
            steps=500, extra=["critic_measurements.enabled=false", "run_role=dev"]
        ),
        config_name="base_exp1_official",
    )
    patches = patch_wandb()
    trainer = None
    try:
        trainer = Exp12Trainer(cfg, str(tmp_path))
        trainer.start()
        probes = RunProbes(trainer, str(tmp_path))
        seen = []

        def captured(k, critics):
            seen.append((k, trainer.interaction_step, trainer.update_step))
            return {name: {"score": np.ones(10, np.float32) * 0.5} for name in critics}

        with mock.patch.object(probes, "_probe", side_effect=captured):
            trainer.train(
                10,
                before_first_update=probes.capture_fresh,
                after_step=probes.maybe_check,
            )
        assert seen[0] == (0, 10, 0)
        assert seen[1][0:2] == (1, 10) and seen[1][2] > 0
    finally:
        if trainer is not None:
            trainer.close()
        for patch in patches:
            patch.stop()


def test_actual_ten_target_artifact_passes_curve_contract_and_truncation_fails():
    from experiments.exp12.probe import ProbeConfig, probe_round, probe_key
    from analysis.exp12_validation import _curves

    class Tiny(nn.Module):
        @nn.compact
        def __call__(self, observations, actions):
            return nn.Dense(1)(jnp.concatenate([observations, actions], axis=-1))

    net = Tiny()
    obs = jnp.arange(32, dtype=jnp.float32).reshape(16, 2) / 32
    act = jnp.zeros((16, 1), jnp.float32)
    params = net.init(jax.random.PRNGKey(1), obs, act)["params"]
    cfg = ProbeConfig(10, 2000, 50, 16, 4, 100000.0, 8)
    scores = []
    losses = []
    for r in range(10):
        out = probe_round(
            net,
            {"fresh": (net, params)},
            optax.adamw(1e-4),
            obs,
            act,
            probe_key(1, 0, r),
            cfg,
        )["fresh"]
        scores.append(float(out["score"]))
        losses.append(np.asarray(out["losses"]))
    record = {
        "check_index": 0,
        **{f"score_fresh_r{r}": v for r, v in enumerate(scores)},
    }
    config = {
        "official_exp1_version": "no-hb-v1",
        "probe": {"rounds": 10, "steps": 2000, "checks": 50},
    }

    def snapshot(array):
        buf = io.BytesIO()
        np.savez(
            buf, **{"check_00/fresh_losses": array, "check_00/fresh_score": scores}
        )
        return mock.Mock(read=lambda _: buf.getvalue())

    _curves(snapshot(np.array(losses)), "fixture.npz", [0], False, [record], config)
    with pytest.raises(ValueError, match="shape"):
        _curves(
            snapshot(np.array(losses)[:, :1000]),
            "fixture.npz",
            [0],
            False,
            [record],
            config,
        )
    bad = {**config, "probe": {**config["probe"], "rounds": 5}}
    with pytest.raises(ValueError, match="contract"):
        _curves(snapshot(np.array(losses)), "fixture.npz", [0], False, [record], bad)


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


@pytest.mark.parametrize(
    "override",
    [
        "probe.rounds=5",
        "probe.steps=1000",
        "probe.checks=20",
        "seed=11",
        "env_name=h1-run-v0",
        "env.num_env_steps=500000",
    ],
)
def test_confirmatory_official_profile_rejects_legacy_or_foreign_overrides(override):
    with pytest.raises(ValueError, match="official Exp1"):
        compose_config(CONFIG_PATH, "base_exp1_official", [override])


def test_frozen_manifest_uses_33_configs_and_rejects_unknown_version(tmp_path):
    from scripts.freeze_exp12_analysis_manifest import build

    frozen = build(
        str(tmp_path), "a" * 40, "A100", None, {}, protocol_version="no-hb-v1"
    )
    assert len(frozen["configs"]) == 33
    assert frozen["protocol_version"] == "no-hb-v1"
    assert all(v["probe"]["rounds"] == 10 for v in frozen["configs"].values())
    assert all(not key.split("/")[1].startswith("h1-") for key in frozen["configs"])
    with pytest.raises(ValueError, match="version"):
        build(str(tmp_path), "a" * 40, "A100", None, {}, protocol_version="unknown")


def test_missing_official_evidence_fails_closed_for_all_330_parents(tmp_path):
    import json
    from scripts.freeze_exp12_analysis_manifest import build
    from analysis.exp12_validation import validate, ValidationError

    frozen = build(
        str(tmp_path / "checkpoints"),
        "a" * 40,
        "A100",
        None,
        {},
        protocol_version="no-hb-v1",
    )
    path = tmp_path / "study.json"
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValidationError) as failed:
        validate(str(tmp_path / "results"), str(path), "exp1")
    assert len(failed.value.census) == 330
    assert {row["status"] for row in failed.value.census} == {"invalid_incomplete"}
    with pytest.raises(ValidationError, match="Exp2"):
        validate(str(tmp_path / "results"), str(path), "exp2")


def test_default_exp1_entrypoint_selects_official_profile_before_model_creation(
    tmp_path,
):
    from experiments import exp1

    with mock.patch.object(
        exp1, "compose_config", side_effect=RuntimeError("stop before model")
    ) as compose:
        with pytest.raises(RuntimeError, match="stop before model"):
            exp1.run(
                {
                    "checkpoint_dir": str(tmp_path),
                    "config_path": CONFIG_PATH,
                    "config_name": "base_sac",
                    "overrides": [],
                }
            )
    assert compose.call_args.args[1] == "base_exp1_official"


def test_complete_synthetic_330_parent_certificate(tmp_path):
    # Synthetic packaging/IO evidence only: no GPU result or natural trigger.
    from tests import exp12_complete_fixture as fixture
    from scripts.freeze_exp12_analysis_manifest import build
    from analysis.exp12_validation import validate
    from experiments.exp12.official_exp1 import ARCHITECTURES, ENVIRONMENTS, SEEDS
    import json

    runtime = {
        suite: {
            **{k: "SYNTHETIC" for k in fixture.v.RUNTIME_KEYS},
            "simulator_versions": {"synthetic": "1"},
        }
        for suite in ("dmc", "myosuite")
    }
    study = build(
        str(tmp_path / "checkpoints"),
        fixture.COMMIT,
        "SYNTHETIC",
        None,
        runtime,
        protocol_version="no-hb-v1",
    )
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    for arch, _, _ in ARCHITECTURES:
        for env, _ in ENVIRONMENTS:
            for seed in SEEDS:
                fixture.parent(tmp_path, study, arch, env, seed)
    _, certificate = validate(str(tmp_path / "results"), str(path), "exp1")
    assert certificate["pass"] and len(certificate["census"]) == 330
    # Actual source mismatch must not be treated as a source-qualified result.
    metadata = tmp_path / "checkpoints/exp1/D2W512/dog-run/seed_1/run_metadata.json"
    content = json.loads(metadata.read_text())
    content["code"]["commit"] = "b" * 40
    metadata.write_text(json.dumps(content))
    with pytest.raises(fixture.v.ValidationError, match="revision|source|commit"):
        validate(str(tmp_path / "results"), str(path), "exp1")
