"""Fail-closed CPU contracts for the separate Exp96 intervention queue."""

import json
from unittest import mock

import pytest

from scripts import exp96_exp2_arms as arms


def _approval(tmp_path):
    control = tmp_path / "positive.json"
    control.write_text(
        json.dumps(
            {
                "status": "m_chosen",
                "chosen_m": "half",
                "source_code": {"commit": "a" * 40, "dirty": False},
                "analysis_code": {"commit": "b" * 40, "dirty": False},
                "allow_any_setting": False,
                "trigger_check": {"forced": False},
                "run_identity": {
                    "run_role": "dev",
                    "architecture": "D4W1536",
                    "environment": "dog-run",
                    "seed": 991,
                },
                "probe": {
                    name: {"score_rounds": [value] * 10}
                    for name, value in (
                        ("fresh", 1.0),
                        ("degraded", 0.75),
                        ("injected_last", 0.875),
                        ("injected_half", 0.96875),
                        ("injected_all", 0.984375),
                    )
                },
                "loss_rounds": {
                    name: [value] * 10
                    for name, value in (
                        ("degraded", 0.25),
                        ("injected_last", 0.125),
                        ("injected_half", 0.03125),
                        ("injected_all", 0.015625),
                    )
                },
                "probe_configuration": {
                    "rounds": 10,
                    "steps": 2000,
                    "checks": 50,
                    "pool_size": 25600,
                    "batch_size": 256,
                },
                "qualification_criteria": {
                    "natural_unforced_trigger": True,
                    "healthy_reference": "fresh critic at check 0",
                    "noise_threshold": 0.10,
                    "similar_within": 0.10,
                    "rounds": 10,
                },
            }
        )
    )
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "queue.json").write_text("{}")
    data = {
        "approved": True,
        "study_version": arms.exp96_campaign.VERSION,
        "expected_commit": "a" * 40,
        "scheduler_commit": "b" * 40,
        "queue_sha256": arms.sha(queue / "queue.json"),
        "decision_reference": "owner",
        "injection_m": "half",
        "trainer_checkout": str(tmp_path / "frozen"),
        "positive_control": {"path": str(control), "sha256": arms.sha(control)},
        "resources": {
            "nodes": 1,
            "gpus_per_node": 4,
            "account": "biqc-delta-gpu",
            "partition": "gpuA100x4",
            "cpus_per_node": 64,
            "memory_gb_per_node": 256,
            "wall_minutes": 480,
        },
        "admission": {
            "max_processes_per_gpu": 1,
            "gpu_headroom_gb": 2,
            "host_headroom_gb": 8,
            "stop_margin_minutes": 15,
            "architectures": {
                name: {
                    "cpus": 16,
                    "host_ram_gb": 8,
                    "gpu_memory_gb": 8,
                    "min_interactions_per_second": 1,
                    "estimated_arm_minutes_by_group": {
                        "dmc_hard": 60,
                        "myosuite_simba": 60,
                    },
                }
                for name in arms.ARCHS
            },
        },
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(data))
    return queue, path, data, control


def test_exp2_owner_approval_and_positive_control_are_source_bound(tmp_path):
    queue, path, data, control = _approval(tmp_path)
    with mock.patch.object(
        arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
    ), mock.patch.object(arms, "clean_checkout"), mock.patch.object(
        arms.subprocess, "check_output", side_effect=["b" * 40, b""]
    ):
        assert arms.approved(queue, path)[2] == data
    control.write_text(control.read_text() + " ")
    with mock.patch.object(
        arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
    ), mock.patch.object(arms.subprocess, "check_output", side_effect=["b" * 40, b""]):
        with pytest.raises(ValueError, match="positive control"):
            arms.approved(queue, path)
    data["positive_control"]["sha256"] = arms.sha(control)
    data["expected_commit"] = "c" * 40
    path.write_text(json.dumps(data))
    with mock.patch.object(
        arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
    ):
        with pytest.raises(ValueError, match="source-bound"):
            arms.approved(queue, path)


def test_one_arm_per_gpu_and_two_node_resource_fence(tmp_path):
    _, _, approval, _ = _approval(tmp_path)
    tokens = ["0", "1", "2", "3"]
    assert arms.resources(approval, tokens, set(range(64)))[0]["nodes"] == 1
    approval["admission"]["max_processes_per_gpu"] = 2
    with pytest.raises(ValueError, match="unqualified"):
        arms.resources(approval, tokens, set(range(64)))
    approval["admission"]["max_processes_per_gpu"] = 1
    approval["resources"]["nodes"] = 2
    assert arms.resources(approval, tokens, set(range(64)))[0]["nodes"] == 2
    with pytest.raises(ValueError, match="actual allocation"):
        arms.resources(approval, tokens[:3], set(range(64)))


def test_five_target_or_confirmatory_seed_positive_control_is_rejected(tmp_path):
    queue, path, data, control = _approval(tmp_path)
    report = json.loads(control.read_text())
    report["probe"]["fresh"]["score_rounds"] = [0.0] * 5
    control.write_text(json.dumps(report))
    data["positive_control"]["sha256"] = arms.sha(control)
    path.write_text(json.dumps(data))
    with mock.patch.object(
        arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
    ), mock.patch.object(arms, "clean_checkout"), mock.patch.object(
        arms.subprocess, "check_output", side_effect=["b" * 40, b""]
    ):
        with pytest.raises(ValueError, match="ten-target positive control"):
            arms.approved(queue, path)
    report["probe"]["fresh"]["score_rounds"] = [0.0] * 10
    report["run_identity"]["seed"] = 6
    control.write_text(json.dumps(report))
    data["positive_control"]["sha256"] = arms.sha(control)
    path.write_text(json.dumps(data))
    with mock.patch.object(
        arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
    ), mock.patch.object(arms, "clean_checkout"), mock.patch.object(
        arms.subprocess, "check_output", side_effect=["b" * 40, b""]
    ):
        with pytest.raises(ValueError, match="independent"):
            arms.approved(queue, path)


def test_positive_control_report_cannot_forge_m_or_unpaired_loss(tmp_path):
    queue, path, data, control = _approval(tmp_path)
    original = json.loads(control.read_text())
    for change, message in (
        ({"chosen_m": "last"}, "inconsistent"),
        (
            {"loss_rounds": {**original["loss_rounds"], "degraded": [0.0] * 10}},
            "malformed",
        ),
    ):
        report = {**original, **change}
        control.write_text(json.dumps(report))
        data["positive_control"]["sha256"] = arms.sha(control)
        data["injection_m"] = report["chosen_m"]
        path.write_text(json.dumps(data))
        with mock.patch.object(
            arms.exp96_campaign, "load", return_value=({"commit": "a" * 40}, {})
        ), mock.patch.object(arms, "clean_checkout"), mock.patch.object(
            arms.subprocess, "check_output", side_effect=["b" * 40, b""]
        ):
            with pytest.raises(ValueError, match=message):
                arms.approved(queue, path)


def test_no_eligible_fork_means_no_submission_command(tmp_path):
    queue, path, approval, _ = _approval(tmp_path)
    with mock.patch.object(
        arms, "approved", return_value=({}, {}, approval)
    ), mock.patch.object(arms, "discover", return_value=[]):
        with pytest.raises(ValueError, match="no naturally eligible"):
            arms.submission_command(queue, path)
