"""CPU-only admission checks for a normally queued, approval-pending allocation."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import exp96_campaign as campaign


def _stage(tmp_path, monkeypatch):
    commit = "a" * 40
    monkeypatch.setattr(campaign, "_source", lambda _: None)
    root, queue = tmp_path / "campaign", tmp_path / "queue"
    study, allocation = tmp_path / "study.json", tmp_path / "allocation.json"
    campaign.prepare(root, commit, study, queue, staged=True)
    return commit, root, queue, study, allocation


def _approve(commit, queue, study, allocation, tmp_path):
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"verdict": "PASS", "expected_commit": commit,
                                 "study_version": campaign.VERSION, "gpu": campaign.GPU}))
    gates = {name: {"status": "APPROVED", "decision_reference": "test-only"}
             for name in ("ten_function_method", "scientific_calibration", "gpu_smoke",
                          "allocation_and_storage", "cross_seed_analysis")}
    gates["gpu_smoke"]["evidence"] = {"path": str(smoke), "sha256": campaign.sha(smoke)}
    study.write_text(json.dumps({"approved": True, "expected_commit": commit,
                                 "study_version": campaign.VERSION, "parent_count": 96,
                                 "decision_reference": "test-only", "gates": gates}))
    profiles = {arch: {"cpus": 2, "host_ram_gb": 4, "gpu_memory_gb": 4,
                       "min_interactions_per_second": .1,
                       "estimated_minutes_by_group": {"dmc_hard": 60, "myosuite_simba": 60}}
                for arch in ("D2W512", "D4W1024", "D4W1536")}
    allocation.write_text(json.dumps({
        "approved": True, "expected_commit": commit,
        "queue_sha256": campaign.sha(queue / "queue.json"),
        "study_approval_sha256": campaign.sha(study),
        "decision_reference": "test-only",
        "resources": {"gpus": 1, "cpus": 16, "memory_gb": 64, "wall_minutes": 120},
        "admission": {"packing": "ten_target", "validated_concurrency": True,
                      "max_processes_per_gpu": 6, "gpu_headroom_gb": 2,
                      "host_headroom_gb": 4, "stop_margin_minutes": 5,
                      "architectures": profiles}}))
    return smoke


def test_staged_queue_is_complete_and_missing_approval_blocks_without_claim(tmp_path, monkeypatch):
    commit, root, queue, study, allocation = _stage(tmp_path, monkeypatch)
    frozen, plan = campaign.load(queue, require_approval=False)
    assert len(frozen["parents"]) == len({p["id"] for p in frozen["parents"]}) == 96
    assert plan["approval_sha256"] is None
    with pytest.raises(FileNotFoundError):
        campaign.load(queue)
    command = campaign.pending_command(queue, allocation, cpus=16,
                                       memory_gb=64, wall_minutes=120)
    assert "SBATCH_HOLD" in command and "--hold" not in command
    assert "--gres=gpu:1" in command and "--time=120" in command
    assert "scripts/sbatch_exp96_allocation.sh" in command
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("EXPECTED_COMMIT", commit)
    with pytest.raises(FileNotFoundError):
        campaign.startup_gate_receipt(queue, allocation)
    receipt = json.loads((queue / "slurm/startup_gate_12345.json").read_text())
    assert receipt["verdict"] == "BLOCKED"
    assert not (queue / "ledger").exists()
    assert not (queue / "attempts").exists()
    assert not list(root.rglob("DONE"))


def test_startup_gate_binds_approvals_resources_and_a100_smoke(tmp_path, monkeypatch):
    commit, _, queue, study, allocation = _stage(tmp_path, monkeypatch)
    smoke = _approve(commit, queue, study, allocation, tmp_path)
    monkeypatch.setattr(campaign, "require_pinned_stack", lambda: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *_, **__: "TimeLimit=02:00:00")
    monkeypatch.setenv("SLURM_JOB_ID", "12346")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    assert campaign.startup_gate_receipt(queue, allocation) == campaign.sha(allocation)
    receipt = json.loads((queue / "slurm/startup_gate_12346.json").read_text())
    assert receipt["verdict"] == "PASS" and receipt["approval_sha256"] == campaign.sha(allocation)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
    with pytest.raises(ValueError, match="actual CPU/memory"):
        campaign.startup_gate(queue, allocation)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")
    study.write_text(study.read_text() + " ")
    with pytest.raises(ValueError, match="hash-bound"):
        campaign.startup_gate(queue, allocation)
    study.write_text(study.read_text()[:-1])
    smoke.write_text(json.dumps({"verdict": "PASS", "expected_commit": commit,
                                 "study_version": campaign.VERSION, "gpu": "NVIDIA T4"}))
    decision = json.loads(study.read_text())
    decision["gates"]["gpu_smoke"]["evidence"]["sha256"] = campaign.sha(smoke)
    study.write_text(json.dumps(decision))
    alloc_decision = json.loads(allocation.read_text())
    alloc_decision["study_approval_sha256"] = campaign.sha(study)
    allocation.write_text(json.dumps(alloc_decision))
    with pytest.raises(ValueError, match="A100 GPU smoke"):
        campaign.startup_gate(queue, allocation)


def test_wrapper_runs_gate_before_srun_and_checks_source():
    path = Path(__file__).resolve().parents[1] / "scripts/sbatch_exp96_allocation.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    body = path.read_text()
    assert body.index("exp12_check_checkout") < body.index("startup-gate") < body.index("srun ")
    assert 'exit 3' in body and 'export EXP96_APPROVAL_SHA256' in body
    assert 'if [ -f main/run.py ]; then' in body
