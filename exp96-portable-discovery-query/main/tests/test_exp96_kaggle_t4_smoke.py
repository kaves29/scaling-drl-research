"""CPU contract checks for the Kaggle T4 engineering notebook and runner."""

import json
import tarfile
from pathlib import Path

import numpy as np

from scripts import exp96_gpu_smoke as original
from scripts import exp96_kaggle_t4_smoke as kaggle


def test_original_dev_commands_and_full_probes_are_reused(tmp_path):
    assert len(kaggle.CASES) == 4
    for case in kaggle.CASES:
        assert kaggle.command(case, tmp_path) == original.command(case, tmp_path)
        cmd = kaggle.command(case, tmp_path)
        assert "base_exp12_ten" in cmd
        assert "env.num_env_steps=40000" in cmd
        assert "probe.checks=2" in cmd
        assert "run_role=dev" in cmd and "fork.enabled=false" in cmd
        assert not any("probe.steps=" in str(x) or "probe.rounds=" in str(x) for x in cmd)


def test_t4_measurement_and_checkpoint_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle.subprocess, "check_output", lambda *_, **__: "Tesla T4, 550.1, 15109, 1200, 45\n")
    assert kaggle.gpu_measurement()["total_mb"] == 15109
    run = tmp_path / "medium"
    (run / "state").mkdir(parents=True)
    (run / "probes").mkdir()
    (run / "state/LATEST").write_text("step_000010000_a\n")
    assert not kaggle._first_probe_ready(tmp_path, "medium")
    (run / "probes/check_01.npz").write_bytes(b"test")
    assert kaggle._first_probe_ready(tmp_path, "medium")
    (run / "state/LATEST").write_text("step_000009999_a\n")
    assert not kaggle._first_probe_ready(tmp_path, "medium")


def test_full_fidelity_artifact_shapes_and_endpoint(tmp_path):
    run = tmp_path / "small"
    (run / "probes").mkdir(parents=True)
    (run / "state").mkdir()
    for index in range(3):
        names = ("fresh_q1_losses", "fresh_q2_losses") if index == 0 else (
            "fresh_q1_losses", "fresh_q2_losses", "current_q1_losses", "current_q2_losses")
        np.savez_compressed(run / "probes" / f"check_{index:02d}.npz",
                            **{name: np.zeros((10, 2000), np.float32) for name in names})
    (run / "probes/probe_checks.csv").write_text("check_index\n")
    (run / "state/LATEST").write_text("step_000020000_a\n")
    (run / "DONE").write_text(json.dumps({"interaction_step": 20_000}))
    (run / "run_metadata.json").write_text(json.dumps({"launches": [
        {"resumed": False}, {"resumed": True}]}))
    evidence = kaggle._probe_artifacts(tmp_path, "small")
    assert evidence["endpoint_20000"] and evidence["csv"] and evidence["latest"]
    assert evidence["resumed_launch"]
    assert all(evidence["checks"][str(i)]["full_fidelity"] for i in range(3))
    np.savez_compressed(run / "probes/check_02.npz", fresh_q1_losses=np.zeros((5, 1000)))
    assert not kaggle._probe_artifacts(tmp_path, "small")["checks"]["2"]["full_fidelity"]


def test_failure_writes_real_receipt_and_bundle_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "_source", lambda _: None)
    monkeypatch.setattr(kaggle, "gpu_preflight", lambda: (_ for _ in ()).throw(RuntimeError("no T4")))
    root = tmp_path / "evidence"
    assert kaggle.run(root, "a" * 40, 600, 600) == 2
    report = json.loads((root / "summary.json").read_text())
    assert report["verdict"] == "FAIL" and "no T4" in report["error"]
    assert report["attempts"] == [] and report["a100_qualification"] == "UNVERIFIED"
    with tarfile.open(tmp_path / "evidence_evidence.tar.gz") as archive:
        assert "evidence/summary.json" in archive.getnames()


def test_notebook_json_paths_and_cell_syntax():
    path = Path(__file__).resolve().parents[1] / "notebooks/exp96_kaggle_t4_engineering_smoke.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(codes) >= 5
    for code in codes:
        compile(code, str(path), "exec")
    source = "\n".join(codes)
    assert "MAIN = REPO / 'main'" in source
    assert "SOURCE_SHA" in source and "checkout', '--quiet', '--detach', SOURCE_SHA" in source
    assert "jax[cuda12]==0.4.34" in source
    assert "gpu_preflight" in source
    assert "--max-seconds', '18000'" in source
    assert "_evidence.tar.gz" in source
