"""CPU-only fail-closed launcher tests. All PASS evidence here is synthetic."""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from audits.probe_discrepancy import launch
from audits.probe_discrepancy.plan import digest, plan_hash, validate_design

DRAFT = Path(__file__).parents[1] / "audits/probe_discrepancy/pilot_manifest.DRAFT.json"


def write(path, value):
    path.write_text(json.dumps(value))


def test_exact_ten_balanced_pairs_and_preserved_configs():
    m = launch.read(DRAFT)
    validate_design(m)
    assert m["count"] == 10 and m["schedule"]["processes_per_gpu"] == [5, 5]
    assert not m["launch_approved"] and not m["executable"]
    assert len({r["original_run_id"] for r in m["runs"]}) == 5
    for left, right in zip(m["runs"][::2], m["runs"][1::2]):
        assert left["source_scientific_config"] == right["source_scientific_config"]
        assert left["gpu"] == 0 and right["gpu"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "nine",
        "eleven",
        "six_four",
        "waves",
        "replacement",
        "same_output",
        "same_wandb",
        "confounded_seed",
        "changed_target",
    ],
)
def test_design_rejects_schedule_or_identity_or_method_changes(kind):
    m = launch.read(DRAFT)
    if kind in ("nine", "eleven"):
        m["runs"] = m["runs"][:-1] if kind == "nine" else m["runs"] + [m["runs"][0]]
    elif kind == "six_four":
        m["runs"][1]["gpu"] = 0
    elif kind == "waves":
        m["schedule"]["waves"] = 2
    elif kind == "replacement":
        m["schedule"]["replacement_runs"] = True
    elif kind == "same_output":
        m["runs"][1]["output_relative"] = m["runs"][0]["output_relative"]
    elif kind == "same_wandb":
        m["runs"][1]["wandb_id"] = m["runs"][0]["wandb_id"]
    else:
        r = m["runs"][1]
        if kind == "confounded_seed":
            r["seed"] = r["source_scientific_config"]["seed"] = r[
                "source_scientific_config"
            ]["agent"]["seed"] = r["source_scientific_config"]["env"]["seed"] = 8
        else:
            r["source_scientific_config"]["probe"]["target_scale"] = 1
        r["source_scientific_config_sha256"] = digest(r["source_scientific_config"])
    with pytest.raises(ValueError):
        validate_design(m)


@pytest.fixture
def qualified_fixture(tmp_path):
    """Fake future evidence to exercise validation, never actual qualification."""
    m = launch.read(DRAFT)
    checkout, root = tmp_path / "checkout", tmp_path / "new-pilot"
    checkout.mkdir()
    root.mkdir()
    for name in (*launch.SOURCE_FILES, "main/fixture_adapter.py"):
        p = checkout / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic fixture, not an implemented diagnostic adapter\n")
    m["resource_estimate"]["packing_qualified"] = True
    m.update(
        launch_approved=True,
        executable=True,
        new_campaign_root=str(root),
        diagnostic_implementation_source="1" * 40,
    )
    manifest, approval = tmp_path / "manifest.json", tmp_path / "approval.json"
    write(manifest, m)
    runtime = {
        "python": sys.executable,
        "versions": {
            k: "fixture" for k in ("python", "jax", "jaxlib", "optax", "cuda", "driver")
        },
    }
    a = {
        "approved": True,
        "decision_reference": "synthetic-test-only",
        "expected_commit": "1" * 40,
        "manifest_sha256": launch.sha(manifest),
        "plan_sha256": plan_hash(m),
        "source_files": {
            name: launch.sha(checkout / name) for name in launch.SOURCE_FILES
        },
        "adapter": {
            "entrypoint": "main/fixture_adapter.py",
            "protocol": "echocritic-observation-only-v1",
            "sha256": launch.sha(checkout / "main/fixture_adapter.py"),
        },
        "runtime": runtime,
        "allocation": m["allocation_proposal"],
        "node_name": "fixture-node",
        "wandb_namespace": "fixture-only",
        "wandb_project": "fixture-only",
        "new_campaign_root": str(root),
        "limits": {
            k: 1
            for k in (
                "gpu_headroom_MiB",
                "host_headroom_GiB",
                "storage_headroom_GiB",
                "barrier_timeout_seconds",
                "max_exec_skew_seconds",
                "stop_margin_seconds",
                "telemetry_interval_seconds",
                "idle_gpu_memory_limit_MiB",
            )
        },
        "gates": {},
    }
    raw = tmp_path / "raw-evidence.txt"
    raw.write_text("synthetic fixture, no GPU qualification\n")
    for name in launch.GATES:
        r = {
            "verdict": "PASS",
            "decision_reference": "synthetic-only",
            "expected_commit": a["expected_commit"],
            "plan_sha256": plan_hash(m),
            "runtime": runtime,
            "allocation": a["allocation"],
            "raw_evidence": [{"path": str(raw), "sha256": launch.sha(raw)}],
        }
        if name == "five_process_per_gpu":
            r.update(
                simultaneous_all_ten=True,
                max_concurrent_processes_per_gpu=[5, 5],
                waves=1,
                run_config_hashes=sorted(
                    row["source_scientific_config_sha256"] for row in m["runs"]
                ),
                node_name=a["node_name"],
                gpu_uuids=["GPU-fixture0", "GPU-fixture1"],
                cpu_affinity=list(range(32)),
                peak_gpu_memory_MiB=[100, 100],
                gpu_total_memory_MiB=[40000, 40000],
                peak_host_memory_GiB=1,
                projected_endpoint_seconds=100,
                minimum_interactions_per_second=1,
                paired_probe_seconds=1,
                cold_compile_seconds=1,
                **{
                    k: True
                    for k in (
                        "cold_compile",
                        "steady_training",
                        "paired_full_probe",
                        "checkpoint_io",
                        "binding_verified",
                        "on_off_equivalence",
                    )
                },
            )
        if name == "live_delta_layout":
            r.update(
                node_name=a["node_name"],
                valid_until_utc=(
                    dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
                ).isoformat(),
                scheduler_layout_verified=True,
                cpu_memory_cgroup_verified=True,
                storage_verified=True,
            )
        path = tmp_path / f"{name}.json"
        write(path, r)
        a["gates"][name] = {"path": str(path), "sha256": launch.sha(path)}
    write(approval, a)
    return m, a, manifest, approval, checkout


def preflight_fixture(fixture):
    _, _, m, a, checkout = fixture
    return launch.preflight(m, a, checkout, check_git=False)


def test_complete_synthetic_fixture_passes_validation(qualified_fixture):
    preflight_fixture(qualified_fixture)


@pytest.mark.parametrize("name", launch.GATES)
def test_each_missing_gate_fails_closed(qualified_fixture, name):
    _, a, _, path, _ = qualified_fixture
    del a["gates"][name]
    write(path, a)
    with pytest.raises(ValueError, match="missing gate"):
        preflight_fixture(qualified_fixture)


@pytest.mark.parametrize(
    "change",
    [
        "old_source",
        "bad_hash",
        "wave_profile",
        "wrong_mix",
        "vram",
        "host_ram",
        "walltime",
        "expired",
        "adapter",
        "closed_flags",
    ],
)
def test_unqualified_evidence_cannot_unlock(qualified_fixture, change):
    m, a, manifest, approval, _ = qualified_fixture
    key = "live_delta_layout" if change == "expired" else "five_process_per_gpu"
    path = Path(a["gates"][key]["path"])
    r = launch.read(path)
    if change == "old_source":
        r["expected_commit"] = "2" * 40
    elif change == "bad_hash":
        a["gates"][key]["sha256"] = "0" * 64
    elif change == "wave_profile":
        r["simultaneous_all_ten"] = False
    elif change == "wrong_mix":
        r["run_config_hashes"] = []
    elif change == "vram":
        r["peak_gpu_memory_MiB"] = [40000, 40000]
    elif change == "host_ram":
        r["peak_host_memory_GiB"] = 128
    elif change == "walltime":
        r["projected_endpoint_seconds"] = 99999999
    elif change == "expired":
        r["valid_until_utc"] = "2000-01-01T00:00:00+00:00"
    elif change == "adapter":
        a["adapter"]["sha256"] = "0" * 64
    elif change == "closed_flags":
        m["launch_approved"] = m["executable"] = False
        write(manifest, m)
        a["manifest_sha256"] = launch.sha(manifest)
    if change not in ("bad_hash", "adapter", "closed_flags"):
        write(path, r)
        a["gates"][key]["sha256"] = launch.sha(path)
    write(approval, a)
    with pytest.raises(ValueError):
        preflight_fixture(qualified_fixture)


def test_command_is_exact_two_gpu_one_task_no_array_and_never_submits(
    qualified_fixture, monkeypatch
):
    _, _, manifest, approval, checkout = qualified_fixture
    original_preflight = launch.preflight
    monkeypatch.setattr(
        launch,
        "preflight",
        lambda *args: original_preflight(manifest, approval, checkout, check_git=False),
    )
    args = launch.sbatch_argv(manifest, approval, checkout)
    assert "--nodes=1" in args and "--ntasks=1" in args and "--gres=gpu:2" in args
    assert (
        "--cpus-per-task=32" in args and "--mem=128G" in args and "--time=2880" in args
    )
    assert not any("array" in arg for arg in args)


@pytest.mark.parametrize("value", ["", "0", "0,0", "0,1,2", "GPU-a,MIG-b"])
def test_invalid_gpu_visibility_refused(value):
    with pytest.raises(ValueError):
        launch.allocated_tokens({"CUDA_VISIBLE_DEVICES": value})


def test_binding_uses_inherited_tokens_and_distinct_wandb(qualified_fixture):
    m, a, *_ = qualified_fixture
    left, right = m["runs"][:2]
    env0 = launch.child_environment(m, a, left, ["3", "7"])
    env1 = launch.child_environment(m, a, right, ["3", "7"])
    assert env0["CUDA_VISIBLE_DEVICES"] == "3" and env1["CUDA_VISIBLE_DEVICES"] == "7"
    assert env0["WANDB_RUN_ID"] != env1["WANDB_RUN_ID"]
    assert env0["WANDB_DIR"] != env1["WANDB_DIR"]
    assert env0["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


@pytest.mark.parametrize("spawn_failure", [False, True])
def test_controller_starts_all_ten_before_release_no_replacement(
    qualified_fixture, monkeypatch, spawn_failure
):
    m, a, manifest, approval, checkout = qualified_fixture
    original_preflight = launch.preflight
    monkeypatch.setattr(
        launch,
        "preflight",
        lambda *args: original_preflight(manifest, approval, checkout, check_git=False),
    )
    monkeypatch.setattr(launch.os, "sched_getaffinity", lambda pid: set(range(32)))
    monkeypatch.setattr(launch.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(launch.signal, "signal", lambda *args: None)
    monkeypatch.setattr(launch, "inspect", lambda: {"fixture": True})
    monkeypatch.setattr(
        launch,
        "cgroup_memory",
        lambda: {"limit_bytes": 128 * 1024**3, "usage_bytes": 0},
    )
    monkeypatch.setattr(
        launch,
        "gpu_info",
        lambda t: {
            "uuid": "GPU-fixture" + t,
            "name": m["allocation_proposal"]["gpu_model"],
            "used_MiB": 0,
            "total_MiB": 40000,
        },
    )
    for key, value in {
        "SLURM_JOB_ID": "fixture",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURMD_NODENAME": a["node_name"],
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "131072",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }.items():
        monkeypatch.setenv(key, value)
    root = Path(m["new_campaign_root"])
    calls = []
    instances = []

    class Process:
        def __init__(self, argv, **kwargs):
            if spawn_failure and len(calls) == 2:
                raise OSError("injected startup failure")
            instances.append(self)
            self.rid = argv[argv.index("--run-id") + 1]
            self.pid = 90000 + len(calls)
            self.returncode = None
            self.polls_after_go = 0
            assert not (root / "barrier/GO").exists()
            calls.append((self.rid, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            row = next(r for r in m["runs"] if r["pilot_id"] == self.rid)
            write(root / "barrier" / f"{self.rid}.ready.json", {"fixture": True})
            write(
                root / row["output_relative"] / "diagnostic_completion.json",
                {
                    "expected_commit": a["expected_commit"],
                    "plan_sha256": plan_hash(m),
                    "pilot_id": self.rid,
                    "config_sha256": row["source_scientific_config_sha256"],
                    "diagnostics_complete": True,
                    "scientific_endpoint_complete": True,
                },
            )

        def poll(self):
            if (root / "barrier/GO").exists():
                assert len(calls) == 10
                self.polls_after_go += 1
                write(root / "barrier" / f"{self.rid}.exec.json", {"time": time.time()})
                if self.polls_after_go > 1:
                    self.returncode = 0
            return self.returncode

        def wait(self):
            return self.returncode

    def terminate(pid, sig):
        next(p for p in instances if p.pid == pid).returncode = -15

    monkeypatch.setattr(launch.os, "killpg", terminate)
    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    args = SimpleNamespace(
        manifest=str(manifest),
        approval=str(approval),
        checkout=str(checkout),
        manifest_sha=launch.sha(manifest),
        approval_sha=launch.sha(approval),
    )
    if spawn_failure:
        with pytest.raises(OSError, match="injected startup failure"):
            launch.worker(args)
        assert len(calls) == 2
        assert not (root / "barrier/GO").exists()
        assert (root / "barrier/ABORT").exists()
        assert (
            launch.read(root / "pilot_processes.json")["verdict"] == "FAILED_OR_PARTIAL"
        )
        assert all((root / "runs" / rid / "process.log").exists() for rid, _ in calls)
        return
    launch.worker(args)
    assert len(calls) == 10
    assert (
        [gpu for _, gpu in calls].count("0")
        == [gpu for _, gpu in calls].count("1")
        == 5
    )


def test_blocked_worker_starts_no_process(tmp_path, monkeypatch):
    approval = tmp_path / "unapproved.json"
    write(approval, {"approved": False})
    monkeypatch.setattr(
        launch.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("blocked launch spawned a process"),
    )
    args = SimpleNamespace(
        manifest=str(DRAFT),
        approval=str(approval),
        checkout=str(Path(__file__).parents[2]),
    )
    with pytest.raises(ValueError):
        launch.worker(args)


def test_draft_cli_preflight_returns_nonzero_and_no_sbatch(
    tmp_path, monkeypatch, capsys
):
    approval = tmp_path / "unapproved.json"
    write(approval, {"approved": False})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch",
            "preflight",
            "--manifest",
            str(DRAFT),
            "--approval",
            str(approval),
            "--checkout",
            str(Path(__file__).parents[2]),
        ],
    )
    assert launch.main() == 2
    out = capsys.readouterr()
    assert out.out == "" and "BLOCKED" in out.err


def test_both_replicas_cannot_silently_change_original_target(qualified_fixture):
    m = launch.read(DRAFT)
    for r in m["runs"][:2]:
        r["source_scientific_config"]["probe"]["target_scale"] = 1
        r["source_scientific_config_sha256"] = digest(r["source_scientific_config"])
    with pytest.raises(ValueError, match="pinned original"):
        validate_design(m)


@pytest.mark.parametrize(
    "phase",
    [
        "cold_compile",
        "steady_training",
        "paired_full_probe",
        "checkpoint_io",
        "binding_verified",
        "on_off_equivalence",
    ],
)
def test_each_required_packing_phase(qualified_fixture, phase):
    _, a, _, approval, _ = qualified_fixture
    binding = a["gates"]["five_process_per_gpu"]
    path = Path(binding["path"])
    r = launch.read(path)
    r[phase] = False
    write(path, r)
    binding["sha256"] = launch.sha(path)
    write(approval, a)
    with pytest.raises(ValueError, match="packing phase"):
        preflight_fixture(qualified_fixture)


def test_finished_process_telemetry_does_not_abort_other_runs():
    assert launch.process_counters(999999999) == {}
