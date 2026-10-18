#!/usr/bin/env python3
"""Targeted new-launcher qualification, at most eight A100s for fifteen minutes.

This does not rerun the four-case20k smoke or alter the scientific experiment.
Old500k dog checkpoints are engineering inputs. Replay is deliberately filled
by repeating those transitions to test capacity, never called scientific data.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
from scripts import exp96_multigpu as m

PILOT = Path("/work/hdd/biqc/skaveti1/exp12_prod_pilot7_20261014T213210Z/checkpoints")
INPUTS = dict(
    zip(
        m.ARCHS,
        (
            "run1_D2W512_dog-run_seed1",
            "run2_D4W1024_dog-run_seed1",
            "run3_D4W1536_dog-run_seed1",
        ),
    )
)
SMOKE = Path("/projects/biqc/skaveti1/exp96/smoke_be1ffae_20261017T141608Z")


def claim_child(root, label):
    from scripts import exp12_dynamic_scheduler as d

    item = {
        "kind": "exp1",
        "id": "race",
        "row": {"checkpoint_dir": str(root / "no_parent")},
    }
    owner = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(),
        "host": "qualification",
        "token": uuid.uuid4().hex,
        "claimed_at": time.time(),
    }
    won = d.Ledger(root, query=lambda job: True).claim(item, owner, lambda item: None)
    m.write(root / (label + ".json"), {"won": won, "owner": owner})


def child(root, arch, label, representative):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.exp1 import compose_config
    from experiments.exp12.precision import (
        set_matmul_precision,
        configure_compilation_cache,
        runtime_info,
    )
    from experiments.exp12.probe import probe_config, critic_optimizer
    from experiments.exp12.run_probes import original_critic_def, run_probe_networks
    from experiments.exp12.state import latest_state_dir, state_differences
    from experiments.exp12.trainer import Exp12Trainer
    from scripts.exp12_fullwidth_gate import placement, gpu_placement_ok

    start = time.monotonic()
    root.mkdir()
    assigned = os.environ["CUDA_VISIBLE_DEVICES"]
    if len(jax.devices("gpu")) != 1 or jax.default_backend() != "gpu":
        raise ValueError("child did not initialize exactly one GPU")
    expected = set(map(int, os.environ["EXP96_ASSIGNED_CPU_IDS"].split(",")))
    if os.sched_getaffinity(0) != expected or len(expected) != m.CPUS[arch]:
        raise ValueError("child CPU binding mismatch")
    physical = m.checked(
        ["nvidia-smi", "-i", assigned, "--query-gpu=uuid,name", "--format=csv,noheader"]
    ).split(",")
    if (
        physical[0].strip() != assigned
        or physical[1].strip() != "NVIDIA A100-SXM4-40GB"
    ):
        raise ValueError("child physical GPU mismatch")
    set_matmul_precision()
    configure_compilation_cache()
    blocks, width = {"D2W512": (2, 512), "D4W1024": (4, 1024), "D4W1536": (4, 1536)}[
        arch
    ]
    seed = 10000 + int(label.split("_")[-1]) + 100 * int(label.split("_")[0])
    cfg = compose_config(
        str(MAIN / "configs"),
        "base_exp12_ten",
        [
            f"critic_num_blocks={blocks}",
            f"critic_hidden_dim={width}",
            "env_name=dog-run",
            "env=dmc_hard",
            f"seed={seed}",
            "run_role=dev",
            f'results_root={root / "results"}',
        ],
    )
    trainer = Exp12Trainer(cfg, str(root))
    input_root = PILOT / INPUTS[arch]
    state = latest_state_dir(input_root / "state")
    if state is None or not state.name.startswith("step_000500000_"):
        raise ValueError("retained input is not the verified500k state")
    trainer.restore(state, new_wandb_run=True)
    # Touch full1M replay capacity, using explicitly repeated real transitions.
    # This bounds the official <=595k occupancy without inventing a trained state.
    b = trainer.buffer
    n = b._num_in_buffer
    capacity = len(b._observations)
    if capacity != 1_000_000 or n != 500_000:
        raise ValueError("unexpected replay shape")
    for key in (
        "_observations",
        "_actions",
        "_rewards",
        "_terminateds",
        "_truncateds",
        "_next_observations",
    ):
        a = getattr(b, key)
        a[n:] = a[: capacity - n]
    b._num_in_buffer, b._current_idx = capacity, 0
    restored_placement = placement(trainer)
    if not gpu_placement_ok(restored_placement):
        raise ValueError("restored model/optimizer is not FP32 GPU-backed")
    # The first call includes SAC compilation and the diagnostic path; subsequent
    # calls measure the complete warmed update and CPU dispatch under real packing.
    trainer.agent.update(
        trainer.update_step, b.sample(), compute_actor_grad_cosine=True
    )
    times = []
    for _ in range(12):
        tick = time.monotonic()
        info = trainer.agent.update(
            trainer.update_step, b.sample(), compute_actor_grad_cosine=False
        )
        if not all(np.isfinite(x) for x in info.values()):
            raise ValueError("nonfinite SAC update")
        trainer.update_step += 1
        times.append(time.monotonic() - tick)
    net = original_critic_def(cfg)
    obs_dim, act_dim = (
        trainer.train_env.observation_space.shape[0],
        trainer.train_env.action_space.shape[0],
    )
    if (obs_dim, act_dim) != (223, 38):
        raise ValueError("dog engineering shape changed")
    fresh = net.init(
        jax.random.PRNGKey(seed),
        observations=jnp.zeros((1, obs_dim)),
        actions=jnp.zeros((1, act_dim)),
    )["params"]
    tx = critic_optimizer(cfg.agent)
    m.write(root / "ready.json", {"pid": os.getpid(), "gpu": assigned})
    while not (root.parent / "PROBES_GO.json").exists():
        time.sleep(0.2)
    tick = time.monotonic()
    result = run_probe_networks(
        trainer.agent,
        b,
        net,
        {"current": (net, trainer._sac_agent.critic.params), "fresh": (net, fresh)},
        tx,
        tx,
        seed,
        50,
        probe_config(cfg),
    )
    probe_seconds = time.monotonic() - tick
    for name in ("fresh", "current"):
        if result[name]["losses"].shape != (10, 2000):
            raise ValueError("fitting budget changed")
        if any(not np.isfinite(v).all() for v in result[name].values()):
            raise ValueError("nonfinite full probe")
    curves = root / "full_paired_probe.npz"
    np.savez_compressed(
        curves,
        **{
            f"{name}_{key}": value
            for name, values in result.items()
            for key, value in values.items()
        },
    )
    roundtrip = None
    save_seconds = 0
    if representative:
        tick = time.monotonic()
        a = trainer.save(root / "state_before")
        trainer.restore(a, new_wandb_run=True)
        c = trainer.save(root / "state_after")
        roundtrip = state_differences(a, c)
        save_seconds = time.monotonic() - tick
        if roundtrip:
            raise ValueError("checkpoint roundtrip changed state: " + repr(roundtrip))
    statistics = jax.devices("gpu")[0].memory_stats()
    if not statistics or "peak_bytes_in_use" not in statistics:
        raise ValueError("JAX peak allocator statistics unavailable")
    report = {
        "source": m.checked(["git", "rev-parse", "HEAD"], cwd=MAIN),
        "pid": os.getpid(),
        "architecture": arch,
        "label": label,
        "gpu_uuid": assigned,
        "cpu_ids": sorted(expected),
        "status": "PASS",
        "input_state": str(state),
        "replay_note": "engineering stress:1M capacity filled by repeated500k transitions",
        "runtime": runtime_info(),
        "restored_placement": restored_placement,
        "probe_seconds": probe_seconds,
        "warm_update_seconds": max(times[-8:]),
        "save_restore_seconds": save_seconds,
        "elapsed": time.monotonic() - start,
        "roundtrip_differences": roundtrip,
        "representative": representative,
        "device_memory_stats": {k: int(v) for k, v in statistics.items()},
        "curve_path": str(curves),
        "curve_sha256": m.digest(curves),
    }
    m.write(root / "child_receipt.json", report)
    while not (root.parent / "EXIT_GO.json").exists():
        time.sleep(0.2)
    trainer.close()


def assignment_probe(root):
    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise ValueError("production child did not isolate one GPU")
    jax.block_until_ready(jnp.ones((8, 8)) @ jnp.ones((8, 8)))
    expected = sorted(map(int, os.environ["EXP96_ASSIGNED_CPU_IDS"].split(",")))
    if sorted(os.sched_getaffinity(0)) != expected:
        raise ValueError("production child affinity changed")
    root.mkdir()
    m.write(
        root / "ready.json",
        {
            "pid": os.getpid(),
            "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "cpu_ids": expected,
        },
    )
    while not (root.parent / "EXIT.json").exists():
        time.sleep(0.1)


def production_assignment(root, commit, gpus, cpu_lanes):
    """Exercise the real durable _start path; rows are isolated qualification probes."""
    from scripts import exp12_dynamic_scheduler as d

    root.mkdir()
    ledger = d.Ledger(root, query=lambda job: True)
    approval = {
        "expected_commit": commit,
        "admission": {"architectures": {"D2W512": {"cpus": 2}}},
    }
    jobs = []
    try:
        for i, (gpu, lane) in enumerate(zip(gpus, cpu_lanes)):
            item = {
                "id": f"assignment_{i}",
                "kind": "exp1",
                "row": {
                    "architecture": "D2W512",
                    "group": "dmc_hard",
                    "run_key": f"assignment_{i}",
                    "checkpoint_dir": str(root / f"non_parent_{i}"),
                    "command": [
                        "python",
                        "-B",
                        str(Path(__file__).resolve()),
                        "assign",
                        "--root",
                        str(root / str(i)),
                        "--label",
                        str(i),
                    ],
                },
            }
            job = m.start_child(
                item,
                gpu["uuid"],
                approval,
                root,
                ledger,
                os.environ["SLURM_JOB_ID"],
                lane[:2],
            )
            if job is None:
                raise ValueError("production child launch failed")
            job.update(label=str(i), cpus=lane[:2])
            jobs.append(job)
        until = time.monotonic() + 75
        while time.monotonic() < until:
            if any(j["process"].poll() is not None for j in jobs):
                raise ValueError("assignment probe ended before observation")
            if all((root / j["label"] / "ready.json").exists() for j in jobs):
                _, apps, _ = sample(jobs)
                for j in jobs:
                    r = m.read(root / j["label"] / "ready.json")
                    if (
                        r["pid"] != j["process"].pid
                        or r["gpu"] != j["gpu"]
                        or r["cpu_ids"] != j["cpus"]
                        or apps.get(j["process"].pid, (None,))[0] != j["gpu"]
                    ):
                        raise ValueError(
                            "production child physical GPU/CPU isolation failed"
                        )
                m.write(root / "EXIT.json", {"observed": time.time()})
                for j in jobs:
                    if j["process"].wait(timeout=15) != 0:
                        raise ValueError("assignment probe failed")
                    ledger.finish(j["item"], j["owner"])
                return [str(root / j["label"] / "ready.json") for j in jobs]
            time.sleep(0.5)
        raise TimeoutError("production assignment probes did not become observable")
    finally:
        for j in jobs:
            if j["process"].poll() is None:
                os.killpg(j["process"].pid, signal.SIGTERM)


def sample(processes):
    import psutil

    device = {}
    for row in m.checked(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines():
        token, total, used = [x.strip() for x in row.split(",")]
        device[token] = (int(total), int(used))
    applications = {}
    text = m.checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in text.splitlines():
        if not row:
            continue
        pid, token, memory = [x.strip() for x in row.split(",")]
        applications[int(pid)] = (token, int(memory))
    rss = {}
    for record in processes:
        p = record["process"]
        if p.poll() is None:
            try:
                rss[p.pid] = sum(
                    x.memory_info().rss
                    for x in [
                        psutil.Process(p.pid),
                        *psutil.Process(p.pid).children(recursive=True),
                    ]
                )
            except psutil.NoSuchProcess:
                pass
    return device, applications, rss


def run(root, commit):
    root = Path(root).resolve()
    work = root / "qualification"
    work.mkdir()
    deadline = time.monotonic() + 720
    records = []
    report = {
        "schema": m.SCHEMA,
        "expected_commit": commit,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "verdict": "INCOMPLETE",
        "limits": m.LIMITS,
        "compositions": [list(c) for c in m.COMPOSITIONS],
        "original_smoke": str(SMOKE / "smoke.json"),
        "claims_race_passed": False,
        "all_devices_isolated": False,
        "evidence_sha256": {},
    }
    peak_device, peak_pid, peak_rss = {}, {}, {}
    try:
        m.source(commit)
        from scripts.exp12_launch_plan import require_pinned_stack

        require_pinned_stack()
        r = {
            "partition": "gpuA100x8",
            "nodes": 1,
            "gpus": 8,
            "cpus": 128,
            "memory_gb": 2007,
            "wall_minutes": 15,
        }
        m.verify_actual(r)
        m.reserved_job(os.environ["EXP96_BUDGET"], os.environ["EXP96_RESERVATION"], r)
        report["training_fingerprint"] = m.training_fingerprint(commit)
        if report["training_fingerprint"] != m.training_fingerprint(m.BASE_SMOKE_SHA):
            raise ValueError("scientific execution/config/dependency bytes changed")
        if m.digest(SMOKE / "smoke.json") != m.BASE_SMOKE_DIGEST:
            raise ValueError("original smoke changed")
        # The existing corrected validator re-reads the real arrays once; no GPU
        # training is repeated and the original development horizon is retained.
        for label in ("small", "medium", "large", "replacement"):
            run_dir = SMOKE / label
            cfg = m.read(run_dir / "run_metadata.json")["resolved_config"]
            if m.core().validate_probe_artifacts(run_dir, cfg, 20000) != 2:
                raise ValueError("original artifact validation failed")
        gpus = m.gpu_inventory(8)
        cpu_lanes = m.cpu_partition(os.sched_getaffinity(0), 8)
        report["gpus"] = gpus
        report["cpu_lanes"] = cpu_lanes
        assignments = production_assignment(
            work / "production_assignment", commit, gpus, cpu_lanes
        )
        report["production_orchestration_passed"] = True
        report["evidence_sha256"].update({p: m.digest(p) for p in assignments})
        race = work / "claims_race"
        race.mkdir()
        race_children = []
        for i in range(12):
            race_children.append(
                subprocess.Popen(
                    [
                        "taskset",
                        "--cpu-list",
                        str(cpu_lanes[0][i]),
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "claim",
                        "--root",
                        str(race),
                        "--label",
                        str(i),
                    ],
                    cwd=MAIN,
                )
            )
        for p in race_children:
            if p.wait(timeout=45) != 0:
                raise ValueError("claim race child failed")
        winners = [m.read(race / f"{i}.json") for i in range(12)]
        report["claims_race_passed"] = sum(x["won"] for x in winners) == 1
        if not report["claims_race_passed"]:
            raise ValueError("duplicate/missing claim ownership")
        for lane, (gpu, cpus, composition) in enumerate(
            zip(gpus, cpu_lanes, m.COMPOSITIONS)
        ):
            offset = 0
            ordinal = 0
            for arch, count in zip(m.ARCHS, composition):
                for index in range(count):
                    ids = cpus[offset : offset + m.CPUS[arch]]
                    offset += len(ids)
                    label = f"{lane}_{ordinal}"
                    cmd = [
                        "taskset",
                        "--cpu-list",
                        ",".join(map(str, ids)),
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "child",
                        "--root",
                        str(work / label),
                        "--arch",
                        arch,
                        "--label",
                        label,
                    ]
                    if ordinal == 0:
                        cmd.append("--representative")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu["uuid"],
                        JAX_PLATFORMS="cuda,cpu",
                        EXP96_ASSIGNED_CPU_IDS=",".join(map(str, ids)),
                        WANDB_MODE="disabled",
                        EXP12_JAX_CACHE_DIR=str(work / (label + "_jax_cache")),
                        WANDB_DIR=str(work),
                        PYTHONDONTWRITEBYTECODE="1",
                    )
                    env.pop("JAX_PLATFORM_NAME", None)
                    stream = (work / (label + ".log")).open("xb")
                    p = subprocess.Popen(
                        cmd,
                        cwd=MAIN,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    stream.close()
                    records.append(
                        {
                            "process": p,
                            "label": label,
                            "arch": arch,
                            "gpu": gpu["uuid"],
                            "cpus": ids,
                        }
                    )
                    ordinal += 1
        while time.monotonic() < deadline:
            device, apps, rss = sample(records)
            for token, (_, used) in device.items():
                peak_device[token] = max(peak_device.get(token, 0), used)
            for p in records:
                pid = p["process"].pid
                if pid in apps:
                    if apps[pid][0] != p["gpu"]:
                        raise ValueError("child escaped its assigned GPU")
                    peak_pid[pid] = max(peak_pid.get(pid, 0), apps[pid][1])
                peak_rss[pid] = max(peak_rss.get(pid, 0), rss.get(pid, 0))
            if not (work / "PROBES_GO.json").exists() and all(
                (work / p["label"] / "ready.json").exists() for p in records
            ):
                if not all(
                    p["process"].pid in apps and apps[p["process"].pid][0] == p["gpu"]
                    for p in records
                ):
                    raise ValueError(
                        "not all packed CUDA contexts observed at fitting barrier"
                    )
                m.write(
                    work / "PROBES_GO.json",
                    {"children": len(records), "released": time.time()},
                )
            if not (work / "EXIT_GO.json").exists() and all(
                (work / p["label"] / "child_receipt.json").exists() for p in records
            ):
                m.write(work / "EXIT_GO.json", {"released": time.time()})
            if any(p["process"].poll() not in (None, 0) for p in records):
                raise ValueError("packed child failed; inspect logs")
            if all(p["process"].poll() == 0 for p in records):
                break
            time.sleep(1)
        else:
            raise TimeoutError("targeted720s workload cap reached")
        profiles = {}
        receipts = []
        for p in records:
            path = work / p["label"] / "child_receipt.json"
            x = m.read(path)
            if (
                x["source"] != commit
                or x["gpu_uuid"] != p["gpu"]
                or x["cpu_ids"] != p["cpus"]
                or x["status"] != "PASS"
                or p["process"].pid not in peak_pid
            ):
                raise ValueError("child receipt/telemetry provenance mismatch")
            report["evidence_sha256"][str(path)] = m.digest(path)
            report["evidence_sha256"][x["curve_path"]] = x["curve_sha256"]
            x["peak_driver_mb"] = peak_pid[p["process"].pid]
            x["peak_rss_bytes"] = peak_rss[p["process"].pid]
            receipts.append(x)
        for arch in m.ARCHS:
            children = [x for x in receipts if x["architecture"] == arch]
            # Driver-resident sampling includes context; allocator peak also
            # catches transient arrays between nvidia-smi samples. Add headroom.
            peak = max(
                max(
                    x["peak_driver_mb"],
                    x["device_memory_stats"]["peak_bytes_in_use"] / 2**20 + 384,
                )
                for x in children
            )
            gpu_mb = math.ceil(peak * 1.15)
            host = math.ceil(
                max(6, max(x["peak_rss_bytes"] for x in children) / 2**30 * 1.3 + 2)
            )
            update = max(x["warm_update_seconds"] for x in children)
            probe = max(x["probe_seconds"] for x in children)
            cold = max(x["elapsed"] for x in children)
            full = math.ceil(
                (2 * (595000 - 4999) * update + 60 * probe) * 1.25 / 60 + 15
            )
            checkpoint = math.ceil((20002 * update + probe + cold) * 1.25 / 60 + 2)
            profiles[arch] = {
                "cpus": m.CPUS[arch],
                "host_ram_gb": host,
                "gpu_memory_mb": gpu_mb,
                "estimated_minutes": full,
                "checkpoint_minutes": checkpoint,
                "warm_update_seconds": update,
                "paired_probe_seconds": probe,
                "measurement_scope": "full1M replay, retained500k dog, concurrent full-width fits; estimates include25% time margin",
            }
        headroom = 2048
        min_total = min(x["memory_mb"] for x in gpus)
        for gpu, composition in zip(gpus, m.COMPOSITIONS):
            if peak_device[gpu["uuid"]] + headroom > gpu["memory_mb"]:
                raise ValueError("actual packed peak lacks2GiB device headroom")
            if (
                sum(
                    n * profiles[a]["gpu_memory_mb"]
                    for a, n in zip(m.ARCHS, composition)
                )
                + headroom
                > min_total
            ):
                raise ValueError(
                    "conservative per-process reservations cannot support requested packing"
                )
        m.source(commit)
        report["resources"] = r
        report["fitting_barrier_sha256"] = m.digest(work / "PROBES_GO.json")
        report.update(
            verdict="PASS",
            profiles=profiles,
            gpu_headroom_mb=headroom,
            all_devices_isolated=True,
            peak_device_mb=peak_device,
            peak_host_rss_gb=sum(peak_rss.values()) / 2**30,
            original_smoke_sha256=m.BASE_SMOKE_DIGEST,
            reused_scope="unchanged trainer/probe/config and corrected actual artifact validation; this receipt newly qualifies launcher/packing only",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for p in records:
            if p["process"].poll() is None:
                try:
                    os.killpg(p["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in records:
            if p["process"].poll() is None:
                try:
                    p["process"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p["process"].pid, signal.SIGKILL)
        report["attempts"] = [
            {
                "label": p["label"],
                "arch": p["arch"],
                "gpu": p["gpu"],
                "returncode": p["process"].poll(),
            }
            for p in records
        ]
        m.write(root / "qualification.json", report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["verdict"] == "PASS" else 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("child", "claim", "assign"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arch", choices=m.ARCHS)
    p.add_argument("--label", required=True)
    p.add_argument("--representative", action="store_true")
    a = p.parse_args()
    if a.action == "assign":
        assignment_probe(a.root)
    elif a.action == "claim":
        claim_child(a.root, a.label)
    else:
        child(a.root, a.arch, a.label, a.representative)


if __name__ == "__main__":
    main()
