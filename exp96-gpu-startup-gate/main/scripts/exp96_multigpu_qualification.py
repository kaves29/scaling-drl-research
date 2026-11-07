#!/usr/bin/env python3
"""Required packing qualification, sequential phases on one A100.

This does not rerun the four-case20k smoke or alter the scientific experiment.
Old500k dog checkpoints are engineering inputs. Replay is deliberately filled
by repeating those transitions to test capacity, never called scientific data.
"""
import argparse
import ast
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
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


REUSABLE_SOURCE = "a5c376d740b072d5211e55d94860a61dbd054fec"


def engineering_dimensions(env, buffer):
    """Use unbatched vector-environment spaces and cross-check replay geometry."""
    obs = getattr(env, "single_observation_space", env.observation_space).shape
    act = getattr(env, "single_action_space", env.action_space).shape
    replay = (buffer._observations.shape[1:], buffer._actions.shape[1:])
    if (obs, act) != replay or (obs, act) != ((223,), (38,)):
        raise ValueError("dog engineering space/replay geometry changed")
    return obs[0], act[0]


def implementation_nodes(commit, path, names):
    tree = ast.parse(m.checked(["git", "show", f"{commit}:{path}"], cwd=MAIN))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    }


def reusable_checks(path, commit, expected=None):
    """Certify only independently passed components of the failed prior job."""
    path = Path(path).resolve()
    q = m.read(path)
    if (
        q.get("expected_commit") != REUSABLE_SOURCE
        or str(q.get("job_id")) != "22990946"
        or q.get("verdict") != "INCOMPLETE"
        or not q.get("production_orchestration_passed")
        or not q.get("claims_race_passed")
        or q.get("training_fingerprint") != m.training_fingerprint(commit)
        or q.get("limits") != m.LIMITS
        or not all(list(c) in q.get("compositions", []) for c in m.COMPOSITIONS)
    ):
        raise ValueError("prior receipt does not establish reusable components")
    unchanged = {
        "main/scripts/exp96_multigpu.py": (
            "start_child",
            "gpu_inventory",
            "gpu_tokens",
            "cpu_partition",
        ),
        "main/scripts/exp96_multigpu_qualification.py": (
            "claim_child",
            "production_assignment",
            "assignment_probe",
            "sample",
        ),
    }
    for file, names in unchanged.items():
        old = implementation_nodes(REUSABLE_SOURCE, file, names)
        new = implementation_nodes(commit, file, names)
        if len(old) != len(names) or old != new:
            raise ValueError("reused component implementation changed: " + file)
    for file in (
        "main/scripts/exp12_dynamic_scheduler.py",
        "main/scripts/exp96_campaign.py",
    ):
        if m.checked(
            ["git", "rev-parse", f"{REUSABLE_SOURCE}:{file}"], cwd=MAIN
        ) != m.checked(["git", "rev-parse", f"{commit}:{file}"], cwd=MAIN):
            raise ValueError("reused queue/validator implementation changed")
    evidence = dict(q["evidence_sha256"])
    ready = []
    for i in range(8):
        file = (
            path.parent
            / "qualification"
            / "production_assignment"
            / str(i)
            / "ready.json"
        )
        if str(file) not in evidence:
            raise ValueError("missing original assignment hash")
        ready.append(m.read(file))
    if len({x["gpu"] for x in ready}) != 8 or len({x["pid"] for x in ready}) != 8:
        raise ValueError("old assignment identities not distinct")
    cpus = [cpu for x in ready for cpu in x["cpu_ids"]]
    if len(cpus) != 16 or len(set(cpus)) != 16:
        raise ValueError("old assignment CPUs not isolated")
    claims = []
    for i in range(12):
        file = path.parent / "qualification" / "claims_race" / f"{i}.json"
        evidence[str(file)] = m.digest(file)
        claims.append(m.read(file))
    if (
        sum(x["won"] is True for x in claims) != 1
        or len({x["owner"]["token"] for x in claims}) != 12
        or any(str(x["owner"]["job_id"]) != "22990946" for x in claims)
    ):
        raise ValueError("old race evidence not a unique owner")
    for file, sha in evidence.items():
        if m.digest(file) != sha:
            raise ValueError("reused evidence changed: " + file)
    result = {
        "path": str(path),
        "sha256": m.digest(path),
        "evidence_sha256": evidence,
        "checks": ["production_assignment", "claims_race", "original_smoke_validation"],
    }
    if expected is not None and result != expected:
        raise ValueError("reused component binding changed")
    return result


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

    def progress(stage):
        m.write(
            root / "progress.json",
            dict(stage=stage, monotonic=time.monotonic(), pid=os.getpid()),
            replace=True,
        )

    progress("GPU/CPU isolation")
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
    progress("construct trainer and verify replay geometry")
    trainer = Exp12Trainer(cfg, str(root))
    obs_dim, act_dim = engineering_dimensions(trainer.train_env, trainer.buffer)
    input_root = PILOT / INPUTS[arch]
    state = latest_state_dir(input_root / "state")
    if state is None or not state.name.startswith("step_000500000_"):
        raise ValueError("retained input is not the verified500k state")
    progress("restore retained checkpoint")
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
    progress("SAC warm updates")
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
    fresh = net.init(
        jax.random.PRNGKey(seed),
        observations=jnp.zeros((1, obs_dim)),
        actions=jnp.zeros((1, act_dim)),
    )["params"]
    tx = critic_optimizer(cfg.agent)
    m.write(root / "ready.json", {"pid": os.getpid(), "gpu": assigned})
    while not (root.parent / "PROBES_GO.json").exists():
        time.sleep(0.2)
    progress("full paired ten-target fitting")
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
    validate_probe_result(result)
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
        progress("checkpoint save/restore/save and bitwise comparison")
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
    progress("complete; awaiting phase shutdown")
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


def cuda_visible_identity():
    """Query CUDA's actual visible UUID without allocating a context or tensors."""
    driver = ctypes.CDLL("libcuda.so.1")

    class CuUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.POINTER(CuUuid), ctypes.c_int],
        "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
        "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
    }
    for name, args in signatures.items():
        f = getattr(driver, name)
        f.argtypes, f.restype = args, ctypes.c_int

    def call(name, *args):
        result = getattr(driver, name)(*args)
        if result != 0:
            raise RuntimeError(f"{name} failed with CUDA error {result}")

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise ValueError(f"CUDA must expose exactly one GPU, found {count.value}")
    device = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), 0)
    token = CuUuid()
    call("cuDeviceGetUuid", ctypes.byref(token), device)
    bus = ctypes.create_string_buffer(32)
    call("cuDeviceGetPCIBusId", bus, len(bus), device)
    context = ctypes.c_void_p()
    call("cuCtxGetCurrent", ctypes.byref(context))
    if context.value is not None:
        raise ValueError("discovery unexpectedly created a CUDA context")
    version = ctypes.c_int()
    call("cuDriverGetVersion", ctypes.byref(version))
    return dict(
        uuid="GPU-" + str(uuid.UUID(bytes=bytes(token.bytes))),
        pci_bus_id=bus.value.decode("ascii"),
        visible_count=count.value,
        driver_api_version=version.value,
        controller_context=None,
    )


def qualification_device(diagnostic_path=None):
    """Match CUDA UUID to the inventory query already proven on Delta."""
    evidence = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "SLURM_GPUS_ON_NODE",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
            )
        }
    }
    try:
        ownership = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
            "SLURM_STEP_GPUS"
        )
        allocated = m.gpu_tokens(ownership) if ownership else []
        if (
            os.environ.get("SLURM_GPUS_ON_NODE") != "1"
            or (ownership and len(allocated) != 1)
            or not os.environ.get("CUDA_VISIBLE_DEVICES")
        ):
            raise ValueError("one Slurm GPU and its inherited CUDA visibility required")
        evidence["nvidia_smi"] = m.checked(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        )
        evidence["cuda"] = cuda_visible_identity()
        identity = evidence["cuda"]
        devices = []
        for line in evidence["nvidia_smi"].splitlines():
            index, token, name, memory = [x.strip() for x in line.split(",")]
            if token.lower() == identity["uuid"].lower():
                if name != "NVIDIA A100-SXM4-40GB" or not 39000 <= int(memory) <= 42000:
                    raise ValueError("unqualified GPU model/capacity: " + name)
                if (
                    allocated
                    and allocated[0].startswith("GPU-")
                    and allocated[0].lower() != token.lower()
                ):
                    raise ValueError("Slurm UUID disagrees with CUDA")
                devices.append(
                    dict(
                        index=index,
                        uuid=token,
                        memory_mb=int(memory),
                        pci_bus_id=identity["pci_bus_id"],
                    )
                )
        if len(devices) != 1:
            raise ValueError("CUDA UUID not uniquely matched by NVIDIA inventory")
        evidence["selected"] = devices[0]
        return devices[0]
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            evidence["query_output"] = exc.output
        raise
    finally:
        if diagnostic_path is not None:
            m.write(diagnostic_path, evidence)


def gpu_startup_worker(root):
    """Exercise the new partial-GPU identity and a real synchronized GPU kernel."""
    root.mkdir()
    identity = cuda_visible_identity()
    assigned = os.environ["CUDA_VISIBLE_DEVICES"]
    if identity["uuid"].lower() != assigned.lower():
        raise ValueError("startup CUDA UUID differs from assigned physical GPU")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.exp12.precision import set_matmul_precision, runtime_info

    expected = sorted(map(int, os.environ["EXP96_ASSIGNED_CPU_IDS"].split(",")))
    if sorted(os.sched_getaffinity(0)) != expected:
        raise ValueError("startup CPU affinity mismatch")
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise ValueError("startup did not initialize exactly one JAX GPU")
    set_matmul_precision()
    value = jnp.ones((8, 8), dtype=jnp.float32)
    result = jax.block_until_ready(value @ value)
    host = np.asarray(result)
    if host.dtype != np.float32 or not np.array_equal(
        host, np.full((8, 8), 8, np.float32)
    ):
        raise ValueError("startup GPU kernel failed its exact finite FP32 check")
    m.write(
        root / "ready.json",
        dict(
            source=m.checked(["git", "rev-parse", "HEAD"], cwd=MAIN),
            pid=os.getpid(),
            gpu=assigned,
            cpu_ids=expected,
            cuda_identity=identity,
            runtime=runtime_info(),
            scope="startup only; not packing qualification",
            kernel_finite_fp32=True,
        ),
    )
    while not (root / "EXIT.json").exists():
        time.sleep(0.1)


def gpu_startup_gate(root, commit, gpu, cpu_ids):
    """Run before any retained-state loading/fitting; use the actual telemetry path."""
    cpu_ids = list(cpu_ids[:2])
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu["uuid"],
        JAX_PLATFORMS="cuda,cpu",
        EXP96_ASSIGNED_CPU_IDS=",".join(map(str, cpu_ids)),
        EXP12_JAX_CACHE_DIR="off",
        WANDB_MODE="disabled",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("JAX_PLATFORM_NAME", None)
    log = root.with_suffix(".log")
    with log.open("xb") as stream:
        process = subprocess.Popen(
            [
                "taskset",
                "--cpu-list",
                ",".join(map(str, cpu_ids)),
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "startup",
                "--root",
                str(root),
                "--label",
                "startup",
            ],
            cwd=MAIN,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ValueError("GPU startup worker exited; inspect startup log")
            ready = root / "ready.json"
            if ready.exists():
                receipt = m.read(ready)
                device, applications, rss = sample([dict(process=process)])
                if (
                    receipt["source"] != commit
                    or receipt["pid"] != process.pid
                    or receipt["gpu"] != gpu["uuid"]
                    or receipt["cpu_ids"] != cpu_ids
                    or not receipt.get("kernel_finite_fp32")
                    or receipt["runtime"]["platform"] != "gpu"
                    or receipt["runtime"]["jax"] != "0.4.34"
                    or receipt["runtime"]["matmul_precision"] != "tensorfloat32"
                    or applications.get(process.pid, (None,))[0] != gpu["uuid"]
                    or gpu["uuid"] not in device
                    or process.pid not in rss
                ):
                    raise ValueError(
                        "GPU startup identity/kernel/runtime/telemetry verification failed"
                    )
                m.write(root / "EXIT.json", dict(observed=time.time()))
                if process.wait(timeout=15) != 0:
                    raise ValueError("GPU startup worker did not exit successfully")
                proof = dict(
                    ready_path=str(ready),
                    ready_sha256=m.digest(ready),
                    log_path=str(log),
                    log_sha256=m.digest(log),
                    observed_gpu_uuid=applications[process.pid][0],
                    observed_driver_mb=applications[process.pid][1],
                    observed_rss_bytes=rss[process.pid],
                    passed=True,
                    source=commit,
                )
                m.write(root / "verified.json", proof)
                return proof
            time.sleep(0.2)
        raise TimeoutError("GPU startup gate exceeded90s; no fitting allowed")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def validate_probe_result(result):
    """Check the real single-critic output contract without changing fit budgets."""
    import numpy as np

    for name in ("fresh", "current"):
        if result[name]["losses"].shape != (10, 2000):
            raise ValueError("fitting budget changed")
        if any(not np.isfinite(v).all() for v in result[name].values()):
            raise ValueError("nonfinite full probe")


def run(root, commit):
    root = Path(root).resolve()
    work = root / "qualification"
    work.mkdir()
    wall_minutes = int(os.environ.get("EXP96_QUALIFICATION_MINUTES", "30"))
    workload_seconds = wall_minutes * 60 - 120
    deadline = time.monotonic() + workload_seconds
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
            "partition": "gpuA100x4",
            "nodes": 1,
            "gpus": 1,
            "cpus": 16,
            "memory_gb": 60,
            "wall_minutes": wall_minutes,
        }
        report["resources"] = r
        report["stage"] = "allocation/source verification"
        m.resources_valid(r, qualification=True)
        if workload_seconds <= 0:
            raise ValueError("insufficient qualification cleanup margin")
        m.verify_actual(r)
        m.reserved_job(os.environ["EXP96_BUDGET"], os.environ["EXP96_RESERVATION"], r)
        report["training_fingerprint"] = m.training_fingerprint(commit)
        if report["training_fingerprint"] != m.training_fingerprint(m.BASE_SMOKE_SHA):
            raise ValueError("scientific execution/config/dependency bytes changed")
        if m.digest(SMOKE / "smoke.json") != m.BASE_SMOKE_DIGEST:
            raise ValueError("original smoke changed")
        previous = os.environ.get("EXP96_REUSE_CHECKS")
        if not previous:
            raise ValueError("targeted continuation requires passed component evidence")
        reused = reusable_checks(previous, commit)
        report["reused_checks"] = reused
        report["environment_input_bounds"] = m.read(previous)[
            "environment_input_bounds"
        ]
        report["production_orchestration_passed"] = True
        report["claims_race_passed"] = True
        report["evidence_sha256"].update(reused["evidence_sha256"])
        report["cpu_topology"] = m.checked(["lscpu", "--json"])
        report["stage"] = "GPU discovery"
        gpus = [qualification_device(root / "gpu_discovery.json")]
        report["evidence_sha256"][str(root / "gpu_discovery.json")] = m.digest(
            root / "gpu_discovery.json"
        )
        cpu_lanes = m.cpu_partition(os.sched_getaffinity(0), 1)
        report["gpus"] = gpus
        report["cpu_lanes"] = cpu_lanes
        report["stage"] = "real GPU startup gate"
        startup = gpu_startup_gate(root / "gpu_startup", commit, gpus[0], cpu_lanes[0])
        report["gpu_startup_passed"] = True
        report["gpu_startup"] = startup
        report["evidence_sha256"].update(
            {
                startup["ready_path"]: startup["ready_sha256"],
                startup["log_path"]: startup["log_sha256"],
                str(root / "gpu_startup" / "verified.json"): m.digest(
                    root / "gpu_startup" / "verified.json"
                ),
            }
        )
        print(
            "Real GPU startup kernel, identity and telemetry passed; starting required packing phases",
            flush=True,
        )
        report["execution_scope"] = (
            "one GPU; homogeneous phases sequential; eight-GPU assignment inherited only"
        )
        report["phase_peak_rss_gb"] = {}
        report["fitting_barrier_sha256"] = {}
        for lane, composition in enumerate(m.COMPOSITIONS):
            report["stage"] = f"packed phase {lane}: {composition}"
            print(report["stage"], flush=True)
            gpu, cpus = gpus[0], cpu_lanes[0]
            phase = work / f"phase_{lane}"
            phase.mkdir()
            phase_records = []
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
                        str(phase / label),
                        "--arch",
                        arch,
                        "--label",
                        label,
                    ]
                    if ordinal == 0 and lane < 3:
                        cmd.append("--representative")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu["uuid"],
                        JAX_PLATFORMS="cuda,cpu",
                        EXP96_ASSIGNED_CPU_IDS=",".join(map(str, ids)),
                        WANDB_MODE="disabled",
                        EXP12_JAX_CACHE_DIR=str(phase / (label + "_jax_cache")),
                        WANDB_DIR=str(work),
                        PYTHONDONTWRITEBYTECODE="1",
                    )
                    env.pop("JAX_PLATFORM_NAME", None)
                    stream = (phase / (label + ".log")).open("xb")
                    p = subprocess.Popen(
                        cmd,
                        cwd=MAIN,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    stream.close()
                    record = dict(
                        process=p,
                        label=label,
                        arch=arch,
                        gpu=gpu["uuid"],
                        cpus=ids,
                        path=phase / label,
                    )
                    records.append(record)
                    phase_records.append(record)
                    ordinal += 1
            while time.monotonic() < deadline:
                device, apps, rss = sample(phase_records)
                for token, (_, used) in device.items():
                    if token == gpu["uuid"]:
                        peak_device[token] = max(peak_device.get(token, 0), used)
                for p in phase_records:
                    pid = p["process"].pid
                    if pid in apps:
                        if apps[pid][0] != p["gpu"]:
                            raise ValueError("child escaped its assigned GPU")
                        peak_pid[pid] = max(peak_pid.get(pid, 0), apps[pid][1])
                    peak_rss[pid] = max(peak_rss.get(pid, 0), rss.get(pid, 0))
                if not (phase / "PROBES_GO.json").exists() and all(
                    (p["path"] / "ready.json").exists() for p in phase_records
                ):
                    if not all(
                        p["process"].pid in apps
                        and apps[p["process"].pid][0] == p["gpu"]
                        for p in phase_records
                    ):
                        raise ValueError(
                            "not all packed CUDA contexts observed at fitting barrier"
                        )
                    m.write(
                        phase / "PROBES_GO.json",
                        {"children": len(phase_records), "released": time.time()},
                    )
                if not (phase / "EXIT_GO.json").exists() and all(
                    (p["path"] / "child_receipt.json").exists() for p in phase_records
                ):
                    m.write(phase / "EXIT_GO.json", {"released": time.time()})
                if any(p["process"].poll() not in (None, 0) for p in phase_records):
                    raise ValueError("packed child failed; inspect logs")
                if all(p["process"].poll() == 0 for p in phase_records):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"targeted{workload_seconds}s workload cap reached")
            children = [m.read(p["path"] / "child_receipt.json") for p in phase_records]
            peak = max(
                max(
                    peak_pid[p["process"].pid],
                    child["device_memory_stats"]["peak_bytes_in_use"] / 2**20 + 384,
                )
                for p, child in zip(phase_records, children)
            )
            if (
                len(children) * math.ceil(peak * 1.15) + 2048 > gpu["memory_mb"]
                or peak_device[gpu["uuid"]] + 2048 > gpu["memory_mb"]
            ):
                raise ValueError(
                    "homogeneous packing lacks measured memory headroom; stop remaining phases"
                )
            barrier = phase / "PROBES_GO.json"
            report["fitting_barrier_sha256"][str(barrier)] = m.digest(barrier)
            report["evidence_sha256"][str(barrier)] = m.digest(barrier)
            report["phase_peak_rss_gb"][str(lane)] = (
                sum(peak_rss[p["process"].pid] for p in phase_records) / 2**30
            )
            m.write(
                phase / "phase_complete.json",
                dict(composition=composition, children=len(phase_records)),
            )
        profiles = {}
        receipts = []
        for p in records:
            path = p["path"] / "child_receipt.json"
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
        for composition in m.COMPOSITIONS:
            gpu = gpus[0]
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
        report["workload_cap_seconds"] = workload_seconds
        report.update(
            verdict="PASS",
            profiles=profiles,
            gpu_headroom_mb=headroom,
            all_devices_isolated=True,
            peak_device_mb=peak_device,
            peak_host_rss_gb=max(report["phase_peak_rss_gb"].values()),
            original_smoke_sha256=m.BASE_SMOKE_DIGEST,
            reused_scope="unchanged trainer/probe/config and corrected actual artifact validation; this receipt newly qualifies launcher/packing only",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
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
    p.add_argument("action", choices=("child", "claim", "assign", "startup"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arch", choices=m.ARCHS)
    p.add_argument("--label", required=True)
    p.add_argument("--representative", action="store_true")
    a = p.parse_args()
    if a.action == "startup":
        gpu_startup_worker(a.root)
    elif a.action == "assign":
        assignment_probe(a.root)
    elif a.action == "claim":
        claim_child(a.root, a.label)
    else:
        child(a.root, a.arch, a.label, a.representative)


if __name__ == "__main__":
    main()
