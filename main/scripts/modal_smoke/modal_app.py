"""Modal app for the Angle 2A/2B/2C minimal end-to-end smoke test.

All outputs (checkpoints, pool, ledgers, 2A/2B/2C results, logs) go to the
Modal Volume mounted at the repo's results/ directory; nothing is kept on
container-local disk.

    modal run scripts/modal_smoke/modal_app.py --step volume_check
    modal run scripts/modal_smoke/modal_app.py --step env_check
    modal run scripts/modal_smoke/modal_app.py --step preflight   # then pool, ledger, angle2a, angle2b, angle2c, inspect
"""

import subprocess
import time
import uuid
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parents[2]
REMOTE_REPO = "/root/repo"
RESULTS_MOUNT = f"{REMOTE_REPO}/results"
GPU = "A10G"
# Modal list prices (USD/hr) used only for the running spend estimate.
GPU_RATE = 1.10
CPU_CORES, MEM_GIB = 4, 16
CPU_RATE_PER_CORE, MEM_RATE_PER_GIB = 0.0473, 0.0080

volume = modal.Volume.from_name("echocritic-smoke-results", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libegl1", "libosmesa6", "libglib2.0-0")
    .pip_install_from_requirements(str(LOCAL_REPO / "requirements.txt"))
    .pip_install("jax[cuda12]==0.4.34")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        str(LOCAL_REPO), REMOTE_REPO,
        ignore=["results", "results/**", "logs", "logs/**", "**/__pycache__", "**/*.pyc", "wandb", "wandb/**"],
    )
)

app = modal.App("echocritic-smoke", image=image)

STAGE_TIMEOUTS = {
    "preflight": 30 * 60,
    "pool": 75 * 60,
    "ledger": 10 * 60,
    "angle2a": 150 * 60,
    "angle2b": 30 * 60,
    "angle2c": 30 * 60,
    "inspect": 10 * 60,
}


@app.function(volumes={RESULTS_MOUNT: volume}, timeout=300)
def volume_write_probe() -> dict:
    token = {"token": uuid.uuid4().hex, "written_at": time.time()}
    path = Path(RESULTS_MOUNT) / "smoke" / "volume_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(token))
    volume.commit()
    return {"path": str(path), **token, "hostname": subprocess.check_output(["hostname"], text=True).strip()}


@app.function(volumes={RESULTS_MOUNT: volume}, timeout=300)
def volume_read_probe() -> dict:
    volume.reload()
    path = Path(RESULTS_MOUNT) / "smoke" / "volume_probe.json"
    return {
        "content": path.read_text() if path.exists() else None,
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "mounts": subprocess.check_output(["sh", "-c", f"mount | grep -F '{RESULTS_MOUNT}' || true"], text=True),
    }


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=15 * 60)
def env_check() -> str:
    script = r"""
import importlib.metadata as md, subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
for p in ["jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "numpy", "pandas", "dm_control", "MyoSuite",
          "orbax-checkpoint", "flax", "optax", "mujoco", "scipy", "ml_dtypes"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
import jax, jax.numpy as jnp
print("jax.devices():", jax.devices())
x = jnp.ones((2048, 2048))
print("matmul on", (x @ x).devices(), "sum", float((x @ x).sum()))
from scale_rl.envs import create_envs
tr, ev = create_envs(env_type="dmc", seed=0, env_name="cheetah-run", num_train_envs=1, num_eval_envs=1,
                     rescale_action=True, no_termination=False, action_repeat=2, reward_scale=1.0,
                     max_episode_steps=1000)
obs, _ = tr.reset()
print("cheetah-run obs shape", obs.shape, "action space", tr.action_space)
tr.close(); ev.close()
"""
    out = subprocess.run(
        ["python", "-c", script], cwd=REMOTE_REPO, capture_output=True, text=True,
    )
    freeze = subprocess.run(["pip", "check"], capture_output=True, text=True)
    return out.stdout + out.stderr + "\n--- pip check ---\n" + freeze.stdout + freeze.stderr


def _run_driver(stages: list[str]) -> dict:
    log_dir = Path(RESULTS_MOUNT) / "smoke" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{'_'.join(stages)}_{int(time.time())}.log"
    t0 = time.time()
    tail = []
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                ["python", "-u", "scripts/modal_smoke/smoke_driver.py", *stages],
                cwd=REMOTE_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                tail = (tail + [line])[-80:]
            rc = proc.wait()
    finally:
        volume.commit()
    return {"stages": stages, "returncode": rc, "seconds": time.time() - t0, "log": str(log_path),
            "tail": "".join(tail)}


@app.function(
    gpu=GPU, cpu=CPU_CORES, memory=MEM_GIB * 1024, volumes={RESULTS_MOUNT: volume}, secrets=[wandb_secret],
    timeout=max(STAGE_TIMEOUTS.values()),
)
def run_stage(stage: str) -> dict:
    volume.reload()
    return _run_driver([stage])


@app.local_entrypoint()
def main(step: str = "volume_check"):
    t0 = time.time()
    if step == "volume_check":
        written = volume_write_probe.remote()
        read = volume_read_probe.remote()
        print("wrote:", written)
        print("read :", read)
        ok = read["content"] is not None and written["token"] in read["content"]
        print(f"VOLUME PERSISTENCE {'PASS' if ok else 'FAIL'} "
              f"(writer host {written['hostname']} vs reader host {read['hostname']})")
        return
    if step == "env_check":
        print(env_check.remote())
        rate = GPU_RATE + CPU_CORES * CPU_RATE_PER_CORE + MEM_GIB * MEM_RATE_PER_GIB
    else:
        if step not in STAGE_TIMEOUTS:
            raise SystemExit(f"unknown step {step!r}")
        result = run_stage.remote(step)
        print(f"\n[modal_app] stage={step} returncode={result['returncode']} "
              f"container_seconds={result['seconds']:.0f} log={result['log']}")
        rate = GPU_RATE + CPU_CORES * CPU_RATE_PER_CORE + MEM_GIB * MEM_RATE_PER_GIB
        if result["returncode"] != 0:
            print(result["tail"])
    elapsed = time.time() - t0
    print(f"[modal_app] wall={elapsed:.0f}s est_cost<=${elapsed / 3600 * rate:.2f} (at ${rate:.2f}/hr, incl. startup)")
