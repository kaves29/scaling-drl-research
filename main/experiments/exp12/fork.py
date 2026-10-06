"""Experiment 2 fork machinery shared by the control arm (inside exp1) and the separate arm jobs.

Layout under the Exp 1 run directory:
    fork/state/                 complete training state at f*_run (never deleted)
    fork/fork.json              fork step, check index, horizons, paths, device that produced the fork
    fork/panel.npz              fixed panel of 256 replay (s, a) pairs, drawn once at the fork
    fork/check1_pre.npz         Q and dQ/da on the panel: the critic just before injection
    fork/check1_control.npz     the same for the control critic right after its restore
    fork/FORK_READY             written last; arm jobs refuse to start without it
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp12.envs import create_eval_env
from experiments.exp12.precision import runtime_info
from experiments.exp12.trainer import evaluate_episodes
from experiments.exp12.twin import is_twin
from utils.atomic_io import atomic_write_text

PANEL_STREAM = 0x50414E4C  # "PANL"
EVAL_STREAM = 0x50464556  # "PFEV"
PANEL_SIZE = 256
FORK_DIR, READY = "fork", "FORK_READY"


class ValidationDone(Exception):
    """The identity snapshot is saved and testing.stop_after_identity_snapshot asks to stop."""


def check_validation_flags(cfg) -> None:
    if cfg.testing.stop_after_identity_snapshot:
        if cfg.run_role != "dev":
            raise ValueError("testing.stop_after_identity_snapshot is validation-only and requires run_role=dev.")
        if cfg.fork.identity_snapshot_steps is None:
            raise ValueError("testing.stop_after_identity_snapshot needs fork.identity_snapshot_steps.")


def fork_dir(run_dir) -> Path:
    return Path(run_dir) / FORK_DIR


def is_ready(run_dir) -> bool:
    return (fork_dir(run_dir) / READY).exists()


FORK_JSON_INFO = ("run_dir", "fresh_critic_dir", "device")
CHECK1_PRECISION = "highest"


def _read_fork_json(run_dir) -> Dict:
    with open(fork_dir(run_dir) / "fork.json") as f:
        return json.load(f)


def read_fork(run_dir) -> Dict:
    """The fork plan (with run_key); fork.json's paths and device are for readers and checks only."""
    return {k: v for k, v in _read_fork_json(run_dir).items() if k not in FORK_JSON_INFO}


def device_info() -> Dict:
    d = jax.devices()[0]
    return {"platform": d.platform, "device_kind": d.device_kind, "device_count": jax.device_count()}


def check_same_device(run_dir) -> None:
    """Both arms of a fork must run on the GPU model that produced the fork state (decision 5, Phase 4)."""
    recorded = _read_fork_json(run_dir).get("device")
    current = device_info()
    if recorded is None or (recorded["platform"], recorded["device_kind"]) != (current["platform"],
                                                                                current["device_kind"]):
        raise RuntimeError(f"fork in {run_dir} was produced on {recorded}, but this job runs on {current}; "
                           "both arms of a fork must run on the same device model")


def matmul_precision_report() -> Dict:
    """Measured float32 matmul error under the current precision setting (A0; for the record)."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((256, 256)), rng.standard_normal((256, 256))
    exact = a @ b
    got = np.asarray(jax.jit(jnp.matmul)(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)), np.float64)
    return {**runtime_info(), "float32_matmul_max_rel_error": float(np.abs(got - exact).max() / np.abs(exact).max())}


def fork_plan(cfg, fork_step: int, check_index: int) -> Dict:
    n = int(cfg.num_interaction_steps)
    horizon = int(round(float(cfg.fork.horizon_fraction) * n))
    every = int(round(float(cfg.fork.eval_every_fraction) * n))
    if horizon % every:
        raise ValueError("fork.horizon_fraction must be a multiple of fork.eval_every_fraction")
    arm_end = fork_step + horizon
    if arm_end > float(cfg.fork.max_total_fraction) * n + 1e-9:
        raise ValueError(f"fork at {fork_step} leaves no full {horizon}-step horizon within the maximum budget")
    return {"fork_step": int(fork_step), "fork_check_index": int(check_index), "num_interaction_steps": n,
            "horizon_steps": horizon, "arm_end_step": arm_end, "control_end_step": max(n, arm_end),
            "eval_every_steps": every, "eval_episodes": int(cfg.fork.eval_episodes)}


def sample_panel(trainer, seed: int, check_index: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng([seed, PANEL_STREAM, check_index])
    idx = rng.integers(0, trainer.buffer._num_in_buffer, size=PANEL_SIZE)
    raw = trainer.buffer._observations[idx]
    norm = trainer.agent._normalize(raw) if hasattr(trainer.agent, "_normalize") else raw
    return {"buffer_index": idx, "observation_raw": raw, "observation": np.asarray(norm, np.float32),
            "action": trainer.buffer._actions[idx]}


def panel_q_and_grad(critic, panel) -> Dict[str, np.ndarray]:
    """Q and dQ/da on the panel in full FP32 (Check 1 is the one place that does not use TF32)."""
    obs, act = jnp.asarray(panel["observation"]), jnp.asarray(panel["action"])
    q_fn = lambda a: critic.network_def.apply({"params": critic.params}, obs, a)
    if is_twin(critic.params):  # Q of both networks and dQ/da of min(Q1, Q2) (amendment (z))
        with jax.default_matmul_precision(CHECK1_PRECISION):
            q, grad = jax.jit(lambda a: (q_fn(a), jax.grad(lambda b: jnp.minimum(*q_fn(b)).sum())(a)))(act)
        return {"q": np.asarray(q).reshape(-1), "dq_da": np.asarray(grad)}
    with jax.default_matmul_precision(CHECK1_PRECISION):
        q, grad = jax.jit(lambda a: (q_fn(a), jax.grad(lambda b: q_fn(b).sum())(a)))(act)
    return {"q": np.asarray(q).reshape(-1), "dq_da": np.asarray(grad)}


def check1(pre: Dict, after: Dict, control: Dict, tolerance_eps: float, injected: bool) -> Dict:
    """Check 1 on the panel (values from panel_q_and_grad, full FP32). The control (and an identity arm)
    must equal the pre-fork critic bit for bit; an injected critic may differ by tolerance_eps * float32
    eps * the pre-injection max magnitude (the reverse-mode summation order of dQ/da changes)."""
    eps = float(np.finfo(np.float32).eps)
    scale_q, scale_g = float(np.abs(pre["q"]).max()), float(np.abs(pre["dq_da"]).max())
    out = {"tolerance_eps": tolerance_eps, "after_is_injected": injected, "matmul_precision": CHECK1_PRECISION,
           "pairs": {}}
    named = {"pre": pre, "after": after, "control": control}
    for a, b in (("pre", "after"), ("pre", "control"), ("after", "control")):
        tol = tolerance_eps if injected and "after" in (a, b) else 0.0
        dq = float(np.abs(named[a]["q"] - named[b]["q"]).max())
        dg = float(np.abs(named[a]["dq_da"] - named[b]["dq_da"]).max())
        out["pairs"][f"{a}_vs_{b}"] = {
            "max_abs_dq": dq, "max_abs_d_dq_da": dg, "tolerance_eps": tol,
            "dq_eps_units": dq / (eps * scale_q) if scale_q else 0.0,
            "d_dq_da_eps_units": dg / (eps * scale_g) if scale_g else 0.0,
            "pass": dq <= tol * eps * scale_q and dg <= tol * eps * scale_g,
        }
    out["max_eps_units"] = max(max(p["dq_eps_units"], p["d_dq_da_eps_units"]) for p in out["pairs"].values())
    out["pass"] = all(p["pass"] for p in out["pairs"].values())
    return out


def save_npz(path: Path, arrays: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def write_fork(trainer, run_dir, plan: Dict, run_key: str, fresh_critic_dir: str) -> Path:
    """Complete state, panel and pre-injection Check 1 values; FORK_READY comes later."""
    d = fork_dir(run_dir)
    if d.exists() and not (d / READY).exists():
        import shutil

        shutil.rmtree(d)  # an earlier, interrupted fork attempt
    d.mkdir(parents=True, exist_ok=True)
    trainer.save(d / "state")
    panel = sample_panel(trainer, int(trainer.cfg.seed), plan["fork_check_index"])
    save_npz(d / "panel.npz", panel)
    save_npz(d / "check1_pre.npz", panel_q_and_grad(trainer._sac_agent.critic, panel))
    atomic_write_text(d / "fork.json", json.dumps({**plan, "run_key": run_key, "run_dir": str(run_dir),
                                                   "fresh_critic_dir": fresh_critic_dir,
                                                   "device": device_info()}, indent=2))
    return d


def post_fork_eval(trainer, plan: Dict, eval_index: int, arm: str) -> list:
    """One post-fork evaluation that cannot influence training and is identical across arms
    given the same policy: a fresh env, a JAX key and the global RNGs (HumanoidBench draws
    Reach goals from np.random) all seeded from (seed, index); the training states are
    restored afterwards."""
    import random

    seed = int(trainer.cfg.seed)
    env_seed = int(np.random.SeedSequence([seed, EVAL_STREAM, eval_index]).generate_state(1)[0] % (2**31 - 1))
    env_cfg = dict(trainer.cfg.env)
    env_cfg["seed"] = env_seed
    core = trainer._sac_agent
    saved_key, np_state, py_state = core._rng, np.random.get_state(), random.getstate()
    env = None
    t0 = time.perf_counter()
    try:
        np.random.seed(env_seed)
        random.seed(env_seed)
        env = create_eval_env(**env_cfg)
        core._rng = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), EVAL_STREAM), eval_index)
        _, returns, lengths, successes = evaluate_episodes(trainer.agent, env, plan["eval_episodes"])
    finally:
        core._rng = saved_key
        np.random.set_state(np_state)
        random.setstate(py_state)
        if env is not None:
            env.close()
    print(f"[{arm}] post-fork evaluation {eval_index}: {time.perf_counter() - t0:.1f} s for "
          f"{plan['eval_episodes']} episodes", flush=True)
    steps_since = eval_index * plan["eval_every_steps"]
    return [{"arm": arm, "eval_index": eval_index, "steps_since_fork": steps_since,
             "interaction_step": plan["fork_step"] + steps_since, "episode": i, "return": float(r),
             "length": int(n), "success": float(s)} for i, (r, n, s) in enumerate(zip(returns, lengths, successes))]


def eval_due(plan: Dict, interaction_step: int) -> Optional[int]:
    since = interaction_step - plan["fork_step"]
    if 0 <= since <= plan["horizon_steps"] and since % plan["eval_every_steps"] == 0:
        return since // plan["eval_every_steps"]
    return None
