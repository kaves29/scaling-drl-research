"""Staged full-width engineering qualification; no submissions or forced triggers."""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))


def write(path, value):
    from utils.atomic_io import atomic_write_text

    atomic_write_text(Path(path), json.dumps(value, indent=2, allow_nan=False))


def validate_spec(spec):
    required = {
        "expected_commit",
        "environment",
        "seed",
        "checkpoint_step",
        "crash_step",
        "end_step",
    }
    if not required <= spec.keys() or not re.fullmatch(
        "[0-9a-f]{40}", str(spec["expected_commit"])
    ):
        raise ValueError("complete explicit gate specification required")
    from generate_manifest import EXP12_ENVS

    if spec["environment"] not in dict(EXP12_ENVS):
        raise ValueError("unknown approved environment")
    if any(
        type(spec[k]) is not int
        for k in ("seed", "checkpoint_step", "crash_step", "end_step")
    ):
        raise ValueError("integer seed and diagnostic extent required")
    # These are diagnostic stopping points, not replacements for the scientific budget.
    if not 5000 < spec["checkpoint_step"] < spec["crash_step"] < spec["end_step"]:
        raise ValueError("gate must cross actual warmup then checkpoint < crash < end")
    if spec["seed"] in range(1, 6):
        raise ValueError("qualification seed must be outside confirmatory seeds")


def placement(trainer):
    import jax
    import numpy as np

    result = {}
    for name in ("_actor", "_critic", "_target_critic", "_temperature"):
        network = getattr(trainer._sac_agent, name)
        for field in ("params", "opt_state"):
            leaves = jax.tree_util.tree_leaves(getattr(network, field))
            # Optimizer scalar counters can be host integers; floating moment arrays must be GPU-backed.
            floating = [
                x
                for x in leaves
                if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)
            ]
            platforms = sorted({d.platform for x in floating for d in x.devices()})
            result[f"{name}.{field}"] = {
                "floating_leaves": len(floating),
                "platforms": platforms,
                "dtypes": sorted({str(x.dtype) for x in floating}),
                "elements": sum(int(x.size) for x in floating),
            }
    return result


def gpu_placement_ok(record):
    # Target networks have no optimizer moments by design; all parameter trees are required.
    expected = {
        f"{n}.{f}"
        for n in ("_actor", "_critic", "_target_critic", "_temperature")
        for f in ("params", "opt_state")
    }
    return set(record) == expected and all(
        (
            v["platforms"] == ["gpu"] and v.get("dtypes") == ["float32"]
            if v["floating_leaves"]
            else k == "_target_critic.opt_state"
        )
        for k, v in record.items()
    )


def cache_observer(original, counters, observations):
    """Observe pinned JAX cache events per backend while returning the literal executable."""

    def wrapped(backend, *args, **kwargs):
        hit = counters.get("/jax/compilation_cache/cache_hits", 0)
        miss = counters.get("/jax/compilation_cache/cache_misses", 0)
        try:
            return original(backend, *args, **kwargs)
        finally:
            observations.append(
                {
                    "platform": backend.platform,
                    "hits": counters.get("/jax/compilation_cache/cache_hits", 0) - hit,
                    "misses": counters.get("/jax/compilation_cache/cache_misses", 0)
                    - miss,
                }
            )

    return wrapped


def classify(records, differences):
    for name in ("reference_cold", "reference_warm", "crash", "resume"):
        r = records.get(name)
        if not r or not r.get("complete") or r.get("backend") != "gpu":
            return "INCOMPLETE"
        if not all(gpu_placement_ok(v) for v in r.get("placements", {}).values()):
            return "INFRASTRUCTURE_FAILURE"
        if not r.get("placements") or r.get("scientific_invariant_failures"):
            return "SCIENTIFIC_INVARIANT_FAILURE"
        if name != "crash" and "peak_bytes_in_use" not in (
            r.get("device_memory_stats") or {}
        ):
            return "INCOMPLETE"
    if (
        not records["reference_cold"].get("cache_initially_empty")
        or records["reference_cold"].get("gpu_cache_misses", 0) <= 0
    ):
        return "INCOMPLETE"
    if records["reference_warm"].get("gpu_cache_hits", 0) <= 0:
        return "INCOMPLETE"
    if differences["references"]:
        return "NONDETERMINISTIC_BACKEND"
    if (
        differences["resume_vs_cold"]
        or differences["resume_vs_warm"]
        or differences["roundtrip"]
    ):
        return "RESUME_DEFECT"
    return "PASS"


def execute_stage(root, spec, stage):
    import jax
    import jaxlib
    import numpy as np
    import omegaconf
    from experiments.exp1 import compose_config
    from experiments.exp12 import fork
    from experiments.exp12.precision import (
        configure_compilation_cache,
        runtime_info,
        set_matmul_precision,
        _cache_events,
    )
    from experiments.exp12.state import latest_state_dir, state_differences
    from experiments.exp12.trainer import Exp12Trainer
    from generate_manifest import EXP12_ENVS, exp12_overrides
    from utils.run_metadata import code_version
    from scripts.exp12_launch_plan import require_pinned_stack

    require_pinned_stack()
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError("pinned Delta Python3.12.13 required for GPU execution")

    validate_spec(spec)
    if code_version() != {"commit": spec["expected_commit"], "dirty": False}:
        raise ValueError("exact clean source required")
    if jax.__version__ != jaxlib.__version__ or jax.__version__ != "0.4.34":
        raise ValueError("pinned JAX/JAXLIB0.4.34 required")
    devices = jax.devices()
    if (
        len(devices) != 1
        or devices[0].platform != "gpu"
        or devices[0].device_kind != "NVIDIA A100-SXM4-40GB"
    ):
        raise ValueError("one actual A100-SXM4-40GB required")
    jax.local_devices(backend="cpu")
    if jax.config.jax_enable_x64:
        raise ValueError("approved FP32 policy requires x64 disabled")
    cache = root / "persistent_cache"
    if stage == "reference_cold" and cache.exists():
        raise ValueError(
            "cold gate requires a new empty cache, never delete existing evidence"
        )
    if stage != "reference_cold" and not cache.is_dir():
        raise ValueError("cold reference must execute first")
    os.environ["EXP12_JAX_CACHE_DIR"] = str(cache)
    set_matmul_precision()
    configure_compilation_cache()
    stage_dir = root / stage
    stage_dir.mkdir(exist_ok=False)
    receipt = {
        "stage": stage,
        "complete": False,
        "backend": jax.default_backend(),
        "python": sys.version,
        "cache_initially_empty": stage == "reference_cold",
        "expected_commit": spec["expected_commit"],
        "spec_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest(),
        "runtime": runtime_info(),
        "placements": {},
        "scientific_invariant_failures": [],
        "timings": {},
    }
    publish = lambda: write(stage_dir / "receipt.json", receipt)
    publish()
    cfg = compose_config(
        str(MAIN / "configs"),
        "base_exp12",
        exp12_overrides(
            ("D4W1536", 4, 1536),
            spec["environment"],
            dict(EXP12_ENVS)[spec["environment"]],
            spec["seed"],
            str(root / "results"),
        )
        + ["run_role=dev"],
    )
    if spec["end_step"] >= int(cfg.num_interaction_steps):
        raise ValueError("qualification extent must be below the scientific budget")
    write(
        stage_dir / "resolved_config.json",
        omegaconf.OmegaConf.to_container(cfg, resolve=True),
    )
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    def timed(name, action):
        start = time.perf_counter()
        value = action()
        jax.block_until_ready(trainer._sac_agent._critic.params)
        receipt["timings"][name] = time.perf_counter() - start
        publish()
        return value

    # Pinned JAX compilation timing observes existing calls, without replacing kernels.
    from jax._src import compiler

    original_compile = compiler.backend_compile
    original_cached = compiler.compile_or_get_cached
    compilation = []
    cache_operations = []

    def observe_compile(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_compile(*args, **kwargs)
        finally:
            compilation.append(time.perf_counter() - start)

    compiler.backend_compile = observe_compile
    compiler.compile_or_get_cached = cache_observer(
        original_cached, _cache_events, cache_operations
    )
    try:
        init_start = time.perf_counter()
        trainer = Exp12Trainer(cfg, str(stage_dir))
        jax.block_until_ready(trainer._sac_agent._critic.params)
        receipt["timings"]["initialization"] = time.perf_counter() - init_start
        if stage == "resume":
            source = latest_state_dir(root / "crash" / "state")
            if source is None:
                raise ValueError("crash checkpoint missing")
            timed("restore", lambda: trainer.restore(source))
            if trainer.interaction_step != spec["checkpoint_step"]:
                raise ValueError("wrong checkpoint restored")
            receipt["restored_step"] = trainer.interaction_step
            receipt["restored_update"] = trainer.update_step
            receipt["placements"]["restored"] = placement(trainer)
        else:
            timed("start", trainer.start)
            timed("prefix", lambda: trainer.train(spec["checkpoint_step"]))
            receipt["placements"]["trained"] = placement(trainer)
            timed("checkpoint", lambda: trainer.save(stage_dir / "state"))
        if stage == "crash":
            timed("post_checkpoint", lambda: trainer.train(spec["crash_step"]))
            receipt.update(
                complete=True,
                simulated_crash=True,
                checkpoint_step=spec["checkpoint_step"],
                interaction_step=trainer.interaction_step,
                update_step=trainer.update_step,
            )
            receipt["placements"]["pre_crash"] = placement(trainer)
            publish()
            os._exit(3)  # abrupt process exit; no final checkpoint or graceful cleanup
        before = trainer.interaction_step
        timed("warm_train", lambda: trainer.train(spec["end_step"]))
        receipt["warm_iterations_per_second"] = (spec["end_step"] - before) / receipt[
            "timings"
        ]["warm_train"]
        receipt["training_only_projected_hours"] = (
            int(cfg.num_interaction_steps)
            / receipt["warm_iterations_per_second"]
            / 3600
        )
        receipt["placements"]["saved"] = placement(trainer)
        final = timed("final_save", lambda: trainer.save(stage_dir / "final"))
        # Cache evidence for the common training workload excludes subsequent fork reinitialization.
        receipt["cache_hits"] = _cache_events.get(
            "/jax/compilation_cache/cache_hits", 0
        )
        receipt["cache_misses"] = _cache_events.get(
            "/jax/compilation_cache/cache_misses", 0
        )
        receipt["cache_operations"] = list(cache_operations)
        receipt["gpu_cache_hits"] = sum(
            x["hits"] for x in cache_operations if x["platform"] in ("gpu", "cuda")
        )
        receipt["gpu_cache_misses"] = sum(
            x["misses"] for x in cache_operations if x["platform"] in ("gpu", "cuda")
        )
        # Only the cold reference performs independent full-width fork/identity roundtrip.
        # Diagnostic construction is not a natural trigger and never produces FORK_READY.
        if stage == "reference_cold":
            plan = fork.fork_plan(cfg, trainer.interaction_step, 1)
            timed(
                "fork_write",
                lambda: fork.write_fork(
                    trainer,
                    stage_dir,
                    plan,
                    "qualification_only",
                    "not_a_scientific_fresh_critic",
                ),
            )
            panel = fork.load_npz(stage_dir / "fork" / "panel.npz")
            pre = {
                name: fork.panel_q_and_grad(getattr(trainer._sac_agent, name), panel)
                for name in ("_critic", "_target_critic")
            }
            restored = Exp12Trainer(cfg, str(stage_dir / "restored"))
            timed(
                "fork_restore",
                lambda: restored.restore(
                    latest_state_dir(stage_dir / "fork" / "state")
                ),
            )
            receipt["placements"]["fork_restored"] = placement(restored)
            for name, values in pre.items():
                after = fork.panel_q_and_grad(getattr(restored._sac_agent, name), panel)
                check = fork.check1(values, after, after, 0.0, injected=False)
                write(stage_dir / f"check1{name}.json", check)
                if not check["pass"]:
                    receipt["scientific_invariant_failures"].append(name)
            roundtrip = restored.save(stage_dir / "roundtrip")
            receipt["roundtrip_differences"] = state_differences(final, roundtrip)
            receipt["parent_after_fork_differences"] = state_differences(
                final, latest_state_dir(stage_dir / "fork" / "state")
            )
            if receipt["parent_after_fork_differences"]:
                receipt["scientific_invariant_failures"].append(
                    "fork changed complete parent state"
                )
        receipt.update(
            complete=True,
            interaction_step=trainer.interaction_step,
            update_step=trainer.update_step,
            compilation_calls=len(compilation),
            compilation_seconds=sum(compilation),
            device_memory_stats=devices[0].memory_stats(),
            checkpoint_bytes=sum(
                p.stat().st_size for p in stage_dir.rglob("*") if p.is_file()
            ),
        )
        import resource

        receipt["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        publish()
    finally:
        compiler.backend_compile = original_compile
        compiler.compile_or_get_cached = original_cached


def compare(root, spec):
    from experiments.exp12.state import latest_state_dir, state_differences, load_meta

    records = {
        s: json.loads((root / s / "receipt.json").read_text())
        for s in ("reference_cold", "reference_warm", "crash", "resume")
    }
    for name, r in records.items():
        if (
            r.get("expected_commit") != spec["expected_commit"]
            or r.get("spec_sha256")
            != hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        ):
            r["complete"] = False
        expected = spec["crash_step"] if name == "crash" else spec["end_step"]
        if r.get("interaction_step") != expected or r.get("update_step") != 2 * (
            expected - 4999
        ):
            r["complete"] = False
        attempts = list((root / "attempts").glob(f"{name}_*/process.json"))
        if len(attempts) != 1 or json.loads(attempts[0].read_text()).get("exit") != (
            3 if name == "crash" else 0
        ):
            r["complete"] = False
    checkpoint = latest_state_dir(root / "crash" / "state")
    saved = load_meta(checkpoint)
    if (saved["interaction_step"], saved["update_step"]) != (
        records["resume"].get("restored_step"),
        records["resume"].get("restored_update"),
    ):
        records["resume"]["complete"] = False
    states = {
        n: latest_state_dir(root / n / "final")
        for n in ("reference_cold", "reference_warm", "resume")
    }
    differences = {
        "references": state_differences(
            states["reference_cold"], states["reference_warm"]
        ),
        "resume_vs_cold": state_differences(states["reference_cold"], states["resume"]),
        "resume_vs_warm": state_differences(states["reference_warm"], states["resume"]),
        "roundtrip": records["reference_cold"].get(
            "roundtrip_differences", ["missing roundtrip"]
        ),
    }
    verdict = classify(records, differences)
    report = {
        "verdict": verdict,
        "qualification": "fullwidth_resume_identity",
        "coverage": [
            {
                "architecture": "D4W1536",
                "suite": (
                    "humanoid_bench"
                    if spec["environment"].startswith("h1-")
                    else "myosuite" if spec["environment"].startswith("myo-") else "dmc"
                ),
                "environment": spec["environment"],
            }
        ],
        "expected_commit": spec["expected_commit"],
        "records": records,
        "differences": differences,
        "scope": "full-width engineering, not natural-trigger/probe/scientific qualification",
    }
    write(root / "validation.json", report)
    return 0 if verdict == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("reference_cold", "reference_warm", "crash", "resume", "compare"),
        required=True,
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    from utils.run_metadata import code_version
    import subprocess

    if (
        code_version() != {"commit": spec["expected_commit"], "dirty": False}
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=MAIN
        ).strip()
    ):
        raise ValueError(
            "every stage, including CPU comparison, requires the exact clean source"
        )
    root = args.out.resolve()
    if not args.out.is_absolute() or root.is_relative_to(MAIN.parent):
        raise ValueError("absolute evidence directory outside checkout required")
    root.mkdir(parents=True, exist_ok=True)
    canonical = root / "spec.json"
    if canonical.exists() and json.loads(canonical.read_text()) != spec:
        raise ValueError("gate specification changed across stages")
    if not canonical.exists():
        write(canonical, spec)
    try:
        if args.stage == "compare":
            return compare(root, spec)
        execute_stage(root, spec, args.stage)
        return 0
    except Exception as exc:
        write(
            root / f"{args.stage}_error.json",
            {"verdict": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
