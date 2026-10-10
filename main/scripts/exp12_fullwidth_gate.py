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


STAGES = ("reference_cold", "reference_warm", "crash", "resume")


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def attempt_accounting(root, spec, records):
    """Bind one accepted process per stage to the immutable launcher checksum receipt.

    Failed attempts remain audited. Earlier manifests describe historical global
    files, so only their own attempt files are checked against current bytes.
    Accepted manifests must additionally bind every current stage artifact.
    Checksums establish consistency, not authenticity against wholesale forgery.
    """
    root = Path(root)
    audit = {stage: {"attempts": [], "accepted": None} for stage in STAGES}
    directories = sorted((root / "attempts").iterdir())
    jobs = set()
    for directory in directories:
        match = re.fullmatch(
            r"(reference_cold|reference_warm|crash|resume)_([0-9]+)", directory.name
        )
        if not match or not directory.is_dir() or directory.is_symlink():
            raise ValueError("malformed attempt directory: " + directory.name)
        stage, job = match.groups()
        if job in jobs:
            raise ValueError("job ID reused across stages: " + job)
        jobs.add(job)
        process = strict_json(directory / "process.json")
        if not isinstance(process, dict):
            raise ValueError("process must be an object")
        code = process.get("exit")
        if (
            type(code) is not int
            or not 0 <= code <= 255
            or process.get("stage") != stage
        ):
            raise ValueError("malformed process/stage: " + directory.name)
        verdict = {
            0: "PROCESS_COMPLETE",
            2: "INFRASTRUCTURE_FAILURE",
            124: "TIMEOUT",
            137: "KILLED_OR_TIMEOUT",
        }.get(code, "INCOMPLETE")
        if code == 3 and stage == "crash":
            verdict = "EXPECTED_CRASH"
        if (
            process.get("verdict") != verdict
            or type(process.get("timeout")) is not bool
            or process["timeout"] != (code in (124, 137))
        ):
            raise ValueError("inconsistent process verdict: " + directory.name)
        if (directory / "exit_status.txt").read_text().strip() != str(code):
            raise ValueError("exit status mismatch: " + directory.name)
        if (directory / "commit.txt").read_text().strip() != spec[
            "expected_commit"
        ] or json.dumps(
            strict_json(directory / "submitted-spec.json"), sort_keys=True
        ) != json.dumps(
            spec, sort_keys=True
        ):
            raise ValueError("attempt provenance mismatch: " + directory.name)
        if (directory / "SHA256SUMS").is_symlink():
            raise ValueError("symlink checksum manifest")
        manifest = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            m = re.fullmatch(r"([0-9a-f]{64})  (\./.+)", line)
            if not m:
                raise ValueError("malformed checksum record")
            digest, name = m.groups()
            relative = Path(name)
            if ".." in relative.parts or relative.is_absolute() or name in manifest:
                raise ValueError("unsafe or duplicate checksum record")
            if name != "./" + relative.as_posix():
                raise ValueError("noncanonical checksum path")
            manifest[name] = digest
        accepted = code == (3 if stage == "crash" else 0)
        required = [
            p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
        ]
        # Check deleted files too: a current file enumeration alone misses them.
        own_prefix = "./attempts/" + directory.name + "/"
        required += [
            root / name[2:]
            for name in manifest
            if name.startswith(own_prefix) and not name.endswith("/SHA256SUMS")
        ]
        if accepted:
            for name in ("dependencies.txt", "nvidia-smi.csv", "workload.log"):
                if not (directory / name).is_file():
                    raise ValueError("missing accepted process evidence: " + name)
            stage_root = root / stage
            if stage_root.is_symlink() or not stage_root.is_dir():
                raise ValueError("invalid stage directory")
            required += [
                root / name[2:]
                for name in manifest
                if name.startswith("./" + stage + "/")
            ]
            required += [p for p in stage_root.rglob("*") if p.is_file()]
            if not (stage_root / "receipt.json").is_file():
                raise ValueError("missing stage receipt")
        for path in required:
            if path.is_symlink() or any(
                parent.is_symlink()
                for parent in path.parents
                if parent != root and root in parent.parents
            ):
                raise ValueError("symlink evidence")
            name = "./" + path.relative_to(root).as_posix()
            if manifest.get(name) != file_sha(path):
                raise ValueError("missing or corrupt checksummed evidence: " + name)
        entry = {
            "job_id": job,
            "directory": directory.name,
            "exit": code,
            "verdict": verdict,
            "manifest_sha256": file_sha(directory / "SHA256SUMS"),
            "expected_commit": spec["expected_commit"],
            "spec_sha256": hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()
            ).hexdigest(),
        }
        audit[stage]["attempts"].append(entry)
        if accepted:
            if audit[stage]["accepted"] is not None:
                raise ValueError("ambiguous successful attempts: " + stage)
            receipt = records[stage]
            if (
                not isinstance(receipt, dict)
                or type(receipt.get("complete")) is not bool
            ):
                raise ValueError("malformed stage receipt")
            if any(
                type(receipt.get(k)) is not int
                for k in ("interaction_step", "update_step")
            ):
                raise ValueError("malformed stage counters")
            if (
                receipt.get("stage") != stage
                or receipt.get("expected_commit") != spec["expected_commit"]
                or receipt.get("spec_sha256") != entry["spec_sha256"]
            ):
                raise ValueError("stage receipt provenance mismatch: " + stage)
            entry["receipt_sha256"] = file_sha(root / stage / "receipt.json")
            audit[stage]["accepted"] = entry
    if any(not value["accepted"] for value in audit.values()):
        raise ValueError("missing accepted attempt")
    return audit


def compare(root, spec, report_path, comparator_commit):
    from experiments.exp12.state import latest_state_dir, state_differences, load_meta

    if Path(report_path).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("report must be outside immutable evidence")
    records = {
        s: strict_json(root / s / "receipt.json")
        for s in ("reference_cold", "reference_warm", "crash", "resume")
    }
    try:
        accounting = attempt_accounting(root, spec, records)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        write(
            report_path,
            {
                "verdict": "INCOMPLETE",
                "error": str(exc),
                "expected_commit": spec["expected_commit"],
                "comparator_commit": comparator_commit,
            },
        )
        return 1
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
        "attempt_accounting": accounting,
        "comparator_commit": comparator_commit,
        "differences": differences,
        "scope": "full-width engineering, not natural-trigger/probe/scientific qualification",
    }
    write(report_path, report)
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
    parser.add_argument(
        "--comparison-commit", help="exact clean CPU reviewer revision; compare only"
    )
    parser.add_argument(
        "--report", type=Path, help="new comparison report outside immutable evidence"
    )
    args = parser.parse_args()
    spec = strict_json(args.spec)
    validate_spec(spec)
    from utils.run_metadata import code_version
    import subprocess

    if args.comparison_commit and (
        args.stage != "compare"
        or not re.fullmatch("[0-9a-f]{40}", args.comparison_commit)
    ):
        raise ValueError("comparison revision is valid only for CPU compare")
    execution_commit = args.comparison_commit or spec["expected_commit"]
    if (
        code_version() != {"commit": execution_commit, "dirty": False}
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
    if args.stage == "compare":
        if (
            not args.report
            or not args.report.is_absolute()
            or args.report.resolve().is_relative_to(root)
            or args.report.resolve().is_relative_to(MAIN.parent)
            or args.report.exists()
        ):
            raise ValueError(
                "CPU comparison requires a fresh absolute report outside evidence and checkout"
            )
        if strict_json(root / "spec.json") != spec:
            raise ValueError("gate specification changed")
        try:
            return compare(root, spec, args.report, execution_commit)
        except Exception as exc:
            write(
                args.report,
                {
                    "verdict": "INCOMPLETE",
                    "error": f"{type(exc).__name__}: {exc}",
                    "expected_commit": spec["expected_commit"],
                    "comparator_commit": execution_commit,
                },
            )
            return 1
    if args.report:
        raise ValueError("external report is compare-only")
    root.mkdir(parents=True, exist_ok=True)
    canonical = root / "spec.json"
    if canonical.exists() and json.loads(canonical.read_text()) != spec:
        raise ValueError("gate specification changed across stages")
    if not canonical.exists():
        write(canonical, spec)
    try:
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
