"""Run every repository test module in isolated CPU processes with resumable receipts."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]


def fingerprint(main):
    """Hash executable/configuration inputs, including newly added untracked sources."""
    names = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(main.parent),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "main",
            ]
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = main.parent / name
        if not path.is_file() or (
            path.suffix not in (".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".ini")
            and "/.claude/" not in name
        ):
            continue
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def save_json(path, value):
    """Publish a complete receipt atomically."""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def check_provenance(path, provenance, resume):
    """Refuse stale receipts instead of assigning old results to a different tree."""
    if path.exists():
        if not resume or json.loads(path.read_text()) != provenance:
            raise ValueError(
                "validation output/provenance differs; preserve it and use a fresh directory"
            )
    else:
        save_json(path, provenance)


def test_ids(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from test_ids(case)
        else:
            yield case.id()


def worker(module, receipt, source_hash):
    """Use actual unittest counters, not inferred log text, as the completion receipt."""
    sys.path.insert(0, str(MAIN))
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    identifiers = list(test_ids(suite))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    successful = result.wasSuccessful() and result.testsRun == len(identifiers)
    save_json(
        receipt,
        {
            "module": module,
            "source_hash": source_hash,
            "success": successful,
            "tests_run": result.testsRun,
            "test_ids": identifiers,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skips": [
                {"test": test.id(), "reason": reason} for test, reason in result.skipped
            ],
            "unexpected_successes": len(result.unexpectedSuccesses),
        },
    )
    return 0 if successful else 1


def run_module(module, out, source_hash, environment):
    attempt = out / module / str(time.time_ns())
    attempt.mkdir(parents=True)
    log, receipt = attempt / "unittest.log", attempt / "receipt.json"
    with log.open("w") as stream:
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                module,
                "--receipt",
                str(receipt),
                "--source-hash",
                source_hash,
            ],
            cwd=MAIN,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    data = (
        json.loads(receipt.read_text())
        if receipt.exists()
        else {"module": module, "success": False, "tests_run": 0}
    )
    data.update(
        exit_code=process.returncode,
        log=str(log),
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    )
    data["success"] = bool(
        data["success"]
        and process.returncode == 0
        and data.get("source_hash") == source_hash
    )
    return data


def reusable(module, data, source_hash, out):
    """Require a completed unchanged module and its intact log before skipping work."""
    try:
        log = Path(data["log"]).resolve()
        return bool(
            data["module"] == module
            and data["success"] is True
            and data["exit_code"] == 0
            and data["source_hash"] == source_hash
            and data["failures"] == data["errors"] == 0
            and data.get("unexpected_successes", 0) == 0
            and data["tests_run"] == len(data["test_ids"])
            and out.resolve() in log.parents
            and hashlib.sha256(log.read_bytes()).hexdigest() == data["log_sha256"]
        )
    except (KeyError, OSError, TypeError):
        return False


def main(argv=None):
    """Validate code only; never submit jobs or modify scientific settings.

    A resumed run reuses only completed successful module receipts with the
    same source/configuration bytes, commit, Python, dependencies and CPU env.
    Failed/incomplete attempts remain on disk. Worker logs preserve warnings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-hash", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.receipt, args.source_hash)
    if not args.out_dir or not args.out_dir.is_absolute() or args.jobs < 1:
        parser.error("require absolute --out-dir and positive --jobs")
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "PYTHONDONTWRITEBYTECODE": "1",
        "EXP12_JAX_CACHE_DIR": "off",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "WANDB_MODE": "disabled",
    }
    modules = [
        "tests." + path.stem for path in sorted((MAIN / "tests").glob("test_*.py"))
    ]
    source_hash = fingerprint(MAIN)
    versions = {}
    for name in (
        "jax",
        "jaxlib",
        "numpy",
        "flax",
        "optax",
        "orbax-checkpoint",
        "gymnasium",
        "mujoco",
        "dm-control",
        "humanoid-bench",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hb_source = None
    spec = importlib.util.find_spec("humanoid_bench")
    if spec is not None and spec.origin:
        hb_root = Path(spec.origin).resolve().parent.parent
        commit = subprocess.run(
            ["git", "-C", str(hb_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(hb_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
        )
        hb_source = {
            "root": str(hb_root),
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=MAIN, text=True
        ).strip(),
        "source_hash": source_hash,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "versions": versions,
        "humanoid_bench_source": hb_source,
        "modules": modules,
        "jobs": args.jobs,
        "environment": {
            key: environment.get(key)
            for key in (
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "EGL_PLATFORM",
                "PYTHONPATH",
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "NVIDIA_TF32_OVERRIDE",
                "EXP12_JAX_CACHE_DIR",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    check_provenance(args.out_dir / "provenance.json", provenance, args.resume)
    ledger = args.out_dir / "modules.json"
    results = json.loads(ledger.read_text()) if ledger.exists() else {}
    pending = [
        module
        for module in modules
        if not reusable(module, results.get(module, {}), source_hash, args.out_dir)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_module, module, args.out_dir, source_hash, environment
            ): module
            for module in pending
        }
        for future in concurrent.futures.as_completed(futures):
            module = futures[future]
            try:
                results[module] = future.result()
            except Exception as error:
                results[module] = {
                    "module": module,
                    "success": False,
                    "tests_run": 0,
                    "error": repr(error),
                }
            save_json(ledger, results)
            print(
                json.dumps(
                    {
                        "module": module,
                        "success": results[module]["success"],
                        "tests_run": results[module]["tests_run"],
                    }
                ),
                flush=True,
            )
    unchanged = fingerprint(MAIN) == source_hash
    passed = unchanged and all(
        results.get(module, {}).get("success") for module in modules
    )
    summary = {
        "success": bool(passed),
        "source_unchanged": unchanged,
        "provenance": provenance,
        "modules": len(modules),
        "tests_run": sum(results[module]["tests_run"] for module in modules),
        "skips": [
            skip for module in modules for skip in results[module].get("skips", [])
        ],
        "failed_modules": [
            module for module in modules if not results[module]["success"]
        ],
    }
    save_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
