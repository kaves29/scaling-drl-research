"""Explicit tiny engineering validation on an already allocated GPU; never sbatch."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    if not out.is_absolute():
        raise ValueError("absolute external evidence path required")
    if root == out or root in out.resolve().parents:
        raise ValueError("evidence must be outside the checkout")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.expected_commit or len(head) != 40:
        raise ValueError("GPU validation source revision mismatch")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip():
        raise ValueError("GPU validation requires a clean checkout")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "main"))
    import jax
    import jaxlib
    from experiments.exp12.precision import (
        configure_compilation_cache,
        runtime_info,
        set_matmul_precision,
    )

    if jax.__version__ != "0.4.34" or jaxlib.__version__ != "0.4.34":
        raise ValueError("pinned JAX/JAXLIB0.4.34 required")
    if (
        jax.default_backend() != "gpu"
        or len(jax.devices()) != 1
        or jax.devices()[0].device_kind != "NVIDIA A100-SXM4-40GB"
    ):
        raise ValueError("one actual A100-SXM4-40GB required; CPU fallback refused")
    if not jax.local_devices(backend="cpu"):
        raise ValueError("SAC explicit CPU initialization requires the CPU backend too")
    set_matmul_precision()
    configure_compilation_cache()
    (out / "backend.json").write_text(
        json.dumps(
            {
                "commit": head,
                "runtime": runtime_info(),
                "device_kind": jax.devices()[0].device_kind,
                "python": sys.version,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            indent=2,
        )
    )
    # Existing tiny CPU fixture constants are engineering tests, not Exp3 settings.
    # GPU qualification must execute every selected assertion; skips cannot qualify.
    names = [
        "tests.test_exp3_pilots.GuidanceTest",
        "tests.test_exp3_pilots.TargetTest",
        "tests.test_exp3_streams",
    ]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open(out / "tests.log", "x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    verdict = (
        "PASS"
        if result.wasSuccessful() and not result.skipped and result.testsRun > 0
        else "FAIL_OR_INCOMPLETE"
    )
    (out / "validation.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "commit": head,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skips": len(result.skipped),
                "scope": "tiny single/twin diagnostic gradients, targets and stream integrity; not full-width/source-run qualification",
            },
            indent=2,
        )
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
