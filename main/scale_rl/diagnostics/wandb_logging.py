"""Publish already-computed critic diagnostics; never evaluate a model."""

import hashlib
from pathlib import Path

import wandb

from scale_rl.diagnostics.critic_metrics import PREFIX


def scalar_payload(record, *, environment_step):
    """Preserve scalar values/nulls and isolate all keys from historical metrics."""
    if record["schema"] != PREFIX:
        raise ValueError("unsupported critic diagnostic schema")
    metrics = record["metrics"]
    if any(not key.startswith(PREFIX + "/") for key in metrics):
        raise ValueError("critic diagnostics must not overwrite historical keys")
    if any(
        value is not None and type(value) not in (float, int, bool)
        for value in metrics.values()
    ):
        raise ValueError("critic diagnostic history requires scalar values")
    return {
        **metrics,
        f"{PREFIX}/interaction_step": int(record["interaction_step"]),
        f"{PREFIX}/optimizer_step": int(record["optimizer_step"]),
        f"{PREFIX}/raw_environment_step": int(environment_step),
    }


def publish_masks(record, output, *, environment_step):
    """Queue a per-run artifact version; offline SDK spools it locally.

    Disabled W&B retains local sidecars only. No extra wandb.log call, step
    increment, sampling, model operation or wait for server completion.
    """
    run = wandb.run
    if run is None or run.disabled:
        return None
    output = Path(output).resolve()
    mask = output / record["mask_file"]
    if mask.resolve().parent != output or mask.name != record["mask_file"]:
        raise ValueError("mask artifact must be inside diagnostic output")
    if hashlib.sha256(mask.read_bytes()).hexdigest() != record["mask_sha256"]:
        raise ValueError("critic diagnostic mask checksum mismatch")
    receipt = mask.with_suffix(".json")
    artifact = wandb.Artifact(
        name=f"critic-diagnostics-{run.id}",
        type="critic-diagnostics",
        description="Versioned critic_diag/v1 per-unit masks and measurement provenance",
        metadata={
            "schema": record["schema"],
            "interaction_step": record["interaction_step"],
            "optimizer_step": record["optimizer_step"],
            "raw_environment_step": int(environment_step),
            "mask_sha256": record["mask_sha256"],
            "inputs": record["inputs"],
        },
    )
    for path in (mask, receipt, output / "session.json", output / "frozen_inputs.npz"):
        if path in (mask, receipt) or path.exists():
            artifact.add_file(str(path), name=path.name, policy="immutable")
    return run.log_artifact(
        artifact,
        aliases=["latest", f"interaction-{record['interaction_step']}"],
    )
