"""Prepare immutable plans and execute only explicitly approved stages; never sbatch."""

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
BASE = "844e3c0cc24998b25c3d1e667bcbfe056b1b336c"
GATES = {
    "pilot": (
        "fullwidth_resume_identity",
        "pilot_natural_trigger_disposition",
        "pilot_allocation",
    ),
    "parents": (
        "cuda_identity_scaled_suites",
        "fresh_null_disposition",
        "hopper_range_disposition",
        "positive_control",
        "suite_runtime_storage",
        "grid_allocation",
        "cuda_numeric_bounds",
        "study_methodology_freeze",
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_pinned_stack():
    from importlib.metadata import version

    pins = {
        "jax": "0.4.34",
        "jaxlib": "0.4.34",
        "flax": "0.8.4",
        "optax": "0.2.3",
        "orbax-checkpoint": "0.5.3",
        "numpy": "1.26.4",
    }
    for package, expected in pins.items():
        if version(package) != expected:
            raise ValueError(
                f"pinned requirement mismatch: {package} must be {expected}"
            )


def allocation_matches(resources, environment, job_description):
    """Check actual job allocation, not just requested CLI words."""
    if (
        int(environment.get("SLURM_CPUS_PER_TASK", "0")) != resources["cpus"]
        or int(environment.get("SLURM_MEM_PER_NODE", "0"))
        != resources["memory_gb"] * 1024
    ):
        raise ValueError("actual CPU/memory allocation differs from approval")
    match = re.search(r"\bTimeLimit=(\S+)", job_description)
    if not match:
        raise ValueError("cannot verify actual Slurm wall time")
    limit = match[1]
    days, clock = limit.split("-", 1) if "-" in limit else ("0", limit)
    parts = [int(x) for x in clock.split(":")]
    if (
        len(parts) != 3
        or int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
        != resources["wall_minutes"] * 60
    ):
        raise ValueError("actual wall time differs from approval")


def array_matches(stage, index, resume, environment):
    """Refuse accidentally launching only index0 of the195-parent plan."""
    if stage == "pilot":
        if environment.get("SLURM_ARRAY_TASK_ID") is not None:
            raise ValueError("development pilot must not be an array")
        return
    if int(environment.get("SLURM_ARRAY_TASK_ID", "-1")) != index:
        raise ValueError("parent execution requires the declared array index")
    if not resume and any(
        int(environment.get(k, "-1")) != v
        for k, v in {
            "SLURM_ARRAY_TASK_COUNT": 195,
            "SLURM_ARRAY_TASK_MIN": 0,
            "SLURM_ARRAY_TASK_MAX": 194,
            "SLURM_ARRAY_TASK_STEP": 1,
        }.items()
    ):
        raise ValueError("initial parent submission must enumerate exactly0-194")


def approve(approval, stage, commit):
    if (
        stage not in GATES
        or approval.get("approved") is not True
        or approval.get("stage") != stage
        or approval.get("expected_commit") != commit
    ):
        raise ValueError("explicit stage/source approval required")
    for name in GATES[stage]:
        gate = approval.get("gates", {}).get(name, {})
        if gate.get("status") != "APPROVED" or not gate.get("decision_reference"):
            raise ValueError(f"unresolved mandatory gate: {name}")
        if name.startswith("fullwidth") or name.startswith("cuda_identity"):
            coverage = set()
            paths = gate.get("evidence", [])
            if not paths:
                raise ValueError(f"verified evidence required: {name}")
            for evidence in paths:
                p = Path(evidence["path"])
                data = json.loads(p.read_text())
                if (
                    sha(p) != evidence["sha256"]
                    or data.get("verdict") != "PASS"
                    or data.get("expected_commit") != commit
                ):
                    raise ValueError(f"stale/failed evidence: {p}")
                required_kind = (
                    "cuda_identity_fork"
                    if name.startswith("cuda_identity")
                    else "fullwidth_resume_identity"
                )
                if data.get("qualification") != required_kind:
                    raise ValueError("GPU evidence has the wrong qualification scope")
                for cell in data.get("coverage", []):
                    coverage.add(
                        (cell["architecture"], cell["suite"], cell.get("environment"))
                    )
            if name.startswith("cuda_identity"):
                # Later amendment(p) explicitly names both scaled sizes in every suite.
                required = {
                    (a, s)
                    for a in ("D4W1024", "D4W1536")
                    for s in ("dmc", "myosuite", "humanoid_bench")
                }
                if not required <= {(a, s) for a, s, _ in coverage}:
                    raise ValueError(
                        "missing scaled-architecture/suite identity coverage"
                    )
            elif ("D4W1536", "dmc", "dog-run") not in coverage:
                raise ValueError("pilot needs full-width dog-run evidence")
    resources = approval.get("resources", {})
    for key in ("cpus", "memory_gb", "wall_minutes", "concurrency"):
        if type(resources.get(key)) is not int or resources[key] <= 0:
            raise ValueError(f"explicit approved allocation needed: {key}")
    if resources.get("gpus") != 1:
        raise ValueError("one A100 per parent is the prepared execution scope")
    if resources["concurrency"] > (1 if stage == "pilot" else 195):
        raise ValueError("concurrency exceeds stage population")
    return resources


def config_hash(overrides):
    import omegaconf
    from experiments.exp1 import compose_config
    from utils.run_metadata import _hash, _strip_locations

    cfg = compose_config(str(MAIN / "configs"), "base_exp12", overrides)
    return _hash(_strip_locations(omegaconf.OmegaConf.to_container(cfg, resolve=True)))


def cells(root, stage):
    from generate_manifest import (
        EXP12_ARCHS,
        EXP12_ENVS,
        EXP12_SEEDS,
        EXP12_BUDGETS,
        exp12_overrides,
        exp12_paths,
    )

    entries = []
    population = [
        (a, e, g, s) for e, g in EXP12_ENVS for a in EXP12_ARCHS for s in EXP12_SEEDS
    ]
    if stage == "pilot":
        population = [(EXP12_ARCHS[2], "dog-run", "dmc_hard", 102)]
    for a, e, g, seed in population:
        checkpoint, _ = exp12_paths(str(root / "checkpoints"), a[0], e, seed)
        overrides = exp12_overrides(a, e, g, seed, str(root / "results"))
        if stage == "pilot":
            overrides += ["run_role=dev"]
        command = [
            sys.executable,
            "-u",
            "run.py",
            "--experiment",
            "exp1",
            "--config_name",
            "base_exp12",
        ]
        for override in overrides:
            command += ["--overrides", override]
        interval = EXP12_BUDGETS[g] // 2 // 20
        command += [
            "--checkpoint_dir",
            checkpoint,
            "--checkpoint_interval",
            str(interval),
            "--checkpoint_start_frac",
            "0.0",
        ]
        entries.append(
            {
                "index": len(entries),
                "run_key": f"exp1_{a[0]}_{e}_seed{seed}",
                "architecture": a[0],
                "environment": e,
                "seed": seed,
                "group": g,
                "checkpoint_dir": checkpoint,
                "budget": EXP12_BUDGETS[g],
                "config_hash": config_hash(overrides),
                "overrides": overrides,
                "command": command,
            }
        )
    return entries


def prepare(stage, root, commit, approval_path, out):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("full source SHA required")
    root = Path(root)
    if not root.is_absolute() or root.resolve().is_relative_to(MAIN.parent):
        raise ValueError(
            "absolute role-separated campaign root outside checkout required"
        )
    approval = json.loads(Path(approval_path).read_text())
    resources = approve(approval, stage, commit)
    if root.exists():
        raise ValueError(
            "fresh campaign root required; explicit recovery uses original plan"
        )
    rows = cells(root, stage)
    if len(rows) != (1 if stage == "pilot" else 195) or len(
        {r["run_key"] for r in rows}
    ) != len(rows):
        raise ValueError("population incomplete or duplicated")
    plan = {
        "schema_version": 1,
        "stage": stage,
        "commit": commit,
        "root": str(root),
        "approval": str(Path(approval_path).resolve()),
        "approval_sha256": sha(approval_path),
        "resources": resources,
        "entries": rows,
    }
    with Path(out).open("x") as stream:
        json.dump(plan, stream, indent=2)
    return plan


def submission_command(plan_path):
    """Print only; allocation values come from the immutable approved plan."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if sha(plan["approval"]) != plan["approval_sha256"]:
        raise ValueError("approval changed after preparation")
    resources = approve(
        json.loads(Path(plan["approval"]).read_text()), plan["stage"], plan["commit"]
    )
    if (
        resources != plan["resources"]
        or cells(Path(plan["root"]), plan["stage"]) != plan["entries"]
    ):
        raise ValueError("plan differs from approved configuration/allocation")
    logs = Path(plan["root"]) / "slurm"
    command = [
        "env",
        f"EXPECTED_COMMIT={plan['commit']}",
        f"LAUNCH_PLAN={plan_path}",
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory_gb']}G",
        f"--time={resources['wall_minutes']}",
        f"--output={logs}/%x_%A_%a.out",
        f"--error={logs}/%x_%A_%a.err",
    ]
    if plan["stage"] == "parents":
        command.append(f"--array=0-194%{resources['concurrency']}")
    command.append("scripts/sbatch_exp12_prepared_parent.sh")
    reservation = str(plan_path) + ".submission-reservation"
    return (
        shlex.join(command)
        + " > "
        + shlex.quote(reservation + "/job-id.txt")
        + " 2> "
        + shlex.quote(reservation + "/submission.stderr")
    )


def execute(plan_path, index, resume=False):
    from utils.run_metadata import code_version

    require_pinned_stack()
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError("pinned Delta Python3.12.13 required for execution")

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if sha(plan["approval"]) != plan["approval_sha256"]:
        raise ValueError("approval changed after preparation")
    approve(
        json.loads(Path(plan["approval"]).read_text()), plan["stage"], plan["commit"]
    )
    if code_version() != {"commit": plan["commit"], "dirty": False}:
        raise ValueError("source not exact approved clean commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=MAIN
    ).strip():
        raise ValueError("untracked or changed checkout refused")
    expected = cells(Path(plan["root"]), plan["stage"])
    if expected != plan["entries"] or not 0 <= index < len(expected):
        raise ValueError("plan commands or population changed")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("prepared execution requires an approved Slurm allocation")
    description = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True
    )
    allocation_matches(plan["resources"], os.environ, description)
    array_matches(plan["stage"], index, resume, os.environ)
    import jax
    import jaxlib

    if jax.__version__ != jaxlib.__version__ or jax.__version__ != "0.4.34":
        raise ValueError("pinned JAX required")
    if (
        len(jax.devices()) != 1
        or jax.devices()[0].device_kind != "NVIDIA A100-SXM4-40GB"
        or jax.default_backend() != "gpu"
    ):
        raise ValueError("actual approved A100 GPU required")
    jax.local_devices(backend="cpu")
    row = expected[index]
    run = Path(row["checkpoint_dir"])
    if (run / "DONE").exists():
        raise ValueError(
            "completed marker requires scientific post-validation; never rerun automatically"
        )
    if run.exists() and not resume:
        raise ValueError(
            "existing run requires explicit --resume after recovery review"
        )
    if resume:
        from experiments.exp12.state import latest_state_dir, load_meta
        from utils.run_metadata import load_run_metadata

        meta = load_run_metadata(run / "run_metadata.json")
        if (
            not meta
            or meta["config_hash"] != row["config_hash"]
            or meta["code"] != {"commit": plan["commit"], "dirty": False}
        ):
            raise ValueError("resume provenance mismatch")
        state = latest_state_dir(run / "state")
        if state is None:
            raise ValueError(
                "no committed resume state; inspect fork-only recovery separately"
            )
        load_meta(state)
    claims = Path(plan["root"]) / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims / row["run_key"]
    claim.mkdir()  # single atomic claim; never steal stale claims automatically
    (claim / "owner.json").write_text(
        json.dumps(
            {"job_id": os.environ.get("SLURM_JOB_ID"), "plan_sha256": sha(plan_path)}
        )
    )
    attempts = Path(plan["root"]) / "attempts" / row["run_key"]
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / (
        os.environ.get("SLURM_JOB_ID", "local")
        + "_"
        + str(os.environ.get("SLURM_RESTART_COUNT", "0"))
    )
    attempt.mkdir()
    os.environ["EXP12_JAX_CACHE_DIR"] = str(attempt / "jax_cache")
    with (attempt / "training.log").open("x") as log:
        code = subprocess.call(
            row["command"], cwd=MAIN, stdout=log, stderr=subprocess.STDOUT
        )
    (attempt / "exit_status.txt").write_text(str(code) + "\n")
    # Claims intentionally retained on all outcomes until evidence/recovery is reviewed.
    return code


def validate_replay_normalization(state):
    """Read required auxiliary checkpoint payloads; never construct a trainer."""
    import pickle
    import numpy as np
    from experiments.exp12.state import BUFFER_ARRAYS

    with (state / "obs_rms.pkl").open("rb") as stream:
        rms = pickle.load(stream)
    if set(rms) != {"mean", "var", "count"} or any(
        not np.isfinite(np.asarray(v)).all() for v in rms.values()
    ):
        raise ValueError("invalid normalization checkpoint")
    if np.asarray(rms["mean"]).shape != np.asarray(rms["var"]).shape:
        raise ValueError("normalization shape mismatch")
    with (state / "buffer_meta.pkl").open("rb") as stream:
        replay = pickle.load(stream)
    if not {"num_in_buffer", "current_idx", "n_step_transitions"} <= replay.keys():
        raise ValueError("missing replay metadata fields")
    if any(
        type(replay[k]) is not int or replay[k] < 0
        for k in ("num_in_buffer", "current_idx")
    ):
        raise ValueError("invalid replay counters")
    with np.load(state / "buffer.npz", allow_pickle=False) as data:
        if set(data.files) != {k.lstrip("_") for k in BUFFER_ARRAYS}:
            raise ValueError("replay array members differ")
        for key in data.files:
            value = data[key]
            if (
                not value.shape
                or value.shape[0] != replay["num_in_buffer"]
                or not np.isfinite(value).all()
            ):
                raise ValueError("invalid replay payload")


def postvalidate(run, commit, out):
    """Engineering completeness is separate from canonical scientific certification."""
    import numpy as np
    from experiments.exp12.state import latest_state_dir, load_meta, load_agent_tree
    from utils.run_metadata import _hash, _strip_locations

    run = Path(run)
    errors = []
    report = {
        "engineering_verdict": "INCOMPLETE",
        "scientific_verdict": "NOT_QUALIFIED",
        "expected_commit": commit,
        "errors": errors,
    }
    try:
        metadata = json.loads((run / "run_metadata.json").read_text())
        cfg = metadata["resolved_config"]
        from generate_manifest import EXP12_ARCHS, EXP12_ENVS, exp12_overrides

        identity = metadata["identity"]
        architecture = next(a for a in EXP12_ARCHS if a[0] == identity["architecture"])
        expected_overrides = exp12_overrides(
            architecture,
            identity["environment"],
            dict(EXP12_ENVS)[identity["environment"]],
            identity["seed"],
            cfg["results_root"],
        )
        if identity["run_role"] == "dev":
            expected_overrides += ["run_role=dev"]
        elif identity["run_role"] != "confirmatory" or identity["seed"] not in range(
            1, 6
        ):
            raise ValueError("invalid production role/seed")
        if metadata["code"] != {"commit": commit, "dirty": False} or any(
            x["commit"] != commit or x["dirty"] for x in metadata["launches"]
        ):
            raise ValueError("source/launch provenance mismatch")
        if metadata["config_hash"] != _hash(_strip_locations(cfg)):
            raise ValueError("resolved configuration fingerprint mismatch")
        if metadata["config_hash"] != config_hash(expected_overrides):
            raise ValueError(
                "settings differ from the approved grid/development configuration"
            )
        done = json.loads((run / "DONE").read_text())
        state = latest_state_dir(run / "state")
        meta = load_meta(state)
        required = {
            "interaction_step",
            "update_step",
            "update_counter",
            "numpy_rng_state",
            "python_rng_state",
            "train_env",
            "eval_env",
            "observations",
            "timestep",
            "meters",
            "media",
            "metrics_rows",
            "eval_rows",
            "extra_state",
            "agent_window_buffers",
            "actor_diagnostics",
        }
        if not required <= meta.keys():
            raise ValueError("checkpoint complete-state contract missing fields")
        expected = int(cfg["num_interaction_steps"])
        if (run / "fork" / "fork.json").exists():
            plan = json.loads((run / "fork" / "fork.json").read_text())
            expected = plan["control_end_step"]
            if not (run / "fork" / "FORK_READY").exists():
                raise ValueError("incomplete fork")
            from experiments.exp12.fork import check1, load_npz

            pre, control = load_npz(run / "fork/check1_pre.npz"), load_npz(
                run / "fork/check1_control.npz"
            )
            if not check1(pre, control, control, 0, False)["pass"]:
                raise ValueError("bit-exact control Check1 failed")
        if meta["interaction_step"] != expected or done["interaction_step"] != expected:
            raise ValueError("premature endpoint")
        if meta["update_step"] != 2 * (expected - 4999) or meta["update_counter"] != 0:
            raise ValueError(
                "training update counters inconsistent with approved warmup/UTD"
            )
        import jax

        tree = load_agent_tree(state)
        if (
            not {
                "rng",
                "actor",
                "critic",
                "target_critic",
                "temperature",
                "churn_ref_batch",
            }
            <= tree.keys()
        ):
            raise ValueError("agent checkpoint contract missing fields")
        for value in jax.tree_util.tree_leaves(tree):
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("nonfinite saved model/optimizer state")
        validate_replay_normalization(state)
        import pandas as pd

        records = pd.read_csv(run / "probes/probe_checks.csv")
        last_check = expected // (
            int(cfg["num_interaction_steps"]) // int(cfg["probe"]["checks"])
        )
        if records["check_index"].duplicated().any() or set(
            records["check_index"]
        ) != set(range(last_check + 1)):
            raise ValueError("missing scheduled probes")
        for check in range(last_check + 1):
            with np.load(run / "probes" / f"check_{check:02d}.npz") as data:
                if not data.files or any(
                    not np.isfinite(data[k]).all() for k in data.files
                ):
                    raise ValueError(f"missing/nonfinite probe {check}")
        report.update(
            engineering_verdict="COMPLETE",
            run_role=metadata["identity"]["run_role"],
            endpoint=expected,
            scientific_verdict=(
                "DEV_EXCLUDED"
                if metadata["identity"]["run_role"] == "dev"
                else "CANONICAL_CERTIFICATION_REQUIRED"
            ),
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    with Path(out).open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


def certify_identity(run, arm, commit, out):
    """Read existing canonical identity artifacts; never run training or reconstruct state."""
    from scripts.compare_identity_fork import compare
    from analysis.exp12_validation import suite
    from experiments.exp12.fork import check1, load_npz
    from experiments.exp12.state import latest_state_dir, load_meta

    run, arm = Path(run), Path(arm)
    metadata = [json.loads((p / "run_metadata.json").read_text()) for p in (run, arm)]
    for m in metadata:
        if m["code"] != {"commit": commit, "dirty": False} or not m["launches"]:
            raise ValueError("identity source is incompatible")
        for launch in m["launches"]:
            if (
                launch.get("commit") != commit
                or launch.get("dirty") is not False
                or launch.get("device", {}).get("platform") != "gpu"
                or launch.get("device", {}).get("device_kind")
                != "NVIDIA A100-SXM4-40GB"
            ):
                raise ValueError(
                    "identity artifacts do not establish the approved GPU/source"
                )
            if (
                launch.get("runtime", {}).get("jax") != "0.4.34"
                or launch.get("runtime", {}).get("jaxlib") != "0.4.34"
                or launch.get("runtime", {}).get("matmul_precision") != "tensorfloat32"
            ):
                raise ValueError("identity precision/runtime policy mismatch")
        c = m["resolved_config"]
        dimensions = {"D4W1024": (4, 1024), "D4W1536": (4, 1536)}
        if dimensions.get(m["identity"]["architecture"]) != (
            c["critic_num_blocks"],
            c["critic_hidden_dim"],
        ) or (c["actor_num_blocks"], c["actor_hidden_dim"]) != (1, 128):
            raise ValueError("identity architecture is not the actual approved width")
        import omegaconf
        from experiments.exp1 import compose_config
        from generate_manifest import EXP12_ARCHS, EXP12_ENVS, exp12_overrides

        architecture = next(
            a for a in EXP12_ARCHS if a[0] == m["identity"]["architecture"]
        )
        canonical = omegaconf.OmegaConf.to_container(
            compose_config(
                str(MAIN / "configs"),
                "base_exp12",
                exp12_overrides(
                    architecture,
                    m["identity"]["environment"],
                    dict(EXP12_ENVS)[m["identity"]["environment"]],
                    m["identity"]["seed"],
                    c["results_root"],
                )
                + [f"run_role={m['identity']['run_role']}"],
            ),
            resolve=True,
        )
        for key in (
            "agent",
            "buffer",
            "env",
            "probe",
            "trigger",
            "checks",
            "diagnostics",
            "num_env_steps",
            "num_interaction_steps",
            "updates_per_interaction_step",
        ):
            if c[key] != canonical[key]:
                raise ValueError(f"identity does not use full approved settings: {key}")
    identities = [
        (
            m["identity"]["architecture"],
            m["identity"]["environment"],
            m["identity"]["seed"],
        )
        for m in metadata
    ]
    if identities[0] != identities[1]:
        raise ValueError("identity parent/arm are unmatched")
    if metadata[0]["protocol_hash"] != metadata[1]["protocol_hash"]:
        raise ValueError("identity training protocols differ")
    result = compare(run, arm)
    original, restored = load_npz(run / "fork/check1_pre.npz"), load_npz(
        run / "fork/check1_control.npz"
    )
    if not check1(original, restored, restored, 0.0, False)["pass"]:
        raise ValueError("control differs from original pre-fork Check1")
    for p in (run / "fork/identity_control", arm / "identity_identity"):
        state = latest_state_dir(p)
        if load_meta(state)["update_step"] <= 0:
            raise ValueError("identity snapshot is untrained")
        verify_saved_gpu_sharding(state)
    files = [
        run / "run_metadata.json",
        arm / "run_metadata.json",
        run / "fork/check1_pre.npz",
        run / "fork/check1_control.npz",
        arm / "check1_after.npz",
    ]
    for p in (Path(result["control_snapshot"]), Path(result["identity_snapshot"])):
        files += [f for f in p.rglob("*") if f.is_file()]
    report = dict(
        verdict="PASS" if result["pass"] else "SCIENTIFIC_INVARIANT_FAILURE",
        qualification="cuda_identity_fork",
        expected_commit=commit,
        coverage=[
            dict(
                architecture=identities[0][0],
                suite=suite(identities[0][1]),
                environment=identities[0][1],
            )
        ],
        native_comparison=result,
        inputs={str(f): sha(f) for f in files},
    )
    with Path(out).open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


def verify_saved_gpu_sharding(state):
    """Pinned Orbax saved placement: parameters and AdamW moments, not host counters."""
    encoded = json.loads((Path(state) / "agent_ckpt/_sharding").read_text())
    shards = {base64.b64decode(k).decode(): json.loads(v) for k, v in encoded.items()}
    required = [
        n + ".params." for n in ("actor", "critic", "target_critic", "temperature")
    ]
    for prefix in required:
        group = [v for k, v in shards.items() if k.startswith(prefix)]
        if not group or any(
            not re.fullmatch(r"(?:cuda|gpu):\d+", v.get("device_str", ""))
            for v in group
        ):
            raise ValueError("saved parameter placement does not establish GPU backing")
    for network in ("actor", "critic", "temperature"):
        for moment in (".mu.", ".nu."):
            group = [
                v
                for k, v in shards.items()
                if k.startswith(network + ".opt_state.") and moment in k
            ]
            if not group or any(
                not re.fullmatch(r"(?:cuda|gpu):\d+", v.get("device_str", ""))
                for v in group
            ):
                raise ValueError("saved optimizer moments do not establish GPU backing")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--stage", choices=tuple(GATES), required=True)
    for name in ("root", "commit", "approval", "out"):
        build.add_argument("--" + name, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--plan", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--resume", action="store_true")
    command = sub.add_parser("submission-command")
    command.add_argument("--plan", required=True)
    check = sub.add_parser("validate-run")
    for name in ("run", "commit", "out"):
        check.add_argument("--" + name, required=True)
    identity = sub.add_parser("certify-identity")
    for name in ("run", "arm", "commit", "out"):
        identity.add_argument("--" + name, required=True)
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.stage, args.root, args.commit, args.approval, args.out)
        return 0
    if args.action == "execute":
        return execute(args.plan, args.index, args.resume)
    if args.action == "submission-command":
        print(submission_command(args.plan))
        return 0
    if args.action == "certify-identity":
        return (
            0
            if certify_identity(args.run, args.arm, args.commit, args.out)["verdict"]
            == "PASS"
            else 1
        )
    return (
        0
        if postvalidate(args.run, args.commit, args.out)["engineering_verdict"]
        == "COMPLETE"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
