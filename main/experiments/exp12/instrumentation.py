"""Optional sidecars observing existing runner results without sampling or fitting."""

from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path

_active = ContextVar("echocritic_session", default=None)
ENVIRONMENT_VARIABLE = "ECHOCRITIC_INSTRUMENTATION_ROOT"


def entrypoint(kind):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            output = os.environ.get(ENVIRONMENT_VARIABLE)
            if not output:
                return function(*args, **kwargs)
            from analysis.campaign.provenance import DiagnosticSession

            repository = Path(__file__).resolve().parents[3]
            destination = Path(output)
            if not destination.is_absolute():
                raise ValueError("instrumentation output must be absolute")
            destination = destination.resolve()
            protected = [repository]
            training_args = (
                (args[0] if args else kwargs["args"]) if kind == "training" else None
            )
            if kind == "training":
                checkpoint = training_args.get("checkpoint_dir")
                if checkpoint:
                    protected.append(Path(checkpoint).resolve())
            if kind == "diagnostic":
                diagnostic_args = args[0] if args else kwargs.get("args")
                campaign = getattr(diagnostic_args, "campaign", None)
                if campaign:
                    protected.append(Path(campaign).resolve())
            if any(destination == p or p in destination.parents for p in protected):
                raise ValueError(
                    "instrumentation output must be outside source/checkpoints/frozen campaign"
                )
            with DiagnosticSession(
                destination, repository, {}, runner_kind=kind
            ) as session:
                token = _active.set(session)
                try:
                    result = function(*args, **kwargs)
                    if kind == "training":
                        from experiments.angle_1 import DONE_MARKER

                        done = Path(training_args["checkpoint_dir"]) / DONE_MARKER
                        session.completion_status = (
                            "complete" if done.exists() else "partial"
                        )
                    elif result not in (None, 0):
                        session.completion_status = "partial"
                    return result
                finally:
                    _active.reset(token)

        return run

    return decorate


def training_state(trainer, metrics, checkpoint=None):
    """Observe host metrics and actual live state; never run extra model operations."""
    session = _active.get()
    if session is None:
        return
    from omegaconf import OmegaConf
    from experiments.exp12.precision import runtime_info
    from analysis.campaign.provenance import parameter_count

    cfg = OmegaConf.to_container(trainer.cfg, resolve=True)
    agent = cfg["agent"]
    checkpoint = str(checkpoint) if checkpoint is not None else "in_memory"
    fields = {
        "optimizer": {
            name: {
                "transformation": "optax.adamw",
                "learning_rate": agent[f"{name}_learning_rate"],
                "weight_decay": agent[f"{name}_weight_decay"],
                "schedule": "constant",
            }
            for name in ("actor", "critic", "temp")
        },
        "architectures": {
            name: {k: v for k, v in agent.items() if k.startswith(name + "_")}
            for name in ("actor", "critic")
        },
        "parameter_counts": {
            "actor": parameter_count(trainer._sac_agent.actor.params),
            "critic": parameter_count(trainer._sac_agent.critic.params),
        },
        "environment": cfg["env"],
        "replay": dict(cfg["buffer"]),
        "target": {
            "kind": "online_bellman",
            "gamma": cfg["gamma"],
            "n_step": cfg["n_step"],
            "target_tau": agent["target_tau"],
        },
        "task_id": "online_training",
        "seed": cfg["seed"],
        "precision": runtime_info(),
        "checkpoint": checkpoint,
        "interaction_step": int(trainer.interaction_step),
        "optimizer_update": int(trainer.update_step),
    }
    if "injection" in trainer.extra_state:
        fields["optimizer"]["critic"][
            "transformation"
        ] = "injected_optimizer: AdamW trunk/new; zero old/copy"
        fields["architectures"]["injection"] = trainer.extra_state["injection"]
    fields["replay"]["populated_transitions"] = int(trainer.buffer._num_in_buffer)
    session.configure(cfg, **fields)
    # Numeric values are serialized verbatim, including explicit nonfinite markers.
    session.record_observation(
        metrics, fields["interaction_step"], fields["optimizer_update"], checkpoint
    )


def diagnostic_state(runner):
    """Mirror already-produced fixed-task rows; unknown counters stay unknown."""
    session = _active.get()
    if session is None:
        return
    from experiments.exp12.precision import runtime_info

    campaign = Path(runner.args.campaign).resolve()
    if session.output == campaign or campaign in session.output.parents:
        raise ValueError("instrumentation output must be outside the frozen campaign")
    session.configure(
        {"plan": runner.plan, "plan_sha256": runner.plan_sha256},
        optimizer={"bindings": runner.optimizer_provenance},
        checkpoint={"campaign": str(campaign)},
        precision=runtime_info(),
    )
    if not runner.rows:
        return
    session.record_observation(
        {
            "rows": runner.rows,
            "inputs": runner.inputs,
            "counter_semantics": "per-row fitting steps; training counters unavailable",
        },
        None,
        None,
        {"campaign": str(campaign)},
    )
