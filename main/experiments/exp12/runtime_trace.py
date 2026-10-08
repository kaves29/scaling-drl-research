"""Opt-in stage timing for short diagnostic executions of the real pipeline."""

import contextlib
import functools
import json
import os
import time
from pathlib import Path
from unittest import mock

import jax


class RuntimeTrace:
    def __init__(self, path, synchronize=False, progress_every=100, *,
                 boundary_detail=False, stream=None):
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        self.path = Path(path)
        self.synchronize = synchronize
        self.progress_every = progress_every
        self.boundary_detail = boundary_detail
        self.stream = stream
        self.flush_depth = 0
        self.training = False
        self.totals = {}
        self.trainer = None
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        self.file = self.stream if self.stream is not None else self.stack.enter_context(self.path.open("x"))
        self.emit("trace_start", synchronize=self.synchronize, pid=os.getpid())
        self.emit("begin", stage="backend_init")
        start = time.perf_counter()
        try:
            device = str(jax.devices()[0])
        except BaseException as exc:
            self.emit(
                "end",
                stage="backend_init",
                error=type(exc).__name__,
                seconds=time.perf_counter() - start,
            )
            self.stack.close()
            raise
        self.emit(
            "end",
            stage="backend_init",
            error=None,
            device=device,
            seconds=time.perf_counter() - start,
        )
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.emit(
                "summary",
                totals=self.totals,
                error=None if kind is None else kind.__name__,
            )
        finally:
            self.stack.close()

    def emit(self, event, **fields):
        self.file.write(
            json.dumps(
                {
                    "event": event,
                    "wall_time": time.time(),
                    "monotonic_time": time.perf_counter(),
                    **fields,
                }
            )
            + "\n"
        )
        self.file.flush()

    def ready(self, result=None, *, log=False, phase=None):
        if self.synchronize:
            if log:
                self.emit("begin", stage="synchronization", phase=phase)
            start = time.perf_counter()
            error = None
            try:
                jax.block_until_ready(result)
                if self.trainer is not None and hasattr(self.trainer, "agent"):
                    agent = self.trainer._sac_agent
                    jax.block_until_ready(
                        (agent._actor, agent._critic, agent._target_critic,
                         agent._temperature, agent._rng)
                    )
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                elapsed = time.perf_counter() - start
                total = self.totals.setdefault("synchronization", {"calls": 0, "seconds": 0.0})
                total["calls"] += 1
                total["seconds"] += elapsed
                if log:
                    self.emit("end", stage="synchronization", phase=phase,
                              seconds=elapsed, error=error)

    def near_logging_boundary(self, label):
        if not self.boundary_detail or self.trainer is None or not self.training:
            return False
        if label not in {"sample_actions", "train_env_step", "replay_sample", "update_many"}:
            return False
        cfg = self.trainer.cfg
        cadence = getattr(cfg, "logging_per_interaction_step", None)
        if not cadence:
            return False
        remaining = (-self.trainer.interaction_step) % cadence
        return remaining < min(self.progress_every, cadence)

    @staticmethod
    def payload_metadata(tree):
        # Shape/dtype inspection only: no array conversion, transfer, or synchronization.
        leaves = jax.tree_util.tree_leaves(tree)
        return {"array_leaves": sum(hasattr(x, "nbytes") for x in leaves),
                "array_bytes": sum(int(x.nbytes) for x in leaves if hasattr(x, "nbytes"))}

    def wrap(self, obj, name, label, coarse=True, *, sync=True, metadata=None):
        original = getattr(obj, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            log_call = coarse or self.totals.get(label, {}).get("calls", 0) < 3 or self.near_logging_boundary(label)
            details = {"call_index": self.totals.get(label, {}).get("calls", 0) + 1,
                       "begin_recorded": log_call}
            if self.trainer is not None:
                details["interaction_step"] = self.trainer.interaction_step
            if metadata is not None:
                details.update(metadata(*args, **kwargs))
            if label == "_probe":
                details["check_index"] = kwargs.get(
                    "k", args[1] if len(args) > 1 else None
                )
            if label == "backend_compile":
                module = kwargs.get("module", args[1] if len(args) > 1 else None)
                if module is not None:
                    details["module"] = str(module.operation.attributes["sym_name"])
            if log_call:
                self.emit("begin", stage=label, **details)
            wall_start = time.perf_counter()
            start = None
            phase = "before_sync"
            error = None
            try:
                if sync:
                    self.ready(log=log_call, phase="before:" + label)
                start = time.perf_counter()
                phase = "call"
                result = original(*args, **kwargs)
                if label == "cache_read":
                    details["cache_hit"] = result[0] is not None
                if label == "configure_compilation_cache":
                    details["cache_dir"] = result
                if label == "trainer_init":
                    self.trainer = args[0]
                if sync:
                    phase = "after_sync"
                    self.ready(result, log=log_call, phase="after:" + label)
                phase = "complete"
                return result
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                elapsed = 0.0 if start is None else time.perf_counter() - start
                total = self.totals.setdefault(
                    label, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}
                )
                total["calls"] += 1
                total["seconds"] += elapsed
                total["max_seconds"] = max(total["max_seconds"], elapsed)
                if log_call or error is not None:
                    self.emit(
                        "end", stage=label, seconds=elapsed, error=error,
                        phase=phase, wall_seconds=time.perf_counter() - wall_start, **details
                    )

        self.stack.enter_context(mock.patch.object(obj, name, timed))

    def progress(self, trainer):
        self.ready(log=True, phase="progress")
        self.emit(
            "progress",
            interaction_step=trainer.interaction_step,
            env_step=trainer.interaction_step
            * trainer.cfg.action_repeat
            * trainer.cfg.num_train_envs,
            update_step=trainer.update_step,
            evaluation_episodes=len(trainer.eval_rows),
            post_fork_evaluation_episodes=len(
                trainer.extra_state.get("post_fork_evals", [])
            ),
            last_check=max(
                (
                    r["check_index"]
                    for r in trainer.extra_state.get("probe_records", [])
                ),
                default=None,
            ),
            totals=self.totals,
        )

    def install(self):
        from jax._src import compilation_cache, compiler

        from experiments import exp1, exp2_arm
        from experiments.exp12 import fork, precision, probe, run_probes, trainer
        from scale_rl.agents.sac.sac_agent import SACAgent
        from scale_rl.buffers.numpy_buffer import NpyUniformBuffer
        from experiments.exp12.diagnostics import ActorDiagnostics, DiagnosticPendingUpdateMetrics
        from experiments.angle_1 import PendingUpdateMetrics
        from scale_rl.common.logger import WandbTrainerLogger

        def pending_metadata(pending):
            return {"pending_groups": len(pending._pending),
                    **self.payload_metadata([info for _, info in pending._pending])}

        # These observer wrappers add no readiness calls or new transfers.
        self.wrap(DiagnosticPendingUpdateMetrics, "flush", "diagnostic_metric_flush",
                  sync=False, metadata=pending_metadata)
        self.wrap(PendingUpdateMetrics, "_replay", "metric_replay", sync=False,
                  metadata=lambda pending, host_infos: {
                      "pending_groups": len(pending._pending),
                      **self.payload_metadata(host_infos)})
        original_flush = DiagnosticPendingUpdateMetrics.flush

        def flush(pending):
            self.flush_depth += 1
            try:
                return original_flush(pending)
            finally:
                self.flush_depth -= 1

        self.stack.enter_context(mock.patch.object(DiagnosticPendingUpdateMetrics, "flush", flush))
        original_get = jax.device_get

        def device_get(value):
            if not self.flush_depth:
                return original_get(value)
            details = self.payload_metadata(value)
            self.emit("begin", stage="metric_device_get", **details)
            start = time.perf_counter()
            error = None
            try:
                return original_get(value)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self.emit("end", stage="metric_device_get", seconds=time.perf_counter() - start,
                          error=error, **details)

        self.stack.enter_context(mock.patch.object(jax, "device_get", device_get))
        self.wrap(ActorDiagnostics, "window_metrics", "diagnostic_window", sync=False)
        self.wrap(WandbTrainerLogger, "log_metric", "logger_log", sync=False)
        self.wrap(WandbTrainerLogger, "reset", "logger_reset", sync=False)

        for obj, names in (
            (
                trainer.Exp12Trainer,
                ("__init__", "start", "_evaluate", "save", "restore", "inject"),
            ),
            (run_probes.RunProbes, ("__init__", "capture_fresh", "_probe")),
            (probe, ("probe_round", "_base_targets", "_fit", "_mean_prediction")),
            (trainer, ("save_buffer", "load_buffer", "save_meta", "load_meta")),
            (fork, ("write_fork", "post_fork_eval", "panel_q_and_grad", "check1")),
            (exp2_arm, ("check2",)),
            (precision, ("configure_compilation_cache",)),
            (exp1, ("configure_compilation_cache",)),
            (exp2_arm, ("configure_compilation_cache",)),
        ):
            for name in names:
                label = (
                    ("trainer_init" if obj is trainer.Exp12Trainer else "probe_init")
                    if name == "__init__"
                    else name
                )
                self.wrap(obj, name, label)
        self.wrap(SACAgent, "update_many", "update_many", coarse=False)
        self.wrap(SACAgent, "sample_actions", "sample_actions", coarse=False)
        self.wrap(SACAgent, "get_metrics", "structural_metrics")
        self.wrap(SACAgent, "save_checkpoint", "agent_save")
        self.wrap(SACAgent, "load_checkpoint", "agent_restore")
        self.wrap(NpyUniformBuffer, "sample", "replay_sample", coarse=False)
        self.wrap(probe, "sample_pool", "probe_pool", coarse=False)
        self.wrap(compilation_cache, "get_executable_and_time", "cache_read")
        self.wrap(compilation_cache, "put_executable_and_time", "cache_write")
        self.wrap(compiler, "backend_compile", "backend_compile")

        original_init = trainer.Exp12Trainer.__init__

        def init(t, *args, **kwargs):
            original_init(t, *args, **kwargs)
            for label, env in (
                ("train_env_step", t.train_env),
                ("eval_env_step", t.eval_env),
            ):
                self.wrap(env, "step", label, coarse=False)

        self.stack.enter_context(
            mock.patch.object(trainer.Exp12Trainer, "__init__", init)
        )
        original_train = trainer.Exp12Trainer.train

        def train(t, last_step, after_step=None, before_first_update=None):
            self.trainer = t
            previous_training = self.training
            self.training = True
            self.emit(
                "train_begin",
                interaction_step=t.interaction_step,
                last_step=int(last_step),
            )

            def after(current):
                if after_step is not None:
                    after_step(current)
                if current.interaction_step % self.progress_every == 0:
                    self.progress(current)

            error = None
            try:
                return original_train(t, last_step, after, before_first_update)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self.training = previous_training
                try:
                    self.progress(t)
                except BaseException as exc:
                    if error is None:
                        raise
                    self.emit("progress_error", error=type(exc).__name__)
                self.emit(
                    "train_exit",
                    error=error,
                    counter_semantics="Loop counters; current interaction may be incomplete on exception",
                )

        self.stack.enter_context(
            mock.patch.object(trainer.Exp12Trainer, "train", train)
        )
