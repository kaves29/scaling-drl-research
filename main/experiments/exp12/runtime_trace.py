"""Opt-in stage timing for short diagnostic executions of the real pipeline."""

import contextlib
import functools
import json
import time
from pathlib import Path
from unittest import mock

import jax


class RuntimeTrace:
    def __init__(self, path, synchronize=False, progress_every=100):
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        self.path = Path(path)
        self.synchronize = synchronize
        self.progress_every = progress_every
        self.totals = {}
        self.trainer = None
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        self.file = self.stack.enter_context(self.path.open("x"))
        self.emit(
            "trace_start", synchronize=self.synchronize, device=str(jax.devices()[0])
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
            json.dumps({"event": event, "wall_time": time.time(), **fields}) + "\n"
        )
        self.file.flush()

    def ready(self, result=None):
        if self.synchronize:
            start = time.perf_counter()
            jax.block_until_ready(result)
            if self.trainer is not None and hasattr(self.trainer, "agent"):
                agent = self.trainer._sac_agent
                jax.block_until_ready(
                    (
                        agent._actor,
                        agent._critic,
                        agent._target_critic,
                        agent._temperature,
                        agent._rng,
                    )
                )
            total = self.totals.setdefault(
                "synchronization", {"calls": 0, "seconds": 0.0}
            )
            total["calls"] += 1
            total["seconds"] += time.perf_counter() - start

    def wrap(self, obj, name, label, coarse=True):
        original = getattr(obj, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            log_call = coarse or self.totals.get(label, {}).get("calls", 0) < 3
            details = {}
            if label == "backend_compile":
                module = kwargs.get("module", args[1] if len(args) > 1 else None)
                if module is not None:
                    details["module"] = str(module.operation.attributes["sym_name"])
            if log_call:
                self.emit("begin", stage=label, **details)
            self.ready()
            start = time.perf_counter()
            error = None
            try:
                result = original(*args, **kwargs)
                if label == "trainer_init":
                    self.trainer = args[0]
                self.ready(result)
                return result
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                elapsed = time.perf_counter() - start
                total = self.totals.setdefault(
                    label, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}
                )
                total["calls"] += 1
                total["seconds"] += elapsed
                total["max_seconds"] = max(total["max_seconds"], elapsed)
                if log_call or error is not None:
                    self.emit(
                        "end", stage=label, seconds=elapsed, error=error, **details
                    )

        self.stack.enter_context(mock.patch.object(obj, name, timed))

    def progress(self, trainer):
        self.ready()
        self.emit(
            "progress",
            interaction_step=trainer.interaction_step,
            env_step=trainer.interaction_step
            * trainer.cfg.action_repeat
            * trainer.cfg.num_train_envs,
            update_step=trainer.update_step,
            evaluation_episodes=len(trainer.eval_rows),
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

        for obj, names in (
            (
                trainer.Exp12Trainer,
                ("__init__", "start", "_evaluate", "save", "restore"),
            ),
            (run_probes.RunProbes, ("capture_fresh", "_probe")),
            (probe, ("probe_round", "_base_targets", "_fit", "_mean_prediction")),
            (trainer, ("save_buffer", "load_buffer", "save_meta", "load_meta")),
            (fork, ("write_fork", "post_fork_eval", "panel_q_and_grad", "check1")),
            (exp2_arm, ("check2",)),
            (precision, ("configure_compilation_cache",)),
            (exp1, ("configure_compilation_cache",)),
            (exp2_arm, ("configure_compilation_cache",)),
        ):
            for name in names:
                self.wrap(obj, name, "trainer_init" if name == "__init__" else name)
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
                self.progress(t)
                self.emit(
                    "train_exit",
                    error=error,
                    counter_semantics="Loop counters; current interaction may be incomplete on exception",
                )

        self.stack.enter_context(
            mock.patch.object(trainer.Exp12Trainer, "train", train)
        )
