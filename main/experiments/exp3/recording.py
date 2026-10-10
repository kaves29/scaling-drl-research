"""Process-local opt-in recording of existing Exp2 arm entry points, unchanged loop."""

import contextlib
from pathlib import Path
from unittest.mock import patch

from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp3.artifacts import digest, state_fingerprint
from experiments.exp3.retention import retain_state, validate_retention
from experiments.exp3.streams import (
    RecordingBuffer,
    StreamReader,
    StreamWriter,
    benchmark_source,
    write_json,
)
from utils.run_metadata import code_version, load_run_metadata


@contextlib.contextmanager
def record_exp2_scope(root, chunk_size, benchmark=None, resume=False, retain=None):
    """Invoke the existing Exp2 run inside this scope; no calls to trainer.save.

    This instrumentation is opt-in and process-local. It writes new output only.
    Multiple source train calls require a contiguous committed stream cursor;
    crash recovery ahead of a source checkpoint fails rather than inventing data.

    retain (optional, no defaults): {"root", "until_step", "replay"}. Every routine
    save the source already makes under <run_dir>/state at a step in
    [fork_step, until_step] is kept as an immutable snapshot (retention.py). This
    includes an Exp2 arm's post-injection save at the fork step.
    """
    retain = validate_retention(retain)
    original = Exp12Trainer.train
    original_class_save = Exp12Trainer.save
    writer = None

    def save_retaining(trainer, root, keep_previous=False):
        path = original_class_save(trainer, root, keep_previous=keep_previous)
        plan = trainer.extra_state.get("fork")
        if (
            retain is not None
            and plan is not None
            and Path(root).resolve() == (trainer.run_dir / "state").resolve()
            and int(plan["fork_step"]) <= trainer.interaction_step <= retain["until_step"]
        ):
            retain_state(path, retain["root"], trainer.interaction_step, retain["replay"])
        return path

    def recorded(trainer, last_step, after_step=None, before_first_update=None):
        nonlocal writer
        # Exp1's pre-fork training must run unchanged and is not a post-fork stream.
        # Its normal ForkNow handler enters control, then invokes train again.
        plan = trainer.extra_state.get("fork")
        if plan is None:
            return original(trainer, last_step, after_step, before_first_update)
        metadata = trainer.run_dir / "run_metadata.json"
        info = load_run_metadata(metadata)
        if not info:
            raise ValueError("source recorder needs actual run provenance")
        parent = (
            Path(str(trainer.cfg.fork.source))
            if trainer.cfg.fork.source
            else trainer.run_dir
        )
        fork_metadata = parent / "run_metadata.json"
        fork_state = latest_state_dir(parent / "fork" / "state")
        if fork_state is None:
            raise ValueError("source recorder requires immutable complete fork state")
        if writer is None:
            # A parent that crashed before its fork has no stream yet: start fresh;
            # the fork-step check below still refuses any missing historical prefix.
            resume_stream = resume and (Path(root) / "manifest.json").exists()
            if not resume_stream and trainer.interaction_step != int(plan["fork_step"]):
                raise ValueError(
                    "historical prefix unavailable: new recording must begin at the exact fork"
                )
            provenance = {
                "start_step": int(plan["fork_step"]),
                "fork_metadata_sha256": digest(fork_metadata),
                "config_hash": load_run_metadata(fork_metadata)["config_hash"],
                "source_metadata_sha256": digest(metadata),
                "source_code": code_version(),
                "source_identity": info["identity"],
                "fork_state": str(fork_state),
                "fork_state_sha256": state_fingerprint(fork_state),
                "arm": trainer.extra_state.get("arm", "control"),
                # The passive I learner must reproduce the source arm's actual injection.
                "injection": trainer.extra_state.get("injection"),
                "captures": ["agent_key"],
            }
            if resume_stream:
                previous = StreamReader(root, require_complete=False).manifest[
                    "provenance"
                ]
                # run_metadata records a new launch on restart. Retain original
                # provenance but verify immutable fork/source/arm identity.
                for k in (
                    "start_step",
                    "fork_metadata_sha256",
                    "config_hash",
                    "fork_state_sha256",
                    "source_identity",
                    "source_code",
                    "arm",
                    "injection",
                    "captures",
                ):
                    if previous.get(k) != provenance[k]:
                        raise ValueError(f"source recording resume mismatch: {k}")
                write_json(
                    Path(root) / "resume_launch.json",
                    {
                        "metadata_sha256": digest(metadata),
                        "source_code": code_version(),
                        "interaction_step": trainer.interaction_step,
                    },
                )
                provenance = previous
            writer = StreamWriter(
                root, provenance, chunk_size, resume=resume_stream, autoflush=False
            )
        expected = (
            writer.manifest["provenance"]["start_step"] + writer.manifest["count"]
        )
        if trainer.interaction_step != expected:
            raise ValueError("source state and committed stream cursor differ")
        original_buffer = trainer.buffer
        adapter = RecordingBuffer(trainer, writer)
        trainer.buffer = adapter
        try:

            def hook(t):
                adapter.after_step(t)
                if after_step:
                    after_step(t)

            original_save = trainer.save

            def save_with_stream(root, *args, **kwargs):
                # Publish arrivals only with a routine (restorable) save; validation
                # snapshots elsewhere must not advance the committed stream.
                if Path(root).resolve() == (trainer.run_dir / "state").resolve():
                    writer.flush()
                return original_save(root, *args, **kwargs)

            with patch.object(trainer, "save", save_with_stream), benchmark_source(
                trainer, benchmark
            ):
                original(trainer, last_step, hook, before_first_update)
            # No flush here: the source's next routine save (or seal() at scope exit)
            # publishes the tail, so a crash cannot leave the stream ahead of a save.
        finally:
            trainer.buffer = original_buffer

    with patch.object(Exp12Trainer, "train", recorded), patch.object(
        Exp12Trainer, "save", save_retaining
    ):
        yield
    if writer is None:
        raise ValueError("source did not execute a training continuation")
    writer.seal()
    if benchmark:
        write_json(Path(root) / "benchmark.json", benchmark.report(root))
