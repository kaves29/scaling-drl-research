"""Process-local opt-in recording of existing Exp2 arm entry points, unchanged loop."""

import contextlib
from pathlib import Path
from unittest.mock import patch

from experiments.exp12.state import latest_state_dir
from experiments.exp12.trainer import Exp12Trainer
from experiments.exp3.artifacts import digest, state_fingerprint, metadata_fingerprint
from experiments.exp3.retention import retain_state, validate_retention, retention_end
from experiments.exp3.streams import (
    RecordingBuffer,
    StreamReader,
    StreamWriter,
    benchmark_source,
    write_json,
    recover_checkpoint_publication,
)
from utils.run_metadata import code_version, load_run_metadata
from utils.atomic_io import atomic_write_text


@contextlib.contextmanager
def _record_exp2_scope(root, chunk_size, benchmark=None, resume=False, retain=None):
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
    recorded_trainer = None

    def save_retaining(trainer, root, keep_previous=False):
        routine = Path(root).resolve() == (trainer.run_dir / "state").resolve()
        publish = writer is not None and trainer is recorded_trainer and routine
        if publish:
            writer.prepare_checkpoint(trainer.interaction_step)
        path = original_class_save(trainer, root, keep_previous=keep_previous)
        if publish:
            from experiments.exp12.state import load_meta

            writer.publish_checkpoint(load_meta(path)["interaction_step"])
        plan = trainer.extra_state.get("fork")
        if (
            retain is not None
            and plan is not None
            and Path(root).resolve() == (trainer.run_dir / "state").resolve()
            and int(plan["fork_step"])
            <= trainer.interaction_step
            <= retention_end(retain, plan)
        ):
            retain_state(
                path, retain["root"], trainer.interaction_step, retain["replay"]
            )
        return path

    def recorded(trainer, last_step, after_step=None, before_first_update=None):
        nonlocal writer, recorded_trainer
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
                "fork_metadata_fingerprint": metadata_fingerprint(fork_metadata),
                "config_hash": load_run_metadata(fork_metadata)["config_hash"],
                "source_metadata_sha256": digest(metadata),
                "source_metadata_fingerprint": metadata_fingerprint(metadata),
                "source_code": code_version(),
                "source_identity": info["identity"],
                "fork_state": str(fork_state),
                "fork_state_sha256": state_fingerprint(fork_state),
                "arm": trainer.extra_state.get("arm", "control"),
                # The passive I learner must reproduce the source arm's actual injection.
                "injection": trainer.extra_state.get("injection"),
                "captures": ["agent_key"],
                "retention": retain,
            }
            if resume_stream:
                previous = StreamReader(root, require_complete=False).manifest[
                    "provenance"
                ]
                # run_metadata records a new launch on restart. Retain original
                # provenance but verify immutable fork/source/arm identity.
                for k in (
                    "start_step",
                    "fork_metadata_fingerprint",
                    "config_hash",
                    "fork_state_sha256",
                    "source_metadata_fingerprint",
                    "source_identity",
                    "source_code",
                    "arm",
                    "injection",
                    "captures",
                    "retention",
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
            recover_checkpoint_publication(root, trainer.interaction_step)
            writer = StreamWriter(
                root, provenance, chunk_size, resume=resume_stream, autoflush=False
            )
            recorded_trainer = trainer
            if not resume_stream:
                atomic_write_text(
                    Path(root) / "fork_metadata.json", fork_metadata.read_text()
                )
                atomic_write_text(
                    Path(root) / "source_metadata.json", metadata.read_text()
                )
        if trainer is not recorded_trainer:
            raise ValueError("one stream cannot combine multiple source trainers")
        # Recover a death after source LATEST publication but before snapshot retention.
        if resume and retain is not None:
            state = latest_state_dir(trainer.run_dir / "state")
            if state is not None and int(
                plan["fork_step"]
            ) <= trainer.interaction_step <= retention_end(retain, plan):
                retain_state(
                    state, retain["root"], trainer.interaction_step, retain["replay"]
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

            with benchmark_source(trainer, benchmark):
                original(trainer, last_step, hook, before_first_update)
            # Only the source's existing routine save publishes pending arrivals.
        finally:
            trainer.buffer = original_buffer

    with patch.object(Exp12Trainer, "train", recorded), patch.object(
        Exp12Trainer, "save", save_retaining
    ):
        yield
    if writer is None:
        raise ValueError("source did not execute a training continuation")
    if writer.pending:
        raise ValueError(
            "cannot seal arrivals without a matching routine source checkpoint"
        )
    writer.seal()
    if benchmark:
        write_json(Path(root) / "benchmark.json", benchmark.report(root))


@contextlib.contextmanager
def record_exp2_scope(root, chunk_size, benchmark=None, resume=False, retain=None):
    """One source writer per stream, including resumed processes."""
    import fcntl

    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink a lock inode: another process may already be waiting on it.
    with (root.parent / (root.name + ".capture.lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "source stream already has an active recording writer"
            ) from exc
        try:
            with _record_exp2_scope(root, chunk_size, benchmark, resume, retain):
                yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def finalize_capture(root, run):
    """Seal a completed source's capture after a death between DONE and sealing.

    This verifies capture/source binding; it is not scientific run certification.
    It never loads a trainer, saves a checkpoint, or advances a learner.
    """
    import fcntl
    import json
    from experiments.exp12.state import load_meta

    root, run = Path(root).resolve(), Path(run).resolve()
    with (root.parent / (root.name + ".capture.lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("capture finalization refuses an active writer") from exc
        reader = StreamReader(root, require_complete=False)
        p = reader.manifest["provenance"]
        metadata = run / "run_metadata.json"
        if (
            metadata_fingerprint(metadata) != p.get("source_metadata_fingerprint")
            or load_run_metadata(metadata)["identity"] != p["source_identity"]
        ):
            raise ValueError("finalization source provenance differs")
        state = latest_state_dir(run / "state")
        if state is None:
            raise ValueError("completed source checkpoint unavailable")
        meta = load_meta(state)
        done = json.loads((run / "DONE").read_text())
        if done["interaction_step"] != meta["interaction_step"]:
            raise ValueError("DONE/source checkpoint endpoint differs")
        if (
            meta["extra_state"].get("arm", "control") != p["arm"]
            or meta["extra_state"].get("injection") != p["injection"]
        ):
            raise ValueError("completed source arm/injection differs")
        recover_checkpoint_publication(root, meta["interaction_step"])
        reader = StreamReader(root, require_complete=False)
        if p["start_step"] + reader.manifest["count"] != meta["interaction_step"]:
            raise ValueError("completed source has an unavailable captured tail")
        if sum(1 for _ in reader) != reader.manifest["count"]:
            raise ValueError("completed capture length differs")
        retain = p.get("retention")
        if retain is not None and p["start_step"] <= meta[
            "interaction_step"
        ] <= retention_end(retain, meta["extra_state"]["fork"]):
            retain_state(
                state, retain["root"], meta["interaction_step"], retain["replay"]
            )
        if not reader.manifest["complete"]:
            manifest = dict(reader.manifest, complete=True)
            write_json(root / "manifest.json", manifest)
        return {
            "capture_verdict": "COMPLETE",
            "scientific_qualification": False,
            "interaction_step": meta["interaction_step"],
            "count": reader.manifest["count"],
        }
