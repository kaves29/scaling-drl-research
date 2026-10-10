"""All-passive matched replay learners using the existing ordinary SAC update."""

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from experiments.exp12.state import load_buffer, save_buffer
from experiments.exp12.diagnostics import ActorDiagnostics
from experiments.exp3.artifacts import core, tree_equal
from experiments.exp3.streams import atomic_pickle, write_json


def replay_batch(buffer, indices):
    names = (
        "observations",
        "actions",
        "rewards",
        "terminateds",
        "truncateds",
        "next_observations",
    )
    fields = (
        "observation",
        "action",
        "reward",
        "terminated",
        "truncated",
        "next_observation",
    )
    return {
        field: np.array(getattr(buffer, "_" + name)[indices], copy=True)
        for field, name in zip(fields, names)
    }


def assert_replay_equal(a, b, recent_only=False):
    from experiments.exp12.state import _deep_equal

    if (a._num_in_buffer, a._current_idx) != (b._num_in_buffer, b._current_idx):
        raise ValueError("passive replay counters differ")
    if not _deep_equal(list(a._n_step_transitions), list(b._n_step_transitions)):
        raise ValueError("passive n-step queues differ")
    for name in (
        "observations",
        "actions",
        "rewards",
        "terminateds",
        "truncateds",
        "next_observations",
    ):
        indices = (
            ((a._current_idx - np.arange(1, a._add_batch_size + 1)) % a._max_length)
            if recent_only and len(a)
            else slice(0, a._num_in_buffer)
        )
        if not tree_equal(
            getattr(a, "_" + name)[indices], getattr(b, "_" + name)[indices]
        ):
            raise ValueError(f"passive replay contents differ: {name}")


class PassivePair:
    """U/I copies of one common fork. No environment stepping or policy sampling API."""

    def __init__(
        self,
        untreated,
        injected,
        rng_state,
        normalization,
        stream_fingerprint,
        benchmark=None,
    ):
        if normalization not in ("source_statistics", "frozen_fork"):
            raise ValueError("explicit passive normalization policy required")
        self.learners = {"u": untreated, "i": injected}
        self.cfg = untreated.cfg
        self.normalization = normalization
        self.stream_fingerprint = stream_fingerprint
        self.benchmark = benchmark
        self.cursor = 0
        self.update_step = untreated.meta["update_step"]
        self.rng = np.random.RandomState()
        self.rng.set_state(rng_state)
        assert_replay_equal(untreated.buffer, injected.buffer)
        ua, ia = core(untreated.agent), core(injected.agent)
        if (
            not tree_equal(ua._actor, ia._actor)
            or not tree_equal(ua._temperature, ia._temperature)
            or not tree_equal(ua._rng, ia._rng)
        ):
            raise ValueError(
                "passive actor/optimizer/temperature/RNG initialization differs"
            )
        if not tree_equal(vars(untreated.agent.obs_rms), vars(injected.agent.obs_rms)):
            raise ValueError("passive fork normalization differs")
        if untreated.meta["interaction_step"] != injected.meta["interaction_step"]:
            raise ValueError("passive fork steps differ")
        self.start_step = untreated.meta["interaction_step"]
        self.last_indices = None
        self.diagnostics = {
            k: ActorDiagnostics(v.cfg) for k, v in self.learners.items()
        }
        for k, v in self.learners.items():
            self.diagnostics[k].load_state(v.meta["actor_diagnostics"])
        self.last_metrics = {}

    def deliver(self, record, updates_per_arrival):
        if record["step"] != self.start_step + self.cursor + 1:
            raise ValueError("passive arrival order differs")
        if type(updates_per_arrival) is not int or updates_per_arrival < 0:
            raise ValueError("explicit passive updates per arrival required")
        # Source timing is preserved; a mismatch is refused rather than silently rescheduled.
        if record["updates"] != updates_per_arrival:
            raise ValueError("source update timing differs from passive protocol")
        if self.normalization == "source_statistics":
            for learner in self.learners.values():
                for k, value in record["normalization"].items():
                    old = np.asarray(getattr(learner.agent.obs_rms, k))
                    new = np.asarray(value)
                    if old.shape != new.shape or old.dtype != new.dtype:
                        raise ValueError("source normalization shape/dtype differs")
        for learner in self.learners.values():
            if self.benchmark:
                self.benchmark.measure(
                    "transition_delivery",
                    learner.buffer.add,
                    copy.deepcopy(record["transition"]),
                )
            else:
                learner.buffer.add(copy.deepcopy(record["transition"]))
            if self.normalization == "source_statistics":
                for k, value in record["normalization"].items():
                    original = getattr(learner.agent.obs_rms, k)
                    restored = (
                        value.item()
                        if isinstance(original, float) and np.asarray(value).ndim == 0
                        else copy.deepcopy(value)
                    )
                    setattr(learner.agent.obs_rms, k, restored)
        u, i = self.learners.values()
        assert_replay_equal(u.buffer, i.buffer, recent_only=True)
        if updates_per_arrival and not u.buffer.can_sample():
            raise ValueError("passive replay cannot supply required update")
        if updates_per_arrival:
            indices = self.rng.randint(
                0,
                len(u.buffer),
                size=(updates_per_arrival, int(self.cfg.buffer.sample_batch_size)),
            )
            self.last_indices = indices
            batches = replay_batch(u.buffer, indices)
            if not tree_equal(batches, replay_batch(i.buffer, indices)):
                raise ValueError("matched replay sampling produced different batches")
            for label, learner in self.learners.items():
                # Wrapper normalizes fresh copies; each learner owns actor/critic/alpha/optimizers.
                view = SimpleNamespace(
                    agent=learner.agent,
                    buffer=learner.buffer,
                    interaction_step=record["step"],
                )
                args = (
                    self.update_step,
                    copy.deepcopy(batches),
                    int(self.cfg.actor_grad_cosine_every),
                )
                kwargs = self.diagnostics[label].update_kwargs(view)
                info = (
                    self.benchmark.measure(
                        "sac_update", learner.agent.update_many, *args, **kwargs
                    )
                    if self.benchmark
                    else learner.agent.update_many(*args, **kwargs)
                )
                self.diagnostics[label].collect(
                    {"train/actor_gnorm": np.asarray(info["train/actor_gnorm"])}
                )
            self.update_step += updates_per_arrival
        self.cursor += 1
        self.last_metrics = {}
        if record["step"] % int(self.cfg.logging_per_interaction_step) == 0:
            indices = self.rng.randint(
                0, len(u.buffer), size=int(self.cfg.buffer.sample_batch_size)
            )
            for label, learner in self.learners.items():
                self.last_metrics[label] = {
                    **learner.agent.get_metrics(
                        self.update_step, replay_batch(learner.buffer, indices)
                    ),
                    **self.diagnostics[label].window_metrics(),
                }

    def save(self, root):
        import uuid

        root = Path(root).resolve()
        path = root / f"cursor_{self.cursor:09d}_{uuid.uuid4().hex}"
        path.mkdir(parents=True)
        for label, learner in self.learners.items():
            d = path / label
            d.mkdir()
            learner.agent.save_checkpoint(str(d))
            save_buffer(learner.buffer, d)
            a = core(learner.agent)
            atomic_pickle(
                d / "windows.pkl",
                {
                    k: copy.deepcopy(getattr(a, k))
                    for k in (
                        "actor_loss_buffer",
                        "actor_entropy_buffer",
                        "churn_buffer",
                    )
                },
            )
        atomic_pickle(
            path / "passive.pkl",
            {
                "cursor": self.cursor,
                "update_step": self.update_step,
                "start_step": self.start_step,
                "rng": self.rng.get_state(),
                "normalization": self.normalization,
                "stream_fingerprint": self.stream_fingerprint,
                "diagnostics": {k: v.state() for k, v in self.diagnostics.items()},
                "config_hash": self.learners["u"].info["config_hash"],
                "interventions": {k: v.intervention for k, v in self.learners.items()},
            },
        )
        write_json(root / "LATEST.json", {"directory": path.name})
        return path

    def restore(self, root):
        root = Path(root).resolve()
        name = json.loads((root / "LATEST.json").read_text())["directory"]
        path = (root / name).resolve()
        if Path(name).name != name or path.parent != root:
            raise ValueError("passive checkpoint path escapes root")
        with open(path / "passive.pkl", "rb") as f:
            meta = pickle.load(f)
        for key, expected in (
            ("normalization", self.normalization),
            ("stream_fingerprint", self.stream_fingerprint),
            ("start_step", self.start_step),
            ("config_hash", self.learners["u"].info["config_hash"]),
            ("interventions", {k: v.intervention for k, v in self.learners.items()}),
        ):
            if meta[key] != expected:
                raise ValueError(f"passive resume mismatch: {key}")
        for label, learner in self.learners.items():
            d = path / label
            learner.agent.load_checkpoint(str(d))
            load_buffer(learner.buffer, d)
            with open(d / "windows.pkl", "rb") as f:
                for k, v in pickle.load(f).items():
                    setattr(core(learner.agent), k, v)
        self.cursor, self.update_step = meta["cursor"], meta["update_step"]
        self.rng.set_state(meta["rng"])
        for k, v in meta["diagnostics"].items():
            self.diagnostics[k].load_state(v)
        assert_replay_equal(*(learner.buffer for learner in self.learners.values()))

    def evaluate(self, callback):
        """Outcome interface must use evaluation-only environments; restore learner/global RNG."""
        import random

        np_state, py_state, sample_state = (
            np.random.get_state(),
            random.getstate(),
            self.rng.get_state(),
        )
        keys = {k: core(v.agent)._rng for k, v in self.learners.items()}
        try:
            return {k: callback(v.agent) for k, v in self.learners.items()}
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            self.rng.set_state(sample_state)
            for k, learner in self.learners.items():
                core(learner.agent)._rng = keys[k]
