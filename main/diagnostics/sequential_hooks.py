"""Explicit instance-local opt-in hook; no registration or production edits."""

from pathlib import Path

from .sequential_probe import Settings, create_bank, create_reference, evaluate, verify


class Observer:
    def __init__(
        self,
        *,
        enabled=False,
        bank=None,
        reference=None,
        output=None,
        teacher=None,
        teacher_architecture=None,
        architecture=None,
        environment=None,
        seed=None,
        source=None,
        bank_producer=False,
    ):
        self.enabled = enabled
        self.options = dict(
            bank=bank,
            reference=reference,
            output=output,
            teacher=teacher,
            teacher_architecture=teacher_architecture,
            architecture=architecture,
            environment=environment,
            seed=seed,
            source=source,
            bank_producer=bank_producer,
        )
        self.fresh = None

    def initialize(self, trainer, saved_fresh):
        if not self.enabled:
            return None
        o = self.options
        if trainer.update_step != 0:
            raise ValueError("alternative must initialize before SAC updates")
        cfg = trainer.cfg
        if cfg.agent.critic_use_cdq or cfg.agent.critic_sparsity:
            raise ValueError(
                "single dense critic only; twin/sparse adapters unqualified"
            )
        self.settings = Settings(
            steps=int(cfg.probe.steps),
            batch_size=int(cfg.probe.batch_size),
            chunk=int(cfg.probe.eval_chunk),
            learning_rate=float(cfg.agent.critic_learning_rate),
            weight_decay=float(cfg.agent.critic_weight_decay),
        )
        if int(cfg.seed) != o["seed"] or cfg.env_name != o["environment"]:
            raise ValueError("matched-block identity mismatch")
        if not Path(o["bank"]).exists():
            if not o["bank_producer"]:
                raise FileNotFoundError(
                    "designated block producer must persist bank first"
                )
            agent = trainer.agent
            rms = None
            if hasattr(agent, "_normalize"):
                rms = dict(
                    mean=agent.obs_rms.mean,
                    var=agent.obs_rms.var,
                    count=agent.obs_rms.count,
                    epsilon=agent.epsilon,
                )
            create_bank(
                o["bank"],
                observations=trainer.buffer._observations,
                actions=trainer.buffer._actions,
                num_filled=trainer.buffer._num_in_buffer,
                warmup=int(cfg.buffer.min_length),
                rms=rms,
                environment=o["environment"],
                seed=o["seed"],
                teacher=o["teacher"],
                teacher_architecture=o["teacher_architecture"],
                pool_size=int(cfg.probe.pool_size),
                scale=float(cfg.probe.target_scale),
                teacher_chunk=int(cfg.probe.eval_chunk),
                source=o["source"],
            )
        bank = verify(o["bank"])
        if (
            bank["environment"] != o["environment"]
            or bank["seed"] != o["seed"]
            or bank["teacher_architecture"] != o["teacher_architecture"]
            or bank["teacher_definition"] != repr(o["teacher"])
            or bank["scale"] != float(cfg.probe.target_scale)
            or bank["source"] != o["source"]
        ):
            raise ValueError("shared target-bank contract mismatch")
        self.network = trainer._sac_agent.critic.network_def
        self.fresh = saved_fresh
        if not Path(o["reference"]).exists():
            create_reference(
                o["reference"],
                o["bank"],
                self.network,
                saved_fresh,
                architecture=o["architecture"],
                sac_updates=0,
            )
        return self.observe(trainer)

    def restore(self, trainer, saved_fresh):
        from .sequential_probe import tree_hash

        if not self.enabled:
            return None
        o = self.options
        cfg = trainer.cfg
        if int(cfg.seed) != o["seed"] or cfg.env_name != o["environment"]:
            raise ValueError("resume training config identity mismatch")
        if cfg.agent.critic_use_cdq or cfg.agent.critic_sparsity:
            raise ValueError("resume requires a single dense critic")
        bank = verify(o["bank"])
        ref = verify(o["reference"])
        if (
            bank["environment"],
            bank["seed"],
            bank["teacher_architecture"],
            bank["source"],
        ) != (o["environment"], o["seed"], o["teacher_architecture"], o["source"]):
            raise ValueError("resume bank identity mismatch")
        if (
            ref["fresh_hash"] != tree_hash(saved_fresh)
            or ref["architecture"] != o["architecture"]
        ):
            raise ValueError("resume reference identity mismatch")
        self.network = trainer._sac_agent.critic.network_def
        if repr(self.network) != ref["network"]:
            raise ValueError("resume learner architecture mismatch")
        self.settings = Settings(
            steps=int(cfg.probe.steps),
            batch_size=int(cfg.probe.batch_size),
            chunk=int(cfg.probe.eval_chunk),
            learning_rate=float(cfg.agent.critic_learning_rate),
            weight_decay=float(cfg.agent.critic_weight_decay),
        )
        self.fresh = saved_fresh
        # evaluate() additionally binds bank hash, runtime and checkpoint identity.
        return None

    def observe(self, trainer):
        if not self.enabled:
            return None
        if self.fresh is None:
            raise ValueError("saved fresh initialization not captured")
        return evaluate(
            enabled=True,
            bank=self.options["bank"],
            reference=self.options["reference"],
            output=Path(self.options["output"])
            / f"checkpoint_{trainer.interaction_step:09d}",
            network=self.network,
            current=trainer._sac_agent.critic.params,
            fresh=self.fresh,
            checkpoint=int(trainer.interaction_step),
            settings=self.settings,
        )


class Hooks:
    """Use only in a dedicated alternative run, after trainer/probes construction.

    Disabled context does not inspect/patch either object. Enabled attaches to
    this trainer/probe instance only, preserving original methods and outputs.
    Observation happens after fresh capture and after existing evaluations.
    """

    def __init__(self, trainer, probes, observer):
        self.trainer, self.probes, self.observer = trainer, probes, observer
        self.original = []

    def __enter__(self):
        if not self.observer.enabled:
            return self
        if self.original:
            raise ValueError("hooks already installed")
        original_fresh = self.probes.capture_fresh
        original_evaluate = self.trainer._evaluate

        def fresh(trainer):
            result = original_fresh(trainer)
            if self.observer.fresh is None:
                self.observer.initialize(trainer, self.probes.fresh)
            return result

        def evaluation(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            self.observer.observe(self.trainer)
            return result

        for obj, name, replacement in (
            (self.probes, "capture_fresh", fresh),
            (self.trainer, "_evaluate", evaluation),
        ):
            self.original.append(
                (obj, name, name in obj.__dict__, obj.__dict__.get(name))
            )
            setattr(obj, name, replacement)
        return self

    def __exit__(self, *args):
        for obj, name, owned, value in reversed(self.original):
            if owned:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self.original.clear()
