from typing import Dict, Optional

from omegaconf import OmegaConf

import wandb

# Canonical architecture-id mapping used for both WandB run naming and the
# onset ledger's `architecture` field. Anything not explicitly named here
# falls back to a dynamic "D{critic_num_blocks}W{critic_hidden_dim}" id, so
# arbitrary architectures are supported without touching this map.
_CRITIC_SIZE_MAP = {
    (128, 1): "Small",
    (1024, 3): "XXL",
}


def get_architecture_id(cfg) -> str:
    """Derives a stable architecture identifier from the agent config.

    Single source of truth for "architecture" naming, shared by WandB
    run/group naming (see get_run_info) and the onset ledger.
    """
    return _CRITIC_SIZE_MAP.get(
        (cfg.agent.critic_hidden_dim, cfg.agent.critic_num_blocks),
        f"D{cfg.agent.critic_num_blocks}W{cfg.agent.critic_hidden_dim}",
    )


def get_run_info(args):
    critic_size_label = get_architecture_id(args)
    utd = args.updates_per_interaction_step
    env = args.env.env_name
    group = f"{env}_{critic_size_label}_UTD{utd}"
    job_type = critic_size_label
    name = (
        f"{env}_{critic_size_label}_UTD{utd}"
        f"_CD{args.agent.critic_num_blocks}_CW{args.agent.critic_hidden_dim}"
        f"_AD{args.agent.actor_num_blocks}_AW{args.agent.actor_hidden_dim}"
        f"_seed{args.seed}"
    )
    return {
        "group": group,
        "job_type": job_type,
        "name": name,
    }
class WandbTrainerLogger(object):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        run_info = get_run_info(cfg)

        wandb.init(
            project=cfg.project_name,
            group=run_info["group"],
            config=dict_cfg,
            job_type=run_info["job_type"],
            name=run_info["name"], 
        )

        self.reset()

    @property
    def run_name(self) -> Optional[str]:
        """Authoritative exact WandB run name (None if wandb is disabled)."""
        return wandb.run.name if wandb.run is not None else None

    @property
    def run_id(self) -> Optional[str]:
        """Authoritative WandB run id (None if wandb is disabled)."""
        return wandb.run.id if wandb.run is not None else None

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        wandb.log(log_data, step=step)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}


class AverageMeterDict(object):
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
