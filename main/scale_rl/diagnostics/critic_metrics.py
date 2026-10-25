"""Versioned critic measurements, separate from historical training metrics.

Reference formulas and activation mappings are pinned in
main/docs/reference_critic_diagnostics.md. No parameter updates or RNG draws.
"""

from pathlib import Path
import hashlib
import json
import os
import uuid

import flax.linen as nn
import jax
import numpy as np

PREFIX = "critic_diag/v1"
COLLECTION = "critic_diagnostics_v1"
ENV_ROOT = "ECHOCRITIC_CRITIC_DIAGNOSTICS_ROOT"


def _features(values, dtype):
    x = np.asarray(values, dtype=dtype)
    if x.ndim < 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("nonempty finite examples-by-units tensor required")
    return x


def activity(values, thresholds=(0.0, 0.1, 0.2)):
    """ReDo relative absolute activity; all leading axes are examples/spatial.

    Float32 CPU reductions; reference epsilon and inclusive threshold retained.
    Exact-zero mask is elementwise equality, not an isclose approximation.
    """
    x = _features(values, np.float32)
    thresholds = tuple(float(t) for t in thresholds)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not np.isfinite(t) or t < 0 for t in thresholds)
    ):
        raise ValueError("unique finite nonnegative thresholds required")
    axes = tuple(range(x.ndim - 1))
    absolute_mean = np.mean(np.abs(x), axis=axes, dtype=np.float32)
    denominator = absolute_mean.mean(dtype=np.float32) + np.float32(1e-9)
    relative = absolute_mean / denominator
    if not np.isfinite(relative).all() or not np.isfinite(denominator):
        raise ValueError("nonfinite float32 activity reduction")
    masks = {
        "relative_tau_" + str(t).removesuffix(".0"): relative <= np.float32(t)
        for t in thresholds
    }
    masks["exact_zero"] = np.all(x == 0, axis=axes)
    return {
        "units": x.shape[-1],
        "relative_activity": relative,
        "mean_absolute_activity": absolute_mean,
        "masks": masks,
    }


def aggregate_activity(layers):
    """Two explicit aggregates; never mix affine and post-ReLU populations."""
    if not layers:
        return {}
    names = set(next(iter(layers.values()))["masks"])
    if any(set(layer["masks"]) != names for layer in layers.values()):
        raise ValueError("inconsistent activity thresholds")
    total = sum(layer["units"] for layer in layers.values())
    result = {"units": total, "layers": len(layers)}
    for name in sorted(names):
        counts = [
            int(np.count_nonzero(layer["masks"][name])) for layer in layers.values()
        ]
        result[name + "/unit_weighted_percent"] = 100 * sum(counts) / total
        result[name + "/layer_mean_percent"] = float(
            np.mean(
                [
                    100 * count / layer["units"]
                    for count, layer in zip(counts, layers.values())
                ]
            )
        )
    return result


def rank_metrics(features, *, center=False):
    """CPU float64 SVD, uncentered by default; no width normalization.

    Stable rank is sum(s^2)/max(s)^2. LoP entropy rank uses p=s/sum(s).
    Threshold counts are kept distinct. Zero spectrum: stable=0 by convention;
    LoP entropy/count fallbacks and historical singular-sum count return 1.
    """
    x = _features(features, np.float64)
    x = x.reshape(-1, x.shape[-1])
    # Scale before centering to avoid overflow in the mean as well as SVD.
    scale = float(np.max(np.abs(x)))
    if scale:
        x = x / scale
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False) if scale else np.zeros(min(x.shape))
    total = singular.sum()
    result = {
        "zero_spectrum": bool(total == 0),
        "examples": x.shape[0],
        "features": x.shape[1],
        "batch_ceiling": min(x.shape[0] - int(center), x.shape[1]),
        "stable_rank": 0.0,
        "entropy_effective_rank_lop": 1.0,
        "squared_energy99_count_lop": 1,
        "singular_sum99_count_analysis64": 1,
    }
    if total:
        p = singular[singular > 0] / total
        result["stable_rank"] = float(np.sum((singular / singular[0]) ** 2))
        result["entropy_effective_rank_lop"] = float(np.exp(-np.sum(p * np.log(p))))
        for name, weights in (
            ("singular_sum99_count_analysis64", singular),
            ("squared_energy99_count_lop", singular**2),
        ):
            result[name] = min(
                len(weights),
                int(
                    np.searchsorted(
                        np.cumsum(weights), 0.99 * weights.sum(), side="left"
                    )
                )
                + 1,
            )
    return result


def mask_transition(previous, current):
    """New diagnostic, not a paper-defined plasticity or chronicity estimator."""
    previous, current = np.asarray(previous), np.asarray(current)
    if (
        previous.dtype != bool
        or current.dtype != bool
        or previous.ndim != 1
        or previous.shape != current.shape
        or not previous.size
    ):
        raise ValueError("matching nonempty per-unit boolean masks required")
    both = int(np.count_nonzero(previous & current))
    old = int(previous.sum())
    union = int(np.count_nonzero(previous | current))
    return {
        "persistent_count": both,
        "reactivated_count": int(np.count_nonzero(previous & ~current)),
        "newly_dormant_count": int(np.count_nonzero(~previous & current)),
        "previous_dormant_count": old,
        "persistent_fraction_of_previous": both / old if old else None,
        "reactivated_fraction_of_previous": (old - both) / old if old else None,
        "jaccard": both / union if union else None,
    }


def capture_critic(critic, batch):
    """Observe actual module outputs; unsupported architectures fail explicitly."""
    from scale_rl.agents.sac.sac_network import SACCritic
    from scale_rl.networks.metrics import flatten_dict

    if type(critic.network_def) is not SACCritic:
        raise ValueError(
            "v1 capture supports the single SACCritic only; no mixed twin axis"
        )
    observations = _features(batch["observation"], np.float32)
    actions = _features(batch["action"], np.float32)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("matching observation/action matrices required")

    # Include only encoder Dense affines plus encoder output. Custom sow sites
    # observe actual ReLUs without adding modules, parameters or random keys.
    def select(module, method):
        return method == "__call__" and (
            isinstance(module, nn.Dense) or type(module).__name__ == "SACEncoder"
        )

    output, state = critic.network_def.apply(
        {"params": critic.params},
        observations,
        actions,
        capture_intermediates=select,
        mutable=["intermediates", COLLECTION],
    )
    flat = flatten_dict(jax.device_get(state), sep="/")
    populations = {"dense_affine": {}, "post_relu": {}}
    features = None
    for path, values in flat.items():
        # Every module/site is invoked once in the supported critic.
        if not isinstance(values, tuple) or len(values) != 1:
            raise ValueError("unexpected repeated activation site")
        x = np.asarray(values[0])
        if path.startswith("intermediates/encoder/") and "/Dense_" in path:
            populations["dense_affine"][path.removeprefix("intermediates/")] = x
        elif path == "intermediates/encoder/__call__":
            features = x
        elif path.startswith(COLLECTION + "/encoder/"):
            populations["post_relu"][path.removeprefix(COLLECTION + "/")] = x
    if (
        features is None
        or not populations["dense_affine"]
        or not populations["post_relu"]
    ):
        raise ValueError("critic capture sites missing; refuse fabricated zero metrics")
    return output, populations, features


def input_hash(batch):
    digest = hashlib.sha256()
    for key in ("observation", "action"):
        x = np.ascontiguousarray(batch[key])
        digest.update(key.encode())
        digest.update(str((x.shape, str(x.dtype))).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


class CriticMeasurements:
    """Optional sidecars; caller supplies existing preprocessed replay batch.

    Frozen inputs are copied, preprocessed once, and never renormalized. A new
    process/session starts a new persistence series, independently of training
    checkpoints. Saved masks permit explicit cross-session offline comparison.
    """

    def __init__(
        self,
        output,
        *,
        frozen_batch=None,
        freeze_first=False,
        compute_rank=False,
        thresholds=(0.0, 0.1, 0.2),
        every=1,
    ):
        self.output = Path(output)
        if not self.output.is_absolute():
            raise ValueError("diagnostic output must be absolute")
        repository = Path(__file__).resolve().parents[3]
        if (
            self.output.resolve() == repository
            or repository in self.output.resolve().parents
        ):
            raise ValueError("diagnostic output must be outside repository")
        if not isinstance(every, int) or isinstance(every, bool) or every <= 0:
            raise ValueError("every must be a positive logging-event count")
        if frozen_batch is not None and freeze_first:
            raise ValueError("select frozen batch or freeze-first, not both")
        activity(np.ones((1, 1)), thresholds)  # validate before output mutation
        self.thresholds = tuple(thresholds)
        self.every, self.compute_rank = every, bool(compute_rank)
        self.freeze_first, self.frozen = freeze_first, None
        if frozen_batch is not None:
            self.frozen = self._copy_batch(frozen_batch)
        self.previous, self.events, self.last_step = {}, 0, None
        self.output.mkdir(parents=True, exist_ok=False)
        if self.frozen is not None:
            np.savez(self.output / "frozen_inputs.npz", **self.frozen)

    @staticmethod
    def _copy_batch(batch):
        copy = {
            key: _features(batch[key], np.float32).copy()
            for key in ("observation", "action")
        }
        if any(x.ndim != 2 for x in copy.values()) or len(copy["observation"]) != len(
            copy["action"]
        ):
            raise ValueError("frozen observations/actions must be matching matrices")
        return {
            key: np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)
            for key, x in copy.items()
        }

    def record(
        self, critic, batch, *, interaction_step, optimizer_step, legacy_metrics=None
    ):
        self.events += 1
        if (self.events - 1) % self.every:
            return None
        if self.last_step is not None and interaction_step <= self.last_step:
            raise ValueError("persistence series requires increasing interaction steps")
        batch = self._copy_batch(batch)
        if self.freeze_first and self.frozen is None:
            self.frozen = self._copy_batch(batch)
            np.savez(self.output / "frozen_inputs.npz", **self.frozen)
        streams = {"current_replay": batch}
        if self.frozen is not None:
            streams["frozen_preprocessed"] = self.frozen
        scalars, arrays, mask_metadata, inputs = {}, {}, {}, {}
        next_previous = {}
        for stream, inputs_batch in streams.items():
            _, populations, features = capture_critic(critic, inputs_batch)
            inputs[stream] = {
                "sha256": input_hash(inputs_batch),
                "examples": len(features),
                "preprocessing": "already preprocessed; no statistics update",
            }
            for population, values in populations.items():
                layers = {
                    path: activity(x, self.thresholds)
                    for path, x in sorted(values.items())
                }
                base = f"{PREFIX}/{stream}/{population}"
                for name, value in aggregate_activity(layers).items():
                    scalars[f"{base}/aggregate/{name}"] = value
                for layer, data in layers.items():
                    for name, mask in data["masks"].items():
                        key = f"{base}/{layer}/{name}"
                        scalars[key + "/percent"] = 100 * float(np.mean(mask))
                        array_name = f"mask_{len(arrays):04d}"
                        arrays[array_name] = mask
                        mask_metadata[array_name] = {"metric": key, "units": len(mask)}
                        next_previous[key] = mask.copy()
                        if key in self.previous:
                            for stat, value in mask_transition(
                                self.previous[key], mask
                            ).items():
                                scalars[key + "/transition_v1_new/" + stat] = value
            if self.compute_rank:
                for name, value in rank_metrics(features).items():
                    scalars[
                        f"{PREFIX}/{stream}/encoder_output_uncentered/analysis64/{name}"
                    ] = value
        if self.previous and set(next_previous) != set(self.previous):
            raise ValueError("activation population changed within persistence session")
        legacy_metrics = legacy_metrics or {}
        if "train/critic_srank" in legacy_metrics:
            scalars[
                f"{PREFIX}/current_replay/encoder_output_uncentered/singular_sum99_count_historical"
            ] = legacy_metrics["train/critic_srank"]
        if "train/critic_gnorm" in legacy_metrics:
            # Window mean of original pre-AdamW TD-gradient norms, not a new gradient.
            scalars[
                f"{PREFIX}/current_replay/td_gradient_pre_adamw_global_l2_window_mean_historical"
            ] = legacy_metrics["train/critic_gnorm"]
        name = f"interaction_{interaction_step:09d}"
        path = self.output / (name + ".npz")
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        record = {
            "schema": PREFIX,
            "interaction_step": int(interaction_step),
            "optimizer_step": int(optimizer_step),
            "inputs": inputs,
            "rank_enabled": self.compute_rank,
            "thresholds": list(self.thresholds),
            "network": {
                "class": type(critic.network_def).__name__,
                "block_type": critic.network_def.block_type,
                "blocks": critic.network_def.num_blocks,
                "width": critic.network_def.hidden_dim,
                "forward_dtype": str(np.dtype(critic.network_def.dtype)),
            },
            "activity_precision": "CPU float32; epsilon=1e-9; inclusive thresholds",
            "rank_precision": "CPU float64 SVD; uncentered; no width normalization",
            "activation_mapping": "encoder Dense affines separate from actual block ReLU outputs; predictor excluded; final encoder includes final LayerNorm for residual critic",
            "previous_interaction_step": self.last_step,
            "mask_file": path.name,
            "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "masks": mask_metadata,
            "metrics": scalars,
            "persistence_attribution": "transition_v1_new is an EchoCritic diagnostic, conditional on stream inputs; no irreversible/chronic inactivity claim",
        }
        with (self.output / (name + ".json")).open("x") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        self.previous, self.last_step = next_previous, interaction_step
        return record


def _flag(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def from_environment(run_dir):
    """Absent root means no extra model call, sampling, SVD or file creation."""
    root = os.environ.get(ENV_ROOT)
    if not root:
        return None
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("diagnostic root must be absolute")
    root = root.resolve()
    run_dir = Path(run_dir).resolve()
    if root == run_dir or run_dir in root.parents:
        raise ValueError("diagnostic output must be outside training checkpoints")
    frozen_path = os.environ.get("ECHOCRITIC_CRITIC_DIAGNOSTICS_REFERENCE")
    frozen = None
    if frozen_path:
        with np.load(frozen_path, allow_pickle=False) as data:
            frozen = {key: data[key].copy() for key in ("observation", "action")}
    run_key = hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
    observer = CriticMeasurements(
        root / run_key / uuid.uuid4().hex,
        frozen_batch=frozen,
        freeze_first=_flag("ECHOCRITIC_CRITIC_DIAGNOSTICS_FREEZE_FIRST"),
        compute_rank=_flag("ECHOCRITIC_CRITIC_DIAGNOSTICS_RANK"),
        every=int(os.environ.get("ECHOCRITIC_CRITIC_DIAGNOSTICS_EVERY", "1")),
    )
    (observer.output / "session.json").write_text(
        json.dumps(
            {
                "schema": PREFIX,
                "run_dir": str(run_dir),
                "reference_file": frozen_path,
                "session_boundary": "persistence resets on new session; retained masks can be compared offline",
            },
            indent=2,
        )
    )
    return observer


def available_gradient_arrays(tree, *, kind):
    """Optional per-layer summaries of already available arrays, never autodiff.

    This is a new diagnostic adapter, not a new SAC return payload. Callers must
    identify array ownership/timing; it cannot reconstruct layers from a scalar
    historical norm. No optimizer update or probe gradient is computed here.
    """
    from flax.traverse_util import flatten_dict

    allowed = {"raw_td_gradient_pre_adamw", "optimizer_update_array", "probe_gradient"}
    if kind not in allowed:
        raise ValueError("explicit supported gradient/update kind required")
    layers = {}
    for path, value in flatten_dict(tree).items():
        x = np.asarray(value, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("finite gradient/update arrays required")
        layers.setdefault("/".join(path[:-1]) or "root", []).append(x.reshape(-1))
    if not layers:
        raise ValueError("nonempty gradient/update tree required")

    def norm(arrays):
        scale = max((float(np.max(np.abs(x))) if x.size else 0) for x in arrays)
        return (
            float(scale * np.sqrt(sum(np.sum((x / scale) ** 2) for x in arrays)))
            if scale
            else 0.0
        )

    result = {
        f"{PREFIX}/available_arrays/{kind}/{name}/l2_analysis64": norm(arrays)
        for name, arrays in sorted(layers.items())
    }
    result[f"{PREFIX}/available_arrays/{kind}/global/l2_analysis64"] = norm(
        [x for arrays in layers.values() for x in arrays]
    )
    return result
