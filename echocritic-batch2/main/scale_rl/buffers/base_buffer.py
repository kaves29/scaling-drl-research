from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym
import numpy as np

from pathlib import Path
import pickle

Batch = Dict[str, np.ndarray]


class BaseBuffer(ABC):
    _EXCLUDE_FROM_CHECKPOINT = {
        "_observation_space",
        "_action_space",
        "_max_length",
        "_min_length",
        "_n_step",
        "_gamma",
        "_add_batch_size",
        "_sample_batch_size",
    }

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        add_batch_size: int,
        sample_batch_size: int,
    ):
        """
        A generic buffer class.

        args:
            observation_shape
            action_shapce
            max_length: maximum length of buffer (max number of experiences stored within the state).
            min_length: minimum number of experiences saved in the buffer state before we can sample.
            add_sequences: indiciator of whether we will be adding data in sequences to the buffer?
            add_batch_size: batch size of data that is added in a single addition call.
            sample_batch_size: batch size of data that is sampled from a single sampling call.
        """

        self._observation_space = observation_space
        self._action_space = action_space
        self._max_length = max_length
        self._min_length = min_length
        self._n_step = n_step
        self._gamma = gamma
        self._add_batch_size = add_batch_size
        self._sample_batch_size = sample_batch_size

    def __len__(self):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, timestep: Dict[str, np.ndarray]) -> None:
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        pass

    @abstractmethod
    def sample(self) -> Batch:
        pass

    @abstractmethod
    def get_observations(self) -> np.ndarray:
        pass

    def save(self, checkpoint_dir: str) -> None:
        """
        Generically persists all subclass instance state (numpy arrays,
        pointers, counters, etc.) without needing to know the subclass's
        exact internal field names.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        exclude = getattr(self, "_EXCLUDE_FROM_CHECKPOINT", set())
        state = {}
        for k, v in self.__dict__.items():
            if k in exclude:
                continue
            if isinstance(v, np.ndarray) and hasattr(self, "_num_in_buffer"):
                state[k] = v[: self._num_in_buffer]
            else:
                state[k] = v

        with open(checkpoint_dir / "buffer_state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, checkpoint_dir: str) -> None:
        """
        Restores whatever state was saved by save(). Numpy arrays are written
        into the existing pre-allocated (full max_length-sized) arrays rather
        than replacing them outright, since save() may have trimmed arrays
        down to only the filled portion.
        """
        checkpoint_path = Path(checkpoint_dir) / "buffer_state.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No buffer checkpoint found at {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        for k, v in state.items():
            if isinstance(v, np.ndarray) and hasattr(self, k):
                current_arr = getattr(self, k)
                if (
                    isinstance(current_arr, np.ndarray)
                    and current_arr.ndim == v.ndim
                    and current_arr.shape[0] >= v.shape[0]
                ):
                    current_arr[: v.shape[0]] = v
                    continue
            setattr(self, k, v)
