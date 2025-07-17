from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from fql.envs.env_utils import make_env_and_datasets
from fql.utils.datasets import Dataset, ReplayBuffer
from task.task import Task


class OfflineTaskWithSimulatedEvaluations(Task):
    def __init__(
        self,
        model: nn.Module,
        params: Dict[str, Any],
        env_name: str,
        buffer_size: int,
        data_directory: Path,
        num_evaluation_envs: int = 50,
        max_episode_steps: int = 1000,
    ):
        self.model = model
        self.max_episode_steps = max_episode_steps
        self.params = params

        _, self.eval_envs, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, data_directory, num_evaluation_envs=num_evaluation_envs
        )

        self.train_dataset = Dataset.create(**self.train_dataset)
        self.train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1)
        )
        self.current_observations = None
        self.episode_steps = 0

    def sample(self, dataset: Literal["train", "val"], batch_size: int):
        return (
            self.train_dataset.sample(batch_size)
            if dataset == "train"
            else self.val_dataset.sample(batch_size)
        )

    def reset(self, seed: int = 0):
        observations, infos = zip(*[eval_env.reset(seed=seed + i) for i, eval_env in enumerate(self.eval_envs)])
        merged_observations = np.array(observations)
        merged_infos = list(infos)

        self.current_observations = merged_observations
        self.episode_steps = 0
        self.invalidate = [False] * len(self.eval_envs)

        return merged_observations, merged_infos

    def step(self, actions):
        next_observations, terminations = self.model.apply(self.params, self.current_observations, actions)

        next_observations = jnp.asarray(next_observations)
        terminations = jnp.asarray(terminations).squeeze() > 0
        reward = jnp.where(terminations, 0, -1)

        self.current_observations = next_observations
        self.episode_steps += 1

        truncations = self.episode_steps >= self.max_episode_steps
        truncations = jnp.full_like(terminations, truncations, dtype=bool)

        infos = [{"success": s} for s in terminations]

        return next_observations, reward, terminations, truncations, infos

    def close(self):
        for eval_env in self.eval_envs:
            eval_env.close()
