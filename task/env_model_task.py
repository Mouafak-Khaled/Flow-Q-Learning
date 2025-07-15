from typing import Literal, Tuple, Dict, Any, Optional
import jax.numpy as jnp
import jax
import numpy as np
from envmodel.wrapper import EnvModelWrapper
from task.task import Task
from flax import linen as nn
from fql.envs.env_utils import make_env_and_datasets
from fql.utils.datasets import Dataset, ReplayBuffer
from pathlib import Path


class OfflineTaskWithSimulatedEvaluations(Task):

    def __init__(
            self,
            model: nn.Module,
            params: Dict[str, Any],
            env_name: str,
            buffer_size: int,
            data_directory: Path,
            max_episode_steps: int=1000
    ):
        self.model = model
        self.max_episode_steps = max_episode_steps
        self.params = params

        _, self.eval_env, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, data_directory
        )
        self.train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1)
        )
        self.current_obs = None
        self.episode_steps = 0

    def reset(self, seed: int= 0):
        obs, info = self.eval_env.reset(seed=seed)
        self.current_obs = obs
        self.episode_steps = 0
        return obs, info

    def step(self, action):
        info = dict()
        next_obs, terminated = self.model.apply(self.params, self.current_obs, action)

        # Convert to jnp arrays (if they aren't already)
        next_obs = jnp.asarray(next_obs)
        terminated = bool(jnp.asarray(terminated).squeeze() > 0)
        reward = -1 if not terminated else 0

        self.current_obs = next_obs
        self.episode_steps += 1

        truncated = self.episode_steps >= self.max_episode_steps
        if truncated or terminated:
            info["success"] = 1 if terminated else 0

        return next_obs, reward, terminated, truncated, info
