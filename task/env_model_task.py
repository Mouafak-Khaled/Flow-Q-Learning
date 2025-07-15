from typing import Literal, Tuple, Dict, Any, Optional
import jax.numpy as jnp
import jax
import numpy as np
from envmodel.wrapper import EnvModelWrapper
from task.task import Task
from flax import linen as nn


class EnvModelTask(Task):
    def __init__(
        self, model: EnvModelWrapper, metadata: Optional[Dict[str, Any]] = None, initial_state_sampler: Optional[callable] = None
    ):
        self.model = model
        self.metadata = metadata

        self.initial_state_sampler = initial_state_sampler
        self.current_obs = None
        self.episode_steps = 0

    def sample(self, dataset: Literal["train", "val"], batch_size: int):
        raise NotImplementedError()


    def reset(self) -> jnp.ndarray:
        if self.initial_state_sampler is None:
            obs_dim = self.metadata.get("obs_dim", 1)
            self.current_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
        else:
            self.current_obs = self.initial_state_sampler()
        self.episode_steps = 0
        return self.current_obs

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, dict]:
        next_obs, reward, terminated = self.model.predict(self.current_obs, action)

        # Convert to jnp arrays (if they aren't already)
        next_obs = jnp.asarray(next_obs)
        reward = float(jnp.asarray(reward).squeeze())
        terminated = bool(jnp.asarray(terminated).squeeze() > 0.5)

        self.current_obs = next_obs
        self.episode_steps += 1

        max_steps = self.metadata.get("max_episode_steps", 1000)
        truncated = self.episode_steps >= max_steps

        return next_obs, reward, terminated, truncated, {}