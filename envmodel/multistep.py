from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from envmodel.baseline import BaselineEnvModel


class Cell(nn.Module):
    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_size: int = 128

    def setup(self):
        self.cell = BaselineEnvModel(
            observation_dimension=self.observation_dimension,
            action_dimension=self.action_dimension,
            hidden_size=self.hidden_size,
        )

    @nn.compact
    def __call__(
        self, observations, actions, **kwargs
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        next_observations, terminations = self.cell(
            observations=observations, actions=actions, **kwargs
        )
        return next_observations, (next_observations, terminations)


class MultistepEnvModel(nn.Module):
    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self, observations, actions, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        model = nn.scan(
            Cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        _, (next_observations, terminations) = model(
            observation_dimension=self.observation_dimension,
            action_dimension=self.action_dimension,
            hidden_size=self.hidden_size,
        )(observations[:, 0, :], actions)

        return next_observations, terminations
