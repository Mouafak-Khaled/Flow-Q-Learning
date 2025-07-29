from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from envmodel.baseline import BaselineStatePredictor


class Cell(nn.Module):
    observation_dimension: int
    action_dimension: int
    hidden_dims: Tuple[int, ...]

    def setup(self):
        self.cell = BaselineStatePredictor(
            observation_dimension=self.observation_dimension,
            action_dimension=self.action_dimension,
            hidden_dims=self.hidden_dims,
        )

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        next_observations, reconstructed_observations = self.cell(
            observations=observations, actions=actions
        )
        return next_observations, (next_observations, reconstructed_observations)


class MultistepStatePredictor(nn.Module):
    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_dims: Tuple[int, ...] = (128, 256, 128)

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        model = nn.scan(
            Cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        _, (next_observations, reconstructed_observations) = model(
            observation_dimension=self.observation_dimension,
            action_dimension=self.action_dimension,
            hidden_dims=self.hidden_dims,
        )(observations[:, 0, :], actions)

        return next_observations, reconstructed_observations
