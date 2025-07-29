from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class StatePredictor(nn.Module):
    """Base class for all state predictors."""

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


class BaselineStatePredictor(StatePredictor):
    """A baseline environment model that predicts the next observation.
    This model is deterministic and does not use any latent variables."""

    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_dims: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observations, actions], axis=-1)

        x = nn.LayerNorm()(x)

        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        next_observations = nn.Dense(self.observation_dimension)(x) + observations

        return next_observations, observations
