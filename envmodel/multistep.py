from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from envmodel.baseline import BaselineEnvModel


class MultistepEnvModel(nn.Module):
    cell: BaselineEnvModel
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def scan_fn(carry, inputs):
            next_observations, terminations = self.cell(
                observations=carry, actions=inputs
            )
            return next_observations, (next_observations, terminations)

        _, (next_observations, terminations) = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
            length=observations.shape[1],
        )(observations, actions)

        return next_observations, terminations
