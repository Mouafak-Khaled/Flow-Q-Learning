from typing import Tuple

from flax import linen as nn


class TerminationPredictor(nn.Module):
    """A termination predictor that predicts whether an episode has terminated based on the next observations and rewards."""

    observation_dimension: int = 28
    hidden_dims: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, observations, **kwargs):
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        terminations = nn.Dense(1)(x)

        return terminations.squeeze(-1)
