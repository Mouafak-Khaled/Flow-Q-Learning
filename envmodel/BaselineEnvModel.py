import jax.numpy as jnp
from flax import linen as nn


class BaselineEnvModel(nn.Module):
    """A baseline environment model that predicts next observations and rewards.
    This model is deterministic and does not use any latent variables."""

    obs_dim: int = 28
    act_dim: int = 5
    hidden_size: int = 128

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)

        x = nn.LayerNorm()(x)

        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        next_obs = nn.Dense(self.obs_dim)(x) + obs
        reward = nn.Dense(1)(x)
        terminated = nn.Dense(1)(x)

        return next_obs, reward, terminated
