import jax.numpy as jnp
from flax import linen as nn


class SimpleCubeEnvModel(nn.Module):
    obs_dim: int = 28
    act_dim: int = 5
    hidden_size: int = 128

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)

        x = nn.LayerNorm()(x)

        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        out = nn.Dense(self.obs_dim + 1)(x)

        next_obs = out[..., :-1]
        reward = out[..., -1]
        return next_obs, reward
