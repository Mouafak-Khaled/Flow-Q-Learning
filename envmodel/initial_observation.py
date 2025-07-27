from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    latent_dimension: int
    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        mean = nn.Dense(self.latent_dimension)(x)
        logvar = nn.Dense(self.latent_dimension)(x)
        return mean, logvar


class Decoder(nn.Module):
    output_dimension: int
    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, z):
        for hidden_dim in self.hidden_dims:
            z = nn.relu(nn.Dense(hidden_dim)(z))
        return nn.Dense(self.output_dimension)(z)


class InitialObservationEnvModel(nn.Module):
    observation_dimension: int
    latent_dimension: int
    hidden_dims: Tuple[int, ...] = (128,)

    def setup(self):
        self.encoder = Encoder(self.latent_dimension, self.hidden_dims)
        self.decoder = Decoder(self.observation_dimension, self.hidden_dims)

    def reparameterize(self, key, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mu + eps * std

    def __call__(
        self, observations, key, **kwargs
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        mu, logvar = self.encoder(observations)
        z = self.reparameterize(key, mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, (mu, logvar)


def vae_loss(
    model: nn.Module, params: Any, rng: jax.Array, batch: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jax.Array]]:
    rng, key = jax.random.split(rng)
    reconstructed_observations, (mu, logvar) = model.apply(params, key=key, **batch)

    reconstruction_loss = jnp.mean(
        (batch["observations"] - reconstructed_observations) ** 2
    )
    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    loss = reconstruction_loss + kl

    logs = {
        "reconstruction_loss": reconstruction_loss,
        "kl": kl,
        "loss": loss,
    }

    return loss, (logs, rng)
