from typing import Tuple, Dict, Any
from envmodel.baseline import BaselineEnvModel
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class Encoder(nn.Module):
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, s: jnp.ndarray):
        x = nn.relu(nn.Dense(self.hidden_dim)(s))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.latent_dim)(x)


class Decoder(nn.Module):
    output_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray):
        x = nn.relu(nn.Dense(self.hidden_dim)(z))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)


class LatentSpaceEnvModel(nn.Module):
    state_predictor: nn.Module

    observation_dim: int
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs):
        encoder = Encoder(self.latent_dim, self.hidden_dim)
        decoder = Decoder(self.observation_dim, self.hidden_dim)

        z = encoder(observations)
        next_z, terminations = self.state_predictor(observations=z, actions=actions)
        reconstructed_observations = decoder(z)
        next_observations = decoder(z)

        return next_observations, terminations, reconstructed_observations


class LatentCell(nn.Module):
    latent_dim: int
    action_dim: int
    hidden_dim: int

    def setup(self):
        self.cell = BaselineEnvModel(
            observation_dimension=self.latent_dim,
            action_dimension=self.action_dim,
            hidden_size=self.hidden_dim,
        )

    @nn.compact
    def __call__(
        self, z, actions, **kwargs
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        next_z, terminations = self.cell(
            observations=z, actions=actions, **kwargs
        )
        return next_z, (next_z, terminations)


class MultistepLatentSpaceEnvModel(nn.Module):
    observation_dim: int
    latent_dim: int
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs):
        encoder = Encoder(self.latent_dim, self.hidden_dim)
        decoder = Decoder(self.observation_dim, self.hidden_dim)
        z = encoder(observations[:, 0, :])
        # Define the recurrent model
        cell = nn.scan(
            LatentCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(self.latent_dim, self.action_dim, self.hidden_dim)

        _, (next_z, terminations) = cell(z, actions)
        next_observations = decoder(next_z)
        reconstructed_observations = decoder(z)

        return next_observations, terminations, reconstructed_observations


def latent_space_loss(
        model: nn.Module,
        params: Any,
        rng: jax.Array,
        batch: Dict[str, jnp.ndarray],
        termination_weight: float,
        termination_true_weight: float,
        reconstruction_weight: float
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jax.Array]]:
    pred_next_obs, pred_terminals, reconstructed_observations = model.apply(params, **batch)

    next_observation_loss = jnp.mean(
        jnp.square(pred_next_obs - batch["next_observations"])
    )
    terminated_loss = optax.sigmoid_binary_cross_entropy(
        logits=jnp.squeeze(pred_terminals), labels=1 + batch["rewards"]
    )
    terminated_loss_true = jnp.where(batch["rewards"] == 0, terminated_loss, 0.0)
    terminated_loss_false = jnp.where(batch["rewards"] != 0, terminated_loss, 0.0)
    terminated_loss = jnp.mean(
        termination_true_weight * terminated_loss_true + terminated_loss_false
    ) / (termination_true_weight + 1)

    observations = batch["observations"]

    if observations.ndim == 3:
        observations = observations[:, 0, :]

    reconstruction_loss = jnp.mean(
        jnp.square(reconstructed_observations - observations)
    )

    loss = (next_observation_loss
            + termination_weight * terminated_loss
            + reconstruction_weight * reconstruction_loss
            ) / (1 + termination_weight + reconstruction_weight)

    logs = {
        "next_observation_loss": next_observation_loss,
        "terminated_loss": terminated_loss,
        "termination_loss_true": jnp.mean(terminated_loss_true),
        "termination_loss_false": jnp.mean(terminated_loss_false),
        "reconstruction_loss": reconstruction_loss,
        "loss": loss,
    }

    return loss, (logs, rng)
