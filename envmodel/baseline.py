from typing import Any, Dict, Tuple
import jax.numpy as jnp
from flax import linen as nn
import optax


class BaselineEnvModel(nn.Module):
    """A baseline environment model that predicts next observations and rewards.
    This model is deterministic and does not use any latent variables."""

    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_size: int = 128

    @nn.compact
    def __call__(self, observations, actions, **kwargs):
        x = jnp.concatenate([observations, actions], axis=-1)

        x = nn.LayerNorm()(x)

        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        next_observations = nn.Dense(self.observation_dimension)(x) + observations
        terminations = nn.Dense(1)(x)

        return next_observations, terminations


def baseline_loss(
    model: nn.Module, params: Any, batch: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Computes the total loss for a given batch."""

    pred_next_obs, pred_terminals = model.apply(params, **batch)

    next_observation_loss = jnp.mean(
        jnp.square(pred_next_obs - batch["next_observations"])
    )
    terminated_loss = optax.sigmoid_binary_cross_entropy(
        logits=jnp.squeeze(pred_terminals), labels=1 + batch["rewards"]
    ).mean()

    loss = next_observation_loss + terminated_loss

    logs = {
        "next_observation_loss": next_observation_loss,
        "terminated_loss": terminated_loss,
        "loss": loss,
    }

    return loss, logs
