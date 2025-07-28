from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


class BaselineEnvModel(nn.Module):
    """A baseline environment model that predicts next observations and rewards.
    This model is deterministic and does not use any latent variables."""

    observation_dimension: int = 28
    action_dimension: int = 5
    hidden_dims: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, observations, actions, **kwargs):
        x = jnp.concatenate([observations, actions], axis=-1)

        x = nn.LayerNorm()(x)

        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        next_observations = nn.Dense(self.observation_dimension)(x) + observations
        terminations = nn.Dense(1)(next_observations)

        return next_observations, terminations


def baseline_loss(
    model: nn.Module,
    params: Any,
    rng: jax.Array,
    batch: Dict[str, jnp.ndarray],
    termination_weight: float,
    termination_true_weight: float,
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jax.Array]]:
    """Computes the total loss for a given batch."""

    pred_next_obs, pred_terminals = model.apply(params, **batch)

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

    loss = (next_observation_loss + termination_weight * terminated_loss) / (1 + termination_weight)

    logs = {
        "next_observation_loss": next_observation_loss,
        "terminated_loss": terminated_loss,
        "termination_loss_true": jnp.sum(terminated_loss_true) / jnp.sum(batch["rewards"] == 0),
        "termination_loss_false": jnp.sum(terminated_loss_false) / jnp.sum(batch["rewards"] != 0),
        "loss": loss,
    }

    return loss, (logs, rng)
