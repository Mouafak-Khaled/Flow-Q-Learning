import jax.numpy as jnp
from typing import Dict, Tuple
import optax


def mean_squared_error(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Computes the squared mean of the difference between predictions and targets."""
    return jnp.mean(jnp.square(predictions - targets))


def weighted_binary_cross_entropy(
    predictions: jnp.ndarray, targets: jnp.ndarray, true_weight: float = 1.0
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Computes the weighted binary cross-entropy loss."""

    loss = optax.sigmoid_binary_cross_entropy(logits=predictions, labels=targets)
    true_loss = jnp.where(targets == 1, loss, 0.0)
    false_loss = jnp.where(targets == 0, loss, 0.0)
    loss = jnp.mean((true_weight * true_loss + false_loss) / (true_weight + 1))

    logs = {
        "true_loss": jnp.sum(true_loss) / jnp.sum(targets == 1),
        "false_loss": jnp.sum(false_loss) / jnp.sum(targets == 0),
    }

    return loss, logs


def state_prediction_loss(
    predicted_next_observations,
    next_observations,
    predicted_termination,
    terminations,
    reconstructed_observations: jnp.ndarray,
    observations: jnp.ndarray,
    true_termination_weight: float,
    termination_weight: float,
    reconstruction_weight: float,
):
    next_observation_loss = mean_squared_error(
        predicted_next_observations, next_observations
    )
    termination_loss, termination_logs = weighted_binary_cross_entropy(
        predicted_termination,
        terminations,
        true_termination_weight,
    ) if termination_weight > 0 else (0.0, {})
    reconstruction_loss = (
        mean_squared_error(reconstructed_observations, observations)
        if reconstruction_weight > 0
        else 0.0
    )

    loss = (
        next_observation_loss
        + termination_weight * termination_loss
        + reconstruction_weight * reconstruction_loss
    ) / (1 + termination_weight + reconstruction_weight)

    logs = {
        "next_observation_loss": next_observation_loss,
        "loss": loss,
    }
    if termination_weight > 0:
        logs["termination_loss"] = termination_loss
        logs["true_termination_loss"] = termination_logs["true_loss"]
        logs["false_termination_loss"] = termination_logs["false_loss"]
    if reconstruction_weight > 0:
        logs["reconstruction_loss"] = reconstruction_loss

    return loss, logs
