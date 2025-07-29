import jax
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


def focal_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Computes focal loss between logits and binary targets.

    Args:
        predictions: [batch_size] — raw logits (not probabilities)
        targets: [batch_size] — binary labels (0 or 1)
        alpha: class balancing factor (typically 0.25)
        gamma: focusing parameter (typically 2.0)
    """
    targets = targets.astype(jnp.float32)
    probabilities = jax.nn.sigmoid(predictions)

    ce_loss = optax.sigmoid_binary_cross_entropy(predictions, targets)

    pt = jnp.where(targets == 1, probabilities, 1 - probabilities)
    modulating_factor = (1.0 - pt) ** gamma

    alpha_factor = jnp.where(targets == 1, alpha, 1 - alpha)

    loss = jnp.mean(alpha_factor * modulating_factor * ce_loss)

    logs = {
        "true_loss": jnp.sum(loss * (targets == 1)) / (jnp.sum(targets == 1) + 1e-8),
        "false_loss": jnp.sum(loss * (targets == 0)) / (jnp.sum(targets == 0) + 1e-8),
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
    termination_loss, termination_logs = (
        weighted_binary_cross_entropy(
            predicted_termination,
            terminations,
            true_termination_weight,
        )
        if termination_weight > 0
        else (0.0, {})
    )
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
