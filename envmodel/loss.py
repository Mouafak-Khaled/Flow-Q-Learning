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
