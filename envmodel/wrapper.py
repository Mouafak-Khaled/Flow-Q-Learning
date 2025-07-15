from abc import ABC
from flax import linen as nn
from typing import Any, Dict


class EnvModelWrapper(ABC):
    """
    Wrapper class for a learned environment model in JAX/Flax.

    This class encapsulates a Flax model and its parameters,
    and provides a simple interface to make predictions (e.g., next state, reward, termination).

    Attributes:
        model (nn.Module): The Flax model representing the environment dynamics.
        params (Dict[str, Any]): The trained parameters of the model.
    """
    def __init__(self, model: nn.Module, params: Dict[str, Any]):
        """
        Initializes the environment model wrapper.

        Args:
            model (nn.Module): A Flax module defining the environment model.
            params (Dict[str, Any]): Parameters for the model, typically loaded from a checkpoint.
        """
        self.model = model
        self.params = params

    def predict(self, state, action):
        """
        Predicts the next observation, reward, and termination signal given current state and action.

        Args:
            state: The current observation/state (e.g., jnp.ndarray).
            action: The action taken from the current state (e.g., jnp.ndarray).

        Returns:
            Tuple: (next_state, reward, terminated), where each is a JAX-compatible array.
        """
        return self.model.apply(self.params, state, action)