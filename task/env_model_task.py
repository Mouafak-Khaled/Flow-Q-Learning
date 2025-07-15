from typing import Literal, Tuple, Callable, Optional
import jax.numpy as jnp
import jax
import numpy as np

from task.task import Task
from flax import linen as nn


class AntSoccerArenaNavigateSingleTask4V0(Task):
    def __init__(
        self,
        model: nn.Module,
        train_state: TrainState,
        val_dataset,
        initial_state_distribution: Callable[[], np.ndarray],
        metadata: dict = None,
    ):
        self.model = WorldModelWrapper(model, train_state, model.apply)
        self.val_dataset = val_dataset
        self.initial_state_distribution = initial_state_distribution
        self.metadata = metadata or {}
        self.current_state = None

    def sample(self, dataset: Literal["train", "val"], batch_size: int):
        if dataset == "val":
            return self.val_dataset.sample(batch_size)
        raise NotImplementedError("WorldModelTask only supports validation sampling.")

    def reset(self) -> np.ndarray:
        self.current_state = self.initial_state_distribution()
        return self.current_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        next_state, reward, terminated = self.model.predict(self.current_state, action)
        self.current_state = next_state
        info = {}
        truncated = False  # Not modeled
        return next_state, reward, terminated, truncated, info


class WorldModelTask(Task):
    def __init__(
        self,
        env_model: nn.Module, # Your trained world model (e.g., an instance of DummyWorldModel)
        obs_dim: int,
        action_dim: int,
        initial_state_sampler: Optional[callable] = None,
        max_episode_steps: Optional[int] = None,
        env_name_suffix: str = "simulated", # Suffix for the environment name
        default_initial_state: Optional[np.ndarray] = None,
    ):
        """
        Initialize the task with a trained world model, acting as a simulated environment.

        Args:
            env_model: The trained world model capable of predicting next_state,
                         reward, and termination. This model should have a method
                         like `predict(observation, action)` that returns
                         (predicted_next_observation, predicted_reward, predicted_terminal_logits).
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            initial_state_sampler: An optional function or object that provides
                                   initial observations for the reset method.
                                   If None and `default_initial_state` is None,
                                   reset will return a zero vector.
            max_episode_steps: Optional maximum number of steps per episode for truncation.
            env_name_suffix: A suffix to append to the base environment name
                             ("antsoccer-arena-navigate-singletask-task4-v0")
                             to differentiate between different world models.
            default_initial_state: An optional fixed NumPy array to use as the initial state
                                   if no `initial_state_sampler` is provided.
        """
        self.env_model = env_model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.initial_state_sampler = initial_state_sampler
        self.max_episode_steps = max_episode_steps
        self.default_initial_state = default_initial_state

        # Define the full environment name based on the base and suffix
        self.env_name = f"antsoccer-arena-navigate-singletask-task4-v0_{env_name_suffix}"

        self.current_observation: Optional[np.ndarray] = None
        self.current_step = 0

    def sample(self, dataset: Literal["train", "val"], batch_size: int) -> Dict[str, Any]:
        """
        In the context of a WorldModelTask, this method is primarily for
        conforming to the `Task` interface. It's not intended for sampling
        from an offline dataset in the traditional sense.
        As this class acts as a simulated environment via `reset` and `step`,
        batch sampling from a static dataset isn't its primary function.
        If you need to generate data (e.g., rollouts) from the world model,
        that functionality would typically be implemented in a separate method
        or utility specific to data generation, not under `sample` as defined by `Task`.
        """
        raise NotImplementedError(
            f"Sampling data from a WorldModelTask ({self.env_name}) is not directly supported."
            "This class acts as a simulated environment via `reset` and `step`."
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the simulated environment to an initial state.

        Returns:
            A tuple containing (initial_observation, info).
        """
        self.current_step = 0
        if self.initial_state_sampler:
            initial_obs = self.initial_state_sampler()
        elif self.default_initial_state is not None:
            initial_obs = self.default_initial_state
        else:
            # Default to a zero observation if no sampler or default state is provided
            initial_obs = np.zeros(self.obs_dim, dtype=np.float32)

        self.current_observation = initial_obs
        info = {"env_name": self.env_name, "current_step": self.current_step}
        return self.current_observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the simulated environment using the trained world model.

        Args:
            action: The action to take (NumPy array).

        Returns:
            A tuple containing (next_observation, reward, terminated, truncated, info).
        """
        if self.current_observation is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        # Ensure action is a NumPy array
        action = np.asarray(action, dtype=np.float32)

        # Predict using the world model
        predicted_next_obs, predicted_reward, predicted_terminal_logits = self.env_model.predict(
            self.current_observation, action
        )

        # Convert terminal logits to boolean. A common way is to threshold.
        # Adjust the threshold (0.5 here) based on how your world model was trained
        # to output 'terminal' (e.g., probability, logit).
        terminated = (predicted_terminal_logits > 0.5)

        self.current_step += 1
        truncated = False
        if self.max_episode_steps is not None and self.current_step >= self.max_episode_steps:
            truncated = True

        info = {"env_name": self.env_name, "current_step": self.current_step}

        self.current_observation = predicted_next_obs # Update current state for the next step
        return predicted_next_obs, predicted_reward, terminated, truncated, info
