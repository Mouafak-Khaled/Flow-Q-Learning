from abc import ABC, abstractmethod
from typing import Literal


class Task(ABC):
    @abstractmethod
    def sample(self, dataset: Literal["train", "val"], batch_size: int):
        """
        Abstract method to sample a batch of data from the specified dataset.

        Args:
            dataset (str): The dataset to sample from, either 'train' or 'val'.
            batch_size (int): The number of samples to return.

        Returns:
            A batch of data sampled from the specified dataset.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Abstract method to reset the task environment to its initial state.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Abstract method to take a step in the task environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing (next_observation, reward, terminated, truncated, info).
        """
        pass
