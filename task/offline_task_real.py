from pathlib import Path
from typing import Literal

from fql.envs.env_utils import make_env_and_datasets
from fql.utils.datasets import Dataset, ReplayBuffer
from task.task import Task


class OfflineTaskWithRealEvaluations(Task):
    def __init__(self, buffer_size: int, env_name: str, data_directory: Path = None):
        """
        Initialize the task with an evaluation environment.

        Args:
            buffer_size: The size of the buffer.
            env_name: The name of the environment to create.
        """
        _, self.eval_env, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, data_directory
        )
        self.train_dataset = Dataset.create(**self.train_dataset)
        self.train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1)
        )

    def sample(self, dataset: Literal["train", "val"], batch_size: int):
        return (
            self.train_dataset.sample(batch_size)
            if dataset == "train"
            else self.val_dataset.sample(batch_size)
        )

    def reset(self):
        return self.eval_env.reset()

    def step(self, action):
        return self.eval_env.step(action)
