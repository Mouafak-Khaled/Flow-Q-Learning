from pathlib import Path
from typing import Literal

from fql.envs.env_utils import make_env_and_datasets
from fql.utils.datasets import Dataset, ReplayBuffer
from task.task import Task
import numpy as np


class OfflineTaskWithRealEvaluations(Task):
    def __init__(self, buffer_size: int, env_name: str, data_directory: Path = None, num_evaluation_envs: int = 50):
        """
        Initialize the task with an evaluation environment.

        Args:
            buffer_size: The size of the buffer.
            env_name: The name of the environment to create.
        """
        _, self.eval_envs, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, data_directory, num_evaluation_envs=num_evaluation_envs
        )

        if num_evaluation_envs == 1:
            self.eval_envs = [self.eval_envs]

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

    def reset(self, seed: int | None = None):
        if len(self.eval_envs) == 1:
            observation, info = self.eval_envs[0].reset(seed=seed)
            observations = [observation]
            infos = [info]
        elif seed is None:
            observations, infos = zip(*[eval_env.reset() for eval_env in self.eval_envs])
        else:
            observations, infos = zip(*[eval_env.reset(seed=seed + i) for i, eval_env in enumerate(self.eval_envs)])

        merged_observations = np.array(observations)
        merged_infos = list(infos)
        self.invalidate = [False] * len(self.eval_envs)

        return merged_observations, merged_infos

    def step(self, actions):
        results = [eval_env.step(action) for action, eval_env in zip(actions, self.eval_envs)]
        next_observations, rewards, terminations, truncations, infos = zip(*results)

        for i in range(len(self.eval_envs)):
            if self.invalidate[i]:
                infos[i]['invalid'] = True
            if terminations[i] or truncations[i]:
                self.invalidate[i] = True

        merged_next_observations = np.array(next_observations)
        merged_rewards = np.array(rewards)
        merged_terminations = np.array(terminations)
        merged_truncations = np.array(truncations)
        merged_infos = list(infos)
        return merged_next_observations, merged_rewards, merged_terminations, merged_truncations, merged_infos

    def close(self):
        for eval_env in self.eval_envs:
            eval_env.close()
