from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from fql.envs.env_utils import make_env_and_datasets
from fql.utils.datasets import Dataset, ReplayBuffer
from task.task import Task
from utils.envmodel import load_model


class OfflineTaskWithSimulatedEvaluations(Task):
    def __init__(
        self,
        env_name: str,
        model: str = "baseline",
        data_directory: Path = None,
        save_directory: Path = Path("exp/"),
        buffer_size: int = 2000000,
        num_evaluation_envs: int = 50,
        max_episode_steps: int = 1000,
    ):
        self.max_episode_steps = max_episode_steps

        _, self.eval_envs, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, data_directory, num_evaluation_envs=num_evaluation_envs
        )

        if num_evaluation_envs == 1:
            self.eval_envs = [self.eval_envs]

        self.train_dataset = Dataset.create(**self.train_dataset)
        self.train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(self.train_dataset), size=max(buffer_size, self.train_dataset.size + 1)
        )

        example_batch = self.train_dataset.sample(1)
        self.state_predictor = load_model(
            save_directory, env_name, model, example_batch
        )

        self.termination_predictor = load_model(
            save_directory, env_name, "termination_predictor", example_batch
        )

        self.current_observations = None
        self.episode_steps = 0
        self.model = model
        self.env_name = env_name

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
            observations, infos = zip(
                *[eval_env.reset() for eval_env in self.eval_envs]
            )
        else:
            observations, infos = zip(
                *[
                    eval_env.reset(seed=seed + i)
                    for i, eval_env in enumerate(self.eval_envs)
                ]
            )

        merged_observations = np.array(observations)
        merged_infos = list(infos)
        self.current_observations = merged_observations
        self.episode_steps = 0
        self.invalidate = [False] * len(self.eval_envs)

        return merged_observations, merged_infos

    def step(self, actions):
        next_observations = self.state_predictor(self.current_observations, actions)
        terminations = self.termination_predictor(next_observations)

        next_observations = jnp.asarray(next_observations)
        terminations = jnp.asarray(terminations) > 0
        reward = jnp.where(terminations, 0, -1)

        self.current_observations = next_observations
        self.episode_steps += 1

        truncations = self.episode_steps >= self.max_episode_steps
        truncations = jnp.full_like(terminations, truncations, dtype=bool)

        infos = [{"success": s} for s in terminations]

        for i in range(len(self.eval_envs)):
            if self.invalidate[i]:
                infos[i]["invalid"] = True
            if terminations[i] or truncations[i]:
                self.invalidate[i] = True

        return next_observations, reward, terminations, truncations, infos

    def close(self):
        for eval_env in self.eval_envs:
            eval_env.close()
