from copy import deepcopy
from dataclasses import asdict
import random
from pathlib import Path
import numpy as np
import pickle
from fql.agents.fql import FQLAgent
from hpo.strategy import HpoStrategy
from task.task import Task
from trainer.config import ExperimentConfig, TrainerConfig
from trainer.experiment import Experiment
from utils.logger import Logger


class Trainer:
    def __init__(self, task: Task, strategy: HpoStrategy, config: TrainerConfig):
        self.task = task
        self.strategy = strategy
        self.config = config

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        strategy.init(
            [ExperimentConfig(alpha=alpha) for alpha in [0.03, 0.1, 0.3, 1, 3, 10]]
        )

        self.experiments = {}

        self.current_step = 0

    def load(self, path: str) -> None:
        """
        Load the trainer state from a file.

        Args:
            path (str): Path to the file containing the trainer state.
        """
        # Implement loading logic here
        load_dir = Path(path)

        # Load strategy
        self.strategy.load_strategy(load_dir)

        # Load trainer metadata
        with open(load_dir / "trainer_state.pkl", "rb") as f:
            trainer_state = pickle.load(f)

        self.current_step = trainer_state["current_step"]
        experiment_configs = trainer_state["experiment_configs"]

        # Restore RNG states
        random.setstate(trainer_state["random_state"])
        np.random.set_state(trainer_state["np_random_state"])

        # Recreate and load experiments
        for config in experiment_configs:
            experiment_name = "_".join(f"{k}_{v}" for k, v in vars(config).items())
            experiment_dir = load_dir / "experiments" / experiment_name

            self.create_experiment(config)
            self.experiments[config].load_agent(experiment_dir)

    def save(self, path: str) -> None:
        """
        Save the trainer state to a file.

        Args:
            path (str): Path to the file where the trainer state will be saved.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.strategy.save_strategy(save_dir, step=self.current_step)

        # Save metadata including RNG states
        trainer_state = {
            "current_step": self.current_step,
            "experiment_configs": list(self.experiments.keys()),
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
        }

        with open(save_dir / "trainer_state.pkl", "wb") as f:
            pickle.dump(trainer_state, f)

        # Save all experiment agents
        for config, experiment in self.experiments.items():
            experiment_name = "_".join(f"{k}_{v}" for k, v in vars(config).items())
            experiment_dir = save_dir / "experiments" / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            experiment.save_agent(experiment_dir)

    def create_experiment(self, experiment_config) -> Experiment:
        example_batch = self.task.sample("train", 1)

        agent_config = deepcopy(self.config.agent)
        for field, value in vars(experiment_config).items():
            if hasattr(agent_config, field):
                setattr(agent_config, field, value)
        experiment_name = "_".join(
            f"{field}_{value}" for field, value in vars(experiment_config).items()
        )

        agent = FQLAgent.create(
            self.config.seed,
            example_batch["observations"],
            example_batch["actions"],
            asdict(agent_config),
        )

        self.experiments[experiment_config] = Experiment(
            agent=agent,
            task=self.task,
            steps=self.config.steps,
            log_interval=self.config.log_interval,
            logger=Logger(
                self.config.save_directory / self.config.env_name / experiment_name
            ),
        )

    def train(self) -> None:
        """
        Train the agents using the specified task and strategy.
        """
        candidates = self.strategy.sample()

        for config in candidates:
            if config not in self.experiments:
                self.create_experiment(config)

        while self.current_step < self.config.steps:
            # Train each experiment
            for config in (
                self.strategy.init_population
                if self.config.evaluation_mode
                else candidates
            ):
                self.experiments[config].train(self.config.eval_interval)

            self.current_step += self.config.eval_interval

            # Evaluate experiments
            for config in (
                self.strategy.init_population
                if self.config.evaluation_mode
                else candidates
            ):
                score = self.experiments[config].evaluate(self.config.eval_episodes)
                self.strategy.update(self.experiments[config], score)

            # Sample new candidates based on the strategy
            new_candidates = self.strategy.sample()

            # Stop experiments that are no longer candidates
            for config in new_candidates:
                if config not in self.experiments:
                    self.create_experiment(config)

            # Stop experiments that are no longer candidates
            for config in candidates:
                if config not in new_candidates:
                    self.experiments[config].stop(self.config.evaluation_mode)

            # Update candidates for the next iteration
            candidates = new_candidates

        for config in candidates:
            experiment_name = "_".join(
                f"{field}_{value}" for field, value in vars(config).items()
            )
            self.experiments[config].save_agent(
                self.config.save_directory / self.config.env_name / experiment_name
            )

        for experiment in self.experiments.values():
            experiment.stop()
