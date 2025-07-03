from hpo.strategy import HpoStrategy
from task.task import Task
from trainer.config import TrainerConfig
from trainer.experiment import Experiment


class Trainer:
    def __init__(self, task: Task, strategy: HpoStrategy, config: TrainerConfig):
        self.task = task
        self.strategy = strategy
        self.config = config

        self.experiments = {}

    def load(self, path: str) -> None:
        """
        Load the trainer state from a file.

        Args:
            path (str): Path to the file containing the trainer state.
        """
        # Implement loading logic here
        pass

    def save(self, path: str) -> None:
        """
        Save the trainer state to a file.

        Args:
            path (str): Path to the file where the trainer state will be saved.
        """
        # Implement saving logic here
        pass

    def create_experiment(self, experiment_config) -> Experiment:
        self.experiments[experiment_config] = Experiment(
            self.task,
            self.config,
            experiment_config,
        )

    def train(self) -> None:
        """
        Train the agents using the specified task and strategy.
        """
        candidates = self.strategy.sample()

        for config in (
            self.strategy.init_population
            if self.config.evaluation_mode
            else candidates
        ):
            if config not in self.experiments:
                self.create_experiment(config)

        while len(candidates) > 0:
            # Train each experiment
            for config in (
                self.strategy.init_population
                if self.config.evaluation_mode
                else candidates
            ):
                if self.experiments[config].train(self.config.eval_interval):
                    self.experiments[config].save_agent()

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

        for experiment in self.experiments.values():
            experiment.stop()
