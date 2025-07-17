import random

import numpy as np
import time
from hpo.strategy import HpoStrategy
from task.task import Task
from trainer.config import TrainerConfig
from trainer.experiment import Experiment


class Trainer:
    def __init__(
        self,
        task: Task,
        strategy: HpoStrategy,
        config: TrainerConfig,
        state_dict: dict | None = None,
    ):
        self.task = task
        self.strategy = strategy
        self.config = config

        self.experiments = {}
        self.candidates = None

        if state_dict is not None:
            for experiment_config, experiment_state in state_dict["experiments"].items():
                self.create_experiment(experiment_config, state_dict=experiment_state)
            self.candidates = state_dict["candidates"]
            self.untrained_candidates = state_dict["untrained_candidates"]
            self.finished_candidates = state_dict["finished_candidates"]
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
        else:
            self.untrained_candidates = []
            self.finished_candidates = []
            self.candidates = self.strategy.sample()
            
            for config in self.candidates:
                self.create_experiment(config)

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the trainer.

        Returns:
            dict: State dictionary containing the experiments and candidates.
        """
        return {
            "experiments": {
                config: experiment.state_dict()
                for config, experiment in self.experiments.items()
            },
            "candidates": self.candidates,
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "untrained_candidates": self.untrained_candidates,
            "finished_candidates": self.finished_candidates
        }

    def create_experiment(self, experiment_config, **kwargs) -> Experiment:
        self.experiments[experiment_config] = Experiment(
            self.task,
            self.config,
            experiment_config,
            **kwargs
        )

    def train(self, max_evaluations: int) -> None:
        """
        Train the agents using the specified task and strategy.
        """
        while max_evaluations > 0:
            # Train each experiment
            untrained_candidates = []
            for config in (self.untrained_candidates if len(self.untrained_candidates) > 0 else self.candidates):
                if max_evaluations == 0:
                    untrained_candidates.append(config)
                    continue
                start_time = time.perf_counter()
                if self.experiments[config].train(self.config.eval_interval):
                    self.experiments[config].save_agent()
                    self.finished_candidates.append(config)
                # Evaluate experiments
                score = self.experiments[config].evaluate()
                self.strategy.update(self.experiments[config], score)
                elapsed_time = time.perf_counter() - start_time
                print(f"Elapsed time: {elapsed_time:.6f} seconds")
                max_evaluations -= 1

            if len(untrained_candidates) > 0:
                self.untrained_candidates = untrained_candidates
                break
            else:
                self.untrained_candidates = []

            unfinished_candidates = [
                config for config in self.candidates if config not in self.finished_candidates
            ]
            if len(unfinished_candidates) == 0:
                break

            # Sample new candidates based on the strategy
            new_candidates = self.strategy.sample()

            # Create new experiments for new candidates
            for config in new_candidates:
                if config not in self.experiments:
                    self.create_experiment(config)

            # Stop experiments that are no longer candidates
            for config in self.candidates:
                if config not in new_candidates:
                    self.experiments[config].stop()

            # Update candidates for the next iteration
            self.candidates = new_candidates

        for experiment in self.experiments.values():
            experiment.stop()
