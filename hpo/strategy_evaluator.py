from collections import defaultdict
from typing import List

from hpo.strategy import HpoStrategy
from hpo.successive_halving import SuccessiveHalving
from trainer.config import ExperimentConfig
from utils.logger import CsvLogger
from pathlib import Path


class HpoStrategyEvaluator(HpoStrategy):
    """
    Evaluates the performance of a hyperparameter optimization strategy.
    """

    def __init__(
        self,
        population: List[ExperimentConfig],
        total_evaluations: int,
        save_directory: Path,
        env_name: str,
        state_dict: dict | None = None,
    ):
        super().__init__(population=population, state_dict=state_dict)
        log_path = save_directory / env_name / "strategies_stopping_times.csv"
        self.logger = CsvLogger(log_path)

        STRATEGIES = {
            "successive_halving_0.5": (
                SuccessiveHalving,
                {
                    "population": population,
                    "total_evaluations": total_evaluations,
                    "fraction": 0.5,
                },
            ),
        }

        if state_dict is not None:
            self.performed_evaluations = state_dict["performed_evaluations"]
            self.strategies = {
                name: STRATEGIES[name][0](**STRATEGIES[name][1], state_dict=state_dict)
                for name, state_dict in state_dict["strategies"].items()
            }
        else:
            self.performed_evaluations = 0
            self.strategies = {
                name: strategy_class(**kwargs)
                for name, (strategy_class, kwargs) in STRATEGIES.items()
            }

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the strategy evaluator.

        Returns:
            dict: State dictionary containing the population and stopped_at dictionary.
        """
        return {
            **super().state_dict(),
            "strategies": {
                name: strategy.state_dict()
                for name, strategy in self.strategies.items()
            },
            "performed_evaluations": self.performed_evaluations,
        }

    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        """
        Update the performance history of a candidate.

        Args:
            candidate (ExperimentConfig): The candidate for which to update the performance history.
            performance (float): The performance score of the candidate.
        """
        for strategy in self.strategies.values():
            if candidate in strategy.population:
                strategy.update(candidate, performance)

    def sample(self) -> List[ExperimentConfig]:
        """
        Sample candidates from the population based on the strategy.

        Returns:
            List[ExperimentConfig]: A list of sampled candidates.
        """
        self.performed_evaluations += 1

        for name, strategy in self.strategies.items():
            old_population = strategy.population
            new_population = strategy.sample()

            for candidate in old_population:
                if candidate not in new_population:
                    self.logger.log({
                        "strategy_name": name,
                        "experiment_name": candidate,
                        "stopped_at": self.performed_evaluations
                    })
        return self.population
