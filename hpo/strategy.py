from abc import ABC, abstractmethod
from typing import List

from trainer.config import ExperimentConfig


class HpoStrategy(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    This class defines the interface for strategies that can be used to sample candidates
    from a population based on their performance history.
    """
    def __init__(self, population: List[ExperimentConfig], total_evaluations: int, state_dict: dict | None = None):
        """
        Initialize the strategy with an optional state dictionary.

        Args:
            population (List[ExperimentConfig]): The initial population of candidates.
            total_evaluations (int): The total number of evaluations performed.
            state_dict (dict | None): Optional state dictionary to restore the strategy's state.
        """
        self.population = population
        self.total_evaluations = total_evaluations

        if state_dict is not None:
            self.population = state_dict["population"]

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the strategy.

        Returns:
            dict: State dictionary containing the population and initial population.
        """
        return {
            "population": self.population
        }

    @abstractmethod
    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        """
        Update the performance history of a candidate.

        Args:
            candidate: The candidate for which to update the performance history.
            performance (float): The performance score of the candidate.
        """
        pass

    @abstractmethod
    def sample(self) -> List[ExperimentConfig]:
        """
        Sample candidates from the population based on the strategy.

        Returns:
            List[ExperimentConfig]: A list of sampled candidates.
        """
        pass
