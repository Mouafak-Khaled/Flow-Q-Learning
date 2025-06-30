from abc import ABC, abstractmethod
from typing import List

from trainer.config import ExperimentConfig


class HpoStrategy(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    This class defines the interface for strategies that can be used to sample candidates
    from a population based on their performance history.
    """

    def populate(self, population: List[ExperimentConfig]) -> None:
        """
        Initialize the strategy with the given population of candidates.

        Args:
            population (list): The initial set of candidates.
        """
        self.population = population
        self.init_population = population.copy()

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
