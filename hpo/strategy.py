from abc import ABC, abstractmethod


class HpoStrategy(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    This class defines the interface for strategies that can be used to sample candidates
    from a population based on their performance history.
    """

    def init(self, population: list) -> None:
        """
        Initialize the strategy with the given population of candidates.

        Args:
            population (list): The initial set of candidates.
        """
        self.population = population
        self.init_population = population.copy()

    @abstractmethod
    def update(self, candidate, performance: float) -> None:
        """
        Update the performance history of a candidate.

        Args:
            candidate: The candidate for which to update the performance history.
            performance (float): The performance score of the candidate.
        """
        pass

    @abstractmethod
    def sample(self) -> list:
        """
        Sample candidates from the population based on the strategy.

        Returns:
            list: A list of sampled candidates.
        """
        pass
