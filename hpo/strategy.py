from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from trainer.config import ExperimentConfig
import pickle

class HpoStrategy(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    This class defines the interface for strategies that can be used to sample candidates
    from a population based on their performance history.
    """

    def __init__(self, population: List[ExperimentConfig]) -> None:
        """
        Initialize the strategy with the given population of candidates.

        Args:
            population (list): The initial set of candidates.
        """
        self.population = population
        self.init_population = population.copy()
        self.current_step = 0

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

    def save_strategy(self, save_directory: Path, step: int = None):
        """
        Save the current state of the hyperparameter optimization strategy to disk.

        The method saves the current population and initial population as a pickle file.
        If `step` is provided, it is used in the filename. Otherwise, it falls back to
        using `self.current_step`.

        Args:
            save_directory (Path): The directory where the strategy should be saved.
            step (int, optional): The training or optimization step to include in the
                filename. If None, `self.current_step` is used.
        """
        save_directory.mkdir(parents=True, exist_ok=True)
        filename = f"strategy_{step}.pkl" if step is not None else f"strategy_{self.current_step}.pkl"
        save_path = save_directory / filename

        save_dict = dict(
            population=self.population,
            init_population=self.init_population,
        )
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

    def load_strategy(self, load_directory: Path, step: int = None):
        """
        Load the state of the hyperparameter optimization strategy from disk.

        This method restores the population and initial population from a previously saved
        pickle file. If `step` is provided, it is used to determine the filename. Otherwise,
        it falls back to using `self.current_step`.

        Args:
            load_directory (Path): The directory from which the strategy should be loaded.
            step (int, optional): The training or optimization step to include in the
                filename. If None, `self.current_step` is used.

        """
        filename = f"strategy_{step}.pkl" if step is not None else f"strategy_{self.current_step}.pkl"
        load_path = load_directory / filename
        with open(load_path, "rb") as f:
            load_dict = pickle.load(f)
        self.population = load_dict["population"]
        self.init_population = load_dict["init_population"]
