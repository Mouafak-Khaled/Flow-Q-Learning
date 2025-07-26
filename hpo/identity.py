from typing import Set

from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig


class Identity(HpoStrategy):
    """
    A strategy that returns the entire population without any modifications.
    """
    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        pass

    def sample(self) -> Set[ExperimentConfig]:
        return self.population
