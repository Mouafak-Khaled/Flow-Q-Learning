from typing import List

from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig


class IdentityStrategy(HpoStrategy):
    """
    A strategy that returns the entire population without any modifications.
    """

    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        pass

    def sample(self) -> List[ExperimentConfig]:
        return self.population
