from typing import List
from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig
from collections import defaultdict


class DiscardWorstStrategy(HpoStrategy):
    """
    A hyperparameter optimization strategy that samples candidates based on the worst performance
    in the population. It removes candidates that have the worst performace.
    """

    def __init__(self, fraction: float = 0.5) -> None:
        self.candidate_scores = defaultdict(float)
        self.fraction = fraction
        self.population = []
        self.init_population = []

    def update(self, candidate, performance: float) -> None:
        self.candidate_scores[candidate] = performance

    def sample(self) -> List[ExperimentConfig]:
        sorted_population = sorted(
            self.candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        to_be_removed = int(len(self.population) * self.fraction)
        best_candidates_with_scores = (
            sorted_population[:-to_be_removed]
            if to_be_removed > 0
            else sorted_population
        )
        self.population = [candidate for candidate, _ in best_candidates_with_scores]
        self.candidate_scores = defaultdict(float, best_candidates_with_scores)
        return self.population
