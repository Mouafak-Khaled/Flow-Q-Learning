from typing import List
from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig
from collections import defaultdict


class DiscardWorstStrategy(HpoStrategy):
    """
    A hyperparameter optimization strategy that samples candidates based on the worst performance
    in the population. It removes candidates that have the worst performace.
    """
    def __init__(self, fraction: float = 0.5, state_dict: dict | None = None) -> None:
        super().__init__(state_dict=state_dict)
        self.candidate_scores = defaultdict(float)
        self.fraction = fraction

        if state_dict is not None:
            self.candidate_scores = defaultdict(float, state_dict["candidate_scores"])
            self.fraction = state_dict["fraction"]

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the strategy.

        Returns:
            dict: State dictionary containing the candidate scores and fraction.
        """
        return {
            **super().state_dict(),
            "candidate_scores": dict(self.candidate_scores),
            "fraction": self.fraction,
        }

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
