from typing import List

import numpy as np
from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig
from collections import defaultdict


class SuccessiveHalving(HpoStrategy):
    """
    A hyperparameter optimization strategy that samples candidates based on the worst performance
    in the population. It removes candidates that have the worst performance.
    """
    def __init__(self, population: List[ExperimentConfig], total_evaluations: int, fraction: float = 0.5, state_dict: dict | None = None) -> None:
        super().__init__(population=population, total_evaluations=total_evaluations, state_dict=state_dict)
        self.fraction = fraction

        self.candidate_scores = defaultdict(float)
        self.performed_evaluations = 0

        self.halving_milestones = self._compute_halving_milestones()

        if state_dict is not None:
            self.candidate_scores = state_dict["candidate_scores"]
            self.performed_evaluations = len(self.candidate_scores)

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the strategy.

        Returns:
            dict: State dictionary containing the candidate scores and fraction.
        """
        return {
            **super().state_dict(),
            "candidate_scores": dict(self.candidate_scores),
        }

    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        self.candidate_scores[candidate] = performance

    def sample(self) -> List[ExperimentConfig]:
        self.performed_evaluations += 1

        if self.performed_evaluations not in self.halving_milestones:
            return self.population

        if len(self.population) <= 1:
            return self.population

        new_length = max(1, int(len(self.population) * self.fraction))

        sorted_population = sorted(
            self.candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_candidates_with_scores = sorted_population[:new_length]
        self.population = [candidate for candidate, _ in best_candidates_with_scores]
        self.candidate_scores = defaultdict(float, best_candidates_with_scores)
        return self.population

    def _compute_halving_milestones(self) -> List[int]:
        """
        Compute evaluation milestones at which to halve the population.
        """
        N = len(self.population)
        f = self.fraction
        T = self.total_evaluations

        R = int(np.floor(np.log(N) / np.log(1 / f)))
        
        milestones = []
        for r in range(R):
            # This is the per-config budget for this round
            budget = int(T * f ** (r - (R - 1)))
            milestones.append(budget)
        
        return milestones
