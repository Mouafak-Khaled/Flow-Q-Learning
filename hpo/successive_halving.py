from collections import defaultdict
from typing import List

import numpy as np

from hpo.strategy import HpoStrategy
from trainer.config import ExperimentConfig


class SuccessiveHalving(HpoStrategy):
    def __init__(
        self,
        population: List[ExperimentConfig],
        total_evaluations: int,
        fraction: float = 0.5,
        history_length: int = 1,
        state_dict: dict | None = None,
    ) -> None:
        """
        Initialize the SuccessiveHalving strategy.

        Args:
            population (List[ExperimentConfig]): Initial population of candidate configurations.
            total_evaluations (int): Total number of allowed evaluations.
            fraction (float): Fraction of the population to prune at each milestone.
            state_dict (dict | None): Optional dictionary to restore internal state.
            history_length (int): Number of evaluations to accumulate before pruning can start.
        """
        super().__init__(population, total_evaluations, state_dict)
        self.history_length = history_length
        self.candidate_scores = defaultdict(list)
        self.performed_evaluations = 0
        self.fraction = fraction
        self.halving_milestones = self.compute_halving_milestones()

        if state_dict is not None:
            self.candidate_scores = state_dict["candidate_scores"]
            self.performed_evaluations = state_dict["performed_evaluations"]

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the strategy.

        Returns:
            dict: State dictionary containing the candidate scores and fraction.
        """
        return {
            **super().state_dict(),
            "candidate_scores": dict(self.candidate_scores),
            "performed_evaluations": self.performed_evaluations,
        }

    def update(self, candidate: ExperimentConfig, performance: float) -> None:
        """
        Update the performance history of a given candidate.

        Args:
            candidate (ExperimentConfig): The evaluated candidate.
            performance (float): The performance score from the evaluation.
        """
        self.candidate_scores[candidate].append(performance)

    def sample(self) -> List[ExperimentConfig]:
        """
        Sample the next set of candidates for evaluation.

        Prunes the population based on average historical performance at
        precomputed halving milestones. Candidates with the lowest average
        scores are removed.

        Returns:
            List[ExperimentConfig]: The updated list of candidate configurations.
        """
        self.performed_evaluations += 1

        if (
            self.performed_evaluations not in self.halving_milestones
            or self.performed_evaluations < self.history_length
        ):
            return self.population

        if len(self.population) <= 1:
            return self.population

        new_length = max(
            1, int(len(self.population) - len(self.population) * self.fraction)
        )

        average_scores = {
            key: sum(values[-self.history_length :]) / self.history_length
            for key, values in self.candidate_scores.items()
            if values
        }

        sorted_population = sorted(
            average_scores.items(), key=lambda x: x[1], reverse=True
        )

        best_candidates_with_scores = sorted_population[:new_length]
        self.population = [candidate for candidate, _ in best_candidates_with_scores]
        self.candidate_scores = defaultdict(list, best_candidates_with_scores)
        return self.population

    def compute_halving_milestones(self) -> List[int]:
        """
        Compute the evaluation steps (milestones) at which the population should be halved,
        as used in Successive Halving algorithm.

        Returns:
            List[int]: A list of evaluation steps (milestones) at which to perform halving,
                    in descending order of aggressiveness (early rounds require more evals).
        """
        N = len(self.population)
        eta = 1 / self.fraction
        T = int(self.total_evaluations * self.fraction)
        R = int(np.floor(np.log(N) / np.log(eta)))  # Number of halving rounds
        return [int(T * eta**-r) for r in range(R)]
