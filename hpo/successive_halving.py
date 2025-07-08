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
        self.candidate_scores[candidate] = performance

    def sample(self) -> List[ExperimentConfig]:
        self.performed_evaluations += 1

        if self.performed_evaluations not in self.halving_milestones:
            return self.population

        if len(self.population) <= 1:
            return self.population

        new_length = max(1, int(len(self.population) - len(self.population) * self.fraction))

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
        eta = 1 / self.fraction
        T = self.total_evaluations
        R = int(np.floor(np.log(N) / np.log(eta)))

        milestones = []
        for r in range(R):
            budget = int(T * eta ** -r)
            milestones.append(budget)
        return milestones



class SuccessiveHalvingWithHistory(HpoStrategy):

    def __init__(
            self,
            population: List[ExperimentConfig],
            total_evaluations: int,
            fraction: float = 0.5,
            history_length: int = 3,
            state_dict: dict | None = None
    ) -> None:
        """
        Initialize the SuccessiveHalvingWithHistory strategy.

        Args:
            population (List[ExperimentConfig]): Initial population of candidate configurations.
            total_evaluations (int): Total number of allowed evaluations.
            fraction (float): Fraction of the population to prune at each milestone.
            state_dict (dict | None): Optional dictionary to restore internal state.
            history_length (int): Number of evaluations to accumulate before pruning can start.
        """
        super().__init__(population, total_evaluations)
        self.fraction = fraction
        self.history_length = history_length

        self.candidate_scores = defaultdict(list)
        self.performed_evaluations = 0

        self.halving_milestones = self._compute_halving_milestones()

        if state_dict is not None:
            self.candidate_scores = state_dict["candidate_scores"]
            self.performed_evaluations = state_dict["performed_evaluations"]

    def state_dict(self) -> dict:
        """
        Return the current internal state of the strategy.

        Returns:
            dict: Dictionary containing the population and candidate score history.
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

        if self.performed_evaluations not in self.halving_milestones or self.performed_evaluations < self.history_length:
            return self.population


        if len(self.population) <= 1:
            return self.population

        new_length = max(1,  int(len(self.population) - len(self.population) * self.fraction))

        average_scores = {
            key: sum(values[-self.history_length:]) / self.history_length
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

    def _compute_halving_milestones(self) -> List[int]:
        """
        Compute the evaluation counts at which pruning should occur.

        The milestones are calculated based on the total number of evaluations,
        the population size, the pruning fraction, and the required history length.

        Returns:
            List[int]: A list of evaluation counts where halving will be triggered.
        """
        N = len(self.population)
        eta = 1 / self.fraction
        T = self.total_evaluations
        R = int(np.floor(np.log(N) / np.log(eta)))

        milestones = []
        for r in range(R):
            budget = int(T * eta ** -r)
            milestones.append(budget)
        return milestones


