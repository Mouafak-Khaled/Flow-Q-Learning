from collections import defaultdict

from hpo.strategy import HpoStrategy
from task.task import Task


class Trainer:
    def __init__(self, task: Task, strategy: HpoStrategy, evaluation_mode: bool = False):
        self.task = task
        self.strategy = strategy
        self.history = defaultdict(list)

        self.evaluation_mode = evaluation_mode
        if self.evaluation_mode:
            self.stopped_at = defaultdict(int)

    def load(self, path: str) -> None:
        """
        Load the trainer state from a file.

        Args:
            path (str): Path to the file containing the trainer state.
        """
        # Implement loading logic here
        pass

    def save(self, path: str) -> None:
        """
        Save the trainer state to a file.

        Args:
            path (str): Path to the file where the trainer state will be saved.
        """
        # Implement saving logic here
        pass

    def train(self, steps: int) -> None:
        """
        Train the agents using the specified task and strategy.
        """
        candidates = self.strategy.sample()
        for step in range(steps):
            # Train each agent
            for agent in (self.strategy.init_population if self.evaluation_mode else candidates):
                train_agent(agent, self.task)

            # Evaluate agents
            for agent in (self.strategy.init_population if self.evaluation_mode else candidates):
                score = evaluate_agent(agent, self.task)
                self.history[agent].append(score)

            # Sample new candidates based on the strategy
            new_candidates = self.strategy.sample()

            # If in evaluation mode, track when agents stopped performing
            if self.evaluation_mode:
                for agent in candidates:
                    if agent not in new_candidates:
                        self.stopped_at[agent] = step

            # Update candidates for the next iteration
            candidates = new_candidates


def train_agent(agent, task) -> None:
    pass


def evaluate_agent(agent, task) -> float:
    return 1.0
