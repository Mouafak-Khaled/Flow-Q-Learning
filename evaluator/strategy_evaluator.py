import matplotlib.pyplot as plt
import seaborn as sns

from evaluator.experiment_from_file import ExperimentFromFile
from hpo.strategy import HpoStrategy
from trainer.config import TrainerConfig
from hpo.successive_halving import SuccessiveHalving


class StrategyEvaluator:
    """
    Evaluates the performance of a hyperparameter optimization strategy.
    """

    def __init__(
        self,
        strategy: HpoStrategy,
        config: TrainerConfig,
    ):
        self.strategy = strategy
        self.config = config

        self.experiments = {}
        self.stopping_step = {}
        self.candidates = None

        self.finished_candidates = []
        self.candidates = self.strategy.sample()
        for config in self.candidates:
            self.experiments[config] = ExperimentFromFile(
                experiment_config=config,
                save_directory=self.config.save_directory,
                env_name=self.config.env_name,
            )

    def evaluate(self) -> None:
        while True:
            for config in self.candidates:
                if self.experiments[config].train(self.config.eval_interval):
                    self.finished_candidates.append(config)

            # Evaluate experiments
            for config in self.candidates:
                score = self.experiments[config].evaluate(self.config.eval_episodes)
                self.strategy.update(config, score)

            # Sample new candidates based on the strategy
            new_candidates = self.strategy.sample()

            done = [config in self.finished_candidates for config in new_candidates]
            if all(done):
                break

            # Stop experiments that are no longer candidates
            for config in self.candidates:
                if config not in new_candidates:
                    self.stopping_step[config] = self.experiments[config].current_step
            
            self.candidates = new_candidates

    def plot(self):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        for experiment in self.experiments.values():
            data = experiment.get_data()
            sns.lineplot(
                data=data,
                x="step",
                y="success",
                label=experiment.get_label(),
                linestyle="--"
                if experiment.current_step in self.stopping_step
                else "-",
            )
            if experiment.current_step in self.stopping_step:
                sns.lineplot(
                    data=data[data["step"] <= experiment.current_step],
                    x="step",
                    y="success",
                    label=experiment.get_label(),
                )
        plt.xlabel("Step")
        plt.ylabel("Success Rate")
        plt.title(get_strategy_title(self.strategy))
        plt.legend()
        plt.savefig(f"report/{get_strategy_title(self.strategy)}.png")
        plt.close()

def get_strategy_title(strategy: HpoStrategy) -> str:
    if isinstance(strategy, SuccessiveHalving):
        return fr"Successive Halving $(f={strategy.fraction}, h={strategy.history_length}, t={strategy.total_evaluations})$"
    else:
        return "Unknown Strategy"
