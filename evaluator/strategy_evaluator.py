import matplotlib.pyplot as plt
import seaborn as sns

from evaluator.experiment_from_file import ExperimentFromFile
from hpo.strategy import HpoStrategy
from hpo.successive_halving import SuccessiveHalving
from trainer.config import TrainerConfig
from utils.tasks import get_task_filename, get_task_title


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

    def evaluate(self) -> dict[int, float]:
        best_score = 0.0
        score_curve = {}
        current_step = 0
        while True:
            for config in self.candidates:
                if self.experiments[config].train(self.config.eval_interval):
                    self.finished_candidates.append(config)

            current_step += self.config.eval_interval

            # Evaluate experiments
            for config in self.candidates:
                score = self.experiments[config].evaluate(self.config.eval_episodes)
                best_score = max(best_score, score)
                self.strategy.update(config, score)

            score_curve[current_step] = best_score

            # Sample new candidates based on the strategy
            self.candidates = self.strategy.sample()

            done = [config in self.finished_candidates for config in self.candidates]
            if all(done):
                break
        
        return score_curve

    def plot(self):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(20, 12))
        for config, experiment in self.experiments.items():
            data = experiment.get_data()
            sns.lineplot(data=data, x="step", y="success", linestyle="--", alpha=0.4)
            sns.lineplot(
                data=data[data["step"] <= self.experiments[config].current_step],
                x="step",
                y="success",
                label=experiment.get_label(),
            )
        plt.xlabel("Step")
        plt.ylabel("Success Rate")
        plt.title(get_strategy_title(self.strategy, self.config.env_name))
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.savefig(
            f"report/{get_strategy_filename(self.strategy, self.config.env_name)}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def get_strategy_title(strategy: HpoStrategy, env_name: str) -> str:
    if isinstance(strategy, SuccessiveHalving):
        return rf"Successive Halving $(f={strategy.fraction}, h={strategy.history_length})$\nFor tuning $\alpha$ on {get_task_title(env_name)} task."
    else:
        return "Unknown Strategy"

def get_strategy_filename(strategy: HpoStrategy, env_name: str) -> str:
    if isinstance(strategy, SuccessiveHalving):
        return f"successive_halving_f{strategy.fraction}_h{strategy.history_length}_{get_task_filename(env_name)}"
    else:
        return "unknown_strategy"