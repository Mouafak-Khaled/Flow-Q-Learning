import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluator.experiment_from_file import ExperimentFromFile
from trainer.config import ExperimentConfig, TrainerConfig


class SensitivityEvaluator:
    def __init__(self, candidates: list[ExperimentConfig], config: TrainerConfig):
        self.config = config
        self.candidates = candidates

    def evaluate(self) -> None:
        success_curves = []
        for candidate in self.candidates:
            experiment = ExperimentFromFile(
                experiment_config=candidate,
                save_directory=self.config.save_directory,
                env_name=self.config.env_name,
            )
            success_curve = experiment.get_data().copy()
            success_curve["alpha"] = candidate.alpha
            success_curve["seed"] = candidate.seed
            success_curves.append(success_curve)
        self.success_curves = pd.concat(success_curves, ignore_index=True)

    def plot_sensitivity_heatmap(self) -> None:
        df = self.success_curves.pivot_table(
            index='alpha', columns='step', values='success', aggfunc='mean'
        )

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Success Rate'})

        ylabels = df.index.values
        ax.set_yticklabels([f"{y:.1f}" for y in ylabels])

        plt.title(f"Success Rate Heatmap for {self.get_task_name()} Task")
        plt.xlabel("Training Steps")
        plt.ylabel(r"$\alpha$")
        plt.savefig(
            f"report/sensitivity_heatmap_{self.get_task_name().lower()}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def plot_sensitivity_curve(self) -> None:
        df = self.success_curves.sort_values(by=['alpha', 'seed', 'step'])

        # get the maximum and final success per experiment
        grouped = df.groupby(['alpha', 'seed'])
        max_success = grouped['success'].max()
        last_success = grouped['success'].last()

        # average the previous results over the seeds
        grouped = pd.DataFrame({
            'max_success': max_success,
            'last_success': last_success
        }).groupby('alpha')
        results = pd.DataFrame({
            'avg_max_success': grouped['max_success'].mean(),
            'avg_last_success': grouped['last_success'].mean()
        }).reset_index()

        # plot the sensitivity curve
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=results, x='alpha', y='avg_last_success', label='Average Final Success', marker='o')
        sns.lineplot(data=results, x='alpha', y='avg_max_success', label='Average Maximum Success', marker='o')
        plt.xscale('log')
        plt.xlabel(r"$\alpha$")
        plt.ylabel("Success Rate")
        plt.title(fr"Sensitivity Curve of $\alpha$ on the {self.get_task_name()} task.")
        plt.legend()
        plt.savefig(
            f"report/sensitivity_curve_{self.get_task_name().lower()}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def get_task_name(self) -> str:
        if self.config.env_name.startswith("cube"):
            return "Cube"
        elif self.config.env_name.startswith("antsoccer"):
            return "Antsoccer"
