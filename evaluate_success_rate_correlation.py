import re

from matplotlib.artist import get
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

from argparser import get_env_model_argparser, build_env_model_config_from_args
from evaluator.env_model_evaluator import EnvModelEvaluator
from success_rate_gaussian_process import SuccessRateGaussianProcess
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations
from trainer.config import ExperimentConfig
from utils.agent import load_agent
from utils.tasks import get_task_filename, get_task_title
from evaluator.evaluation import evaluate_agent


# Usage:
# python evaluate_success_rate_correlation.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --model=baseline \
#        --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
#        --seed=42 --eval_episodes=50

# read arguments from command line
parser = get_env_model_argparser()
parser.add_argument(
    "--eval_episodes", type=int, default=50, help="Number of evaluation episodes."
)
args = parser.parse_args()
config = build_env_model_config_from_args(args)

real_success_rates = pd.read_csv(
    f"results/real_success_rates_{get_task_filename(config.env_name)}.csv"
)

simulated_task = OfflineTaskWithSimulatedEvaluations(
    config.env_name,
    model=config.model,
    data_directory=config.data_directory,
    save_directory=config.save_directory,
    num_evaluation_envs=args.eval_episodes,
)


def evaluate(row):
    exp_path = (
        config.save_directory.parent / "exp-checkpointing" / config.env_name
    ).glob(f"seed_{row['seed']}_alpha_{row['alpha']}_*")[-1]
    agent = load_agent(
        agent_directory=exp_path,
        sample_batch=simulated_task.sample("train", 1),
        agent_filename=f"checkpoint_{row['checkpoint']}.pkl",
    )
    info, _ = evaluate_agent(agent=agent, env=simulated_task)
    return info["success"]


initial_sample = real_success_rates.sample(n=20, random_state=config.seed).copy()
simulated_success_rates = initial_sample.apply(evaluate, axis=1)

gp = SuccessRateGaussianProcess(
    real_success_rates=real_success_rates,
    simulated_success_rates=simulated_success_rates,
    seed=config.seed
)

for i in range(30):
    row = gp.ask()
    gp.tell(evaluate(row))

gp.save_data(
    filename=f"results/success_rate_gp_{config.model}_{get_task_filename(config.env_name)}.csv"
)
gp.plot(
    filename=f"report/success_rate_gp_{config.model}_{get_task_filename(config.env_name)}.gif"
)

simulated_task.close()


# x = df["real_success"]
# y = df["sim_success"]

# r, p = pearsonr(x, y)
# rho, rho_pval = spearmanr(x, y)
# tau, tau_pval = kendalltau(x, y)

# sns.set_theme(style="darkgrid")
# plt.figure(figsize=(12, 8))
# norm = plt.Normalize(df["checkpoint"].min(), df["checkpoint"].max())
# sc = plt.scatter(
#     df["real_success"],
#     df["sim_success"],
#     c=df["checkpoint"],
#     cmap="viridis",
#     norm=norm,
#     alpha=0.7,
# )
# plt.colorbar(sc, label="Checkpoint Step")
# plt.scatter(df["real_success"], df["sim_success"], alpha=0.6)
# plt.xlabel("Real Success Rate (%)")
# plt.ylabel("Simulated Success Rate (%)")
# plt.xticks(range(-10, 110, 20))
# plt.yticks(range(-10, 110, 20))
# plt.title(
#     f"Simulated and Real Success Rate Correlation for {get_task_title(config.env_name)} task"
# )
# plt.tight_layout()
# plt.savefig(
#     f"report/success_rate_correlation_{config.model}_{get_task_filename(config.env_name)}.png",
#     dpi=300,
# )
# plt.close()

# grouped_df = (
#     df.groupby("alpha", dropna=True)[["real_success", "sim_success"]]
#     .mean()
#     .reset_index()
# )
# norm = mcolors.LogNorm(
#     vmin=grouped_df["alpha"].replace(0, 1e-3).min(), vmax=grouped_df["alpha"].max()
# )
# sns.set_theme(style="darkgrid")
# plt.figure(figsize=(12, 8))
# sc = plt.scatter(
#     grouped_df["real_success"],
#     grouped_df["sim_success"],
#     c=grouped_df["alpha"],
#     cmap="viridis",
#     norm=norm,
#     s=100,
# )
# plt.xlabel("Average Real Success Rate (%)")
# plt.ylabel("Average Simulated Success Rate (%)")
# cbar = plt.colorbar(sc)
# cbar.set_label(r"log($\alpha$)")
# plt.suptitle(rf"""Simulated and Real Success Rate Correlation for {get_task_title(config.env_name)} task by $\alpha$
# Pearson's  $r$  = {r:>6.2f} (p = {p:>4.1f})
# Spearman's $\rho$ = {rho:>6.2f} (p = {rho_pval:>4.1f})
# Kendall's  $\tau$ = {tau:>6.2f} (p = {tau_pval:>4.1f})""")
# plt.tight_layout()
# plt.xticks(range(-10, 110, 20))
# plt.yticks(range(-10, 110, 20))
# plt.savefig(
#     f"report/success_rate_correlation_grouped_by_alpha_{config.model}_{get_task_filename(config.env_name)}.png",
#     dpi=300,
# )
# plt.close()


# gdf = (
#     df.groupby(["alpha", "checkpoint"])[["real_success", "sim_success"]]
#     .mean()
#     .reset_index()
# )
# sns.set_theme(style="darkgrid")
# plt.figure(figsize=(12, 8))
# for alpha in sorted(gdf["alpha"].unique()):
#     subset = gdf[gdf["alpha"] == alpha]
#     plt.plot(
#         subset["checkpoint"],
#         subset["real_success"],
#         label=rf"$\alpha$={alpha:.1f} Real",
#         linestyle="--",
#     )
#     plt.plot(
#         subset["checkpoint"],
#         subset["sim_success"],
#         label=rf"$\alpha$={alpha:.1f} Sim",
#         linestyle="-",
#     )
# plt.xlabel("Checkpoint")
# plt.ylabel("Success Rate")
# plt.legend(ncol=2, fontsize=8)
# plt.title(r"Success Rate vs Training Step for Each $\alpha$")
# plt.tight_layout()
# plt.savefig(
#     f"report/success_rate_vs_training_step_correlation_grouped_by_alpha_{config.model}_{get_task_filename(config.env_name)}.png",
#     dpi=300,
# )
# plt.close()


# pivot_real = df.pivot_table(values="real_success", index="alpha", columns="checkpoint")
# pivot_sim = df.pivot_table(values="sim_success", index="alpha", columns="checkpoint")

# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# sns.set_theme(style="darkgrid")
# sns.heatmap(pivot_real, ax=axes[0], cmap="YlGnBu", annot=True, fmt=".1f")
# axes[0].set_title("Real Success Rate")
# axes[0].set_xlabel("Checkpoint")
# axes[0].set_ylabel("Alpha")

# sns.heatmap(pivot_sim, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".1f")
# axes[1].set_title("Simulated Success Rate")
# axes[1].set_xlabel("Checkpoint")

# fig.suptitle(r"Heatmap of Success Rates by $\alpha$ and Checkpoint")
# plt.tight_layout()
# plt.savefig(
#     f"report/heatmap_success_rate_by_alpha_and_checkpoint_{config.model}_{get_task_filename(config.env_name)}.png",
#     dpi=300,
# )
# plt.close()
