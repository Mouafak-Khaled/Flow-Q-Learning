import itertools
import random
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

from argparser import build_env_model_config_from_args, get_env_model_argparser
from evaluator.env_model_evaluator import EnvModelEvaluator
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations
from trainer.config import ExperimentConfig
from utils.agent import load_agent
from utils.tasks import get_task_filename, get_task_title

# Usage:
# python evaluate_success_rate_correlation.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --model=baseline \
#        --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
#        --number_of_seeds=2 --number_of_alphas=20 --eval_episodes=50
# The values for `--number_of_seeds` and `--number_of_alphas` in addition to the seed
# have to be the same as the ones used in the training script.

# read arguments from command line
parser = get_env_model_argparser()

parser.add_argument(
    "--number_of_seeds",
    type=int,
    default=1,
    help="Number of random seeds to use per hyperparameter configuration.",
)
parser.add_argument(
    "--number_of_alphas",
    type=int,
    default=1,
    help="Number of alpha values to use.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of evaluation episodes.",
)
parser.add_argument(
    "--eval_interval", type=int, default=100000, help="Evaluation interval."
)
parser.add_argument(
    "--steps", type=int, default=1000000, help="Number of training steps."
)
parser.add_argument(
    "--eval_episodes", type=int, default=50, help="Number of evaluation episodes."
)

args = parser.parse_args()
config = build_env_model_config_from_args(args)

# generate experiment configurations
random.seed(config.seed)
np.random.seed(config.seed)

alpha_values = np.logspace(
    np.log10(3), np.log10(1000), num=args.number_of_alphas
).tolist()
seeds = random.sample(range(10000), args.number_of_seeds)

combinations = list(itertools.product(alpha_values, seeds))

experiment_configs = [
    ExperimentConfig(seed=seed, alpha=alpha) for alpha, seed in combinations
]

real_task = OfflineTaskWithRealEvaluations(
    config.env_name,
    data_directory=config.data_directory,
    num_evaluation_envs=args.eval_episodes,
)

simulated_task = OfflineTaskWithSimulatedEvaluations(
    config.env_name,
    model=config.model,
    data_directory=config.data_directory,
    save_directory=config.save_directory,
    num_evaluation_envs=args.eval_episodes,
)

df = pd.DataFrame(
    columns=["seed", "alpha", "real_success", "sim_success", "checkpoint"]
)


def extract_timestamp(file_path):
    folder_name = file_path.parent.name  # get folder name, not full path
    match = re.compile(r"_(\d{8}_\d{6})").search(folder_name)
    return match.group(1) if match else ""


def config_to_name(experiment_config: ExperimentConfig) -> str:
    tuned_hyperparams = [
        (field, value)
        for field, value in vars(experiment_config).items()
        if value is not None
    ]
    return "_".join(f"{field}_{value}" for field, value in tuned_hyperparams)


for experiment_config in experiment_configs:
    files = list(
        (config.save_directory / config.env_name).glob(
            f"{config_to_name(experiment_config)}_*"
        )
    )
    file_path = sorted(files, key=extract_timestamp)[-1]
    for checkpoint_idx in range(1, config.eval_episodes + 1):
        checkpoint = checkpoint_idx * config.eval_interval
        if not (file_path / f"checkpoint_{checkpoint}.pkl").exists():
            print(
                f"File: '{file_path / f'checkpoint_{checkpoint}.pkl'}' was not found. Skipping"
            )
            continue
        agent = load_agent(
            agent_directory=file_path,
            sample_batch=real_task.sample("train", 1),
            agent_filename=f"checkpoint_{checkpoint}",
        )

        evaluator = EnvModelEvaluator(
            real_task=real_task,
            simulated_task=simulated_task,
            agent=agent,
            seed=config.seed,
        )

        real_success, sim_success = evaluator.evaluate()
        del evaluator
        del agent
        df.loc[len(df)] = [
            experiment_config.seed,
            experiment_config.alpha,
            real_success,
            sim_success,
            checkpoint,
        ]

real_task.close()
simulated_task.close()

# df.to_csv("report/new_all_data.csv")

x = df["real_success"]
y = df["sim_success"]

r, p = pearsonr(x, y)
rho, rho_pval = spearmanr(x, y)
tau, tau_pval = kendalltau(x, y)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 8))
norm = plt.Normalize(df["checkpoint"].min(), df["checkpoint"].max())
sc = plt.scatter(
    df["real_success"],
    df["sim_success"],
    c=df["checkpoint"],
    cmap="viridis",
    norm=norm,
    alpha=0.7,
)
plt.colorbar(sc, label="Checkpoint Step")
plt.scatter(df["real_success"], df["sim_success"], alpha=0.6)
plt.xlabel("Real Success Rate (%)")
plt.ylabel("Simulated Success Rate (%)")
plt.xticks(range(-10, 110, 20))
plt.yticks(range(-10, 110, 20))
plt.title(
    f"Simulated and Real Success Rate Correlation for {get_task_title(config.env_name)} task"
)
plt.tight_layout()
plt.savefig(
    f"report/success_rate_correlation_{config.model}_{get_task_filename(config.env_name)}.png",
    dpi=300,
)
plt.close()

grouped_df = (
    df.groupby("alpha", dropna=True)[["real_success", "sim_success"]]
    .mean()
    .reset_index()
)
norm = mcolors.LogNorm(
    vmin=grouped_df["alpha"].replace(0, 1e-3).min(), vmax=grouped_df["alpha"].max()
)
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 8))
sc = plt.scatter(
    grouped_df["real_success"],
    grouped_df["sim_success"],
    c=grouped_df["alpha"],
    cmap="viridis",
    norm=norm,
    s=100,
)
plt.xlabel("Average Real Success Rate (%)")
plt.ylabel("Average Simulated Success Rate (%)")
cbar = plt.colorbar(sc)
cbar.set_label(r"log($\alpha$)")
plt.suptitle(rf"""Simulated and Real Success Rate Correlation for {get_task_title(config.env_name)} task by $\alpha$
Pearson's  $r$  = {r:>6.2f} (p = {p:>4.1f})
Spearman's $\rho$ = {rho:>6.2f} (p = {rho_pval:>4.1f})
Kendall's  $\tau$ = {tau:>6.2f} (p = {tau_pval:>4.1f})""")
plt.tight_layout()
plt.xticks(range(-10, 110, 20))
plt.yticks(range(-10, 110, 20))
plt.savefig(
    f"report/success_rate_correlation_grouped_by_alpha_{config.model}_{get_task_filename(config.env_name)}.png",
    dpi=300,
)
plt.close()


gdf = (
    df.groupby(["alpha", "checkpoint"])[["real_success", "sim_success"]]
    .mean()
    .reset_index()
)
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 8))
for alpha in sorted(gdf["alpha"].unique()):
    subset = gdf[gdf["alpha"] == alpha]
    plt.plot(
        subset["checkpoint"],
        subset["real_success"],
        label=rf"$\alpha$={alpha:.1f} Real",
        linestyle="--",
    )
    plt.plot(
        subset["checkpoint"],
        subset["sim_success"],
        label=rf"$\alpha$={alpha:.1f} Sim",
        linestyle="-",
    )
plt.xlabel("Checkpoint")
plt.ylabel("Success Rate")
plt.legend(ncol=2, fontsize=8)
plt.title(r"Success Rate vs Training Step for Each $\alpha$")
plt.tight_layout()
plt.savefig(
    f"report/success_rate_vs_training_step_correlation_grouped_by_alpha_{config.model}_{get_task_filename(config.env_name)}.png",
    dpi=300,
)
plt.close()


pivot_real = df.pivot_table(values="real_success", index="alpha", columns="checkpoint")
pivot_sim = df.pivot_table(values="sim_success", index="alpha", columns="checkpoint")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.set_theme(style="darkgrid")
sns.heatmap(pivot_real, ax=axes[0], cmap="YlGnBu", annot=True, fmt=".1f")
axes[0].set_title("Real Success Rate")
axes[0].set_xlabel("Checkpoint")
axes[0].set_ylabel("Alpha")

sns.heatmap(pivot_sim, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".1f")
axes[1].set_title("Simulated Success Rate")
axes[1].set_xlabel("Checkpoint")

fig.suptitle(r"Heatmap of Success Rates by $\alpha$ and Checkpoint")
plt.tight_layout()
plt.savefig(
    f"report/heatmap_success_rate_by_alpha_and_checkpoint_{config.model}_{get_task_filename(config.env_name)}.png",
    dpi=300,
)
plt.close()
