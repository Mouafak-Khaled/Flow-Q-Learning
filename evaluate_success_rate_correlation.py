import itertools
import pickle
import random

import flax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr

from argparser import build_env_model_config_from_args, get_env_model_argparser
from evaluator.env_model_evaluator import EnvModelEvaluator
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations
from trainer.config import ExperimentConfig
from utils.tasks import get_task_title


def load_agent(agent_directory, sample_batch):
    config_path = agent_directory / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            agent_config = yaml.unsafe_load(f)
    else:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. Please ensure the configuration has been set."
        )

    agent = FQLAgent.create(
        agent_config["seed"],
        sample_batch["observations"],
        sample_batch["actions"],
        agent_config,
    )

    agent_path = agent_directory / "params.pkl"
    if agent_path.exists():
        with open(agent_path, "rb") as f:
            state_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, state_dict["agent"])
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {agent_path}. Please ensure the agent has been trained."
        )

    return agent


# Usage:
# python evaluate_success_rate_correlation.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 \
#        --save_directory=/work/dlclarge2/amriam-fql/exp/ \
#        --number_of_seeds=2 --number_of_alphas=20 --eval_episodes=50 \
#        --model=baseline
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
    "--model",
    type=str,
    default="baseline",
    help="The environment model to evaluate.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of evaluation episodes.",
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
    model=args.model,
    data_directory=config.data_directory,
    save_directory=config.save_directory,
    num_evaluation_envs=args.eval_episodes,
)

df = pd.DataFrame(columns=["seed", "alpha", "real_success", "sim_success"])

for experiment_config in experiment_configs:
    agent = load_agent(
        agent_directory=config.save_directory
        / config.env_name
        / f"seed_{experiment_config.seed}_alpha_{experiment_config.alpha}",
        sample_batch=real_task.sample("train", 1),
    )

    evaluator = EnvModelEvaluator(
        real_task=real_task,
        simulated_task=simulated_task,
        agent=agent,
        seed=config.seed,
    )

    real_success, sim_success = evaluator.evaluate()
    evaluator.close()
    df.loc[len(df)] = [
        experiment_config.seed,
        experiment_config.alpha,
        real_success,
        sim_success,
    ]

x = df["real_success"]
y = df["sim_success"]

r, p = pearsonr(x, y)
rho, rho_pval = spearmanr(x, y)
tau, tau_pval = kendalltau(x, y)

output_path = config.report_directory / config.env_name
output_path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.scatter(df["real_success"], df["sim_success"], alpha=0.6)
plt.xlabel("Real Success Rate (%)")
plt.ylabel("Simulated Success Rate (%)")
plt.xticks(range(-10, 110, 20))
plt.yticks(range(-10, 110, 20))
plt.title(
    f"Simulated and Real Success Rate Correlation for {get_task_title(config.env_name)} task"
)
plt.tight_layout()
plt.savefig(output_path / f"{args.model}_success_rate_correlation.png", dpi=300)
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
plt.figure(figsize=(10, 6))
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
    output_path / f"{args.model}_success_rate_correlation_grouped_alpha.png", dpi=300
)
plt.close()
