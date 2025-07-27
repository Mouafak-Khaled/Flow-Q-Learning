import pickle
from pathlib import Path

import flax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr

from argparser import build_env_model_config_from_args, get_env_model_argparser
from evaluator.evaluation import evaluate_agent
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations


def load_agent(agent_directory, sample_batch) -> FQLAgent:
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


def get_success_rate(agent, simulated_task) -> float:
    sim_info, _ = evaluate_agent(
        agent=agent,
        env=simulated_task,
    )

    return sim_info["success"]


parser = get_env_model_argparser()
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of evaluation episodes for the environment model.",
)

args = parser.parse_args()
config = build_env_model_config_from_args(args)

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


result_paths = [
    f
    for f in (config.save_directory / config.env_name).glob("seed_*_alpha_*_*")
    if f.is_dir()
]

data = []
for result_path in result_paths:
    config_path = result_path / "config.yaml"
    if not config_path.exists():
        print(f"Skipping {result_path}, no config.yaml found.")
        continue

    with open(config_path, "r") as f:
        agent_config = yaml.unsafe_load(f)

    seed = agent_config.get("seed")
    alpha = agent_config.get("alpha")

    agent = load_agent(
        agent_directory=result_path,
        sample_batch=real_task.sample("train", 1),
    )

    eval_path = result_path / "eval.csv"
    if not eval_path.exists():
        print(f"Skipping {eval_path} file not found.")
        continue

    eval_result = pd.read_csv(eval_path)
    if (
        eval_result.empty
        or "success" not in eval_result.columns
        or eval_result["success"].isna().all()
    ):
        print(f"Skipping {eval_path}: Empty, missing, or no valid 'success' values.")
        continue

    real_success_rate = eval_result["success"].iloc[-1]
    sim_success_rate = get_success_rate(
        agent=agent,
        simulated_task=simulated_task,
    )

    data.append(
        {
            "seed": seed,
            "alpha": alpha,
            "real_success": real_success_rate * 100,
            "sim_success": sim_success_rate * 100,
        }
    )

# data = pd.read_csv("exp/exp_real/cube-single-play-singletask-task2-v0/data.csv")

df = pd.DataFrame(data)

df.to_csv(config.save_directory / config.env_name / f"{config.model}_data.csv")
x = df["real_success"]
y = df["sim_success"]

r, p = pearsonr(x, y)
rho, rho_pval = spearmanr(x, y)
tau, tau_pval = kendalltau(x, y)

output_path = Path("report") / config.env_name
output_path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.scatter(df["real_success"], df["sim_success"], alpha=0.6)
plt.xlabel("Real Success Rate (%)")
plt.ylabel("Simulated Success Rate (%)")
plt.xticks(range(0, 110, 20))
plt.yticks(range(0, 110, 20))
plt.title("Sim vs Real Success Rate")
plt.tight_layout()
plt.savefig(output_path / f"{config.model}_success_rate_correlation.png", dpi=300)
plt.close()


grouped_df = (
    df.groupby("alpha", dropna=True)[["real_success", "sim_success"]]
    .mean()
    .reset_index()
)
grouped_df = grouped_df.rename(
    columns={"real_success": "avg_real_success", "sim_success": "avg_sim_success"}
)
norm = mcolors.LogNorm(
    vmin=grouped_df["alpha"].replace(0, 1e-3).min(), vmax=grouped_df["alpha"].max()
)
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    grouped_df["avg_real_success"],
    grouped_df["avg_sim_success"],
    c=grouped_df["alpha"],
    cmap="viridis",
    norm=norm,
    s=100,
)
plt.xlabel("Avg Real Success Rate (%)")
plt.ylabel("Avg Simulated Success Rate (%)")
cbar = plt.colorbar(sc)
cbar.set_label("Alpha (log scale)")
plt.suptitle(
    "Sim vs Real Success Rate by Alpha"
    + "\n"
    + rf"Pearson's  $r$  = {r:>6.2f} (p = {p:>4.1f})"
    + "\n"
    + rf"Spearman's $\rho$ = {rho:>6.2f} (p = {rho_pval:>4.1f})"
    + "\n"
    + rf"Kendall's  $\tau$ = {tau:>6.2f} (p = {tau_pval:>4.1f})",
    fontsize=12,
)
plt.tight_layout()
plt.xticks(range(0, 110, 20))
plt.yticks(range(0, 110, 20))
plt.savefig(
    output_path / f"{config.model}_success_rate_correlation_grouped_alpha.png", dpi=300
)
