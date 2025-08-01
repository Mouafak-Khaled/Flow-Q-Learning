import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

from argparser import build_env_model_config_from_args, get_env_model_argparser
from evaluator.evaluation import evaluate_agent
from success_rate_gaussian_process import SuccessRateGaussianProcess
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations
from utils.agent import load_agent
from utils.tasks import get_task_filename, get_task_title

# Usage:
# python evaluate_success_rate_correlation.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --model=baseline \
#        --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
#        --seed=42 --eval_episodes=50

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
    step = int(row["step"])
    seed = int(row["seed"])
    alpha = row["alpha"]

    exp_path = (
        config.save_directory.parent / "exp-checkpointing" / config.env_name
    ).glob(f"seed_{seed}_alpha_{alpha}_*")

    exp_path = next(exp_path, None)

    if exp_path is None:
        print(f"No experiment found for seed {seed} and alpha {alpha}.")
        return None

    print(f"Loading experiment from {exp_path}/checkpoint_{step}.pkl")

    agent = load_agent(
        agent_directory=exp_path,
        sample_batch=simulated_task.sample("train", 1),
        agent_filename=f"checkpoint_{step}",
        agent_extension=".pkl",
    )
    info, _ = evaluate_agent(agent=agent, env=simulated_task)
    return info["success"]


initial_sample = real_success_rates.sample(n=20, random_state=config.seed).copy()
simulated_success_rates = initial_sample.apply(evaluate, axis=1)

gp = SuccessRateGaussianProcess(
    real_success_rates=real_success_rates,
    simulated_success_rates=simulated_success_rates,
    seed=config.seed,
)

for i in range(30):
    row = gp.ask()
    gp.tell(evaluate(row))

gp.plot(
    filename=f"report/success_rate_gp_{config.model}_{get_task_filename(config.env_name)}.gif"
)

simulated_task.close()

df = gp.get_data()
x = df["success"]
y = df["simulated_success"]

r, p = pearsonr(x, y)
rho, rho_pval = spearmanr(x, y)
tau, tau_pval = kendalltau(x, y)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y)
plt.xlabel("Success Rate on Real Environment")
plt.ylabel("Success Rate on Simulated Environment")
plt.title(
    f"Success Rate Correlation between Real Environment ({get_task_title(config.env_name)}) and Simulated Environment ({config.model})\n"
    f"Pearson r={r:.2f} (p={p:.2g}), Spearman ρ={rho:.2f} (p={rho_pval:.2g}), Kendall τ={tau:.2f} (p={tau_pval:.2g})"
)
plt.tight_layout()
plt.show()
