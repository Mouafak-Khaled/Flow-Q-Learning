import itertools
import random

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from argparser import build_config_from_args, get_argparser
from evaluator.strategy_evaluator import StrategyEvaluator
from hpo.successive_halving import SuccessiveHalving
from trainer.config import ExperimentConfig
from utils.tasks import get_task_filename

# Usage:
# python evaluate_strategy.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 \
#         --save_directory=/work/dlclarge2/amriam-fql/exp/ \
#         --number_of_seeds=2 --number_of_alphas=20 --eval_interval=20000
# The values for `--number_of_seeds`, `--number_of_alphas`, and `--eval_interval` in addition to the seed
# have to be the same as the ones used in the training script.

# read arguments from command line
parser = get_argparser()

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

args = parser.parse_args()
config = build_config_from_args(args)

# generate experiment configurations
random.seed(config.seed)
np.random.seed(config.seed)

alpha_values = np.logspace(
    np.log10(3), np.log10(1000), num=args.number_of_alphas
).tolist()
seeds = random.sample(range(10000), args.number_of_seeds)

combinations = list(itertools.product(alpha_values, seeds))

fractions = [0.25, 0.5, 0.75]
history_lengths = [1, 2, 4, 8]
score_curve = {}
for fraction, history_length in itertools.product(fractions, history_lengths):
    experiment_configs = [
        ExperimentConfig(seed=seed, alpha=alpha) for alpha, seed in combinations
    ]

    # create evaluator
    strategy = SuccessiveHalving(
        population=experiment_configs,
        total_evaluations=config.steps // config.eval_interval,
        fraction=fraction,
        history_length=history_length,
    )

    evaluator = StrategyEvaluator(strategy, config)

    # evaluate the agents
    score_curve[(fraction, history_length)] = evaluator.evaluate()

    # plot the results
    # evaluator.plot()

rows = []
for (fraction, history_length), step_scores in score_curve.items():
    for step, score in step_scores.items():
        rows.append({
            "fraction": fraction,
            "history_length": history_length,
            "step": step,
            "score": score
        })

df = pd.DataFrame(rows)

# Create a unique label for each (fraction, history_length) pair
df["label"] = df.apply(lambda row: f"({row['fraction']}, {row['history_length']})", axis=1)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="step", y="score", hue="label", marker="o", style="label", drawstyle="steps-post")

plt.title("Comparison of Successive Halving Strategies")
plt.xlabel("Step")
plt.ylabel("Best Score")
plt.legend(title="(fraction, history_length)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"report/successive_halving_comparison_{get_task_filename(config.env_name)}.png", dpi=300)
plt.close()
