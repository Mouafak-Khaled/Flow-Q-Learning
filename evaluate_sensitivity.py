import itertools
import random

import numpy as np

from argparser import build_config_from_args, get_argparser
from evaluator.sensitivity_evaluator import SensitivityEvaluator
from trainer.config import ExperimentConfig

# Usage:
# python evaluate_sensitivity.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 \
#        --save_directory=/work/dlclarge2/amriam-fql/exp/ \
#        --number_of_seeds=2 --number_of_alphas=20

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

experiment_configs = [
    ExperimentConfig(seed=seed, alpha=alpha) for alpha, seed in combinations
]

# create evaluator
evaluator = SensitivityEvaluator(experiment_configs, config)

# evaluate the agents
evaluator.evaluate()

# plot the results
evaluator.plot_sensitivity_heatmap()
evaluator.plot_sensitivity_curve()
