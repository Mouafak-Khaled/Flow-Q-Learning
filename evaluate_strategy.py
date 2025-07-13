import itertools
import random

import numpy as np

from argparser import build_config_from_args, get_argparser
from evaluator.strategy_evaluator import StrategyEvaluator
from hpo.successive_halving import SuccessiveHalving
from trainer.config import ExperimentConfig

# read arguments from command line
parser = get_argparser()

parser.add_argument(
    "--strategy",
    type=str,
    default="successive_halving",
    help="The hyperparameter optimization strategy to evaluate.",
)
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
    "--fraction",
    type=float,
    default=0.5,
    help="Fraction of the population to keep after each evaluation.",
)

parser.add_argument(
    "--history_length",
    type=int,
    default=1,
    help="Length of the history to consider for the strategy.",
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
strategy = SuccessiveHalving(
    population=experiment_configs,
    total_evaluations=0,
    fraction=args.fraction,
    history_length=args.history_length,
)

evaluator = StrategyEvaluator(strategy, config)

# evaluate the agents
evaluator.evaluate()


evaluator.plot()
