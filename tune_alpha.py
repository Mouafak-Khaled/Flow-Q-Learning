import itertools
import pickle
import random

import numpy as np

from argparser import build_config_from_args, get_argparser
from hpo.identity import Identity
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import ExperimentConfig
from trainer.experiment import Experiment
from trainer.trainer import Trainer

# read arguments from command line
parser = get_argparser()

parser.add_argument(
    "--max_evaluations",
    type=int,
    default=200,
    help="Maximum number of evaluations.",
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

task = OfflineTaskWithRealEvaluations(
    config.buffer_size, config.env_name, config.data_directory, num_evaluation_envs=config.eval_episodes
)

if args.single_experiment:
    alpha, seed = combinations[args.job_id]
    experiment_config = ExperimentConfig(seed=seed, alpha=alpha)
    experiment = Experiment(task, config, experiment_config)
    done = False
    while not done:
        done = experiment.train(config.eval_interval)
        experiment.evaluate()
else:
    experiment_configs = [
        ExperimentConfig(seed=seed, alpha=alpha) for alpha, seed in combinations
    ]

    # load checkpoint if available
    state_dict = {}
    checkpoint_path = config.save_directory / config.env_name / "checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            state_dict = pickle.load(f)

    # create trainer
    strategy = Identity(
        population=experiment_configs,
        total_evaluations=0,
        state_dict=state_dict.get("strategy"),
    )
    trainer = Trainer(task, strategy, config, state_dict=state_dict.get("trainer"))

    # train the agents
    trainer.train(max_evaluations=args.max_evaluations)

    # save the checkpoint
    state_dict = {
        "trainer": trainer.state_dict(),
        "strategy": trainer.strategy.state_dict(),
    }
    with open(config.save_directory / config.env_name / "checkpoint.pkl", "wb") as f:
        pickle.dump(state_dict, f)
