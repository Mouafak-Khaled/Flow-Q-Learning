import random

import numpy as np

from argparser import build_config_from_args, get_argparser
from hpo.identity import IdentityStrategy
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import ExperimentConfig
from trainer.trainer import Trainer

parser = get_argparser()
parser.add_argument(
    "--number_of_seeds",
    type=int,
    default=1,
    help="Number of random seeds to use.",
)
args = parser.parse_args()

config = build_config_from_args(args)

seeds = random.sample(range(10000), args.number_of_seeds)

strategy = IdentityStrategy()
task = OfflineTaskWithRealEvaluations(
    config.buffer_size, config.env_name, config.data_directory
)

random.seed(config.seed)
np.random.seed(config.seed)

strategy.populate([ExperimentConfig(seed=seed) for seed in seeds])

trainer = Trainer(task, strategy, config)
trainer.train()
