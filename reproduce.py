import random

import numpy as np

from argparser import build_config_from_args, get_args
from hpo.identity import IdentityStrategy
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import ExperimentConfig
from trainer.trainer import Trainer

args = get_args()
config = build_config_from_args(args)

strategy = IdentityStrategy()
task = OfflineTaskWithRealEvaluations(config.buffer_size, config.env_name, config.data_directory)

random.seed(config.seed)
np.random.seed(config.seed)

strategy.populate(
    [
        ExperimentConfig(seed=seed)
        for seed in [8932, 8035, 2463, 4014, 4479, 8443, 2942, 9643]
    ]
)

trainer = Trainer(task, strategy, config)
trainer.train()
