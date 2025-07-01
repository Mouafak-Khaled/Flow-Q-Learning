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

strategy.populate([ExperimentConfig(alpha=alpha, normalize_q_loss=True) for alpha in [0.03, 0.1, 0.3, 1, 3, 10]])

trainer = Trainer(task, strategy, config)
trainer.train()
