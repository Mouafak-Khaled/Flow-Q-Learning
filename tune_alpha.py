import itertools


from hpo.identity import IdentityStrategy
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import ExperimentConfig, TrainerConfig
from trainer.trainer import Trainer

config = TrainerConfig()

grid = {"alpha": [10.0, 300.0, 1000.0]}

values = (grid[k] for k in grid.keys())
combinations = list(itertools.product(*values))

config.env_name = "antsoccer-arena-navigate-singletask-task4-v0"
config.agent.discount = 0.995

strategy = IdentityStrategy()
strategy.init([ExperimentConfig(alpha=alpha) for (alpha,) in combinations])

task = OfflineTaskWithRealEvaluations(config.buffer_size, config.env_name)

trainer = Trainer(task, strategy, config)
trainer.train()
