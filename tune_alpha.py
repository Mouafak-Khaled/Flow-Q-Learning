from hpo.identity import IdentityStrategy
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import TrainerConfig
from trainer.trainer import Trainer

config = TrainerConfig()

config.env_name = "antsoccer-arena-navigate-singletask-task4-v0"
config.agent.discount = 0.995

strategy = IdentityStrategy()

task = OfflineTaskWithRealEvaluations(config.buffer_size, config.env_name)

trainer = Trainer(task, strategy, config)
trainer.train()
