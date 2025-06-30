from hpo.identity import IdentityStrategy
from task.offline_task_real import OfflineTaskWithRealEvaluations
from trainer.config import TrainerConfig
from trainer.trainer import Trainer
from argparser import get_args

config = TrainerConfig()
args = get_args()

config.env_name = args.env_name
config.agent.discount = args.discount
config.buffer_size = args.buffer_size

strategy = IdentityStrategy()

task = OfflineTaskWithRealEvaluations(config.buffer_size, config.env_name)

trainer = Trainer(task, strategy, config)
trainer.train()
