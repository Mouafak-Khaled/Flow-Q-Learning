# import itertools

# from absl import app, flags
# from ml_collections import config_flags

# from run import run_experiment
# from task.offline_task_real import OfflineTaskWithRealEvaluations

# FLAGS = flags.FLAGS

# flags.DEFINE_integer('task_id', 0, 'SLURM task ID.')

# flags.DEFINE_string('run_group', 'Debug', 'Run group.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
# flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

# flags.DEFINE_integer('steps', 1000000, 'Number of offline steps.')
# flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
# flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
# flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
# flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')

# config_flags.DEFINE_config_file('agent', 'fql/agents/fql.py', lock_config=False)

# def main(_):
#     grid = {
#         'seed': [ 8932, 8035, 2463, 4014, 4479, 8443, 2942, 9643 ]
#     }

#     values = (grid[k] for k in grid.keys())
#     combinations = list(itertools.product(*values))

#     task_id = FLAGS.task_id
#     assert 0 <= task_id < len(combinations), "Invalid SLURM_ARRAY_TASK_ID"

#     seed, = combinations[task_id]

#     FLAGS.seed = seed

#     task = OfflineTaskWithRealEvaluations(FLAGS.buffer_size, FLAGS.env_name)
#     run_experiment(FLAGS, task)


# if __name__ == '__main__':
#     app.run(main)


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

strategy.init(
    [
        ExperimentConfig(seed=seed)
        for seed in [8932, 8035, 2463, 4014, 4479, 8443, 2942, 9643]
    ]
)

trainer = Trainer(task, strategy, config)
trainer.train()
