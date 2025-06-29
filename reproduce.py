import itertools

from absl import app, flags
from ml_collections import config_flags

from run import run_experiment 

FLAGS = flags.FLAGS

flags.DEFINE_integer('task_id', 0, 'SLURM task ID.')

flags.DEFINE_string('run_group', 'Debug', 'Run group.')

flags.DEFINE_string('dataset_dir', './data', 'The directory where we store the data.')
flags.DEFINE_string('logging_dir', './logging', 'The directory where we store the logs of weights and bias')

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antsoccer-arena-navigate-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'fql/agents/fql.py', lock_config=False)

def main(_):
    grid = {
        'task': [ 'antsoccer-arena-navigate-singletask-task4-v0', 'antsoccer-arena-navigate-singletask-task2-v0' ],
        'agent_discount': [ 0.995 ],
        'agent_alpha': [ 10.0 ],
        'seed': [ 8932, 8035, 2463, 4014, 4479, 8443, 2942, 9643 ]
    }

    values = (grid[k] for k in grid.keys())
    combinations = list(itertools.product(*values))

    task_id = FLAGS.task_id
    assert 0 <= task_id < len(combinations), "Invalid SLURM_ARRAY_TASK_ID"

    task, discount, alpha, seed = combinations[task_id]

    FLAGS.env_name = task
    FLAGS.agent.discount = discount
    FLAGS.agent.alpha = alpha
    FLAGS.seed = seed

    run_experiment(FLAGS)


if __name__ == '__main__':
    app.run(main)
