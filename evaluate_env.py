import pickle
from dataclasses import asdict

import flax

from argparser import build_config_from_args, get_argparser
from evaluator.env_model_evaluator import EnvModelEvaluator
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations

parser = get_argparser()

args = parser.parse_args()
config = build_config_from_args(args)
task1 = OfflineTaskWithRealEvaluations(
    config.buffer_size, config.env_name, config.data_directory
)
task2 = OfflineTaskWithRealEvaluations(
    config.buffer_size, config.env_name, config.data_directory
)
example_batch = task1.sample("train", 1)
agent = FQLAgent.create(
    config.agent.seed,
    example_batch["observations"],
    example_batch["actions"],
    asdict(config.agent),
)
checkpoint_path = (
    config.save_directory
    / config.env_name
    / "seed_6890_alpha_1000.0_20250705_015142"
    / "params.pkl"
)
if checkpoint_path.exists():
    with open(checkpoint_path, "rb") as f:
        state_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, state_dict["agent"])

env_evaluator = EnvModelEvaluator(task1, task2, agent)
env_evaluator.evaluate()
task1.close()
task2.close()
