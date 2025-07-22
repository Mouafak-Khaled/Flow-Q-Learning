import pickle
from dataclasses import asdict

import flax

from argparser import build_config_from_args, get_argparser
from evaluator.env_model_evaluator import EnvModelEvaluator
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations

parser = get_argparser()

args = parser.parse_args()
config = build_config_from_args(args)

real_task = OfflineTaskWithRealEvaluations(
    config.buffer_size,
    config.env_name,
    config.data_directory,
    num_evaluation_envs=config.eval_episodes,
)

example_batch = real_task.sample("train", 1)
agent = FQLAgent.create(
    config.agent.seed,
    example_batch["observations"],
    example_batch["actions"],
    asdict(config.agent),
)

simulated_task = OfflineTaskWithSimulatedEvaluations(
    config.env_name,
    config.buffer_size,
    config.data_directory,
    config.save_directory,
    num_evaluation_envs=config.eval_episodes,
)

checkpoint_path = config.save_directory / config.env_name / "agent.pkl"
if checkpoint_path.exists():
    with open(checkpoint_path, "rb") as f:
        state_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, state_dict["agent"])

env_model_evaluator = EnvModelEvaluator(
    real_task=real_task,
    simulated_task=simulated_task,
    agent=agent,
    seed=config.agent.seed,
)

env_model_evaluator.compare_trajectories()

env_model_evaluator.close()
