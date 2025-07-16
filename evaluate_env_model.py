import pickle
from dataclasses import asdict

import flax
import jax

from argparser import build_config_from_args, get_argparser
from envmodel.baseline import BaselineEnvModel
from evaluator.env_model_evaluator import EnvModelEvaluator
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations

parser = get_argparser()

args = parser.parse_args()
config = build_config_from_args(args)

real_task = OfflineTaskWithRealEvaluations(
    config.buffer_size, config.env_name, config.data_directory, num_evaluation_envs=config.eval_episodes
)

example_batch = real_task.sample("train", 1)
agent = FQLAgent.create(
    config.agent.seed,
    example_batch["observations"],
    example_batch["actions"],
    asdict(config.agent),
)

env_model = BaselineEnvModel(
    obs_dim=example_batch["observations"].shape[-1],
    act_dim=example_batch["actions"].shape[-1],
    hidden_size=128,
)
params = env_model.init(
    jax.random.PRNGKey(0),
    example_batch["observations"],
    example_batch["actions"]
)
env_model_path = (
    config.save_directory
    / config.env_name
    / "env_models"
    / "baseline.pt"
)
if env_model_path.exists():
    with open(env_model_path, "rb") as f:
        params_bytes = f.read()
params = flax.serialization.from_bytes(params, params_bytes)
simulated_task = OfflineTaskWithSimulatedEvaluations(
    env_model, params, config.env_name, config.buffer_size, config.data_directory
)

checkpoint_path = (
    config.save_directory
    / config.env_name
    / "agent.pkl"
)
if checkpoint_path.exists():
    with open(checkpoint_path, "rb") as f:
        state_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, state_dict["agent"])

env_evaluator = EnvModelEvaluator(real_task, simulated_task, agent)
env_evaluator.evaluate()
real_task.close()
simulated_task.close()
