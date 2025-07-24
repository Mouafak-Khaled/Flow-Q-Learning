import pickle

import flax
import yaml

from argparser import get_env_model_argparser, build_env_model_config_from_args
from evaluator.env_model_evaluator import EnvModelEvaluator
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations


def load_agent(agent_directory, sample_batch):
    config_path = agent_directory / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            agent_config = yaml.unsafe_load(f)
    else:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. Please ensure the configuration has been set."
        )

    agent = FQLAgent.create(
        agent_config["seed"],
        sample_batch["observations"],
        sample_batch["actions"],
        agent_config,
    )

    agent_path = agent_directory / "params.pkl"
    if agent_path.exists():
        with open(agent_path, "rb") as f:
            state_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, state_dict["agent"])
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {agent_path}. Please ensure the agent has been trained."
        )

    return agent


parser = get_env_model_argparser()
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of evaluation episodes for the environment model.",
)

args = parser.parse_args()
config = build_env_model_config_from_args(args)

real_task = OfflineTaskWithRealEvaluations(
    config.env_name,
    data_directory=config.data_directory,
    num_evaluation_envs=args.eval_episodes,
)

simulated_task = OfflineTaskWithSimulatedEvaluations(
    config.env_name,
    model=config.model,
    data_directory=config.data_directory,
    save_directory=config.save_directory,
    num_evaluation_envs=args.eval_episodes,
)

agent = load_agent(
    agent_directory=config.save_directory / config.env_name / "best_run",
    sample_batch=real_task.sample("train", 1),
)

env_model_evaluator = EnvModelEvaluator(
    real_task=real_task,
    simulated_task=simulated_task,
    agent=agent,
    seed=config.seed,
)

env_model_evaluator.evaluate()

env_model_evaluator.close()
