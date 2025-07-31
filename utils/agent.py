import pickle

import flax
import yaml

from fql.agents.fql import FQLAgent


def load_agent(
    agent_directory,
    sample_batch,
    agent_filename: str = "params",
    agent_extension: str = ".pkl",
):
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

    agent_path = agent_directory / agent_filename / agent_extension
    if agent_path.exists():
        with open(agent_path, "rb") as f:
            state_dict = pickle.load(f)
        agent = flax.serialization.from_state_dict(agent, state_dict["agent"])
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {agent_path}. Please ensure the agent has been trained."
        )

    return agent
