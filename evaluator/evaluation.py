import random
from collections import defaultdict
from typing import Callable

import jax
import numpy as np

from fql.agents.fql import FQLAgent
from task.task import Task


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate_agent(
    agent: FQLAgent,
    env: Task,
    seed: int | None = None,
    eval_temperature: float = 0,
) -> tuple[dict[str, float], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate the agent in the environment.

    Args:
        agent: The agent to evaluate.
        env: The environment to evaluate the agent in.
        seed: Random seed for evaluation.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics and transitions.
    """
    actor_fn = supply_rng(
        agent.sample_actions,
        rng=jax.random.PRNGKey(
            seed if seed is not None else np.random.randint(0, 2**32)
        ),
    )
    return evaluate_actor_fn(actor_fn, env, seed, eval_temperature)


def evaluate_actor_fn(
    actor_fn: Callable[[np.ndarray, float], np.ndarray],
    env: Task,
    seed: int | None = None,
    eval_temperature: float = 0,
) -> tuple[dict[str, float], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate the actor in the environment.

    Args:
        actor_fn: Function to sample actions from the actor.
        env: Environment.
        seed: Random seed for evaluation.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    stats = defaultdict(list)
    observations, infos = env.reset(seed=seed)
    done = np.zeros(len(observations), dtype=bool)

    transitions = []

    while not np.all(done):
        actions = actor_fn(observations=observations, temperature=eval_temperature)
        actions = np.array(actions)
        actions = np.clip(actions, -1, 1)

        next_observations, _, terminated, truncated, infos = env.step(actions)
        mask = np.array([info.get("invalid", False) for info in infos], dtype=bool)
        done = np.logical_or(np.logical_or(terminated, truncated), mask)

        for i, info in enumerate(infos):
            if mask[i] or not done[i]:
                continue
            add_to(stats, flatten(info))

        transitions.append((observations, actions, mask))

        observations = next_observations

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, transitions
