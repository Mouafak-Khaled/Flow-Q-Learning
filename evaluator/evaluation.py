from collections import defaultdict

import jax
import numpy as np


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    stats = defaultdict(list) 

    observations, infos = env.reset()
    done = np.zeros(len(observations), dtype=bool)

    while not np.all(done):
        actions = actor_fn(observations=observations, temperature=eval_temperature)
        actions = np.array(actions)
        actions = np.clip(actions, -1, 1)

        next_observations, _, terminated, truncated, infos = env.step(actions)
        done = np.logical_or(terminated, truncated)

        for i, info in enumerate(infos):
            if info.get('invalid', False) or not done[i]:
                continue
            add_to(stats, flatten(info))

        observations = next_observations

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
