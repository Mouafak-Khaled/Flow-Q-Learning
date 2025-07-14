import random
from collections import defaultdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fql.agents.fql import FQLAgent
from task.task import Task


class EnvModelEvaluator:
    def __init__(self, task1: Task, task2: Task, agent: FQLAgent):
        self.task1 = task1
        self.task2 = task2
        self.agent = agent

    def evaluate(self):
        """Evaluate the agent in both task environments."""

        def rollout(task, seed):
            actor_fn = supply_rng(
                self.agent.sample_actions,
                rng=jax.random.PRNGKey(seed),
            )
            random.seed(seed)
            np.random.seed(seed)
            task_transitions = defaultdict(list)
            observation, info = task.reset(seed=seed)
            done = False
            while not done:
                action = actor_fn(observations=observation, temperature=0)
                action = np.array(action)
                action = np.clip(action, -1, 1)

                next_observation, reward, terminated, truncated, info = task.step(
                    action
                )
                done = terminated or truncated

                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
                for k, v in transition.items():
                    task_transitions[k].append(v)
                observation = next_observation
            return task_transitions

        seed = np.random.randint(0, 2**32)

        task1_transitions = rollout(self.task1, seed)
        task2_transitions = rollout(self.task2, seed)

        task1_observations = np.array(task1_transitions["observation"])
        task2_observations = np.array(task2_transitions["observation"])

        max_size = min(task1_observations.shape[0], task2_observations.shape[0])
        squared_diff = np.sum(
            (task1_observations[:max_size] - task2_observations[:max_size]) ** 2, axis=1
        )

        plt.figure(figsize=(8, 6))
        sns.set_theme(style="darkgrid")
        sns.lineplot(data=squared_diff, label="Squared Differences")
        plt.xlabel("Step")
        plt.ylabel("Squared Error")
        plt.title("Comparison of the Environment and the Model Trajectories")
        plt.show()


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped
