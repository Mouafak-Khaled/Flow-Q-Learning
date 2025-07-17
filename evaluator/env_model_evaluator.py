import random
from collections import defaultdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fql.agents.fql import FQLAgent
from task.task import Task

# TODO: refactor this to use the new vectorized task interface
#       this doesn't work at the moment.
class EnvModelEvaluator:
    def __init__(self, reference: Task, evaluated: Task, agent: FQLAgent):
        self.reference = reference
        self.evaluated = evaluated
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
            for i in range(100):
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
                if done:
                    break
            return task_transitions

        reference_observations = []
        evaluated_observations = []
        max_size = 1000
        for i in range(100):
            seed = np.random.randint(0, 2**32)

            reference_transitions = rollout(self.reference, seed)
            evaluated_transitions = rollout(self.evaluated, seed)

            max_size_temp = min(
                len(reference_transitions["observation"]),
                len(evaluated_transitions["observation"]),
            )

            if max_size_temp < 70:
                continue

            max_size = min(max_size, max_size_temp)
            reference_observations.append(
                np.array(reference_transitions["observation"])
            )
            evaluated_observations.append(
                np.array(evaluated_transitions["observation"])
            )

        mse = []
        mae = []
        for i in range(len(reference_observations)):
            mse.append(
                np.sum(
                    (
                        reference_observations[i][:max_size]
                        - evaluated_observations[i][:max_size]
                    )
                    ** 2,
                    axis=1,
                )
            )
            mae.append(
                np.sum(
                    np.abs(
                        reference_observations[i][:max_size]
                        - evaluated_observations[i][:max_size]
                    ),
                    axis=1,
                )
            )

        mse = np.stack(mse)
        mae = np.stack(mae)

        def get_stats(arr):
            mean = np.mean(arr, axis=0)
            min_ = np.min(arr, axis=0)
            max_ = np.max(arr, axis=0)
            return mean, min_, max_

        mse_mean, mse_min, mse_max = get_stats(mse)
        mae_mean, mae_min, mae_max = get_stats(mae)

        steps = np.arange(mse.shape[1])

        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(8, 6))

        plt.plot(steps, mse_mean, label="Mean Squared Error")
        plt.fill_between(steps, mse_min, mse_max, alpha=0.2)

        plt.plot(steps, mae_mean, label="Mean Absolute Error")
        plt.fill_between(steps, mae_min, mae_max, alpha=0.2)

        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.legend()

        plt.title("Comparing Trajectories: Model vs Real Environment")
        plt.tight_layout()
        plt.show()


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped
