import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluator.evaluation import evaluate_actor_fn, evaluate_agent
from fql.agents.fql import FQLAgent
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations


class EnvModelEvaluator:
    def __init__(
        self,
        real_task: OfflineTaskWithRealEvaluations,
        simulated_task: OfflineTaskWithSimulatedEvaluations,
        agent: FQLAgent,
        seed: int,
    ):
        self.real_task = real_task
        self.simulated_task = simulated_task
        self.agent = agent
        self.seed = seed

    def evaluate(self) -> None:
        real_info, real_transitions = evaluate_agent(
            self.agent,
            self.real_task,
            seed=self.seed,
        )

        sim_info, sim_transitions = evaluate_agent(
            self.agent,
            self.simulated_task,
            seed=self.seed,
        )

        def generate_actor_fn(transitions):
            idx = -1

            def actor_fn(observations, temperature=0):
                nonlocal idx
                idx += 1 if idx < len(transitions) - 1 else 0
                return transitions[idx][1]

            return actor_fn

        real2sim_info, _ = evaluate_actor_fn(
            generate_actor_fn(real_transitions),
            self.simulated_task,
            seed=self.seed,
        )

        real_observations, _, real_mask = zip(*real_transitions)
        sim_observations, _, sim_mask = zip(*sim_transitions)
        real_observations = np.array(real_observations)
        sim_observations = np.array(sim_observations)

        i = 0
        while not np.any(real_mask[i]) and not np.any(sim_mask[i]):
            i += 1

        mae = np.abs(real_observations[:i] - sim_observations[:i]).mean(axis=2)
        mse = np.square(real_observations[:i] - sim_observations[:i]).mean(axis=2)

        def get_stats(arr):
            mean = np.mean(arr, axis=1)
            min_ = np.min(arr, axis=1)
            max_ = np.max(arr, axis=1)
            return mean, min_, max_

        mse_mean, mse_min, mse_max = get_stats(mse)
        mae_mean, mae_min, mae_max = get_stats(mae)

        steps = np.arange(mse.shape[0])

        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(8, 6))

        plt.plot(steps, mse_mean, label="Mean Squared Error")
        plt.fill_between(steps, mse_min, mse_max, alpha=0.2)

        plt.plot(steps, mae_mean, label="Mean Absolute Error")
        plt.fill_between(steps, mae_min, mae_max, alpha=0.2)

        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.legend()

        plt.title(
            "Comparing Trajectories: Model vs Real Environment\n"
            f"Success Rate: $real={100 * real_info['success']:.2f}\\%$, $sim={100 * sim_info['success']:.2f}\\%$\n"
            f"$real2sim={100 * real2sim_info.get('success', 0):.2f}\\%$"
        )
        plt.tight_layout()
        plt.show()

    def close(self):
        self.real_task.close()
        self.simulated_task.close()
