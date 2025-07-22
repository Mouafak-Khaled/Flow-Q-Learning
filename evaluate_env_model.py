import pickle
from dataclasses import asdict

import flax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from argparser import build_config_from_args, get_argparser
from evaluator.evaluation import evaluate
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

real_info, real_transitions = evaluate(
    agent,
    real_task,
    seed=config.seed,
)
real_task.close()

sim_info, sim_transitions = evaluate(
    agent,
    simulated_task,
    seed=config.seed,
)
simulated_task.close()

(
    real_observations,
    real_actions,
    real_next_observations,
    real_terminated,
    real_truncated,
    real_mask,
) = zip(*real_transitions)
(
    sim_observations,
    sim_actions,
    sim_next_observations,
    sim_terminated,
    sim_truncated,
    sim_mask,
) = zip(*sim_transitions)
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
    f"Comparing Trajectories: Model vs Real Environment\nSuccess Rate: $real={100 * real_info['success']:.2f}\\%$, $sim={100 * sim_info['success']:.2f}\\%$"
)
plt.tight_layout()
plt.show()
