import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import bootstrap
import argparse

parser = argparse.ArgumentParser(description="Visualize WandB evaluation success statistics.")
parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
parser.add_argument("--project", type=str, required=True, help="WandB project name")
args = parser.parse_args()

api = wandb.Api()
runs = api.runs(f"{args.entity}/{args.project}")

# env_name -> step index -> list of success values
env_step_success = defaultdict(lambda: defaultdict(list))

# Collect data
for run in runs:
    config = run.config
    env_name = config.get("env_name")
    if not env_name:
        continue

    try:
        history = run.history(keys=["evaluation/success"], pandas=False)
        for step_idx, step in enumerate(history):
            if "evaluation/success" in step:
                success = step["evaluation/success"] * 100  # convert to percentage
                env_step_success[env_name][step_idx].append(success)
    except Exception as e:
        print(f"Error fetching history for run {run.id}: {e}")

# Compute stats
def compute_stats(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if len(data) > 1:
        res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_low, ci_high = res.confidence_interval
    else:
        ci_low = ci_high = mean
    return mean, std, ci_low, ci_high

# Plot each environment
for env, steps in env_step_success.items():
    step_idxs = sorted(steps.keys())
    means, stds, ci_lows, ci_highs = [], [], [], []

    for idx in step_idxs:
        mean, std, ci_low, ci_high = compute_stats(steps[idx])
        means.append(mean)
        stds.append(std)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    with open("report/success_stats.csv", "a+") as f:
        print(f"{env}, {means[-1]:.2f}, {stds[-1]:.2f}", file=f)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(step_idxs, means, label="Mean Success", color='blue')
    plt.fill_between(step_idxs, ci_lows, ci_highs, alpha=0.3, color='blue', label="95% CI")
    plt.title(f"Evaluation Success - {env}")
    plt.xlabel("Step")
    plt.ylabel("Success (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"report/{env}_success_plot.png")
