# Efficient Flow Q-Learning

A trainer implementation that tunes the $\alpha$ hyperparameter of the [Flow Q-Learning](https://arxiv.org/abs/2502.02538) model. The trainer starts with a population of values of $\alpha$, and continuously over training eliminates a fraction of the population based on the performance of the trained policy so far. The trainer avoids interaction with the real environment by using a world model trained on the offline dataset.

## Installation

Follow the installation script at [scripts/setup.sh](scripts/setup.sh) to setup the environment. If you're on KISLURM, you can just submit the script to the queue.

> [!NOTE]
> After you do all the steps in the installation script (or after the script finished on KISLURM), you still need to do one final step, that is to login to `wandb`.
> Use `wandb login` (make sure you're in `fql` conda environment), and then paste your API key that you get from your [wandb.ai](https://wandb.ai/) account.

