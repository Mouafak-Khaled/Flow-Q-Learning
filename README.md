# Efficient Flow Q-Learning

A trainer implementation that tunes the $\alpha$ hyperparameter of the [Flow Q-Learning](https://arxiv.org/abs/2502.02538) model. The trainer starts with a population of values of $\alpha$, and continuously over training eliminates a fraction of the population based on the performance of the trained policy so far. The trainer avoids interaction with the real environment by using a world model trained on the offline dataset.
