import pickle
from pathlib import Path

import flax
from tqdm import tqdm

from fql.utils.evaluation import evaluate


class Experiment:
    def __init__(
        self, agent, task, steps: int = 1000000, log_interval: int = 5000, logger=None
    ):
        """
        Initialize the experiment with an agent and a task.

        Args:
            agent: The agent to be trained.
            task: The task environment for training and evaluation.
            steps: Total number of training steps.
            log_interval: Interval for logging metrics.
            logger: Optional logger for logging metrics.
        """
        self.agent = agent
        self.task = task
        self.steps = steps
        self.log_interval = log_interval
        self.logger = logger
        self.current_step = 0

    def train(self, num_steps: int) -> None:
        for _ in tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
            self.current_step += 1
            batch = self.task.sample("train", self.agent.config["batch_size"])
            self.agent, update_info = self.agent.update(batch)

            if self.logger and self.current_step % self.log_interval == 0:
                train_metrics = {f"training/{k}": v for k, v in update_info.items()}
                self.logger.log(train_metrics, step=self.current_step, group="train")

                val_batch = self.task.sample("val", self.agent.config["batch_size"])
                _, val_info = self.agent.total_loss(val_batch, grad_params=None)
                val_metrics = {f"validation/{k}": v for k, v in val_info.items()}
                self.logger.log(val_metrics, step=self.current_step, group="val")

    def evaluate(self, num_episodes: int = 50) -> float:
        eval_metrics = {}
        eval_info, _, _ = evaluate(
            agent=self.agent,
            env=self.task,
            config=self.agent.config,
            num_eval_episodes=num_episodes,
        )
        for k, v in eval_info.items():
            eval_metrics[f"evaluation/{k}"] = v

        if self.logger:
            self.logger.log(eval_metrics, step=self.current_step, group="eval")

        return eval_info["success"]

    def save_agent(self, save_directory: Path):
        """
        Save the agent's state to a file.

        Args:
            save_directory: Directory where the agent will be saved.
            step: Current training step.
        """
        save_dict = dict(
            agent=flax.serialization.to_state_dict(self.agent),
        )
        save_path = save_directory / f"params_{self.current_step}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

    def stop(self, eval_mode: bool = False):
        """
        Stop the experiment.
        """
        if eval_mode:
            self.stopped_at = self.current_step
        else:
            if self.logger:
                self.logger.close()
