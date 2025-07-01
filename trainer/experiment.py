import pickle
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import flax
from tqdm import tqdm

from fql.agents.fql import FQLAgent
from fql.utils.evaluation import evaluate
from task.task import Task
from trainer.config import ExperimentConfig, TrainerConfig
from utils.logger import Logger


class Experiment:
    def __init__(
        self,
        task: Task,
        trainer_config: TrainerConfig,
        experiment_config: ExperimentConfig,
    ):
        """
        Initialize the experiment with an agent and a task.

        Args:
            task (Task): The task to be solved by the agent.
            trainer_config (TrainerConfig): Configuration for the trainer.
            experiment_config (ExperimentConfig): Experiment-specific configurations overriding the default ones.
        """
        tuned_hyperparams = [
            (field, value)
            for field, value in vars(experiment_config).items()
            if value is not None
        ]

        agent_config = deepcopy(trainer_config.agent)
        for field, value in tuned_hyperparams:
            if hasattr(agent_config, field):
                setattr(agent_config, field, value)
        self.experiment_name = "_".join(
            f"{field}_{value}" for field, value in tuned_hyperparams
        )
        self.experiment_name = (
            f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        example_batch = task.sample("train", 1)
        self.agent = FQLAgent.create(
            agent_config.seed,
            example_batch["observations"],
            example_batch["actions"],
            asdict(agent_config),
        )

        self.logger = Logger(
            trainer_config.save_directory,
            trainer_config.env_name,
            self.experiment_name,
            agent_config,
            use_wandb=trainer_config.use_wandb,
        )

        self.task = task
        self.steps = trainer_config.steps
        self.log_interval = trainer_config.log_interval
        self.save_directory = trainer_config.save_directory
        self.env_name = trainer_config.env_name

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

    def save_agent(self):
        """
        Save the agent's state to a file.

        Args:
            save_directory: Directory where the agent will be saved.
            step: Current training step.
        """
        save_dict = dict(
            agent=flax.serialization.to_state_dict(self.agent),
        )
        save_path = self.save_directory / self.env_name / self.experiment_name / f"params_{self.current_step}.pkl"
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
