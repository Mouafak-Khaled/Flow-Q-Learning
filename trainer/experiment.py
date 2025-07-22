import pickle
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import flax
from tqdm import tqdm

from evaluator.evaluation import evaluate
from fql.agents.fql import FQLAgent
from task.task import Task
from trainer.config import ExperimentConfig, TrainerConfig
from utils.logger import Logger


class Experiment:
    def __init__(
        self,
        task: Task,
        trainer_config: TrainerConfig,
        experiment_config: ExperimentConfig,
        state_dict: dict | None = None,
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

        example_batch = task.sample("train", 1)
        self.agent = FQLAgent.create(
            agent_config.seed,
            example_batch["observations"],
            example_batch["actions"],
            asdict(agent_config),
        )

        if state_dict is None:
            self.experiment_name = "_".join(
                f"{field}_{value}" for field, value in tuned_hyperparams
            )
            self.experiment_name = (
                f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            self.current_step = 0
        else:
            self.agent = flax.serialization.from_state_dict(
                self.agent, state_dict["agent"]
            )
            self.experiment_name = state_dict["experiment_name"]
            self.current_step = state_dict["current_step"]

        self.logger = Logger(
            trainer_config.save_directory,
            trainer_config.env_name,
            self.experiment_name,
            agent_config,
            use_wandb=trainer_config.use_wandb,
            state_dict=state_dict["logger"] if state_dict else None,
        )

        self.task = task
        self.steps = trainer_config.steps
        self.log_interval = trainer_config.log_interval
        self.save_directory = trainer_config.save_directory
        self.env_name = trainer_config.env_name

    def state_dict(self):
        """
        Get the state dictionary of the experiment.

        Returns:
            dict: State dictionary containing the experiment's state.
        """
        return {
            "experiment_name": self.experiment_name,
            "logger": self.logger.state_dict(),
            "agent": flax.serialization.to_state_dict(self.agent),
            "current_step": self.current_step,
        }

    def train(self, num_steps: int) -> bool:
        """Trains the agent for a specified number of steps.

        Args:
            num_steps (int): Number of training steps to perform.

        Returns:
            bool: True if training completed, False if stopped before reaching total steps.
        """
        num_steps = min(num_steps, self.steps - self.current_step)
        for _ in tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
            self.current_step += 1
            batch = self.task.sample("train", self.agent.config["batch_size"])
            self.agent, update_info = self.agent.update(batch)

            if self.logger and self.current_step % self.log_interval == 0:
                self.logger.log(update_info, step=self.current_step, group="train")

                val_batch = self.task.sample("val", self.agent.config["batch_size"])
                _, val_info = self.agent.total_loss(val_batch, grad_params=None)
                self.logger.log(val_info, step=self.current_step, group="val")

        return self.current_step == self.steps

    def evaluate(self) -> float:
        eval_info, _ = evaluate(agent=self.agent, env=self.task)

        if self.logger:
            self.logger.log(eval_info, step=self.current_step, group="eval")

        return eval_info["success"]

    def save_agent(self):
        """
        Save the agent's state to a file.
        """
        save_dict = dict(
            agent=flax.serialization.to_state_dict(self.agent),
        )
        save_path = (
            self.save_directory / self.env_name / self.experiment_name / "params.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

    def stop(self):
        """
        Stop the experiment.
        """
        self.logger.close()
