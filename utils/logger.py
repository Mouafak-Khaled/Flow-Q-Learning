import os
from dataclasses import asdict
from pathlib import Path

import wandb
from typing_extensions import Literal
import yaml

from fql.utils.log_utils import CsvLogger
from trainer.config import AgentConfig


class Logger:
    """
    Logger class that logs training, validation, and evaluation metrics
    to CSV files and optionally to Weights & Biases (wandb).

    Attributes:
        train_logger (CsvLogger): Logger for training metrics.
        val_logger (CsvLogger): Logger for validation metrics.
        eval_logger (CsvLogger): Logger for evaluation metrics.
        use_wandb (bool): Whether wandb logging is enabled.
    """

    def __init__(
        self,
        save_directory: Path,
        env_name: str,
        experiment_name: str,
        config: AgentConfig,
        use_wandb: bool = False,
    ):
        """
        Initialize the Logger.

        Args:
            save_directory (Path): Directory where logs will be saved.
            env_name (str): Name of the environment.
            experiment_name (str): Name of the experiment.
            config (AgentConfig): Configuration for the agent.
            use_wandb (bool): Whether to enable wandb logging.
        """
        os.makedirs(save_directory / env_name / experiment_name, exist_ok=True)

        self.train_logger = CsvLogger(save_directory / env_name / experiment_name / "train.csv")
        self.val_logger = CsvLogger(save_directory / env_name / experiment_name / "val.csv")
        self.eval_logger = CsvLogger(save_directory / env_name / experiment_name / "eval.csv")

        with open(save_directory / env_name / experiment_name / "config.yaml", "w") as f:
            yaml.dump(asdict(config), f)

        if use_wandb:
            self.wandb_run = wandb.init(
                project="fql",
                name=f"{env_name}_{experiment_name}",
                config=asdict(config),
                dir=save_directory,
                reinit=True,
            )

    def log(self, metrics: dict, step: int, group: Literal["train", "val", "eval"]):
        """
        Log metrics to CSV files in a save_directory and optionally to wandb.

        Args:
            metrics (dict): Dictionary of metrics to log (e.g., {"loss": 0.1}).
            step (int): The current step or epoch.
            group (Literal): One of 'train', 'val', or 'eval' to specify the logging group.
        """
        if group == "train":
            self.train_logger.log(metrics, step=step)
        elif group == "val":
            self.val_logger.log(metrics, step=step)
        elif group == "eval":
            self.eval_logger.log(metrics, step=step)

        if self.wandb_run:
            with wandb.init(
                project=self.wandb_run.project,
                id=self.wandb_run.id,
                resume="must",
                reinit=True
            ) as wandb_run:
                prefixed_metrics = {f"{group}/{k}": v for k, v in metrics.items()}
                wandb_run.log(prefixed_metrics, step=step)

    def close(self):
        """
        Close all CSV loggers and finalize wandb session if enabled.
        """
        self.train_logger.close()
        self.val_logger.close()
        self.eval_logger.close()
