import os
from dataclasses import asdict
from pathlib import Path

import wandb
from typing_extensions import Literal
import yaml

from trainer.config import AgentConfig


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step=None):
        if step is not None:
            row['step'] = step
    
        if self.file is None:
            self.file = open(self.path, 'w')

        if self.header is None:
            self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
            self.file.write(','.join(self.header) + '\n')

        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')

        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


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
        state_dict: dict | None = None,
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
            kwargs = dict(
                project="fql",
                name=f"{env_name}_{experiment_name}",
                config=asdict(config),
                dir=save_directory,
                reinit='create_new',
                resume="allow"
            )

            if state_dict is not None:
                kwargs["id"] = state_dict["wandb_run"]["id"]
            self.wandb_run = wandb.init(**kwargs)
        else:
            self.wandb_run = None

    def state_dict(self) -> dict:
        """
        Get the state dictionary of the logger.

        Returns:
            dict: State dictionary containing the required state.
        """
        return {
            "wandb_run": {
                "id": self.wandb_run.id,
            } if self.wandb_run else None,
        }

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
            prefixed_metrics = {f"{group}/{k}": v for k, v in metrics.items()}
            self.wandb_run.log(prefixed_metrics, step=step)

    def close(self):
        """
        Close all CSV loggers and finalize wandb session if enabled.
        """
        self.train_logger.close()
        self.val_logger.close()
        self.eval_logger.close()

        if self.wandb_run:
            self.wandb_run.finish()
