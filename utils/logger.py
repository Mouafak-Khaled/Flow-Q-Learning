import os
from pathlib import Path
from  trainer.config import TrainerConfig
from typing_extensions import Literal, Optional
import wandb
from fql.utils.log_utils import CsvLogger
from dataclasses import  asdict

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
         config: TrainerConfig,
         exp_name: str,
         mode: Literal["disabled", "offline", "online"],
         use_wandb: Optional[bool] = False,
    ):
        """
        Initialize the Logger.

        Args:
            save_directory (Path): Directory where CSV logs (and optionally wandb logs) are stored.
            config (TrainerConfig): Configuration object for the training run, passed to wandb.
            exp_name (str): Experiment name for wandb.
            mode (Literal): Wandb mode - 'online', 'offline', or 'disabled'.
            use_wandb (Optional[bool]): If True, enables wandb logging.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.train_logger = CsvLogger(save_directory / "train.csv")
        self.val_logger = CsvLogger(save_directory / "val.csv")
        self.eval_logger = CsvLogger(save_directory / "eval.csv")
        self.use_wandb = use_wandb

        if self.use_wandb:

            self.exp_name = exp_name
            self.config = asdict(config)
            self.mode = mode

            wandb.init(
                project="fql",
                exp_name=self.exp_name,
                config=self.config,
                settings=wandb.Settings(
                    start_method='thread',
                    _disable_stats=False,
                ),
                mode=mode,
                save_code=True,
                dir=save_directory,
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

        if self.use_wandb:
            prefixed_metrics = {f"{group}/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed_metrics, step=step)

    def close(self):
        """
        Close all CSV loggers and finalize wandb session if enabled.
        """
        self.train_logger.close()
        self.val_logger.close()
        self.eval_logger.close()

        if self.use_wandb:
            wandb.finish()



