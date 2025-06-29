from pathlib import Path
from typing_extensions import Literal

from fql.utils.log_utils import CsvLogger


class Logger:
    def __init__(self, save_directory: Path):
        self.train_logger = CsvLogger(save_directory / "train.csv")
        self.val_logger = CsvLogger(save_directory / "val.csv")
        self.eval_logger = CsvLogger(save_directory / "eval.csv")

    def log(self, metrics: dict, step: int, group: Literal["train", "val", "eval"]):
        """
        Log metrics to a file in the save directory.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Current training step.
            group (str): Group name for the metrics.
        """
        if group == "train":
            self.train_logger.log(metrics, step=step)
        elif group == "val":
            self.val_logger.log(metrics, step=step)
        elif group == "eval":
            self.eval_logger.log(metrics, step=step)

    def close(self):
        """
        Close all loggers.
        """
        self.train_logger.close()
        self.val_logger.close()
        self.eval_logger.close()
