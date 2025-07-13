import re
from pathlib import Path

import pandas as pd

from trainer.config import ExperimentConfig


class ExperimentFromFile:
    def __init__(
        self, experiment_config: ExperimentConfig, save_directory: Path, env_name: str
    ):
        self.save_directory = save_directory
        self.env_name = env_name
        self.path = save_directory / env_name / "strategies_stopping_times.csv"
        pattern = re.compile(r"_(\d{8}_\d{6})")

        def extract_timestamp(file_path):
            folder_name = file_path.parent.name  # get folder name, not full path
            match = pattern.search(folder_name)
            return match.group(1) if match else ""

        files = list(
            (save_directory / env_name).glob(
                f"{config_to_name(experiment_config)}_*/eval.csv"
            )
        )
        file_path = sorted(files, key=extract_timestamp)[-1]

        self.experiment_config = experiment_config
        self.current_step = 0
        self.df = pd.read_csv(file_path)
        self.max_steps = self.df["step"].max()

    def train(self, num_steps: int) -> bool:
        self.current_step += num_steps
        return self.current_step >= self.max_steps

    def evaluate(self, num_episodes: int = 50) -> float:
        return self.df[self.df["step"] == self.current_step]["success"]

    def get_label(self) -> str:
        return rf"$seed = {self.experiment_config.seed}, \alpha = {self.experiment_config.alpha:.2f}$"

    def get_data(self) -> pd.DataFrame:
        return self.df[["step", "success"]]


def config_to_name(experiment_config: ExperimentConfig) -> str:
    tuned_hyperparams = [
        (field, value)
        for field, value in vars(experiment_config).items()
        if value is not None
    ]
    return "_".join(f"{field}_{value}" for field, value in tuned_hyperparams)
