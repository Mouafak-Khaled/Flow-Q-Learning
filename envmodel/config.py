from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainerConfig:
    seed: int = 0
    steps: int = 2000
    env_name: str = "cube-single-play-singletask-task2-v0"
    model: str = "baseline"
    termination_weight: float = 1.0
    reconstruction_weight: float = 1.0
    model_config: dict[str, any] = field(default_factory=lambda: {})
    init_learning_rate: float = 1e-3
    batch_size: int = 256
    sequence_length: int = 256
    val_batches: int = 20
    data_directory: Path = Path("data/")
    save_directory: Path = Path("exp/")
