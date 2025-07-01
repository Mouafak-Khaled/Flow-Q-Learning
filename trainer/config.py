from pathlib import Path
from dataclasses import dataclass


@dataclass
class AgentConfig:
    seed: int = 0
    agent_name: str = "fql"
    ob_dims: int = None
    action_dim: int = None
    lr: float = 3e-4
    batch_size: int = 256
    actor_hidden_dims: tuple = (512, 512, 512, 512)
    value_hidden_dims: tuple = (512, 512, 512, 512)
    layer_norm: bool = True
    actor_layer_norm: bool = False
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"
    alpha: float = 10.0
    flow_steps: int = 10
    normalize_q_loss: bool = False
    encoder: str = None


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int | None = None
    alpha: float | None = None
    normalize_q_loss: bool | None = None


@dataclass
class TrainerConfig:
    seed: int = 0
    steps: int = 1000000
    log_interval: int = 5000
    eval_interval: int = 100000
    save_directory: Path = Path("exp/")
    data_directory: Path = Path("data/")
    use_wandb: bool = False
    env_name: str = "cube-double-play-singletask-v0"
    agent: AgentConfig = AgentConfig()
    evaluation_mode: bool = False
    eval_episodes: int = 50
    buffer_size: int = 2000000
