import argparse
from ast import literal_eval
from pathlib import Path

from envmodel.config import TrainerConfig as EnvModelTrainerConfig
from trainer.config import AgentConfig, TrainerConfig


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainer and Agent Configuration")

    # TrainerConfig fields
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for the trainer."
    )
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Number of training steps."
    )
    parser.add_argument(
        "--log_interval", type=int, default=5000, help="Logging interval."
    )
    parser.add_argument(
        "--eval_interval", type=int, default=100000, help="Evaluation interval."
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        default="exp/",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--data_directory", type=str, default="data/", help="Dataset directory."
    )
    parser.add_argument(
        "--report_directory", type=str, default="report/", help="Report directory."
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="cube-single-play-singletask-task2-v0",
        help="Environment (dataset) name.",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=50, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--buffer_size", type=int, default=2000000, help="Replay buffer size."
    )

    # AgentConfig fields (namespaced with agent.)
    parser.add_argument(
        "--agent.seed", type=int, default=0, help="Random seed for the agent."
    )
    parser.add_argument(
        "--agent.agent_name", type=str, default="fql", help="Agent name."
    )
    parser.add_argument(
        "--agent.ob_dims",
        type=int,
        default=None,
        help="Observation dimensions (set automatically).",
    )
    parser.add_argument(
        "--agent.action_dim",
        type=int,
        default=None,
        help="Action dimension (set automatically).",
    )
    parser.add_argument("--agent.lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--agent.batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--agent.actor_hidden_dims",
        type=literal_eval,
        default="(512, 512, 512, 512)",
        help="Actor network hidden dimensions (tuple).",
    )
    parser.add_argument(
        "--agent.value_hidden_dims",
        type=literal_eval,
        default="(512, 512, 512, 512)",
        help="Value network hidden dimensions (tuple).",
    )
    parser.add_argument(
        "--agent.layer_norm", action="store_true", help="Use layer normalization."
    )
    parser.add_argument(
        "--agent.actor_layer_norm",
        action="store_true",
        help="Use layer normalization in actor.",
    )
    parser.add_argument(
        "--agent.discount", type=float, default=0.99, help="Discount factor."
    )
    parser.add_argument(
        "--agent.tau", type=float, default=0.005, help="Target network update rate."
    )
    parser.add_argument(
        "--agent.q_agg",
        type=str,
        default="mean",
        help="Aggregation method for target Q values.",
    )
    parser.add_argument(
        "--agent.alpha",
        type=float,
        default=10.0,
        help="BC coefficient (needs tuning for each environment).",
    )
    parser.add_argument(
        "--agent.flow_steps", type=int, default=10, help="Number of flow steps."
    )
    parser.add_argument(
        "--agent.normalize_q_loss",
        action="store_true",
        help="Whether to normalize the Q loss.",
    )
    parser.add_argument(
        "--agent.encoder",
        type=str,
        default=None,
        help="Visual encoder name (e.g., impala_small).",
    )
    parser.add_argument(
        "--single_experiment",
        action="store_true",
        help="Whether to run a single experiment.",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job ID in a Job Array.",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> TrainerConfig:
    args_dict = vars(args)

    agent_kwargs = {}
    trainer_kwargs = {}

    for k, v in args_dict.items():
        if k.startswith("agent."):
            agent_field = k[len("agent.") :]
            agent_kwargs[agent_field] = v
        else:
            trainer_kwargs[k] = v

    # Adjust directory fields
    trainer_kwargs["save_directory"] = Path(trainer_kwargs["save_directory"])
    trainer_kwargs["data_directory"] = Path(trainer_kwargs["data_directory"])

    # Build AgentConfig
    agent_config_fields = set(AgentConfig.__dataclass_fields__.keys())
    filtered_agent_kwargs = {
        k: v for k, v in agent_kwargs.items() if k in agent_config_fields
    }
    agent_config = AgentConfig(**filtered_agent_kwargs)

    # Build TrainerConfig with agent inside
    trainer_config_fields = set(TrainerConfig.__dataclass_fields__.keys())
    filtered_trainer_kwargs = {
        k: v for k, v in trainer_kwargs.items() if k in trainer_config_fields
    }
    trainer_config = TrainerConfig(**filtered_trainer_kwargs, agent=agent_config)

    return trainer_config


def get_env_model_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The configurations of the environment model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        help="The environment model to be used.",
    )

    parser.add_argument(
        "--model.hidden_dims",
        type=literal_eval,
        default=(128, 128),
        help="The dimension of hidden layers.",
    )

    parser.add_argument(
        "--model.latent_dim",
        type=int,
        default=4,
        help="The dimension of latent representation.",
    )

    parser.add_argument(
        "--model.termination_true_weight",
        type=float,
        default=30.0,
        help="Weight for the true class in the termination loss in the baseline model.",
    )

    parser.add_argument(
        "--model.termination_weight",
        type=float,
        default=1.0,
        help="Weight for the termination loss in the baseline model.",
    )

    parser.add_argument(
        "--model.sequence_length",
        type=int,
        default=256,
        help="The length of sequences for the multistep model.",
    )

    parser.add_argument(
        "--steps", type=int, default=20000, help="The number of training steps."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="cube-single-play-singletask-task2-v0",
        help="The environment task.",
    )

    # Trainer-specific arguments
    parser.add_argument(
        "--init_learning_rate", type=float, default=1e-3, help="Initial learning rate."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")

    parser.add_argument(
        "--val_batches", type=int, default=20, help="Number of validation batches."
    )

    parser.add_argument(
        "--data_directory",
        type=str,
        default="data/",
        help="Path to a directory where training data is stored.",
    )

    parser.add_argument(
        "--save_directory",
        type=str,
        default="exp/",
        help="Path to a directory where model parameters will be saved.",
    )

    return parser


def build_env_model_config_from_args(args: argparse.Namespace) -> EnvModelTrainerConfig:
    args_dict = vars(args)

    model_config = {}
    trainer_kwargs = {}

    for k, v in args_dict.items():
        if k.startswith("model."):
            model_field = k[len("model.") :]
            model_config[model_field] = v
        else:
            trainer_kwargs[k] = v

    # Adjust directory fields
    trainer_kwargs["save_directory"] = Path(trainer_kwargs["save_directory"])
    trainer_kwargs["data_directory"] = Path(trainer_kwargs["data_directory"])
    trainer_kwargs["report_directory"] = Path(trainer_kwargs["report_directory"])

    # Build EnvModelTrainerConfig with model inside
    trainer_config_fields = set(EnvModelTrainerConfig.__dataclass_fields__.keys())
    filtered_trainer_kwargs = {
        k: v for k, v in trainer_kwargs.items() if k in trainer_config_fields
    }
    trainer_config = EnvModelTrainerConfig(
        **filtered_trainer_kwargs, model_config=model_config
    )

    return trainer_config
