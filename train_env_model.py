from dataclasses import asdict
from pathlib import Path

import flax
import ogbench
import wandb
import yaml

from argparser import build_env_model_config_from_args, get_env_model_argparser
from envmodel.baseline import BaselineStatePredictor
from envmodel.loss import mean_squared_error, weighted_binary_cross_entropy
from envmodel.multistep import MultistepStatePredictor
from envmodel.termination_predictor import TerminationPredictor
from envmodel.trainer import Trainer
from utils.data_loader import StepLoader, MultistepLoader

args = get_env_model_argparser().parse_args()
config = build_env_model_config_from_args(args)

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    config.env_name, config.data_directory
)

# --- Model & DataLoader ---
if config.state_predictor == "baseline":
    train_dataloader = StepLoader(train_dataset)
    val_dataloader = StepLoader(val_dataset)

    # Sample once to get shapes
    sample_batch = train_dataloader.sample(config.batch_size)

    state_predictor = BaselineStatePredictor(
        observation_dimension=sample_batch["observations"].shape[-1],
        action_dimension=sample_batch["actions"].shape[-1],
        hidden_dims=config.state_predictor_config["hidden_dims"],
    )
elif config.state_predictor == "multistep":
    train_dataloader = MultistepLoader(
        train_dataset, sequence_length=config.sequence_length
    )
    val_dataloader = MultistepLoader(
        val_dataset, sequence_length=config.sequence_length
    )

    sample_batch = train_dataloader.sample(config.batch_size)

    state_predictor = MultistepStatePredictor(
        observation_dimension=sample_batch["observations"].shape[-1],
        action_dimension=sample_batch["actions"].shape[-1],
        hidden_dims=config.state_predictor_config["hidden_dims"],
    )
else:
    raise ValueError(f"Unknown model type: {config.state_predictor}")


def loss_fn(
    predicted_next_observations,
    next_observations,
    predicted_terminations,
    terminations,
):
    next_observation_loss = mean_squared_error(
        predicted_next_observations, next_observations
    )
    termination_loss, termination_logs = weighted_binary_cross_entropy(
        predicted_terminations,
        terminations,
        config.termination_predictor_config["true_termination_weight"],
    )

    loss = (
        next_observation_loss + config.termination_loss_weight * termination_loss
    ) / (1 + config.termination_loss_weight)
    return (
        loss,
        {
            "next_observation_loss": next_observation_loss,
            "termination_loss": termination_loss,
            "true_termination_loss": termination_logs["true_loss"],
            "false_termination_loss": termination_logs["false_loss"],
            "loss": loss,
        },
    )


termination_predictor = TerminationPredictor(
    observation_dimension=sample_batch["observations"].shape[-1],
    hidden_dims=config.termination_predictor_config["hidden_dims"],
)

# --- Weights and Biases ---
logger = wandb.init(
    name=f"{config.env_name}_{config.state_predictor}",
    project="fql_env_model",
    config=asdict(config),
    dir="exp/",
)

# --- Trainer initialization and training ---
trainer = Trainer(
    state_predictor=state_predictor,
    termination_predictor=termination_predictor,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    loss_fn=loss_fn,
    config=config,
    logger=logger,
)

trainer.train()

save_dir = Path(config.save_directory) / config.env_name / "env_models"
save_dir.mkdir(parents=True, exist_ok=True)

config_path = save_dir / f"{config.state_predictor}_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config.state_predictor_config, f)

model_path = save_dir / f"{config.state_predictor}.pt"
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(trainer.state.params["state_predictor"]))
wandb.save(model_path)

config_path = save_dir / "termination_predictor_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config.termination_predictor_config, f)

model_path = save_dir / "termination_predictor.pt"
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(trainer.state.params["termination_predictor"]))
wandb.save(model_path)

logger.finish()
