from dataclasses import asdict
from functools import partial
from pathlib import Path

import flax
import ogbench
import wandb

from argparser import build_env_model_config_from_args, get_env_model_argparser
from envmodel.baseline import BaselineEnvModel, baseline_loss
from envmodel.initial_observation import InitialObservationEnvModel, vae_loss
from envmodel.multistep import MultistepEnvModel
from envmodel.trainer import Trainer
from utils.data_loader import InitialObservationLoader, StepLoader, MultistepLoader

args = get_env_model_argparser().parse_args()
config = build_env_model_config_from_args(args)

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    config.env_name, config.data_directory
)

# --- Model & DataLoader ---
if config.model == "baseline":
    train_dataloader = StepLoader(train_dataset)
    val_dataloader = StepLoader(val_dataset)

    # Sample once to get shapes
    sample_batch = train_dataloader.sample(config.batch_size)

    env_model = BaselineEnvModel(
        observation_dimension=sample_batch["observations"].shape[-1],
        action_dimension=sample_batch["actions"].shape[-1],
        hidden_size=config.model_config["hidden_dim"],
    )

    loss_fn = partial(
        baseline_loss,
        termination_weight=config.model_config["termination_weight"],
        termination_true_weight=config.model_config["termination_true_weight"],
    )
elif config.model == "multistep":
    train_dataloader = MultistepLoader(train_dataset, sequence_length=config.model_config["sequence_length"])
    val_dataloader = MultistepLoader(val_dataset, sequence_length=config.model_config["sequence_length"])

    sample_batch = train_dataloader.sample(config.batch_size)

    env_model = MultistepEnvModel(
        observation_dimension=sample_batch["observations"].shape[-1],
        action_dimension=sample_batch["actions"].shape[-1],
        hidden_size=config.model_config["hidden_dim"],
    )

    loss_fn = partial(
        baseline_loss,
        termination_weight=config.model_config["termination_weight"],
        termination_true_weight=config.model_config["termination_true_weight"],
    )
elif config.model == "initial_observation":
    train_dataloader = InitialObservationLoader(train_dataset)
    val_dataloader = InitialObservationLoader(val_dataset)

    # Sample once to get shapes
    sample_batch = train_dataloader.sample(config.batch_size)

    env_model = InitialObservationEnvModel(
        observation_dimension=sample_batch["observations"].shape[-1],
        latent_dimension=config.model_config["latent_dim"],
        hidden_size=config.model_config["hidden_dim"],
    )

    loss_fn = vae_loss
else:
    raise ValueError(f"Unknown model type: {args.model}")

# --- Weights and Biases ---
logger = wandb.init(
    name=f"{config.env_name}_{config.model}",
    project="fql_env_model",
    config=asdict(config),
    dir="exp/",
)

# --- Trainer initialization and training ---
trainer = Trainer(
    model=env_model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    loss_fn=loss_fn,
    config=config,
    logger=logger,
)

trainer.train()

save_dir = Path(config.save_directory) / config.env_name / "env_models"
save_dir.mkdir(parents=True, exist_ok=True)
model_path = save_dir / f"{config.model}.pt"
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(trainer.state.params))
wandb.save(model_path)

logger.finish()
