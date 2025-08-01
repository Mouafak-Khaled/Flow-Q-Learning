from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description="Train an environment model for offline RL tasks.")
parser.add_argument(
    "--env_name",
    type=str,
    default="cube-single-play-singletask-task2-v0",
    help="Name of the environment to train the model on.",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["baseline", "multistep", "termination_predictor"],
    default="baseline",
    help="Type of model to train.",
)
parser.add_argument(
    "--save_directory",
    type=str,
    default="exp/",
    help="Directory to save the trained model and logs.",
)

args = parser.parse_args()

if args.model == "termination_predictor":
    from envmodel_hpo.train_termination_predictor import train_termination_predictor

    train_termination_predictor(
        env_name=args.env_name,
        save_directory=Path(args.save_directory),
    )
elif args.model == "baseline":
    from envmodel_hpo.train_baseline import train_baseline

    train_baseline(
        env_name=args.env_name,
        save_directory=Path(args.save_directory),
    )

# from dataclasses import asdict
# from functools import partial
# from pathlib import Path

# import flax
# import ogbench
# import wandb
# import yaml

# from argparser import build_env_model_config_from_args, get_env_model_argparser
# from envmodel.baseline import BaselineStatePredictor
# from envmodel.loss import focal_loss, state_prediction_loss, weighted_binary_cross_entropy
# from envmodel.multistep import MultistepStatePredictor
# from envmodel.state_predictor_trainer import StatePredictorTrainer
# from envmodel.termination_predictor import TerminationPredictor
# from envmodel.termination_predictor_trainer import TerminationPredictorTrainer
# from utils.data_loader import MultistepLoader, StepLoader

# args = get_env_model_argparser().parse_args()
# config = build_env_model_config_from_args(args)

# env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
#     config.env_name, config.data_directory
# )

# # --- Model & DataLoader ---
# if config.model == "baseline":
#     train_dataloader = StepLoader(train_dataset)
#     val_dataloader = StepLoader(val_dataset)

#     # Sample once to get shapes
#     sample_batch = train_dataloader.sample(config.batch_size)

#     model = BaselineStatePredictor(
#         observation_dimension=sample_batch["observations"].shape[-1],
#         action_dimension=sample_batch["actions"].shape[-1],
#         hidden_dims=config.model_config["hidden_dims"],
#     )

#     loss_fn = partial(
#         state_prediction_loss,
#         true_termination_weight=config.true_termination_weight,
#         termination_weight=config.termination_weight,
#         reconstruction_weight=0.0,
#     )
# elif config.model == "multistep":
#     train_dataloader = MultistepLoader(
#         train_dataset, sequence_length=config.sequence_length
#     )
#     val_dataloader = MultistepLoader(
#         val_dataset, sequence_length=config.sequence_length
#     )

#     sample_batch = train_dataloader.sample(config.batch_size)

#     model = MultistepStatePredictor(
#         observation_dimension=sample_batch["observations"].shape[-1],
#         action_dimension=sample_batch["actions"].shape[-1],
#         hidden_dims=config.model_config["hidden_dims"],
#     )

#     loss_fn = partial(
#         state_prediction_loss,
#         true_termination_weight=config.true_termination_weight,
#         termination_weight=config.termination_weight,
#         reconstruction_weight=0.0,
#     )
# elif config.model == "termination_predictor":
#     train_dataloader = StepLoader(train_dataset)
#     val_dataloader = StepLoader(val_dataset)

#     sample_batch = train_dataloader.sample(config.batch_size)

#     model = TerminationPredictor(
#         observation_dimension=sample_batch["observations"].shape[-1],
#         hidden_dims=config.model_config["hidden_dims"],
#     )

#     loss_fn = focal_loss
#     # loss_fn = partial(
#     #     weighted_binary_cross_entropy,
#     #     true_weight=config.true_termination_weight,
#     # )
# else:
#     raise ValueError(f"Unknown model type: {config.model}")


# # --- Weights and Biases ---
# logger = wandb.init(
#     name=f"{config.env_name}_{config.model}",
#     project="fql_env_model",
#     config=asdict(config),
#     dir="exp/",
# )

# # --- Trainer initialization and training ---
# if config.model == "termination_predictor":
#     trainer = TerminationPredictorTrainer(
#         model=model,
#         train_loader=train_dataloader,
#         val_loader=val_dataloader,
#         loss_fn=loss_fn,
#         config=config,
#         logger=logger,
#     )
# else:
#     trainer = StatePredictorTrainer(
#         model=model,
#         train_loader=train_dataloader,
#         val_loader=val_dataloader,
#         loss_fn=loss_fn,
#         config=config,
#         logger=logger,
#     )

# trainer.train()

# save_dir = Path(config.save_directory) / config.env_name / "env_models"
# save_dir.mkdir(parents=True, exist_ok=True)

# config_path = save_dir / f"{config.model}_config.yaml"
# with open(config_path, "w") as f:
#     yaml.dump(config.model_config, f)

# model_path = save_dir / f"{config.model}.pt"
# with open(model_path, "wb") as f:
#     f.write(flax.serialization.to_bytes(trainer.state.params))
# wandb.save(model_path)

# logger.finish()
