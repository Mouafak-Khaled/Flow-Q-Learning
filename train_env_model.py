from pathlib import Path

import flax
import ogbench
import wandb

from argparser import get_env_model_argparser
from envmodel.baseline import BaselineEnvModel
from envmodel.trainer import Trainer
from utils.data_loader import StepLoader

args = get_env_model_argparser().parse_args()

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    args.env_name, args.data_directory
)
train_dataloader = StepLoader(train_dataset)
val_dataloader = StepLoader(val_dataset)

# Sample once to get shapes
sample_batch = train_dataloader.sample(args.batch_size)
obs_sample = sample_batch["observations"]
act_sample = sample_batch["actions"]

# --- Model ---
if args.model == "baseline":
    env_model = BaselineEnvModel(
        obs_dim=obs_sample.shape[-1],
        act_dim=act_sample.shape[-1],
        hidden_size=args.hidden_dim,
    )

# --- Weights and Biases ---
logger = wandb.init(
    name=f"{args.env_name}_{args.model}",
    project="fql_env_model",
    config={
        "env_name": args.env_name,
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "steps": args.steps,
        "init_learning_rate": args.init_learning_rate,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "val_batches": args.val_batches,
        "data_directory": args.data_directory,
        "save_directory": args.save_directory,
    },
    dir="exp/",
)

# --- Trainer initialization and training ---
trainer = Trainer(
    model=env_model,
    init_learning_rate=args.init_learning_rate,
    lr_decay_steps=args.steps,
    seed=args.seed,
    obs_sample=obs_sample,
    act_sample=act_sample,
    logger=logger,
)

trainer.train(
    train_dataloader, val_dataloader, args.batch_size, args.steps, args.val_batches
)

save_dir = Path(args.save_directory) / args.env_name / "env_models"
save_dir.mkdir(parents=True, exist_ok=True)
model_path = save_dir / f"{args.model}.pt"
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(trainer.state.params))
wandb.save(model_path)

logger.finish()
