import os

import flax
import ogbench
import wandb

from envmodel.BaselineEnvModel import BaselineEnvModel
from envmodel.trainer import Trainer
from fql.utils.datasets import Dataset
from argparser import get_env_model_argparser

args = get_env_model_argparser().parse_args()

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.task)
train_dataset = Dataset.create(**train_dataset)
val_dataset = Dataset.create(**val_dataset)

# Sample once to get shapes
sample_batch = train_dataset.sample(args.batch_size)
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
    name=f"{args.task}_{args.model}",
    project="fql_env_model",
    config=args.as_dict(),
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

trainer.train(train_dataset, val_dataset, args.batch_size, args.steps, args.val_batches)

save_dir = f"exp/{args.task}/env_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, f"{args.model}.pt")
with open(model_path, "wb") as f:
    f.write(flax.serialization.to_bytes(trainer.state.params))
wandb.save(model_path)

logger.finish()
