import ogbench

import wandb
from envmodel.BaselineEnvModel import BaselineEnvModel
from envmodel.trainer import Trainer
from fql.utils.datasets import Dataset

env_name = "cube-single-play-singletask-task2-v0"
model = "baseline"

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
train_dataset = Dataset.create(**train_dataset)
val_dataset = Dataset.create(**val_dataset)

# --- Hyperparameters ---
batch_size = 256
init_learning_rate = 1e-3
hidden_size = 128
num_train_steps = 2000
val_batches = 20
seed = 42

# Sample once to get shapes
sample_batch = train_dataset.sample(batch_size)
obs_sample = sample_batch["observations"]
act_sample = sample_batch["actions"]

# --- Model ---
if model == "baseline":
    env_model = BaselineEnvModel(
        obs_dim=obs_sample.shape[-1], act_dim=act_sample.shape[-1], hidden_size=hidden_size
    )

# --- Weights and Biases ---
logger = wandb.init(
    name=f"{env_name}_{model}",
    project="fql_env_model",
    config={
        "env_name": env_name,
        "model": model,
        "init_learning_rate": init_learning_rate,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "seed": seed,
    },
    directory="exp/",
)

# --- Trainer initialization and training ---
trainer = Trainer(
    model=env_model,
    init_learning_rate=init_learning_rate,
    lr_decay_steps=num_train_steps,
    seed=seed,
    obs_sample=obs_sample,
    act_sample=act_sample,
    logger=logger,
)

trainer.train(train_dataset, val_dataset, batch_size, num_train_steps, val_batches)

logger.finish()
