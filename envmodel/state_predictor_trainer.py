from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange
from wandb.sdk.wandb_run import Run

from envmodel.baseline import StatePredictor
from envmodel.config import TrainerConfig
from fql.utils.datasets import Dataset
from utils.data_loader import DataLoader
from utils.envmodel import load_model

LossFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]],
]


class StatePredictorTrainer:
    def __init__(
        self,
        model: StatePredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: LossFn,
        config: TrainerConfig,
        writer: Any | None = None,
        logger: Run | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.writer = writer
        self.logger = logger

        self.rng = jax.random.PRNGKey(self.config.seed)

        sample_batch = self.train_loader.sample(self.config.batch_size)
        self.params = self.model.init(
            self.rng, sample_batch["observations"], sample_batch["actions"]
        )

        self.termination_predictor = (
            load_model(
                self.config.save_directory,
                self.config.env_name,
                "termination_predictor",
                sample_batch,
            )
            if self.config.termination_weight > 0
            else lambda *args, **kwargs: None
        )

        self.schedule = optax.cosine_decay_schedule(
            init_value=self.config.init_learning_rate,
            decay_steps=self.config.steps,
        )
        self.optimizer = optax.adam(self.schedule)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.optimizer
        )

    @partial(jax.jit, static_argnums=0)
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[train_state.TrainState, jnp.ndarray, jax.Array]:
        """Performs a single training step (forward pass, loss calculation, gradients, update)."""

        def loss_fn(params):
            predicted_next_observation, reconstructed_observations = self.model.apply(
                params, batch["observations"], batch["actions"]
            )
            predicted_termination = self.termination_predictor(
                predicted_next_observation, train=False, rng=jax.random.PRNGKey(0)
            )
            loss, logs = self.loss_fn(
                predicted_next_observation,
                batch["next_observations"],
                predicted_termination,
                batch["rewards"] == 0,
                reconstructed_observations,
                batch["observations"],
            )
            return loss, logs

        (_, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        new_state = state.apply_gradients(grads=grads)
        return new_state, logs

    @partial(jax.jit, static_argnums=0)
    def eval_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Evaluates the model on a batch without updating parameters."""
        predicted_next_observation, reconstructed_observations = self.model.apply(
            state.params, batch["observations"], batch["actions"]
        )
        predicted_termination = self.termination_predictor(
            predicted_next_observation, train=False, rng=jax.random.PRNGKey(0)
        )
        _, logs = self.loss_fn(
            predicted_next_observation,
            batch["next_observations"],
            predicted_termination,
            batch["rewards"] == 0,
            reconstructed_observations,
            batch["observations"],
        )
        return logs

    def train(self) -> None:
        """Runs the main training loop."""

        state = self.state

        for step in trange(self.config.steps, desc="Training"):
            if step % 100 == 0:
                val_logs = []
                for _ in range(self.config.val_batches):
                    val_batch_np = self.val_loader.sample(self.config.batch_size)
                    val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

                    # Evaluate using the current state's parameters
                    logs = self.eval_step(state, val_batch)
                    val_logs.append(logs)

                for k in val_logs[0].keys():
                    avg_val_log = np.array(
                        jnp.mean(jnp.array([log[k] for log in val_logs]))
                    )
                    if self.writer:
                        self.writer.add_scalar(
                            f"val/{k}", scalar_value=avg_val_log, global_step=step
                        )
                    if self.logger:
                        self.logger.log({f"val/{k}": avg_val_log}, step=step)

            # Sample a new random batch each step and convert to JAX arrays
            batch_np = self.train_loader.sample(self.config.batch_size)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            state, logs = self.train_step(state, batch)

            if self.writer:
                self.writer.add_scalar(
                    "train/learning_rate",
                    scalar_value=np.array(self.schedule(step)),
                    global_step=step,
                )
                for k, v in logs.items():
                    self.writer.add_scalar(
                        f"train/{k}", scalar_value=np.array(v), global_step=step
                    )
            if self.logger:
                self.logger.log(
                    {
                        "train/learning_rate": np.array(self.schedule(step)),
                        **{f"train/{k}": np.array(v) for k, v in logs.items()},
                    },
                    step=step,
                )

        self.state = state

        val_logs = []
        for _ in range(self.config.val_batches):
            val_batch_np = self.val_loader.sample(self.config.batch_size)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            # Evaluate using the current state's parameters
            logs = self.eval_step(state, val_batch)
            val_logs.append(logs)

        val_metrics = {}
        for k in val_logs[0].keys():
            val_metrics[k] = np.array(jnp.mean(jnp.array([log[k] for log in val_logs])))
            if self.writer:
                self.writer.add_scalar(
                    f"val/{k}", scalar_value=val_metrics[k], global_step=step
                )
            if self.logger:
                self.logger.log({f"val/{k}": val_metrics[k]}, step=step)

        return val_metrics

    def validate(
        self, train_dataset: Dataset, val_dataset: Dataset
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Runs validation without training."""
        state = self.state

        # iterate over the training dataset
        train_logs = []
        for i in range(len(train_dataset["observations"]) // 256):
            # Sample a batch from the training dataset
            if i == len(train_dataset["observations"]) // 256 - 1:
                train_batch_np = train_dataset.get_subset(
                    np.arange(i * 256, len(train_dataset["observations"]))
                )
            else:
                train_batch_np = train_dataset.get_subset(
                    np.arange(i * 256, (i + 1) * 256)
                )
            train_batch = {k: jnp.array(v) for k, v in train_batch_np.items()}

            logs = self.eval_step(state, train_batch)
            train_logs.append(logs)

        train_metrics = {}
        for k in train_logs[0].keys():
            train_metrics[k] = np.array(
                jnp.mean(jnp.array([log[k] for log in train_logs]))
            )

        val_logs = []
        for i in range(len(val_dataset["observations"]) // 256):
            # Sample a batch from the validation dataset
            if i == len(val_dataset["observations"]) // 256 - 1:
                val_batch_np = val_dataset.get_subset(
                    np.arange(i * 256, len(val_dataset["observations"]))
                )
            else:
                val_batch_np = val_dataset.get_subset(np.arange(i * 256, (i + 1) * 256))
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            logs = self.eval_step(state, val_batch)
            val_logs.append(logs)

        val_metrics = {}
        for k in val_logs[0].keys():
            val_metrics[k] = np.array(jnp.mean(jnp.array([log[k] for log in val_logs])))

        return train_metrics, val_metrics
