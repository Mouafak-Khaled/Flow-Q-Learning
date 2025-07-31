from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange
from wandb.sdk.wandb_run import Run

from envmodel.config import TrainerConfig
from envmodel.termination_predictor import TerminationPredictor
from fql.utils.datasets import Dataset
from utils.data_loader import DataLoader

LossFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]],
]


class TerminationPredictorTrainer:
    def __init__(
        self,
        model: TerminationPredictor,
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
            self.rng, sample_batch["observations"], train=True, rng=self.rng
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
            predicted_termination = self.model.apply(
                params, batch["next_observations"], train=True, rng=self.rng
            )
            loss, logs = self.loss_fn(
                predicted_termination,
                batch["rewards"] == 0,
            )
            return loss, logs

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        new_state = state.apply_gradients(grads=grads)
        return new_state, {**logs, "loss": loss}

    @partial(jax.jit, static_argnums=0)
    def eval_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Evaluates the model on a batch without updating parameters."""
        predicted_termination = self.model.apply(
            state.params, batch["next_observations"], train=False, rng=self.rng
        )
        loss, logs = self.loss_fn(
            predicted_termination,
            batch["rewards"] == 0,
        )
        predictions = predicted_termination > 0
        targets = batch["rewards"] == 0

        logs["tp"] = jnp.sum(predictions & targets)
        logs["tn"] = jnp.sum(~predictions & ~targets)
        logs["fp"] = jnp.sum(predictions & ~targets)
        logs["fn"] = jnp.sum(~predictions & targets)

        return {**logs, "loss": loss}

    def train(self) -> Dict[str, float]:
        """Runs the main training loop."""

        state = self.state

        for step in trange(self.config.steps, desc="Training"):
            if step % 100 == 0:
                val_logs = []
                for _ in range(self.config.val_batches):
                    val_batch_np = self.val_loader.sample(256)
                    val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

                    # Evaluate using the current state's parameters
                    logs = self.eval_step(state, val_batch)
                    val_logs.append(logs)

                for k in val_logs[0].keys():
                    if k in ["tp", "tn", "fp", "fn"]:
                        continue
                    avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
                    if self.writer:
                        self.writer.add_scalar(
                            f"val/{k}",
                            scalar_value=np.array(avg_val_log),
                            global_step=step,
                        )
                    if self.logger:
                        self.logger.log({f"val/{k}": avg_val_log}, step=step)
                val_metrics = self._transform_eval_metrics(val_logs)
                for k, v in val_metrics.items():
                    if self.writer:
                        self.writer.add_scalar(
                            f"val/{k}",
                            scalar_value=v,
                            global_step=step,
                        )
                    if self.logger:
                        self.logger.log({f"val/{k}": v}, step=step)

            # Sample a new random batch each step and convert to JAX arrays
            batch_np = self.train_loader.sample(self.config.batch_size)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            state, logs = self.train_step(state, batch)

            if self.writer:
                for k, v in logs.items():
                    self.writer.add_scalar(
                        f"train/{k}", scalar_value=np.array(v), global_step=step
                    )
            if self.logger:
                self.logger.log(
                    {
                        "train/learning_rate": self.schedule(step),
                        **{f"train/{k}": v for k, v in logs.items()},
                    },
                    step=step,
                )

        self.state = state

        val_logs = []
        for _ in range(self.config.val_batches):
            val_batch_np = self.val_loader.sample(256)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            # Evaluate using the current state's parameters
            logs = self.eval_step(state, val_batch)
            val_logs.append(logs)

        for k in val_logs[0].keys():
            if k in ["tp", "tn", "fp", "fn"]:
                continue
            avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
            if self.writer:
                self.writer.add_scalar(
                    f"val/{k}", scalar_value=np.array(avg_val_log), global_step=step
                )
            if self.logger:
                self.logger.log({f"val/{k}": avg_val_log}, step=step)
        val_metrics = self._transform_eval_metrics(val_logs)

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

        train_metrics = self._transform_eval_metrics(train_logs)

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
        val_metrics = self._transform_eval_metrics(val_logs)

        return train_metrics, val_metrics


    def _transform_eval_metrics(self, logs: list[Dict[str, jnp.ndarray]]) -> Dict[str, np.ndarray]:
        total_tp = jnp.sum(jnp.array([log["tp"] for log in logs]))
        total_tn = jnp.sum(jnp.array([log["tn"] for log in logs]))
        total_fp = jnp.sum(jnp.array([log["fp"] for log in logs]))
        total_fn = jnp.sum(jnp.array([log["fn"] for log in logs]))

        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        precision = jnp.where(
            total_tp + total_fp > 0,
            total_tp / (total_tp + total_fp),
            1.0,
        )
        recall = jnp.where(
            total_tp + total_fn > 0,
            total_tp / (total_tp + total_fn),
            1.0,
        )
        f1 = jnp.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

        return {
            "accuracy": np.array(accuracy),
            "precision": np.array(precision),
            "recall": np.array(recall),
            "f1": np.array(f1)
        }