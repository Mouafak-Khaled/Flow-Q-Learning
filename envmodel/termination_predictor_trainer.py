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

        logs["accuracy"] = jnp.mean(predictions == targets)

        true_positives = jnp.sum(predictions & targets)
        predicted_positives = jnp.sum(predictions)
        actual_positives = jnp.sum(targets)

        logs["precision"] = jnp.where(
            predicted_positives > 0,
            true_positives / predicted_positives,
            1.0,
        )
        logs["recall"] = jnp.where(
            actual_positives > 0,
            true_positives / actual_positives,
            1.0,
        )
        logs["f1"] = jnp.where(
            (logs["precision"] + logs["recall"]) > 0,
            2
            * logs["precision"]
            * logs["recall"]
            / (logs["precision"] + logs["recall"]),
            0.0,
        )
        return {**logs, "loss": loss}

    def train(self) -> Dict[str, float]:
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
                    avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
                    if self.writer:
                        self.writer.add_scalar(
                            f"val/{k}", scalar_value=np.array(avg_val_log), global_step=step
                        )
                    if self.logger:
                        self.logger.log({f"val/{k}": avg_val_log}, step=step)

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
            val_batch_np = self.val_loader.sample(self.config.batch_size)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            # Evaluate using the current state's parameters
            logs = self.eval_step(state, val_batch)
            val_logs.append(logs)

        val_metrics = {}
        for k in val_logs[0].keys():
            avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
            val_metrics[k] = np.array(avg_val_log)
            if self.writer:
                self.writer.add_scalar(
                    f"val/{k}", scalar_value=np.array(avg_val_log), global_step=step
                )
            if self.logger:
                self.logger.log({f"val/{k}": avg_val_log}, step=step)

        return val_metrics
