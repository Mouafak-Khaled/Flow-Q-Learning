from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange
from wandb.sdk.wandb_run import Run

from envmodel.config import TrainerConfig
from utils.data_loader import DataLoader

LossFn = Callable[
    [nn.Module, Any, Dict[str, jnp.ndarray]], Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], jax.Array]]
]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: LossFn,
        config: TrainerConfig,
        logger: Run | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger

        self.rng = jax.random.PRNGKey(self.config.seed)

        sample_batch = self.train_loader.sample(self.config.batch_size)
        self.params = self.model.init(self.rng, **sample_batch)

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
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, jnp.ndarray]:
        """Performs a single training step (forward pass, loss calculation, gradients, update)."""

        def loss_fn(params):
            return self.loss_fn(self.model, params, self.rng, batch)

        (loss, (logs, self.rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        new_state = state.apply_gradients(grads=grads)

        return new_state, logs

    @partial(jax.jit, static_argnums=0)
    def eval_step(
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Evaluates the model on a batch without updating parameters."""
        loss, (_, self.rng) = self.loss_fn(self.model, state.params, self.rng, batch)
        return loss

    def train(self) -> None:
        """Runs the main training loop."""

        state = self.state

        for step in trange(self.config.steps, desc="Training"):
            if step % 100 == 0:
                val_losses = []
                for _ in range(self.config.val_batches):
                    val_batch_np = self.val_loader.sample(self.config.batch_size)
                    val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

                    # Evaluate using the current state's parameters
                    val_loss = self.eval_step(state, val_batch)
                    val_losses.append(val_loss)

                avg_val_loss = jnp.mean(jnp.array(val_losses))
                if self.logger:
                    self.logger.log({"val/loss": avg_val_loss}, step=step)

            # Sample a new random batch each step and convert to JAX arrays
            batch_np = self.train_loader.sample(self.config.batch_size)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            state, logs = self.train_step(state, batch)

            if self.logger:
                self.logger.log(
                    {
                        "train/learning_rate": self.schedule(step),
                        **{f"train/{k}": v for k, v in logs.items()},
                    },
                    step=step,
                )

        self.state = state

        val_losses = []
        for _ in range(self.config.val_batches):
            val_batch_np = self.val_loader.sample(self.config.batch_size)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            # Evaluate using the current state's parameters
            val_loss = self.eval_step(state, val_batch)
            val_losses.append(val_loss)

        avg_val_loss = jnp.mean(jnp.array(val_losses))
        if self.logger:
            self.logger.log({"val/loss": avg_val_loss}, step=step)
