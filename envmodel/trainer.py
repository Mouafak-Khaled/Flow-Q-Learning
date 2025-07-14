from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange

from wandb.sdk.wandb_run import Run


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        init_learning_rate: float,
        lr_decay_steps: int,
        seed: int,
        obs_sample: jnp.ndarray,
        act_sample: jnp.ndarray,
        logger: Run | None = None,
    ):
        self.model = model
        self.rng = jax.random.PRNGKey(seed)

        self.params = self.model.init(self.rng, obs_sample, act_sample)

        self.schedule = optax.cosine_decay_schedule(
            init_value=init_learning_rate,
            decay_steps=lr_decay_steps,
        )
        self.optimizer = optax.adam(self.schedule)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.optimizer
        )

        self.logger = logger

    def _compute_loss(
        self, params: Any, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Computes the total loss for a given batch."""

        pred_next_obs, pred_reward, pred_terminals = self.model.apply(
            params, batch["observations"], batch["actions"]
        )

        next_observation_loss = jnp.mean(
            jnp.square(pred_next_obs - batch["next_observations"])
        )
        reward_loss = jnp.mean(jnp.square(pred_reward - batch["rewards"]))
        terminated_loss = optax.sigmoid_binary_cross_entropy(
            logits=jnp.squeeze(pred_terminals), labels=batch["terminals"]
        ).mean()

        loss = next_observation_loss + reward_loss + terminated_loss

        logs = {
            "next_observation_loss": next_observation_loss,
            "reward_loss": reward_loss,
            "terminated_loss": terminated_loss,
            "loss": loss,
        }

        return loss, logs

    @partial(jax.jit, static_argnums=0)
    def train_step(
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, jnp.ndarray]:
        """Performs a single training step (forward pass, loss calculation, gradients, update)."""

        def loss_fn(params):
            return self._compute_loss(params, batch)

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        new_state = state.apply_gradients(grads=grads)

        return new_state, logs

    @partial(jax.jit, static_argnums=0)
    def eval_step(
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Evaluates the model on a batch without updating parameters."""
        loss, _ = self._compute_loss(state.params, batch)
        return loss

    def train(
        self, train_dataset, val_dataset, batch_size, num_train_steps, val_batches
    ):
        """Runs the main training loop."""

        state = self.state

        for step in trange(num_train_steps, desc="Training"):
            if step % 100 == 0:
                val_losses = []
                for _ in range(val_batches):
                    val_batch_np = val_dataset.sample(batch_size)
                    val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

                    # Evaluate using the current state's parameters
                    val_loss = self.eval_step(state, val_batch)
                    val_losses.append(val_loss)

                avg_val_loss = jnp.mean(jnp.array(val_losses))
                if self.logger:
                    self.logger.log({"val/loss": avg_val_loss}, step=step)

            # Sample a new random batch each step and convert to JAX arrays
            batch_np = train_dataset.sample(batch_size)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            state, logs = self.train_step(state, batch)

            if self.logger:
                self.logger.log({
                    "train/learning_rate": self.schedule(step),
                    **{f"train/{k}": v for k, v in logs.items()}
                }, step=step)

        self.state = state

        val_losses = []
        for _ in range(val_batches):
            val_batch_np = val_dataset.sample(batch_size)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}

            # Evaluate using the current state's parameters
            val_loss = self.eval_step(state, val_batch)
            val_losses.append(val_loss)

        avg_val_loss = jnp.mean(jnp.array(val_losses))
        if self.logger:
            self.logger.log({"val/loss": avg_val_loss}, step=step)
