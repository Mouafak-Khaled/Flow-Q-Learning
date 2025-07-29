from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange
from wandb.sdk.wandb_run import Run

from envmodel.baseline import StatePredictor
from envmodel.config import TrainerConfig
from envmodel.termination_predictor import TerminationPredictor
from utils.data_loader import DataLoader

LossFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]],
]


class Trainer:
    def __init__(
        self,
        state_predictor: StatePredictor,
        termination_predictor: TerminationPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: LossFn,
        config: TrainerConfig,
        logger: Run | None = None,
    ):
        self.state_predictor = state_predictor
        self.termination_predictor = termination_predictor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger

        self.rng = jax.random.PRNGKey(self.config.seed)

        sample_batch = self.train_loader.sample(self.config.batch_size)
        self.params = {
            "state_predictor": self.state_predictor.init(
                self.rng, **sample_batch, key=self.rng
            ),
            "termination_predictor": self.termination_predictor.init(
                self.rng, **sample_batch, key=self.rng
            ),
        }

        self.schedule = optax.cosine_decay_schedule(
            init_value=self.config.init_learning_rate,
            decay_steps=self.config.steps,
        )
        self.optimizer = optax.adam(self.schedule)

        self.state = train_state.TrainState.create(
            apply_fn=self.apply, params=self.params, tx=self.optimizer
        )

    def apply(
        self, params: Dict[str, Any], batch: Dict[str, jnp.ndarray], key: jax.Array
    ) -> jnp.ndarray:
        predicted_next_observation = self.state_predictor.apply(
            params["state_predictor"], **batch, key=key
        )
        predicted_termination_ground_truth_based = self.termination_predictor.apply(
            params["termination_predictor"], **batch, key=key
        )
        kwargs = {**batch}
        kwargs["next_observations"] = predicted_next_observation
        predicted_termination_prediction_based = self.termination_predictor.apply(
            params["termination_predictor"], **kwargs, key=key
        )
        return (
            predicted_next_observation,
            predicted_termination_ground_truth_based,
            predicted_termination_prediction_based,
        )

    @partial(jax.jit, static_argnums=0)
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jax.Array,
    ) -> Tuple[train_state.TrainState, jnp.ndarray, jax.Array]:
        """Performs a single training step (forward pass, loss calculation, gradients, update)."""

        def loss_fn(params):
            rng_, key = jax.random.split(rng)
            (
                predicted_next_observation,
                predicted_termination_ground_truth_based,
                predicted_termination_prediction_based,
            ) = self.apply(params, batch, key=key)
            loss, logs = self.loss_fn(
                predicted_next_observation,
                batch["next_observations"],
                predicted_termination_ground_truth_based,
                predicted_termination_prediction_based,
                batch["rewards"] == 0,
            )
            return loss, (logs, rng_)

        (_, (logs, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )

        new_state = state.apply_gradients(grads=grads)
        return new_state, logs, rng

    @partial(jax.jit, static_argnums=0)
    def eval_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jax.Array,
    ) -> Tuple[Dict[str, jnp.ndarray], jax.Array]:
        """Evaluates the model on a batch without updating parameters."""
        rng, key = jax.random.split(rng)
        (
            predicted_next_observation,
            predicted_termination_ground_truth_based,
            predicted_termination_prediction_based,
        ) = self.apply(
            state.params, batch, key=key
        )
        _, logs = self.loss_fn(
            predicted_next_observation,
            batch["next_observations"],
            predicted_termination_ground_truth_based,
            predicted_termination_prediction_based,
            batch["rewards"] == 0,
        )
        return logs, rng

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
                    logs, self.rng = self.eval_step(state, val_batch, self.rng)
                    val_logs.append(logs)

                for k in val_logs[0].keys():
                    avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
                    if self.logger:
                        self.logger.log({f"val/{k}": avg_val_log}, step=step)

            # Sample a new random batch each step and convert to JAX arrays
            batch_np = self.train_loader.sample(self.config.batch_size)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}

            state, logs, self.rng = self.train_step(state, batch, self.rng)

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
            logs, self.rng = self.eval_step(state, val_batch, self.rng)
            val_logs.append(logs)

        for k in val_logs[0].keys():
            avg_val_log = jnp.mean(jnp.array([log[k] for log in val_logs]))
            if self.logger:
                self.logger.log({f"val/{k}": avg_val_log}, step=step)
