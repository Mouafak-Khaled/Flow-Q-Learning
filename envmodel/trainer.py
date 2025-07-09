import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange

import sys
sys.path.append(".")

import ogbench
from fql.utils.datasets import Dataset
from envmodel.SimpleCubeEnvModel import SimpleCubeEnvModel

# --- Load data ---
env, train_dataset, val_dataset = ogbench.make_env_and_datasets("cube-single-play-singletask-task2-v0")
train_dataset = Dataset.create(**train_dataset)
val_dataset = Dataset.create(**val_dataset)

# --- Hyperparameters ---
batch_size = 256
learning_rate = 1e-3
num_train_steps = 10000
val_batches = 20
seed = 42

# --- Model ---
env_model = SimpleCubeEnvModel()
rng = jax.random.PRNGKey(seed)

# Sample once to get shapes
sample_batch = train_dataset.sample(batch_size)
obs_sample = jnp.array(sample_batch["observations"])
act_sample = jnp.array(sample_batch["actions"])

# Initialize parameters
params = env_model.init(rng, obs_sample, act_sample)
print("Model initialized.")

# --- Optimizer ---
optimizer = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=env_model.apply, params=params, tx=optimizer)

# --- Loss function ---
def mse_loss(pred_next_obs, pred_reward, true_next_obs, true_reward):
    obs_loss = jnp.mean((pred_next_obs - true_next_obs) ** 2)
    reward_loss = jnp.mean((pred_reward - true_reward) ** 2)
    return 28 * obs_loss + reward_loss

# --- Training step ---
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        pred_next_obs, pred_reward = env_model.apply(params, batch["observations"], batch["actions"])
        loss = mse_loss(pred_next_obs, pred_reward, batch["next_observations"], batch["rewards"])
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# --- Validation step ---
@jax.jit
def eval_step(params, batch):
    pred_next_obs, pred_reward = env_model.apply(params, batch["observations"], batch["actions"])
    loss = mse_loss(pred_next_obs, pred_reward, batch["next_observations"], batch["rewards"])
    return loss

# --- Training loop ---
for step in trange(num_train_steps, desc="Training"):
    # Sample a new random batch each step
    batch_np = train_dataset.sample(batch_size)
    batch = {k: jnp.array(v) for k, v in batch_np.items()}

    # One training update
    state, loss = train_step(state, batch)

    # Optionally print/validate every N steps
    if (step + 1) % 500 == 0:
        val_losses = []
        for _ in range(val_batches):
            val_batch_np = val_dataset.sample(batch_size)
            val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}
            val_loss = eval_step(state.params, val_batch)
            val_losses.append(val_loss)
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        print(f"Step {step+1}: Train Loss = {loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# Now validate against the env
next_observations = []
rewards = []
pred_next_observations = []
pred_rewards = []

obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    next_obs, reward, done, _, _ = env.step(action)
    next_observations.append(next_obs)
    rewards.append(reward)
    pred_next_obs, pred_reward = env_model.apply(state.params, obs, action)
    pred_next_observations.append(pred_next_obs)
    pred_rewards.append(pred_reward)
    obs = next_obs

# Convert lists to arrays
next_observations = jnp.array(next_observations)
rewards = jnp.array(rewards)
pred_next_observations = jnp.array(pred_next_observations)
pred_rewards = jnp.array(pred_rewards)

# Calculate final validation loss
final_val_loss = mse_loss(pred_next_observations, pred_rewards, next_observations, rewards)
print(f"Final Validation Loss: {final_val_loss:.4f}")
