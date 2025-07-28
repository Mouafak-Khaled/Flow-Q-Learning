import flax
import ogbench
from sklearn.decomposition import PCA
import yaml
from argparser import build_env_model_config_from_args, get_env_model_argparser
from envmodel.initial_observation import Decoder
from utils.data_loader import InitialObservationLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax

from utils.tasks import get_task_title


def plot_pca_distribution(data: np.ndarray, title: str):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        fill=True,
        cmap="viridis",
        bw_adjust=0.5,
        levels=100,
        thresh=0.05,
    )
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


args = get_env_model_argparser().parse_args()
config = build_env_model_config_from_args(args)

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
    config.env_name, config.data_directory
)

train_dataloader = InitialObservationLoader(train_dataset)
val_dataloader = InitialObservationLoader(val_dataset)

real_dataset = np.concatenate(
    [train_dataloader.dataset["observations"], val_dataloader.dataset["observations"]],
    axis=0,
)

plot_pca_distribution(
    real_dataset,
    f"Real Distribution of Initial Observations for {get_task_title(config.env_name)} projected with PCA",
)

env_model_config_path = (
    config.save_directory
    / config.env_name
    / "env_models"
    / "initial_observation_config.yaml"
)
if env_model_config_path.exists():
    with open(env_model_config_path, "r") as f:
        model_config = yaml.unsafe_load(f)
else:
    raise FileNotFoundError(
        f"Configuration file not found at {env_model_config_path}. Please ensure the configuration has been set."
    )

model = Decoder(
    output_dimension=train_dataset["observations"].shape[-1],
    hidden_dims=model_config["hidden_dims"][::-1],
)

env_model_path = (
    config.save_directory / config.env_name / "env_models" / "initial_observation.pt"
)
if env_model_path.exists():
    with open(env_model_path, "rb") as f:
        params_bytes = f.read()
else:
    raise FileNotFoundError(
        f"Model file not found at {env_model_path}. Please ensure the model has been trained."
    )

params = {
    "params": flax.serialization.from_bytes(None, params_bytes)["params"]["decoder"]
}

latent_dim = model_config["latent_dim"]
sampled_noise = jax.random.normal(jax.random.PRNGKey(0), shape=(1100, latent_dim))

generated_samples = model.apply(params, sampled_noise)

plot_pca_distribution(
    generated_samples,
    f"Generated Distribution of Initial Observations for {get_task_title(config.env_name)} projected with PCA",
)
