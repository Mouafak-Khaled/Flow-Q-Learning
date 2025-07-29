from pathlib import Path
from typing import Dict
import flax
import yaml

import jax.numpy as jnp


def load_model(save_directory: Path, env_name: str, model: str, sample_batch: Dict[str, jnp.ndarray]):
    config_path = (
        save_directory / env_name / "env_models" / f"{model}_config.yaml"
    )
    if config_path.exists():
        with open(config_path, "r") as f:
            model_config = yaml.unsafe_load(f)
    else:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. Please ensure the configuration has been set."
        )

    if model in ("baseline", "multistep"):
        from envmodel.baseline import BaselineStatePredictor
        env_model = BaselineStatePredictor(
            observation_dimension=sample_batch["observations"].shape[-1],
            action_dimension=sample_batch["actions"].shape[-1],
            hidden_dims=model_config["hidden_dims"],
        )
    elif model == "termination_predictor":
        from envmodel.termination_predictor import TerminationPredictor
        env_model = TerminationPredictor(
            observation_dimension=sample_batch["observations"].shape[-1],
            hidden_dims=model_config["hidden_dims"],
        )
    else:
        raise ValueError(f"Unknown model type: {model}")

    params_path = save_directory / env_name / "env_models" / f"{model}.pt"
    if params_path.exists():
        with open(params_path, "rb") as f:
            params_bytes = f.read()
    else:
        raise FileNotFoundError(
            f"Model file not found at {params_path}. Please ensure the model has been trained."
        )

    if model == "multistep":
        params = {
            "params": flax.serialization.from_bytes(None, params_bytes)["params"]["ScanCell_0"]["cell"]
        }
    else:
        params = flax.serialization.from_bytes(None, params_bytes)

    def apply_fn(*args, **kwargs):
        return env_model.apply(params, *args, **kwargs)
    
    return apply_fn
