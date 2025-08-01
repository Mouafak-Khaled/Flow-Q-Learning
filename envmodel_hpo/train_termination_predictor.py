import logging
import time
from functools import partial
from pathlib import Path

import flax
import ogbench
import neps
import torch
import yaml

from envmodel.config import TrainerConfig
from envmodel.loss import weighted_binary_cross_entropy

from envmodel.termination_predictor import TerminationPredictor
from envmodel.termination_predictor_trainer import TerminationPredictorTrainer
from fql.utils.datasets import Dataset
from utils.data_loader import BalancedStepLoader, StepLoader


def training_pipeline(
    env_name: str,
    hidden_dim_0: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    hidden_dim_3: int,
    batch_size: int,
    initial_learning_rate: float,
    final_run: bool = False,
    save_directory: Path = Path("exp"),
):
    start = time.time()
    if final_run:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=save_directory / env_name / "env_models" / "termination_predictor"
        )
    else:
        writer = neps.tblogger.ConfigWriter(write_summary_incumbent=True)

    hidden_dims = [hidden_dim_0, hidden_dim_1, hidden_dim_2, hidden_dim_3]
    hidden_dims = [dim for dim in hidden_dims if dim > 0]
    hidden_dims = tuple(hidden_dims)

    config = TrainerConfig(
        env_name=env_name,
        model="termination_predictor",
        model_config={"hidden_dims": hidden_dims},
        init_learning_rate=initial_learning_rate,
        batch_size=batch_size,
        steps=10000,
    )

    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config.env_name, config.data_directory
    )

    train_dataloader = BalancedStepLoader(train_dataset)
    val_dataloader = StepLoader(val_dataset)

    sample_batch = train_dataloader.sample(config.batch_size)

    model = TerminationPredictor(
        observation_dimension=sample_batch["observations"].shape[-1],
        hidden_dims=hidden_dims,
    )

    trainer = TerminationPredictorTrainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        loss_fn=weighted_binary_cross_entropy,
        config=config,
        writer=writer,
    )

    val_metrics = trainer.train()

    if not final_run:
        writer.add_hparams(
            hparam_dict={
                "initial_learning_rate": float(initial_learning_rate),
                "batch_size": int(batch_size),
                "hidden_dim_0": int(hidden_dim_0),
                "hidden_dim_1": int(hidden_dim_1),
                "hidden_dim_2": int(hidden_dim_2),
                "hidden_dim_3": int(hidden_dim_3),
            },
            metric_dict=val_metrics,
        )

    writer.close()

    if final_run:
        return trainer, config
    else:
        return {
            "objective_to_minimize": -val_metrics["f1"],
            "info_dict": val_metrics,
            "cost": time.time() - start,
        }


def train_termination_predictor(
    env_name: str = "antsoccer-arena-navigate-singletask-task4-v0",
    save_directory: Path = Path("exp"),
):
    pipeline_space = {
        "hidden_dim_0": neps.Integer(32, 128, log=True),
        "hidden_dim_1": neps.Integer(64, 512, log=True),
        "hidden_dim_2": neps.Integer(64, 512, log=True),
        "hidden_dim_3": neps.Integer(32, 128, log=True),
        "initial_learning_rate": neps.Float(1e-5, 1e-2, log=True),
        "batch_size": neps.Integer(32, 512, log=True),
    }

    neps.run(
        evaluate_pipeline=partial(
            training_pipeline,
            env_name=env_name,
        ),
        pipeline_space=pipeline_space,
        max_evaluations_total=100,
        optimizer={
            "name": "bayesian_optimization",
            "device": "cpu",
        },
        root_directory=save_directory
        / env_name
        / "env_models"
        / "termination_predictor_neps",
        post_run_summary=True,
    )

    _, summary = neps.status(
        root_directory=save_directory
        / env_name
        / "env_models"
        / "termination_predictor_neps"
    )

    trainer, config = training_pipeline(
        env_name=env_name,
        hidden_dim_0=int(summary["hidden_dim_0"]),
        hidden_dim_1=int(summary["hidden_dim_1"]),
        hidden_dim_2=int(summary["hidden_dim_2"]),
        hidden_dim_3=int(summary["hidden_dim_3"]),
        batch_size=int(summary["batch_size"]),
        initial_learning_rate=float(summary["initial_learning_rate"]),
        final_run=True,
        save_directory=save_directory,
    )

    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config.env_name, config.data_directory
    )

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    training_metrics, validation_metrics = trainer.validate(train_dataset, val_dataset)

    for k, v in training_metrics.items():
        print(f"Training {k}: {v:.4f}")

    for k, v in validation_metrics.items():
        print(f"Validation {k}: {v:.4f}")

    save_dir = Path(config.save_directory) / config.env_name / "env_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / f"{config.model}_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_config, f)

    model_path = save_dir / f"{config.model}.pt"
    with open(model_path, "wb") as f:
        f.write(flax.serialization.to_bytes(trainer.state.params))
