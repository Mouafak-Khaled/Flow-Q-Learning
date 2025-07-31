import time
from functools import partial
from pathlib import Path

import ogbench
import neps

from envmodel.config import TrainerConfig
from envmodel.loss import focal_loss

from envmodel.termination_predictor import TerminationPredictor
from envmodel.termination_predictor_trainer import TerminationPredictorTrainer
from utils.data_loader import StepLoader


def training_pipeline(
    env_name: str,
    hidden_dim_0: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    hidden_dim_3: int,
    batch_size: int,
    initial_learning_rate: float
):
    start = time.time()
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
        steps=2000,
    )

    _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config.env_name, config.data_directory
    )

    train_dataloader = StepLoader(train_dataset)
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
        loss_fn=focal_loss,
        config=config,
        writer=writer,
    )

    val_metrics = trainer.train()

    writer.add_hparams(
        hparam_dict={
            "initial_learning_rate": float(initial_learning_rate),
            "batch_size": int(batch_size),
            "hidden_dim_0": int(hidden_dim_0),
            "hidden_dim_1": int(hidden_dim_1),
            "hidden_dim_2": int(hidden_dim_2),
            "hidden_dim_3": int(hidden_dim_3)
        },
        metric_dict=val_metrics,
    )

    return {
        "objective_to_minimize": -val_metrics["f1"],
        "info_dict": val_metrics,
        "cost": time.time() - start,
    }


if __name__ == "__main__":
    env_name = "antsoccer-arena-navigate-singletask-task4-v0"
    pipeline_space = {
        "hidden_dim_0": neps.Categorical([32, 64, 128]),
        "hidden_dim_1": neps.Categorical([0, 64, 128, 256, 512]),
        "hidden_dim_2": neps.Categorical([0, 64, 128, 256, 512]),
        "hidden_dim_3": neps.Categorical([32, 64, 128]),
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
        root_directory=Path("exp")
        / env_name
        / "env_models"
        / "termination_predictor_neps",
        post_run_summary=True,
    )
