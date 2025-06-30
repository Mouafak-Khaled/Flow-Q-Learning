import argparse

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments to override trainer configuration.

    Returns:
        argparse.Namespace: A namespace containing environment name, discount factor,
                            and buffer size for training configuration.

    Command-line Arguments:
        --env_name (str): Required. The environment to train on. Must be one of:
            - 'antsoccer-arena-navigate-singletask-task2-v0'
            - 'antsoccer-arena-navigate-singletask-task4-v0'
            - 'cube-single-play-singletask-task2-v0'

        --discount (float): Optional. The discount factor for future rewards.
            Defaults to 0.995.

        --buffer_size (int): Optional. The size of the replay buffer.
            Defaults to 100000.

        --use_wandb (flag): Optional. If provided, enables wandb logging.
            Defaults to False.
    """
    parser = argparse.ArgumentParser(description="Trainer Config Overrides")

    parser.add_argument(
        "--env_name",
        type=str,
        help="Environment name",
        required=True,
        choices=[
            'antsoccer-arena-navigate-singletask-task2-v0',
            'antsoccer-arena-navigate-singletask-task4-v0',
            'cube-single-play-singletask-task2-v0'
        ]
    )

    parser.add_argument(
        "--discount",
        type=float,
        default=0.995,
        help="Discount factor"
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100000,
        help="Replay buffer size"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: False)"
    )

    return parser.parse_args()