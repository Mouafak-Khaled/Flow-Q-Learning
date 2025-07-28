from argparser import get_env_model_argparser, build_env_model_config_from_args
from evaluator.env_model_evaluator import EnvModelEvaluator
from task.offline_task_real import OfflineTaskWithRealEvaluations
from task.offline_task_simulated import OfflineTaskWithSimulatedEvaluations
from utils.agent import load_agent


parser = get_env_model_argparser()
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=50,
    help="Number of evaluation episodes for the environment model.",
)

args = parser.parse_args()
config = build_env_model_config_from_args(args)

real_task = OfflineTaskWithRealEvaluations(
    config.env_name,
    data_directory=config.data_directory,
    num_evaluation_envs=args.eval_episodes,
)

simulated_task = OfflineTaskWithSimulatedEvaluations(
    config.env_name,
    model=config.model,
    data_directory=config.data_directory,
    save_directory=config.save_directory,
    num_evaluation_envs=args.eval_episodes,
)

agent = load_agent(
    agent_directory=config.save_directory / config.env_name / "best_run",
    sample_batch=real_task.sample("train", 1),
)

env_model_evaluator = EnvModelEvaluator(
    real_task=real_task,
    simulated_task=simulated_task,
    agent=agent,
    seed=config.seed,
)

env_model_evaluator.evaluate()
env_model_evaluator.plot_comparison()
env_model_evaluator.close()
