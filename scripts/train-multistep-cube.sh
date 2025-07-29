#!/bin/bash

#SBATCH --job-name=TrainMultistepCube_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=32GB
#SBATCH --time=05:00:00
#SBATCH --output=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.out
#SBATCH --error=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python train_env_model.py \
    --env_name=cube-single-play-singletask-task2-v0 --model=multistep --termination_weight=0 --steps=10000 \
    --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/

python evaluate_env_model.py --env_name=cube-single-play-singletask-task2-v0 --model=multistep --eval_episodes=50 \
    --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/

python evaluate_success_rate_correlation.py --env_name=cube-single-play-singletask-task2-v0 --model=multistep \
    --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
    --number_of_seeds=2 --number_of_alphas=20 --eval_episodes=50
