#!/bin/bash

#SBATCH --job-name=EvaluateBaseline_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --output=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.out
#SBATCH --error=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python evaluate_success_rate_correlation.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --model=baseline \
       --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
       --seed=42 --eval_episodes=50

python evaluate_success_rate_correlation.py --env_name=cube-single-play-singletask-task2-v0 --model=baseline \
       --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
       --seed=42 --eval_episodes=50
