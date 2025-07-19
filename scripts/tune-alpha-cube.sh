#!/bin/bash

#SBATCH --array=0-31%1
#SBATCH --job-name=TuneAlphaCube_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --output=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.out
#SBATCH --error=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python tune_alpha.py \
    --env_name=cube-single-play-singletask-task2-v0 --agent.layer_norm \
    --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
    --number_of_seeds=2 --number_of_alphas=20 --use_wandb \
    --eval_interval=20000 --eval_episodes=50 --single_experiment --job_id=$SLURM_ARRAY_TASK_ID
