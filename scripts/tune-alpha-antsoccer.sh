#!/bin/bash

#SBATCH --array=0-26%1
#SBATCH --job-name=TuneAlphaAntSoccer_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/work/dlclarge2/amriam-fql/logs-checkpointing/slurm-%A_%a.out
#SBATCH --error=/work/dlclarge2/amriam-fql/logs-checkpointing/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python tune_alpha.py \
    --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.layer_norm \
    --save_directory=/work/dlclarge2/amriam-fql/exp-checkpointing/ --data_directory=/work/dlclarge2/amriam-fql/data/ \
    --number_of_seeds=2 --number_of_alphas=20 --use_wandb \
    --eval_interval=20000 --eval_episodes=50 --single_experiment --job_id=$SLURM_ARRAY_TASK_ID
