#!/bin/bash

#SBATCH --job-name=TuneAlphaAntSoccer_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python tune_alpha.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.discount=0.995 --evaluation_mode --use_wandb
