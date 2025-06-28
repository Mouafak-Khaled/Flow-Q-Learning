#!/bin/bash

#SBATCH --job-name=ReproduceAntSoccerResults_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --array=0-7%1

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

# Run the experiment with the current SLURM_ARRAY_TASK_ID
python reproduce.py --task_id ${SLURM_ARRAY_TASK_ID} --env_name=cube-single-play-singletask-task2-v0 --agent.alpha=300
