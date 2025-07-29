#!/bin/bash

#SBATCH --job-name=TrainTerminationPredictorCube_FlowQLearning
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=16GB
#SBATCH --time=00:10:00
#SBATCH --output=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.out
#SBATCH --error=/work/dlclarge2/amriam-fql/logs/slurm-%A_%a.err

# Activate your environment
source ~/miniconda3/bin/activate
conda activate fql

python train_env_model.py \
    --env_name=cube-single-play-singletask-task2-v0 --model=termination_predictor --steps=400000 \
    --save_directory=/work/dlclarge2/amriam-fql/exp/ --data_directory=/work/dlclarge2/amriam-fql/data/
