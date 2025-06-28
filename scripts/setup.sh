#!/bin/bash

#SBATCH --partition dllabdlc_gpu-rtx2080
#SBATCH --job-name SetupFlowQLearningDependencies

#SBATCH --output logs/%x-%A-SetupFlowQLearningDependencies.out
#SBATCH --error logs/%x-%A-SetupFlowQLearningDependencies.err

#SBATCH --mem 8GB

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/miniconda3/bin/activate

# Running the job
start=`date +%s`

conda update -y -n base -c defaults conda

conda create -y --name fql python=3.10
conda activate fql

pip install -r fql/requirements.txt
pip install -r requirements.txt

conda install -y -c conda-forge glew
conda install -y -c conda-forge mesalib
conda install -y -c anaconda mesa-libgl-cos6-x86_64
conda install -y -c menpo glfw3

conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate fql

mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

conda env config vars set MJLIB_PATH=$HOME/.mujoco/mujoco210/bin/libmujoco210.so \
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin \
	MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
conda deactivate && conda activate fql

conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda deactivate && conda activate fql

conda install -y patchelf

python -c "import mujoco_py; print(mujoco_py.cymj)"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
