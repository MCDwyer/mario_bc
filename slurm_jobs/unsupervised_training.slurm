#!/bin/bash
#=============================================
# Slurm script
# m.dwyer@soton.ac.uk
#=============================================
# Slurm batch parameters
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.dwyer@soton.ac.uk
#SBATCH --time=60:00:00
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --ntasks=64 # number of cores
#SBATCH --nodes=1  # Number of nodes - keep nodes to minimum to avoid overheads for data transfer between nodes
#SBATCH --mem=10G
#=============================================
# Actual run commands

MODEL_TYPE=$1
AGENT_INDEX=$2
BC_DATA=$3
EXP_ID=$4

export CUDA_VISIBLE_DEVICES=""

cd /scratch/mcd2g19/mario_bc

source activate /home/mcd2g19/.conda/envs/mario_bc_env

python -u unsupervised_training.py $AGENT_INDEX $BC_DATA $MODEL_TYPE $EXP_ID