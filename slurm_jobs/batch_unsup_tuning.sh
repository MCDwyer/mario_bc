#!/bin/bash
# bash script to sbatch bc tuning

if [ "$#" -ne 1 ]; then
  echo "Usage: batch_unsup_tuning.sh <bc_data>"
  exit 1
fi

BC_DATA=$1

# Define a space-separated list
MODEL_NAMES="PPO DQN SAC"

# Iterate over the list
for MODEL_TYPE in $MODEL_NAMES; do
    JOB_NAME="TUNING_${MODEL_TYPE}_${BC_DATA}"
    sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" unsup_tuning.slurm $MODEL_TYPE $BC_DATA
done

