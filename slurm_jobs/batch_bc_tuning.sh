#!/bin/bash
# bash script to sbatch bc tuning

if [ "$#" -ne 1 ]; then
  echo "Usage: batch_bc_tuning.sh <model_type>"
  exit 1
fi

MODEL_TYPE=$1

# Define a space-separated list
BC_DATA_OPTIONS="amalgam expert_distance nonexpert_distance"

# Iterate over the list
for item in $BC_DATA_OPTIONS; do
    JOB_NAME="BC_TUNING_$item_$MODEL_TYPE"
    sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" bc_tuning.slurm $item $MODEL_TYPE
done