#!/bin/bash
# bash script to sbatch bc tuning

if [ "$#" -ne 2 ]; then
  echo "Usage: batch_bc_tuning.sh <model_type>"
  exit 1
fi

MODEL_TYPE=$3
BC_DATA=$4

# Define a space-separated list
BC_DATA_OPTIONS="amalgam expert_distance nonexpert_distance"

# Iterate over the list
for item in $BC_DATA_OPTIONS; do
    sbatch model_tuning.slurm $item $MODEL_TYPE
done