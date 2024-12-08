#!/bin/bash
# bash script to sbatch bc tuning

if [ "$#" -ne 1 ]; then
  echo "Usage: batch_unsup_tuning.sh <model_type> <bc_data>"
  exit 1
fi

MODEL_TYPE=$1
BC_DATA=$2

JOB_NAME="TUNING_${MODEL_TYPE}_${BC_DATA}"
sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" unsup_tuning.slurm $MODEL_TYPE $BC_DATA