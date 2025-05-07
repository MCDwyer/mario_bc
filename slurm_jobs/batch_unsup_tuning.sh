#!/bin/bash
# bash script to sbatch bc tuning

if [ "$#" -ne 2 ]; then
  echo "Usage: batch_unsup_tuning.sh <bc_data> <exp_id>"
  exit 1
fi

BC_DATA=$1
EXP_ID=$2

# Define a space-separated list
# MODEL_NAMES="PPO DQN SAC"
# MODEL_NAMES="PPO DQN"
MODEL_NAMES="DQN"

# Iterate over the list
for MODEL_TYPE in $MODEL_NAMES; do
  for (( i=1; i<20; i++ ))
  do
    JOB_NAME="TUNING_${MODEL_TYPE}_${BC_DATA}_${i}"
    if [ $MODEL_TYPE != "SAC" ]; then
      sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" unsup_tuning.slurm $MODEL_TYPE $BC_DATA $EXP_ID
    else
      sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" sac_unsup_tuning.slurm $MODEL_TYPE $BC_DATA $EXP_ID
    fi
    # sleep 15
  done
done

