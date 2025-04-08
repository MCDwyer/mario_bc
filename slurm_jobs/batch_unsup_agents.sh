#!/bin/bash
# bash script to sbatch multiple agents worth of bc_training

if [ "$#" -ne 4 ]; then
  echo "Usage: batch_unsup_agents.sh <start_index> <model_type> <bc_data> <exp_id>"
  exit 1
fi

# Number of times to run the Python script
START_INDEX=$1
NUM_TIMES=5
MODEL_TYPE=$2
BC_DATA=$3
EXP_ID=$4

OUTPUT_PATH="experiments/$EXP_ID/outputs"

mkdir -p "$OUTPUT_PATH"


# Loop to run the Python script multiple times
for (( i=START_INDEX; i<(NUM_TIMES + START_INDEX); i++ ))
do
    JOB_NAME="${EXP_ID}_TRAINING_${BC_DATA}_${MODEL_TYPE}_${i}"
    sbatch --job-name=$JOB_NAME --output="${JOB_NAME}.out" --error="${JOB_NAME}.err" unsupervised_training.slurm $MODEL_TYPE $i $BC_DATA $EXP_ID
done