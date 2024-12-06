#!/bin/bash
# bash script to sbatch multiple agents worth of bc_training

if [ "$#" -ne 5 ]; then
  echo "Usage: batch_bc_agents.sh <start_index> <number_of_times> <model_type> <bc_data> <exp_id>"
  exit 1
fi

# Number of times to run the Python script
START_INDEX=$1
NUM_TIMES=$2
MODEL_TYPE=$3
BC_DATA=$4
EXP_ID=$5

# Loop to run the Python script multiple times
for (( i=START_INDEX; i<(NUM_TIMES + START_INDEX); i++ ))
do
    sbatch bc_training.slurm $MODEL_TYPE $i $BC_DATA $EXP_ID
done