#!/bin/bash
# bash script to run unsupervised training agent x times

# Check if the required variables are passed to the script
if [ "$#" -ne 3 ]; then
  echo "Usage: ./unsupervised_multi_agent.sh <start_index> <number_of_times> <model>"
  exit 1
fi

# Number of times to run the Python script
START_INDEX=$1
NUM_TIMES=$2
MODEL_NAME=$3

env_path_var=$(conda run -n mario_bc_env bash -c 'echo $CONDA_PREFIX')
source activate $env_path_var

# Loop to run the Python script multiple times
for (( i=START_INDEX; i<(NUM_TIMES + START_INDEX); i++ ))
do
  # nohup python -u unsupervised_training.py $i None $MODEL_NAME > nohup/unsupervised_training_${MODEL_NAME}_${i}.out &
  # echo "Started unsupervised_training.py with start_index $i, output to nohup/unsupervised_training_${MODEL_NAME}_${i}.out"

  # amalgam supervised
  nohup python -u unsupervised_training.py $i amalgam $MODEL_NAME > nohup/supervised_training_${MODEL_NAME}_amalgam_${i}.out &
  echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_${MODEL_NAME}_amalgam_${i}.out"

  # # expert supervised
  # nohup python -u unsupervised_training.py $i expert_distance $MODEL_NAME > nohup/supervised_training_${MODEL_NAME}_expert_distance_${i}.out &
  # echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_${MODEL_NAME}_expert_distance_${i}.out"
  
  # # nonexpert supervised
  # nohup python -u unsupervised_training.py $i nonexpert_distance $MODEL_NAME > nohup/supervised_training_${MODEL_NAME}_nonexpert_distance_${i}.out &
  # echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_${MODEL_NAME}_nonexpert_distance_${i}.out"
done
