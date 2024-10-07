#!/bin/bash
# bash script to run unsupervised training agent x times

# Check if the required variables are passed to the script
if [ "$#" -ne 2 ]; then
  echo "Usage: ./unsupervised_multi_agent.sh <start_index> <number_of_times>"
  exit 1
fi

# Number of times to run the Python script
START_INDEX=$1
NUM_TIMES=$2

# Loop to run the Python script multiple times
for (( i=START_INDEX; i<(NUM_TIMES + START_INDEX); i++ ))
do
  nohup python -u unsupervised_training.py $i None PPO > nohup/unsupervised_training_PPO_${i}.out &
  echo "Started unsupervised_training.py with start_index $i, output to nohup/unsupervised_training_PPO_${i}.out"

  # amalgam supervised
  nohup python -u unsupervised_training.py $i amalgam PPO > nohup/supervised_training_PPO_amalgam_${i}.out &
  echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_PPO_amalgam_${i}.out"

  # # expert supervised
  # nohup python -u unsupervised_training.py $i expert_distance PPO > nohup/supervised_training_PPO_expert_distance_${i}.out &
  # echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_PPO_expert_distance_${i}.out"
  
  # # nonexpert supervised
  # nohup python -u unsupervised_training.py $i nonexpert_distance PPO > nohup/supervised_training_PPO_nonexpert_distance_${i}.out &
  # echo "Started unsupervised_training.py with start_index $i, output to nohup/supervised_training_PPO_nonexpert_distance_${i}.out"
done
