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
  nohup python -u unsupervised_training.py $i > nohup/unsupervised_training_${i}.out &
  echo "Started unsupervised_training.py with start_index $i, output to nohup/unsupervised_training_${i}.out"
done