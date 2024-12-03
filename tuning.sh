#!/bin/bash
# bash script to run unsupervised tuning

nohup python -u model_tuning.py None PPO > nohup/tuning_PPO.out 2>&1 &
echo "Started unsupervised PPO tuning, output to nohup/tuning_PPO.out"

nohup python -u model_tuning.py None DQN > nohup/tuning_DQN.out 2>&1 &
echo "Started unsupervised DQN tuning, output to nohup/tuning_DQN.out"

nohup python -u model_tuning.py None SAC > nohup/tuning_SAC.out 2>&1 &
echo "Started unsupervised SAC tuning, output to nohup/tuning_SAC.out"