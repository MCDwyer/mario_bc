#!/bin/bash
# bash script to run unsupervised tuning

conda run -n mario_bc_env nohup python -u model_tuning.py amalgam PPO > nohup/tuning_PPO_amalgam.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_amalgam.out"

conda run -n mario_bc_env nohup python -u model_tuning.py expert PPO > nohup/tuning_PPO_expert.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_expert.out"

conda run -n mario_bc_env nohup python -u model_tuning.py nonexpert PPO > nohup/tuning_PPO_nonexpert.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_nonexpert.out"
