#!/bin/bash
# bash script to run supervised tuning

env_path_var=$(conda run -n mario_bc_env bash -c 'echo $CONDA_PREFIX')

source activate $env_path_var

# PPO tuning
nohup python -u model_tuning.py amalgam PPO > nohup/tuning_PPO_amalgam.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_amalgam.out"

nohup python -u model_tuning.py expert PPO > nohup/tuning_PPO_expert.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_expert.out"

nohup python -u model_tuning.py nonexpert PPO > nohup/tuning_PPO_nonexpert.out 2>&1 &
echo "Started supervised PPO tuning, output to nohup/tuning_PPO_nonexpert.out"
