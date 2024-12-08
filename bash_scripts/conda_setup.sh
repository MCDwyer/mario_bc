conda env remove -n mario_bc_env && conda env create -f environment.yml

conda run -n mario_bc_env python --version

conda run -n mario_bc_env pip install numpy==1.21.6
conda run -n mario_bc_env pip install pandas==1.3.5
conda run -n mario_bc_env pip install torch==1.13.1
conda run -n mario_bc_env pip install plotly==5.18.0
conda run -n mario_bc_env pip install nbformat==5.8.0
conda run -n mario_bc_env pip install scikit-learn==1.0.2
conda run -n mario_bc_env pip install matplotlib==3.5.3
conda run -n mario_bc_env pip install stable-baselines3==2.0.0
conda run -n mario_bc_env pip install gymnasium==0.28.1
conda run -n mario_bc_env pip install opencv-python==4.10.0.84
conda run -n mario_bc_env pip install opencv-python-headless==4.10.0.84
conda run -n mario_bc_env pip install optuna==3.6.1
conda run -n mario_bc_env pip install tensorboard==2.11.2
conda run -n mario_bc_env pip install scipy==1.7.3
conda run -n mario_bc_env pip install gym==0.25.2
conda run -n mario_bc_env pip install gym-retro==0.8.0
conda run -n mario_bc_env pip install kaleido==0.2.1
conda run -n mario_bc_env pip install tensorflow==2.11.0

# import nes roms
conda run -n mario_bc_env python -m retro.import "./NES_ROMS/"

# get path for moving the data json files
env_path_var=$(conda run -n mario_bc_env bash -c 'echo $CONDA_PREFIX')
full_retro_data_path="$env_path_var/lib/python3.7/site-packages/retro/data/stable/SuperMarioBros-Nes"

# update the json file for SuperMarioBros-Nes
cp gym_retro_json_updates/SuperMarioBros-Nes/data.json $full_retro_data_path

# make empty directories for the random outputs you might need
mkdir nohup
mkdir model_tuning_outputs
mkdir test_gifs