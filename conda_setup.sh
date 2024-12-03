conda create --force -n mario_bc_env python=3.7.2

conda run -n mario_bc_env python --version

# conda install -c conda-forge setuptools=41.2.0 numpy pandas torch

conda run -n mario_bc_env pip install numpy
conda run -n mario_bc_env pip install pandas
conda run -n mario_bc_env pip install torch
conda run -n mario_bc_env pip install plotly
conda run -n mario_bc_env pip install nbformat
conda run -n mario_bc_env pip install scikit-learn
conda run -n mario_bc_env pip install matplotlib
conda run -n mario_bc_env pip install stable-baselines3
conda run -n mario_bc_env pip install gymnasium
conda run -n mario_bc_env pip install opencv-python==4.10.0.84
conda run -n mario_bc_env pip install opencv-python-headless==4.10.0.84
conda run -n mario_bc_env pip install optuna
conda run -n mario_bc_env pip install tensorboard
conda run -n mario_bc_env pip install scipy
conda run -n mario_bc_env pip install gym==0.25.2
conda run -n mario_bc_env pip install gym-retro==0.8.0
conda run -n mario_bc_env pip install kaleido
conda run -n mario_bc_env pip install setuptools==41.2.0