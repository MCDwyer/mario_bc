conda env create --force -f environment.yml

conda run -n mario_bc_env python --version

# conda install -c conda-forge setuptools=41.2.0 numpy pandas torch

# conda run -n mario_bc_env pip --version
# conda run -n mario_bc_env pip install -U pip setuptools==41.2.0
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