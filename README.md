Python 3.7.2
gym-retro 0.8.0
gym 0.25.2
setuptools needs version 41.2.0
numpy

python -m retro.import './NES_ROMS/'

for plotting:
pandas
plotly
nbformat
kaleido
nbformat

for clustering:
scikit-learn


for interactive play:
matplotlib


for models:
stable-baselines3
gymnasium
cv2 (opencv-python):
    opencv-python==4.10.0.84
    opencv-python-headless==4.10.0.84
optuna (for hyper-parameter tuning)

for training stable baselines3 logs:
tensorboard


for statistical testing:
scipy


Initialising environments:
running conda_setup.sh should create and install everything in a conda environment named 'mario_bc_env'
this needs python 3.7.2 though, so need to make sure you can create a conda env with that

it should also initialise the NES roms and update the data.json in the retro package to work with the GymEnvs/retro_env_wrapper

Alternatively, can use the requirements.txt to set up a venv, will need to then run these commands in the venv to set up the gym-retro stuff:
python -m retro.import "./NES_ROMS/"

cp gym_retro_json_updates/SuperMarioBros-Nes/data.json [PATH_TO_VENV]/lib/python3.7/site-packages/retro/data/stable/SuperMarioBros-Nes
