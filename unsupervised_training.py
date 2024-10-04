from stable_baselines3 import PPO, DQN, SAC
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import json
import sys
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

import behavioural_cloning

TIMESTEP_INCREMENT = 1000000
TIMESTEPS = 10000000
# UNSUPERVISED = False
RETRAINING = False
# MODEL_NAME = "PPO"
# MODEL_CLASS = PPO
POLICY = "CnnPolicy"
# TRAINING_DATA_NAME = "amalgam"
# TRAINING_DATA_NAME = "expert_distance"
# TRAINING_DATA_NAME = "nonexpert_distance"
LEVEL_CHANGE = "random"

# TRAINING_FILEPATH = "user_data_processed_for_bc/"
# TRAINING_FILEPATH += TRAINING_DATA_NAME + "_bc_data.obj"
# LEVEL_CHANGE = "single_level_Level1-1"

# training_data_name = "expert_score"
# training_data_name = "nonexpert_score"

# training_data_name = "slower"
# training_data_name = "faster"

def ppo_model(env, log_dir):

    if UNSUPERVISED:
        params = {'rl_learning_rate': 0.0004256977228617011, 'n_steps': 3588, 'gamma': 0.9244431355829242, 'gae_lambda': 0.901757161292598, 'ent_coef': 0.007033089897296567, 'clip_range': 0.31361009245800875, 'vf_coef': 0.824772120758773}
    else:
        params = {'learning_rate': 0.0009422578032986744, 'n_epochs': 15, 'batch_size': 1043, 'rl_learning_rate': 0.00019810071043939884, 'n_steps': 608, 'gamma': 0.9632436461255943, 'gae_lambda': 0.8365664231014514, 'ent_coef': 0.0014620665698483156, 'clip_range': 0.2873084366251664, 'vf_coef': 0.20790175758482327}


    model = PPO(POLICY,
                env,
                batch_size=params['n_steps'],
                learning_rate=params['rl_learning_rate'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                gae_lambda=params['gae_lambda'],
                ent_coef=params['ent_coef'],
                clip_range=params['clip_range'],
                vf_coef=params['vf_coef'],
                verbose=1,
                tensorboard_log=log_dir,
                )
    
    return model, params

def main(agent_index):

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    log_dir = f"training_logs/level_change_{LEVEL_CHANGE}/"

    if UNSUPERVISED:
        log_dir = f"{log_dir}unsupervised/"
    else:
        log_dir = f"{log_dir}supervised/{TRAINING_DATA_NAME}/"

    env = gym.make('MarioEnv-v0')

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Stack frames to allow temporal information to be captured
    env = VecFrameStack(env, n_stack=1)

    string_timesteps = f"{int(TIMESTEPS/1000)}k"

    model_path = log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}"

    if RETRAINING:
        previous_timestep_string = f"{int((TIMESTEPS-TIMESTEP_INCREMENT)/1000)}k"
        previous_model_path = log_dir + f"{previous_timestep_string}_{MODEL_NAME}_{agent_index}"

        model = MODEL_CLASS.load(previous_model_path, env, verbose=1, tensorboard_log=log_dir)
    else:
        # if MODEL_CLASS == PPO:
        #     model, params = ppo_model(env, log_dir)
        # else:
        #     model, params = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=log_dir)

        if MODEL_NAME == "DQN":
            model = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=log_dir, buffer_size=450000)
        else:
            model = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=log_dir)

        if not UNSUPERVISED:
            # bc_model, bc_model_path = behavioural_cloning.behavioural_cloning_with_imitation(env, model_path, TRAINING_FILEPATH)
            # model = MODEL_CLASS.load(bc_model_path, env, verbose=1, tensorboard_log=log_dir)
            # print(env.action_space)
            bc_model_path = f"{model_path}_bc"
            model = behavioural_cloning.behavioural_cloning(model, TRAINING_FILEPATH, bc_model_path) #, params["learning_rate"], params["n_epochs"], params["batch_size"])

    # print(model.policy)

    if UNSUPERVISED:
        name_prefix = f"unsupervised_{MODEL_NAME}_{string_timesteps}_{agent_index}"
    else:
        name_prefix = f"supervised_{MODEL_NAME}_with_{TRAINING_DATA_NAME}_{string_timesteps}_{agent_index}"

    # Configure logging
    tmp_path = log_dir + f"/{name_prefix}/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Define callbacks
    eval_callback = EvalCallback(env, best_model_save_path=f"{log_dir}{name_prefix}/",
                                log_path=log_dir, eval_freq=500,
                                deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=250000, save_path=log_dir,
                                            name_prefix=name_prefix)

    # Train the model
    timesteps = TIMESTEP_INCREMENT if RETRAINING else TIMESTEPS
    model.learn(total_timesteps=timesteps, callback=[eval_callback, checkpoint_callback])
    model.save(model_path)

    # save the history from the env
    history_json_filepath = f"{model_path}.json"

    with open(history_json_filepath, "w") as outfile:
        json.dump(env.history, outfile)

    # env.set_record_option("test_bk2s/.")

    # Load the trained agent to check it's saved properly
    model = MODEL_CLASS.load(model_path)

    # Run a 1000 timesteps to generate a gif just as a check measure (not actual evaluation)
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with an empty image
    im = ax.imshow(np.zeros((84, 84)), cmap='gray', vmin=0, vmax=255)

    global OBS
    OBS, _ = env.reset(options={"level": "Level1-1"})

    # Function to update the image
    def update_img(frame):
        global OBS
        action, _states = model.predict(OBS)

        OBS, _, done, _, _ = env.step(int(action))
        obs = np.array(OBS).squeeze()

        if done:
            OBS, _ = env.reset(options={"level": "Level1-1"})
        # Update the image data
        im.set_data(obs)
        return [im]

    # Create an animation
    ani = animation.FuncAnimation(fig, update_img, frames=1000, blit=True, interval=100)
    animation_path = "test_gifs/"

    if UNSUPERVISED:
        animation_path += "unsupervised"
    else:
        animation_path += f"supervised_w_{TRAINING_DATA_NAME}"

    animation_path += f"_{MODEL_NAME}_{string_timesteps}_{agent_index}.gif"
    ani.save(animation_path, writer='pillow', fps=100)

    # # Show the animation
    # plt.show()

    env.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unsupervised_training.py <index_number> <training_data_name> <model>")
        sys.exit(1)

    agent_index = sys.argv[1]
    TRAINING_DATA_NAME = sys.argv[2]
    MODEL_NAME = sys.argv[3]

    if TRAINING_DATA_NAME == 'None':
        UNSUPERVISED = True
    else:
        UNSUPERVISED = False

    TRAINING_FILEPATH = "user_data_processed_for_bc/"
    TRAINING_FILEPATH += TRAINING_DATA_NAME + "_bc_data.obj"

    if MODEL_NAME == "PPO":
        MODEL_CLASS = PPO
    elif MODEL_NAME == "DQN":
        MODEL_CLASS = DQN
    else:
        MODEL_CLASS = SAC

    print(f"Starting training for agent {agent_index} with model type: {MODEL_NAME} and using {TRAINING_DATA_NAME} training data.")

    main(agent_index)
