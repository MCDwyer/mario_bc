from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import json
import sys

TIMESTEPS = 100
UNSUPERVISED = True
MODEL_NAME = "PPO"
MODEL_CLASS = PPO
POLICY = "CnnPolicy"
TRAINING_DATA_NAME = "amalgam"
LEVEL_CHANGE = "random"

# training_data_name = "expert_dist"
# training_data_name = "nonexpert_dist"

# training_data_name = "expert_score"
# training_data_name = "nonexpert_score"

# training_data_name = "slower"
# training_data_name = "faster"

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

    string_timesteps = f"{int(TIMESTEPS/1000)}k"

    model = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=log_dir)

    # Configure logging
    tmp_path = log_dir + "/tmp/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    if UNSUPERVISED:
        name_prefix = f"unsupervised_{MODEL_NAME}_{agent_index}"
    else:
        name_prefix = f"supervised_{MODEL_NAME}_with_{TRAINING_DATA_NAME}_{agent_index}"

    # Define callbacks
    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=500,
                                deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir,
                                            name_prefix=name_prefix)


    # Train the model
    model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, checkpoint_callback])
    model.save(log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}")

    # save the history from the env
    history_json_filepath = log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}.json"
    with open(history_json_filepath, "w") as outfile:
        json.dump(env.history, outfile)

    # env.set_record_option("test_bk2s/.")

    # Load the trained agent to check it's saved properly
    model = MODEL_CLASS.load(log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}")

    # Run a 1000 timesteps to generate a gif just as a check measure (not actual evaluation)
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with an empty image
    im = ax.imshow(np.zeros((84, 84)), cmap='gray', vmin=0, vmax=255)

    global OBS
    _, OBS = env.initialise_retro_env("Level1-1")

    # Function to update the image
    def update_img(frame):
        global OBS
        action, _states = model.predict(OBS)

        OBS, _, done, _, _ = env.step(int(action))
        obs = np.array(OBS).squeeze()

        if done:
            _, OBS = env.reset()

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
    if len(sys.argv) != 2:
        print("Usage: python unsupervised_training.py <index_number>")
        sys.exit(1)

    agent_index = sys.argv[1]
    main(agent_index)
