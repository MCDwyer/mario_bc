from stable_baselines3 import PPO, DQN, SAC
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from GymEnvs.retro_env_wrapper import DiscreteToBoxWrapper
import json
import sys
import time
# import psutil
import os
import re
from datetime import datetime

import behavioural_cloning

TIMESTEPS = 20000000
POLICY = "CnnPolicy"
LEVEL_CHANGE = "random"
NUM_ACTIONS = 13
ONLY_BC = False

RESUME_TRAINING = True
EVALUATION_FREQ = 200000
SAVE_FREQ = 500000

EXP_RUN_ID = "NO_EXPERIMENT_ID_SET"

MODEL_PARAMETERS = {}


class CheckpointWithOptimizerCallback(CheckpointCallback):
    def __init__(self,
            save_freq: int,
            save_path: str,
            name_prefix: str = "rl_model",
            save_replay_buffer: bool = False,
            save_vecnormalize: bool = False,
            verbose: int = 0):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, save_replay_buffer=save_replay_buffer, save_vecnormalize=save_vecnormalize, verbose=verbose)
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Use the original CheckpointCallback to save the model
            super()._on_step()

            # Save optimizer state
            optimizer_path = os.path.join(self.save_path, f"{self.name_prefix}_optimizer.pth")
            torch.save(self.model.policy.optimizer.state_dict(), optimizer_path)

            if self.verbose:
                print(f"Optimizer state saved: {optimizer_path}")

        return True  # Continue training

class ActionDistributionEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, verbose=0, **kwargs):
        super(ActionDistributionEvalCallback, self).__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose, **kwargs
        )
        self.action_distributions = []
        self.score_rewards = []
        self.dist_rewards = []
        self.eval_info = []

    def _on_step(self) -> bool:
        # Perform evaluation when eval_freq is met
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            action_distribution, score_reward, dist_reward = self._track_action_distribution()
            self.action_distributions.append(action_distribution)
            self.score_rewards.append(score_reward)
            self.dist_rewards.append(dist_reward)

            # Log action distribution to TensorBoard
            self._log_action_distribution(action_distribution)
            # self._log_rewards(score_reward, dist_reward)

            if self.verbose > 0:
                print(f"Action distribution during evaluation: {action_distribution}")
                # print(f"Score reward: {score_reward}, Distance reward: {dist_reward}")

        return super(ActionDistributionEvalCallback, self)._on_step()

    def _track_action_distribution(self):
        # Track action distribution over evaluation episodes
        action_count = {int(i): 0 for i in range(NUM_ACTIONS)}
        total_score_reward = 0
        total_dist_reward = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            # episode_score_reward = 0
            # episode_dist_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                action_count[int(action)] += 1
                obs, rewards, done, info = self.eval_env.step(action)

                # Assuming self.score_reward and self.dist_reward are part of `info`
                # episode_score_reward +=  self.eval_env.envs[0].score_reward #self.eval_env.score_reward
                # episode_dist_reward += self.eval_env.envs[0].dist_reward # self.eval_env.dist_reward

            # total_score_reward += episode_score_reward
            # total_dist_reward += episode_dist_reward
            # eval_info_dict = {"score": info["score"], "x_frame": info["x_frame"], "y_frame": info["y_frame"], "x_position_in_frame": info["x_position_in_frame"], "y_position_in_frame": info["y_position_in_frame"], "lives": info["lives"]}
            # self.eval_info.append(info)
            # print(info)

        avg_score_reward = total_score_reward / self.n_eval_episodes
        avg_dist_reward = total_dist_reward / self.n_eval_episodes

        return action_count, avg_score_reward, avg_dist_reward

    def _log_action_distribution(self, action_distribution):
        # Log as scalars
        for action, count in action_distribution.items():
            self.logger.record(f"eval/action_count_{action}", count)

        # Log as a histogram (requires flattening the distribution)
        actions = [action for action, count in action_distribution.items() for _ in range(count)]
        self.logger.record("eval/action_distribution_histogram", np.array(actions))
        # self.logger.record(f"eval/info", self.eval_info)

    def _log_rewards(self, score_reward, dist_reward):
        # Log score and distance rewards
        self.logger.record("eval/score_reward", score_reward)
        self.logger.record("eval/dist_reward", dist_reward)
        self.logger.record("eval/combined_reward", ((dist_reward+score_reward)/2))

def test_ani(env, model, string_timesteps):
    # Run a 1000 timesteps to generate a gif just as a check measure (not actual evaluation)
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with an empty image
    im = ax.imshow(np.zeros((84, 84)), cmap='gray', vmin=0, vmax=255)

    global OBS
    # OBS, _ = env.reset(options={"level": "Level1-1", "unprocessed_obs": True})
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
    animation_path = f"test_gifs/{EXP_RUN_ID}_"

    if UNSUPERVISED:
        animation_path += "unsupervised"
    else:
        animation_path += f"supervised_w_{TRAINING_DATA_NAME}"

    animation_path += f"_{MODEL_NAME}_{string_timesteps}_{agent_index}.gif"
    ani.save(animation_path, writer='pillow', fps=100)

    env.close()

def set_model_parameters(env, tmp_path):
    
    if USE_TUNED_PARAMS:
        if MODEL_NAME == "PPO":
            # if TRAINING_DATA_NAME == "None":
            #     tuned_params = {'rl_learning_rate': 0.00029297319487593513, 'rl_batch_size': 256, 'clip_range': 0.22222134777250774, 'rl_n_epochs': 9, 'gamma': 0.9072524259577757, 'gae_lambda': 0.9756642643449068}
            # elif TRAINING_DATA_NAME == "amalgam":
            #     {'learning_rate': 0.00010770959628176507, 'n_epochs': 15, 'batch_size': 1024, 'rl_learning_rate': 0.0002522572555081751, 'rl_batch_size': 512, 'clip_range': 0.3258857369171716, 'rl_n_epochs': 10, 'gamma': 0.9049832219200045, 'gae_lambda': 0.9714527136734751}

            tuned_params ={'learning_rate': 0.0034395379546273046, 'n_epochs': 20, 'batch_size': 256, 'rl_learning_rate': 0.0002875045106992828, 'rl_batch_size': 512, 'clip_range': 0.12113817344783348, 'rl_n_epochs': 8, 'gamma': 0.9019615617996611, 'gae_lambda': 0.9137237872887268}

            # if TRAINING_DATA_NAME == "None":
            #     # trial 58
            #     tuned_params = {'rl_learning_rate': 0.00031034455612923554, 'rl_batch_size': 256, 'clip_range': 0.124073405807537, 'rl_n_epochs': 9, 'gamma': 0.918331367428619, 'gae_lambda': 0.9776333153093636}
            #     # # PPO unsupervised trial 29 params:
            #     # tuned_params = {'rl_learning_rate': 0.00023042151449210175, 'rl_batch_size': 256, 'clip_range': 0.26000979068966745, 'rl_n_epochs': 10, 'gamma': 0.9068731710566416, 'gae_lambda': 0.9559237608041501}

            # else:
            #     if TRAINING_DATA_NAME == "amalgam":
            #         # amalgam trial 65
            #         tuned_params = {'learning_rate': 5.889196120332964e-05, 'n_epochs': 5, 'batch_size': 1024, 'rl_learning_rate': 0.0003512114556798816, 'rl_batch_size': 256, 'clip_range': 0.3170306890203502, 'rl_n_epochs': 10, 'gamma': 0.9100668567554057, 'gae_lambda': 0.9221857998682769}
                    
            #     elif TRAINING_DATA_NAME == "expert_distance":
            #         # expert trial 43
            #         # tuned_params = {'learning_rate': 0.0011044312279356197, 'n_epochs': 20, 'batch_size': 2048, 'rl_learning_rate': 0.0007320521243753693, 'rl_batch_size': 512, 'clip_range': 0.12697889784983063, 'rl_n_epochs': 7, 'gamma': 0.9070031522645945, 'gae_lambda': 0.9154795241511263}
            #         # expert trial 75
            #         tuned_params ={'learning_rate': 0.0034395379546273046, 'n_epochs': 20, 'batch_size': 256, 'rl_learning_rate': 0.0002875045106992828, 'rl_batch_size': 512, 'clip_range': 0.12113817344783348, 'rl_n_epochs': 8, 'gamma': 0.9019615617996611, 'gae_lambda': 0.9137237872887268}

            #     else:
            #         # PPO nonexpert trial 8 params:
            #         # tuned_params = {'learning_rate': 0.001080328979605993, 'n_epochs': 5, 'batch_size': 32, 'rl_learning_rate': 0.00020540126558519023, 'rl_batch_size': 128, 'clip_range': 0.25941208801980375, 'rl_n_epochs': 10, 'gamma': 0.9204149567949521, 'gae_lambda': 0.9786072151522366}
            #         # PPO nonexpert trial 71 params:
            #         tuned_params = {'learning_rate': 0.004498813171854396, 'n_epochs': 10, 'batch_size': 64, 'rl_learning_rate': 0.00014499719769529176, 'rl_batch_size': 32, 'clip_range': 0.29992876139327396, 'rl_n_epochs': 5, 'gamma': 0.9187856793929096, 'gae_lambda': 0.930200255152053}

            model = MODEL_CLASS(POLICY, 
                        env,
                        learning_rate=tuned_params['rl_learning_rate'],
                        batch_size=tuned_params['rl_batch_size'],
                        clip_range=tuned_params['clip_range'],
                        n_epochs=tuned_params['rl_n_epochs'],
                        gamma=tuned_params['gamma'],
                        gae_lambda=tuned_params['gae_lambda'],
                        verbose=1,
                        tensorboard_log=tmp_path,
                        )
            
            print(f"Model parameters set to: {tuned_params}")

            return model, tuned_params
    else:
        print(f"Model default parameters being used.")
        model = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=tmp_path)

        params = {}

        params["learning_rate"] = model.learning_rate
        params["n_epochs"] = model.n_epochs if MODEL_CLASS is PPO else 10
        params["batch_size"] = model.batch_size

        return model, params

def find_largest_steps_file(directory, model_prefix):
    # Regex pattern to extract the number of steps from the filename
    pattern = r"_([0-9]+)_steps\.zip$"
    
    largest_steps = 0
    largest_file = None

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a zip file and matches the pattern
        if filename.endswith(".zip") and filename.startswith(model_prefix):
            match = re.search(pattern, filename)
            if match:
                # Extract the number of steps as an integer
                steps = int(match.group(1))
                # Update if this file has the largest number of steps
                if steps > largest_steps:
                    largest_steps = steps
                    largest_file = filename
    
    return largest_file, largest_steps

def main(agent_index):

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    log_dir = f"experiments/{EXP_RUN_ID}/training_logs/level_change_{LEVEL_CHANGE}/"
    model_dir = f"experiments/{EXP_RUN_ID}/saved_models/level_change_{LEVEL_CHANGE}/"

    if UNSUPERVISED:
        log_dir = f"{log_dir}unsupervised/"
    else:
        log_dir = f"{log_dir}supervised/{TRAINING_DATA_NAME}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = gym.make('MarioEnv-v0')

    print("Creating env")

    if EXP_RUN_ID[-1].isdigit():
        # then load the cross validation levels for this index
        with open(f"cross_validation_levels/index_{EXP_RUN_ID[-1]}.pkl", "rb") as file:
            levels_to_use = pickle.load(file)

        env.change_level_set(levels_to_use)

        print(f"Cross validation set {EXP_RUN_ID[-1]} used, training levels set to: {levels_to_use}.")

    value = ""

    if EXP_RUN_ID[0].isdigit():
        for letter in EXP_RUN_ID:
            if letter.isdigit():
                value += letter
            else:
                break
        
    if value.isnumeric():
        value = -(int(value))
        env.set_death_penalty(value)
    else:
        env.set_death_penalty(None)
    
    env.set_reward_function(EXP_RUN_ID)
    
    if MODEL_PARAMETERS:
        env.n_stack = MODEL_PARAMETERS['n_stack']

    if MODEL_NAME == 'SAC':
        env = DiscreteToBoxWrapper(env)

    env = Monitor(env)

    string_timesteps = f"{int(TIMESTEPS/1000000)}M"

    sup_string = "unsupervised" if UNSUPERVISED else f"bc_{TRAINING_DATA_NAME}"
    model_path = model_dir + f"{MODEL_NAME}_{sup_string}_{string_timesteps}_agent_{agent_index}"

    print("Setting up model:\n")

    # Get the current date
    current_date = datetime.now()

    # Format the date as YYYYMMDD
    formatted_date = current_date.strftime("%Y%m%d")
    run_name = f"run_{formatted_date}" 

    name_prefix = f"{MODEL_NAME}_{string_timesteps}_agent_{agent_index}"

    # Configure logging
    tmp_path = log_dir + f"{name_prefix}/{run_name}/"

    model, params = set_model_parameters(env, tmp_path)#MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=tmp_path)

    if not UNSUPERVISED:
        bc_model_path = f"{model_path}_bc_only.zip"

        # check if bc_model exists
        if os.path.exists(bc_model_path):
            print(f"Model file '{bc_model_path}' exists, attempting to load in now:")
            model = MODEL_CLASS.load(bc_model_path, env, verbose=1, tensorboard_log=tmp_path, print_system_info=True)

            print("BC Trained model loaded successfully!")
        else:
            print(f"Model file '{bc_model_path}' does not exist.")
    
            model = behavioural_cloning.behavioural_cloning(MODEL_NAME, model, env.levels_to_use, TRAINING_DATA_NAME, bc_model_path, lr=params["learning_rate"], num_epochs=params["n_epochs"], batch_size=params["batch_size"]) #, params["learning_rate"], params["n_epochs"], params["batch_size"])
            test_ani(env, model, "post_bc_training")
            if ONLY_BC:
                sys.exit()

    print("RL Training Info")
    print(f"Model: {MODEL_NAME}, BC dataset: {TRAINING_DATA_NAME}, timesteps: {TIMESTEPS}")
    print(f"Model Parameters Used: {MODEL_PARAMETERS}\n")
    print(f"Model Parameters (stored on model):\n")

    print(f"\tLearning Rate: {model.learning_rate}")
    print(f"\tBatch Size: {model.batch_size}")

    print(f"\tReward Function: {env.reward_function}")

    print("Model Policy Structure:")
    print(model.policy)
    print()

    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    eval_callback = ActionDistributionEvalCallback(env, 
                                                   eval_freq=EVALUATION_FREQ, 
                                                   verbose=1, 
                                                   best_model_save_path=f"{log_dir}{name_prefix}/", 
                                                   log_path=tmp_path)

    checkpoint_callback = CheckpointWithOptimizerCallback(save_freq=SAVE_FREQ, save_path=log_dir,
                                            name_prefix=name_prefix, save_vecnormalize=True, save_replay_buffer=True)

    timesteps = TIMESTEPS

    if RESUME_TRAINING and os.path.exists(f"{log_dir}{name_prefix}"):
        # model was training already
        print("Model is attempting to resume training.")

        model_loaded = False

        # so then we find the most recent model
        largest_filename, steps = find_largest_steps_file(log_dir, name_prefix)
        if largest_filename is not None:
            prev_model_path = log_dir + "/" + largest_filename

            print(f"Attempting to load in model from: {prev_model_path}")
            try:
                model = MODEL_CLASS.load(prev_model_path, env, verbose=1, tensorboard_log=tmp_path, print_system_info=True)
                timesteps = TIMESTEPS - steps
                model_loaded = True

                optimizer_path = f"{log_dir}{name_prefix}_optimizer.pth"
                optimizer_state = torch.load(optimizer_path)
                model.policy.optimizer.load_state_dict(optimizer_state)
                print("Previous optimiser state has been loaded in.")
            except:
                model_loaded = False

        if not model_loaded:
            print("Model couldn't be loaded in, so will begin training from 0.")
        else:
            print("Model loaded in.")
            test_ani(env, model, f"resuming_training_from_{steps}_timesteps")

    # Train the model
    model.learn(total_timesteps=timesteps, callback=[eval_callback, checkpoint_callback], tb_log_name=name_prefix, reset_num_timesteps=False)
    model.save(model_path)
    print(f"Training finished and model saved to {model_path}.")

    print(f"Levels used during training: {env.levels_used}")

    # Load the trained agent to check it's saved properly
    model = MODEL_CLASS.load(model_path)

    test_ani(env, model, string_timesteps)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python unsupervised_training.py <index_number> <training_data_name> <model>")
        print("OR: python unsupervised_training.py <index_number> <training_data_name> <model> <experiment_run_id>")
        sys.exit(1)

    agent_index = sys.argv[1]
    TRAINING_DATA_NAME = sys.argv[2]
    MODEL_NAME = sys.argv[3]

    if len(sys.argv) == 5:
        EXP_RUN_ID = sys.argv[4]
        EXP_RUN_ID.replace(" ", "_")

        if "default" in EXP_RUN_ID.lower():
            USE_TUNED_PARAMS = False
        else:
            USE_TUNED_PARAMS = True

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

    print(f"Starting training for agent {agent_index} with model type: {MODEL_NAME} and using {TRAINING_DATA_NAME} training data.\n\n")

    main(agent_index)
