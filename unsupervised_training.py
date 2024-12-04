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
import torch
import time
import psutil
import os

import behavioural_cloning

TIMESTEP_INCREMENT = 1000000
TIMESTEPS = 10000000
RETRAINING = False
POLICY = "CnnPolicy"
LEVEL_CHANGE = "random"
NUM_ACTIONS = 13

EVALUATION_FREQ = 100000

class ActionDistributionEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=5, verbose=0, **kwargs):
        super(ActionDistributionEvalCallback, self).__init__(
            eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, verbose=verbose, **kwargs
        )
        self.action_distributions = []

    def _on_step(self) -> bool:
        # Perform evaluation when eval_freq is met
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            action_distribution = self._track_action_distribution()
            self.action_distributions.append(action_distribution)
            
            # Log action distribution to TensorBoard
            self._log_action_distribution(action_distribution)
            
            if self.verbose > 0:
                print(f"Action distribution during evaluation: {action_distribution}")
        return super(ActionDistributionEvalCallback, self)._on_step()

    def _track_action_distribution(self):
        # Track action distribution over evaluation episodes
        action_count = {}
        for i in range(NUM_ACTIONS):
            action_count[int(i)] = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                action_count[int(action)] = action_count.get(int(action), 0) + 1
                obs, _, done, _ = self.eval_env.step(action)
        return action_count

    def _log_action_distribution(self, action_distribution):
        # Log as scalars
        for action, count in action_distribution.items():
            self.logger.record(f"eval/action_count_{action}", count)

        # Log as a histogram (requires flattening the distribution)
        actions = []
        for action, count in action_distribution.items():
            actions.extend([action] * count)
        self.logger.record("eval/action_distribution_histogram", np.array(actions))

def test_ani(env, model, string_timesteps):
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

def compare_params(dict1, dict2): 
    for key in dict1.keys(): 
        if not torch.equal(dict1[key], dict2[key]): 
            return False 
        
    return True

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

    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])

    # # Stack frames to allow temporal information to be captured
    # env = VecFrameStack(env, n_stack=1)

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
            # model = MODEL_CLASS(POLICY, env, verbose=1, tensorboard_log=log_dir)
            params = {'learning_rate': 0.0009422578032986744, 'n_epochs': 15, 'batch_size': 1043, 'rl_learning_rate': 0.00019810071043939884, 'n_steps': 608, 'gamma': 0.9632436461255943, 'gae_lambda': 0.8365664231014514, 'ent_coef': 0.0014620665698483156, 'clip_range': 0.2873084366251664, 'vf_coef': 0.20790175758482327}
# model = PPO(CustomCnnPolicy, env, verbose=1)
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

        if not UNSUPERVISED:
            # Save initial model weights before behavior cloning
            bc_model_path = f"{model_path}_bc"
            model.save(f"{bc_model_path}_initial_weights")

            print(model.policy)

            # bc_model, bc_model_path = behavioural_cloning.behavioural_cloning_with_imitation(env, model_path, TRAINING_FILEPATH)
            # model = MODEL_CLASS.load(bc_model_path, env, verbose=1, tensorboard_log=log_dir)
            # print(env.action_space)
            print("behaviour cloning starting")
            model = behavioural_cloning.behavioural_cloning(MODEL_NAME, model, env, TRAINING_FILEPATH, bc_model_path) #, params["learning_rate"], params["n_epochs"], params["batch_size"])
            
            initial_model = PPO.load(f"{bc_model_path}_initial_weights", env=env)
            bc_trained_model = PPO.load(bc_model_path, env=env)
            
            policy_params_before = initial_model.policy.state_dict()
            
            # Access the updated policy network parameters
            policy_params_after = bc_trained_model.policy.state_dict()
            
            # Compare specific weights (e.g., first layer weights)
            if compare_params(policy_params_before, policy_params_after):
                print("Weights have not changed.")
            else:
                print("Weights have been updated!")

    # print(model.policy)

    if UNSUPERVISED:
        name_prefix = f"unsupervised_{MODEL_NAME}_{string_timesteps}_{agent_index}"
    else:
        name_prefix = f"supervised_{MODEL_NAME}_with_{TRAINING_DATA_NAME}_{string_timesteps}_{agent_index}"

    # Configure logging
    tmp_path = log_dir + f"/{name_prefix}/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    test_ani(env, model, "pre_unsupervised_training")

    # Define callbacks
    # eval_callback = EvalCallback(env, best_model_save_path=f"{log_dir}{name_prefix}/",
    #                             log_path=log_dir, eval_freq=100000,
    #                             deterministic=True, render=False)    
    eval_callback = ActionDistributionEvalCallback(env, 
                                                   eval_freq=100000, 
                                                   verbose=1, 
                                                   best_model_save_path=f"{log_dir}{name_prefix}/", 
                                                   log_path=log_dir)

    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=log_dir,
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

    test_ani(env, model, string_timesteps)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unsupervised_training.py <index_number> <training_data_name> <model>")
        sys.exit(1)

    # Start timing
    start_time = time.time()

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

    end_time = time.time()

    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    print(f"Runtime: {end_time - start_time} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")