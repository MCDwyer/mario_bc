import optuna
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from gymnasium.envs.registration import register
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import behavioural_cloning
from GymEnvs.retro_env_wrapper import DiscreteToBoxWrapper
import sys
import logging

TUNING_FILEPATH = "model_tuning_outputs/"

# TRAINING_FILEPATH_PREFIX = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/user_data_processed_for_bc/"

# TRAINING_DATA_NAME = "amalgam"
# TRAINING_DATA_NAME = "expert_distance"
# TRAINING_DATA_NAME = "nonexpert_distance"

# TRAINING_FILEPATH = TRAINING_FILEPATH_PREFIX + TRAINING_DATA_NAME + "_bc_data.obj"


def make_env(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init


def objective(trial):
    np.random.seed(42) # so same for all tuning tests??

    env = gym.make('MarioEnv-v0')

    env.reset()

    if USE_BC:
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15, 20, 25, 50, 100, 200, 500])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048])

        # Frame stacking
        n_stack = trial.suggest_categorical('n_stack', [1, 2, 4])

        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        env = VecFrameStack(env, n_stack=n_stack)
        model = PPO('CnnPolicy', env)

    else:
        try:
            if MODEL_CLASS == PPO:
                # Hyperparameters to tune
                rl_learning_rate = trial.suggest_float('rl_learning_rate', 1e-5, 1e-3, log=True)
                rl_batch_size = trial.suggest_categorical('rl_batch_size', [128, 256, 512])
                clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
                rl_n_epochs = trial.suggest_int('rl_n_epochs', 3, 10)
                gamma = trial.suggest_float('gamma', 0.9, 0.9999)
                gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)

                # Frame stacking
                n_stack = trial.suggest_categorical('n_stack', [1, 2, 4])

                env = Monitor(env)
                env = DummyVecEnv([lambda: env])

                # Stack frames to allow temporal information to be captured
                env = VecFrameStack(env, n_stack=n_stack)

                # Create the PPO model with CnnPolicy
                model = PPO('CnnPolicy', env,
                            learning_rate=rl_learning_rate,
                            batch_size=rl_batch_size,
                            clip_range=clip_range,
                            n_epochs=rl_n_epochs,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            # policy_kwargs=dict(
                            #     features_extractor_kwargs=dict(
                            #         net_arch=cnn_arch
                            #     )
                            # ),
                            verbose=0)
            elif MODEL_CLASS == DQN:
                # DQN hyperparameters to tune
                rl_learning_rate = trial.suggest_float('rl_learning_rate', 1e-5, 1e-3, log=True)
                rl_batch_size = trial.suggest_categorical('rl_batch_size', [32, 64, 128, 256])
                buffer_size = trial.suggest_int('buffer_size', 50000, 1000000)
                target_update_interval = trial.suggest_int('target_update_interval', 1000, 10000)
                exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.1, 1.0)
                exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
                exploration_fraction = trial.suggest_float('exploration_fraction', 0.01, 0.5)
                
                # CNN architecture: list of tuples (filters, kernel_size, stride)
                # cnn_arch = trial.suggest_categorical('cnn_arch', [
                #     [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                #     [(16, 8, 4), (32, 4, 2)]
                # ])

                # Frame stacking (important for Atari-like environments)
                n_stack = trial.suggest_categorical('n_stack', [1, 2, 4])

                env = Monitor(env)
                env = DummyVecEnv([lambda: env])

                # Stack frames to allow temporal information to be captured
                env = VecFrameStack(env, n_stack=n_stack)

                # Create the DQN model with CnnPolicy
                model = DQN('CnnPolicy', env,
                            learning_rate=rl_learning_rate,
                            batch_size=rl_batch_size,
                            buffer_size=buffer_size,
                            target_update_interval=target_update_interval,
                            exploration_initial_eps=exploration_initial_eps,
                            exploration_final_eps=exploration_final_eps,
                            exploration_fraction=exploration_fraction,
                            # policy_kwargs=dict(
                            #     features_extractor_kwargs=dict(
                            #         net_arch=cnn_arch
                            #     )
                            # ),
                            verbose=0)
            elif MODEL_CLASS == SAC:
                rl_learning_rate = trial.suggest_float('rl_learning_rate', 1e-5, 1e-3, log=True)
                rl_batch_size = trial.suggest_categorical('rl_batch_size', [128, 256, 512, 1024])
                buffer_size = trial.suggest_int('buffer_size', 100000, 1000000)
                gamma = trial.suggest_float('gamma', 0.9, 0.9999)
                tau = trial.suggest_float('tau', 1e-4, 0.005, log=True)
                learning_starts = trial.suggest_int('learning_starts', 1000, 20000)

                # CNN architecture: list of tuples (filters, kernel_size, stride)
                # cnn_arch = trial.suggest_categorical('cnn_arch', [
                #     [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                #     [(16, 8, 4), (32, 4, 2)]
                # ])

                # Frame stacking
                n_stack = trial.suggest_categorical('n_stack', [1, 2, 4])

                env = DiscreteToBoxWrapper(env)

                env = Monitor(env)
                env = DummyVecEnv([lambda: env])

                # Stack frames to allow temporal information to be captured
                env = VecFrameStack(env, n_stack=n_stack)

                # Create the SAC model with CnnPolicy
                model = SAC('CnnPolicy', env,
                    learning_rate=rl_learning_rate,
                    batch_size=rl_batch_size,
                    buffer_size=buffer_size,
                    gamma=gamma,
                    tau=tau,
                    # policy_kwargs=dict(
                    #     features_extractor_kwargs=dict(
                    #         net_arch=cnn_arch
                    #     )
                    # ),
                    learning_starts=learning_starts,
                    verbose=0)

    # except Exception as err:
    #     print(f"Unexpected {err}, {type(err)}")
    #     raise
        except ValueError as err:
            print(err)
            return -10000

    if USE_BC:
        bc_model_path = f"{MODEL_NAME}_tuning_bc"
        start_time = time.time()

        model = behavioural_cloning.behavioural_cloning(MODEL_NAME, model, env, TRAINING_FILEPATH, bc_model_path, lr=learning_rate, num_epochs=num_epochs, batch_size=batch_size, n_stack=n_stack)

        print(f"Took {int(time.time() - start_time)}s to run behavioural cloning.")
        
        # Evaluate the model
        mean_reward, _ = evaluate_model(model, env)
        env.close()

        return mean_reward

    start_time = time.time()
    # # Train the model for a certain number of timesteps
    model.learn(total_timesteps=1000000)

    print(f"Took {int(time.time() - start_time)}s to run 10k timesteps.")
    # Evaluate the model
    mean_reward, _ = evaluate_model(model, env)
    env.close()

    # Return the mean reward as the objective value to be maximized
    return mean_reward


def evaluate_model(model, env, n_episodes=1000):
    """
    Evaluates the model by running it in the environment for n_episodes.
    Returns the mean reward over these episodes.
    """
    episode_rewards = []
    for _ in range(n_episodes):
        # obs, _ = env.reset()
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    mean_reward = sum(episode_rewards) / n_episodes
    return mean_reward, episode_rewards


def run_study(filepath):

    print("Tuning starting")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000, n_jobs=1)

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    output_file_path = f"{filepath}best_trial_output.txt"

    # Open the file in write mode
    with open(output_file_path, 'w') as f:
        # Redirect the output to the file
        f.write('Best trial:\n')
        
        # Get the best trial
        trial = study.best_trial

        # Write the value
        f.write('  Value: {}\n'.format(trial.value))

        # Write the parameters
        f.write('  Params: \n')
        for key, value in trial.params.items():
            f.write('    {}: {}\n'.format(key, value))

    optuna.visualization.plot_optimization_history(study).write_image(f"{filepath}optimisation_hist.png")
    optuna.visualization.plot_optimization_history(study).write_html(f"{filepath}optimisation_hist.html")
    # optuna.visualization.plot_optimization_history(study).show()

    optuna.visualization.plot_param_importances(study).write_image(f"{filepath}param_importance.png")
    optuna.visualization.plot_param_importances(study).write_html(f"{filepath}param_importance.html")
    # optuna.visualization.plot_param_importances(study).show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unsupervised_training.py <training_data_name> <model>")
        sys.exit(1)

    # Define the environment
    register(
        id='MarioEnv-v0',
        entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
    )

    TRAINING_DATA_NAME = sys.argv[1]

    USE_BC = not (TRAINING_DATA_NAME == "None")

    MODEL_NAME = sys.argv[2]

    if MODEL_NAME == "PPO":
        MODEL_CLASS = PPO
    elif MODEL_NAME == "DQN":
        MODEL_CLASS = DQN
    else:
        MODEL_CLASS = SAC


    # if TRAINING_DATA_NAME == 'None':
    #     UNSUPERVISED = True
    # else:
    #     UNSUPERVISED = False

    TRAINING_FILEPATH = "user_data_processed_for_bc/"
    TRAINING_FILEPATH += TRAINING_DATA_NAME + "_bc_data.obj"

    # Configure logging to write to the file
    logging.basicConfig(filename=f"nohup/logging_{MODEL_NAME}.txt", level=logging.INFO)

    if USE_BC:
        filepath = TUNING_FILEPATH + f"{MODEL_NAME}_supervised_{TRAINING_DATA_NAME}_"
        run_study(filepath)
    else:
        filepath = TUNING_FILEPATH + f"{MODEL_NAME}_unsupervised_"
        run_study(filepath)
