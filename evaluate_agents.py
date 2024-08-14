from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from scipy.stats import mannwhitneyu, shapiro, levene, ttest_ind


TIMESTEP_INCREMENT = 1000000
TIMESTEPS = 1000000
UNSUPERVISED = False
RETRAINING = False
MODEL_NAME = "PPO"
MODEL_CLASS = PPO
POLICY = "CnnPolicy"
TRAINING_DATA_NAME = "amalgam"
# TRAINING_DATA_NAME = "expert_distance"
# TRAINING_DATA_NAME = "nonexpert_distance"
LEVEL_CHANGE = "random"

TRAINING_FILEPATH = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/user_data_processed_for_bc/"
TRAINING_FILEPATH += TRAINING_DATA_NAME + "_bc_data.obj"
# LEVEL_CHANGE = "single_level_Level1-1"

# training_data_name = "expert_score"
# training_data_name = "nonexpert_score"

# training_data_name = "slower"
# training_data_name = "faster"

ALL_LEVELS = ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]
TEST_EPISODES = 1000
AGENT_INDICES = list(range(5))

def evaluate_model(model, env, n_episodes=10):
    """
    Evaluates the model by running it in the environment for n_episodes.
    Returns the mean reward over these episodes.
    """
    episode_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, info = env.step(int(action))
            total_reward += reward
        episode_rewards.append(total_reward)
    mean_reward = sum(episode_rewards) / n_episodes
    return mean_reward, episode_rewards

def get_results(model, env):
    mean_rewards = {}
    rewards = {}

    for level in ALL_LEVELS:
        env.level = level

        mean_reward, rewards = evaluate_model(model, env, TEST_EPISODES)

        mean_rewards[level] = mean_reward
        rewards[level] = rewards

    return mean_rewards

def main(agent_index):

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    log_dir = f"training_logs/level_change_{LEVEL_CHANGE}/"

    env = gym.make('MarioEnv-v0')

    string_timesteps = f"{int(TIMESTEPS/1000)}k"

    # env.set_record_option("test_bk2s/.")

    env.level_change_type = "No Change"

    results = {}

    for level in ALL_LEVELS:
        env.level = level
        results[level] = {"unsupervised": [], "supervised": []}

        unsupervised_results = []
        supervised_results = []
        print(level)

        for agent_index in AGENT_INDICES:
            unsupervised_log_dir = f"{log_dir}unsupervised/"
            unsupervised_model_path = unsupervised_log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}"

            supervised_log_dir = f"{log_dir}supervised/{TRAINING_DATA_NAME}/"
            supervised_model_path = supervised_log_dir + f"{string_timesteps}_{MODEL_NAME}_{agent_index}"

            unsupervised_model = MODEL_CLASS.load(unsupervised_model_path, env, verbose=1)
            supervised_model = MODEL_CLASS.load(supervised_model_path, env, verbose=1)

            mean_reward, rewards = evaluate_model(unsupervised_model, env, TEST_EPISODES)

            results[level]["unsupervised"].append(mean_reward)
            unsupervised_results.append(rewards)

            mean_reward, rewards = evaluate_model(supervised_model, env, TEST_EPISODES)
            
            results[level]["supervised"].append(mean_reward)
            supervised_results.append(rewards)
            
        # Check the normality:
        # Run the Shapiro-Wilk test
        stat, p_value = shapiro(results[level]["unsupervised"])

        print(f"Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

        stat, p_value = shapiro(results[level]["supervised"])

        print(f"Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

        # Flatten the lists to combine results from all agents in each group
        unsupervised_flat = [reward for agent in unsupervised_results for reward in agent]
        supervised_flat = [reward for agent in supervised_results for reward in agent]

        # Perform Levene's Test
        stat, p_value = levene(unsupervised_flat, supervised_flat)

        print(f"Levene’s Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")        

        # Example: Assuming 'unsupervised_results' and 'supervised_results' are your performance metrics
        t_stat, p_value = ttest_ind(unsupervised_results, supervised_results)

        print(f"T-statistic: {t_stat}, P-value: {p_value}")

        # Example: Mann-Whitney U Test
        u_stat, p_value = mannwhitneyu(unsupervised_results, supervised_results)

        print(f"U-statistic: {u_stat}, P-value: {p_value}")


    # # Run a 1000 timesteps to generate a gif just as a check measure (not actual evaluation)
    # # Create a figure and axis
    # fig, ax = plt.subplots()

    # # Initialize the plot with an empty image
    # im = ax.imshow(np.zeros((84, 84)), cmap='gray', vmin=0, vmax=255)

    # global OBS
    # OBS, _ = env.reset(options={"level": "Level1-1"})

    # # Function to update the image
    # def update_img(frame):
    #     global OBS
    #     action, _states = model.predict(OBS)

    #     OBS, _, done, _, _ = env.step(int(action))
    #     obs = np.array(OBS).squeeze()

    #     if done:
    #         OBS, _ = env.reset(options={"level": "Level1-1"})
    #     # Update the image data
    #     im.set_data(obs)
    #     return [im]

    # # Create an animation
    # ani = animation.FuncAnimation(fig, update_img, frames=1000, blit=True, interval=100)
    # animation_path = "test_gifs/"

    # if UNSUPERVISED:
    #     animation_path += "unsupervised"
    # else:
    #     animation_path += f"supervised_w_{TRAINING_DATA_NAME}"

    # animation_path += f"_{MODEL_NAME}_{string_timesteps}_{agent_index}.gif"
    # ani.save(animation_path, writer='pillow', fps=100)

    # # Show the animation
    # plt.show()

    env.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python unsupervised_training.py <index_number>")
        sys.exit(1)

    agent_index = sys.argv[1]
    main(agent_index)
