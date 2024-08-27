from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from scipy.stats import mannwhitneyu, shapiro, levene, ttest_ind, ttest_rel, wilcoxon, kruskal, f_oneway
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle
import os

TIMESTEP_INCREMENT = 1000000
TIMESTEPS = 5000000
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

def evaluate_model(model, env, level, record_path, n_episodes=10):
    """
    Evaluates the model by running it in the environment for n_episodes.
    Returns the mean reward over these episodes.
    """
    episode_rewards = []
    episode_trajectories = []
    episode_info = []
 
    for i in range(n_episodes):
        obs, _ = env.reset(options={"level": level, "record_option": record_path})
        done = False
        total_reward = 0.0
        trajectory = []
        scores = []
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, info = env.step(int(action))

            x = info["x_frame"]*256 + info["x_position_in_frame"]
            y = ((info["y_frame"]*256) + info["y_position_in_frame"])
            trajectory.append([x, y])
            scores.append(info["score"])

            total_reward += reward

        print(f"Episode {i} reward: {total_reward}")
        episode_rewards.append(total_reward)
        episode_trajectories.append(np.array(trajectory))
        episode_info.append({'score': np.max(np.array(scores)), 'max_dist': np.max(np.array(trajectory)[:, 0]), 'num_timesteps': len(trajectory)})

    mean_reward = sum(episode_rewards) / n_episodes

    return mean_reward, episode_rewards, episode_trajectories, episode_info

def get_results(model, env):
    mean_rewards = {}
    rewards = {}
    trajectories = {}

    for level in ALL_LEVELS:
        env.level = level

        mean_reward, rewards, episode_trajectories, episode_info = evaluate_model(model, env, level, ".", TEST_EPISODES)

        trajectories[level] = episode_trajectories
        mean_rewards[level] = mean_reward
        rewards[level] = rewards

    return mean_rewards, rewards, trajectories

def two_agent_statistical_tests(unsupervised_results, supervised_results):
    # Check the normality:
    # Run the Shapiro-Wilk test
    stat, p_value = shapiro(unsupervised_results)
    print(f"Unsupervised: Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

    stat, p_value = shapiro(supervised_results)
    print(f"Supervised Shapiro-Wilk Test Statistic: {stat}, P-value: {p_value}")

    # Perform Levene's Test
    stat, p_value = levene(unsupervised_results, supervised_results)

    print(f"Levene’s Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")        

    # Example: Assuming 'unsupervised_results' and 'supervised_results' are your performance metrics
    t_stat, p_value = ttest_ind(unsupervised_results, supervised_results)

    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Example: Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(unsupervised_results, supervised_results)

    print(f"U-statistic: {u_stat}, P-value: {p_value}")

    # Perform paired t-test
    stat, p_value = ttest_rel(unsupervised_results, supervised_results)

    print(f"Paired t-test Statistic: {stat:.4f}, P-value: {p_value:.4f}")

    stat, p_value = wilcoxon(unsupervised_results, supervised_results)

    print(f"Wilcoxon Signed-Rank Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")

def four_agent_statistical_tests(agent_1, agent_2, agent_3, agent_4, mean_rewards=True):

    # Perform Levene's Test across four groups
    stat, p_value = levene(agent_1, agent_2, agent_3, agent_4)

    print(f"Levene’s Test Statistic (on all 4): {stat}, P-value: {p_value}")

    if mean_rewards:
        print("One way ANOVA (best on means):")
        f_stat, p_value = f_oneway(agent_1, agent_2, agent_3, agent_4)

        print(f"F-statistic: {f_stat}, P-value: {p_value}")

        # print("If ANOVA shows significant results, use Tukey’s HSD (Honestly Significant Difference) test to identify which groups are different.")
        # # Combine all rewards into a single array
        # all_rewards = agent_1 + agent_2 + agent_3 + agent_4
        # # Create labels for each group
        # groups = (['Agent1'] * len(agent_1) +
        #         ['Agent2'] * len(agent_2) +
        #         ['Agent3'] * len(agent_3) +
        #         ['Agent4'] * len(agent_4))

        # # Perform Tukey's HSD test
        # tukey = pairwise_tukeyhsd(endog=all_rewards, groups=groups, alpha=0.05)
        # print(tukey)
    else:
        print("Kruskal-Wallis H test")
        # Perform Kruskal-Wallis H Test
        h_stat, p_value = kruskal(agent_1, agent_2, agent_3, agent_4)

        print(f"Kruskal-Wallis H-statistic: {h_stat}, P-value: {p_value}")

    print()
    print()

def load_model_get_results(dir_path, string_timesteps, env, level_results, results, agent_index, level, record_path):
    model_path = dir_path + f"{string_timesteps}_{MODEL_NAME}_{agent_index}"

    model = MODEL_CLASS.load(model_path, env, verbose=1)

    mean_reward, rewards, episode_trajectories, episode_info = evaluate_model(model, env, level, record_path, TEST_EPISODES)

    level_results.append(mean_reward)

    if results is None:
        results = np.array(rewards)
    else:
        results = np.vstack((results, np.array(rewards)))
        
    return level_results, results, episode_trajectories, episode_info

def combine_level_data(results, agent_name, levels=ALL_LEVELS):
    all_means = None
    all_data = None

    for level in results:
        if level in levels:
            if all_means is None:
                all_means = np.array(results[level][agent_name]["means"])
            else:
                all_means = np.vstack((all_means, np.array(results[level][agent_name]["means"])))

            if all_data is None:
                all_data = np.array(results[level][agent_name]["all rewards"])
            else:
                all_data = np.vstack((all_data, np.array(results[level][agent_name]["all rewards"])))

    return all_means, all_data

def main():

    register(
            id='MarioEnv-v0',
            entry_point='GymEnvs.retro_env_wrapper:MarioEnv',
        )

    log_dir = f"training_logs/level_change_{LEVEL_CHANGE}/"

    env = gym.make('MarioEnv-v0')

    string_timesteps = f"{int(TIMESTEPS/1000)}k"

    # env.set_record_option("test_bk2s/.")

    env.level_change_type = "No Change"

    results_dict = {}

    dir_paths = {"unsupervised": f"{log_dir}unsupervised/", 
                 "supervised - amalgam": f"{log_dir}supervised/amalgam/", 
                 "supervised - expert": f"{log_dir}supervised/expert_distance/", 
                 "supervised - nonexpert": f"{log_dir}supervised/nonexpert_distance/"}

    dir_paths = {"supervised - amalgam": f"{log_dir}supervised/amalgam/"}

    trajectories = {}
    info = {}

    for level in ALL_LEVELS:
        env.level = level
        results_dict[level] = {}
        print(level)

        for agent_name in dir_paths:
            print(agent_name)
            results_dict[level][agent_name] = {"means": []}

            means_list = results_dict[level][agent_name]["means"]

            trajectories[agent_name] = {}
            info[agent_name] = {}

            results = None

            for agent_index in AGENT_INDICES:
                record_path = f"test_bk2s/{level}/{agent_name}/{agent_index}/"
                os.makedirs(record_path, exist_ok=True)
                record_path += "."
                trajectories[agent_name][agent_index] = {}
                means_list, results, trajectories, ep_info = load_model_get_results(dir_paths[agent_name], string_timesteps, env, means_list, results, agent_index, level, record_path)
                
                trajectories[agent_name][agent_index][level] = trajectories
                info[agent_name][agent_index] = {}
                info[agent_name][agent_index][level] = ep_info

            results_dict[level][agent_name]["all rewards"] = results
        
    with open('evaluations/trajectories.obj', 'wb') as handle:
        pickle.dump(trajectories, handle)

    with open('evaluations/infos.obj', 'wb') as handle:
        pickle.dump(info, handle)

    with open('evaluations/results.obj', 'wb') as handle:
        pickle.dump(results_dict, handle)

    combined_results = {}
    combined_results_training = {}
    combined_results_test = {}

    training_levels = ["Level1-1", "Level2-1", "Level4-1", "Level5-1", "Level6-1", "Level8-1"]
    test_levels = ["Level3-1", "Level7-1"]

    # run statistical tests on two agents
    for i, agent_name_1 in enumerate(dir_paths):
        for j, agent_name_2, in enumerate(dir_paths):        
            combined_results[agent_name_2] = {}
            combined_results[agent_name_2]["means"], combined_results[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2)

            combined_results_training[agent_name_2] = {}
            combined_results_training[agent_name_2]["means"], combined_results_training[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2, training_levels)

            combined_results_test[agent_name_2] = {}
            combined_results_test[agent_name_2]["means"], combined_results_test[agent_name_2]["all rewards"] = combine_level_data(results, agent_name_2, test_levels)

            if i == j:
                # skip as same agent
                continue

            print(f"2 agent tests on '{agent_name_1}' and '{agent_name_2}'")

            for level in results_dict:
                print(f"Per level tests on {level}")

                print("Tests on the mean rewards:")
                two_agent_statistical_tests(results_dict[level][agent_name_1]["means"], results_dict[level][agent_name_2]["means"])
                print()

                print("Tests on all the rewards:")
                two_agent_statistical_tests(results_dict[level][agent_name_1]["all rewards"], results_dict[level][agent_name_2]["all rewards"])
                print()
                print()


            print("Tests on all levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results[agent_name_1]["means"], combined_results[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results[agent_name_1]["all rewards"], combined_results[agent_name_2]["all rewards"])
            print()
            print()

            print("Tests on training levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results_training[agent_name_1]["means"], combined_results_training[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results_training[agent_name_1]["all rewards"], combined_results_training[agent_name_2]["all rewards"])
            print()
            print()

            print("Tests on test levels combined:")
            print("Tests on the mean rewards:")
            two_agent_statistical_tests(combined_results_test[agent_name_1]["means"], combined_results_test[agent_name_2]["means"])
            print()

            print("Tests on all the rewards:")
            two_agent_statistical_tests(combined_results_test[agent_name_1]["all rewards"], combined_results_test[agent_name_2]["all rewards"])
            print()
            print()

    # more than two agent tests
    if len(dir_paths) > 2:
        
        for level in results_dict:
            print(f"Per level tests on {level}:")

            print("Tests on the mean rewards:")
            four_agent_statistical_tests(results_dict[level]["unsupervised"]["means"], results_dict[level]["supervised - amalgam"]["means"], results_dict[level]["supervised - expert"]["means"], results_dict[level]["supervised - nonexpert"]["means"])

            print("Tests on all the rewards:")
            four_agent_statistical_tests(results_dict[level]["unsupervised"]["all rewards"], results_dict[level]["supervised - amalgam"]["all rewards"], results_dict[level]["supervised - expert"]["all rewards"], results_dict[level]["supervised - nonexpert"]["all rewards"])

            print()
            print()

        print("Tests on all levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results["unsupervised"]["means"], combined_results["supervised - amalgam"]["means"], combined_results["supervised - expert"]["means"], combined_results["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results["unsupervised"]["all rewards"], combined_results["supervised - amalgam"]["all rewards"], combined_results["supervised - expert"]["all rewards"], combined_results["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()
        
        print("Tests on training levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results_training["unsupervised"]["means"], combined_results_training["supervised - amalgam"]["means"], combined_results_training["supervised - expert"]["means"], combined_results_training["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results_training["unsupervised"]["all rewards"], combined_results_training["supervised - amalgam"]["all rewards"], combined_results_training["supervised - expert"]["all rewards"], combined_results_training["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()

        print("Tests on test levels combined:")
        print("Tests on the mean rewards:")
        four_agent_statistical_tests(combined_results_test["unsupervised"]["means"], combined_results_test["supervised - amalgam"]["means"], combined_results_test["supervised - expert"]["means"], combined_results_test["supervised - nonexpert"]["means"])
        print()

        print("Tests on all the rewards:")
        four_agent_statistical_tests(combined_results_test["unsupervised"]["all rewards"], combined_results_test["supervised - amalgam"]["all rewards"], combined_results_test["supervised - expert"]["all rewards"], combined_results_test["supervised - nonexpert"]["all rewards"], mean_rewards=False)
        print()
        print()

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
    main()
